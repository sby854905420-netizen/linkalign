import gc
import logging
import os
import threading
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    SummaryIndex,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    get_response_synthesizer
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import VectorIndexRetriever, SummaryIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from prompts.PipelinePromptStore import *

from llms.ollama.ollamaModel import OllamaModel
from config import GPT_API
from typing import Union, List, Optional
import torch


module_logger = logging.getLogger(__name__)

class RagPipeLines:
    DEFAULT_MODEL = OllamaModel(model_name="ministral-3:8b")
    EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
    _EMBED_MODEL_CACHE = {}
    _EMBED_MODEL_LOCK = threading.Lock()
    _ACTIVE_EMBED_CACHE_KEY = None

    @classmethod
    def _get_preferred_device(cls, device: Optional[str] = None) -> str:
        if device:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def _set_active_embed_model(cls, embed_model, cache_key):
        Settings.embed_model = embed_model
        cls._ACTIVE_EMBED_CACHE_KEY = cache_key
        return embed_model

    @classmethod
    def ensure_embed_model(cls, embed_model_name: str = None, device: str = None):
        if embed_model_name is None:
            embed_model_name = cls.EMBED_MODEL_NAME

        if isinstance(embed_model_name, str) and embed_model_name.startswith("text-embedding-"):
            embed_model = OpenAIEmbedding(
                model=embed_model_name,
                api_key=GPT_API,
            )
            return cls._set_active_embed_model(embed_model, (embed_model_name, "openai"))

        preferred_device = cls._get_preferred_device(device)

        with cls._EMBED_MODEL_LOCK:
            cache_key = (embed_model_name, preferred_device)
            active_key = cls._ACTIVE_EMBED_CACHE_KEY
            if active_key == cache_key and active_key in cls._EMBED_MODEL_CACHE:
                return cls._set_active_embed_model(cls._EMBED_MODEL_CACHE[active_key], active_key)

            if cache_key in cls._EMBED_MODEL_CACHE:
                return cls._set_active_embed_model(cls._EMBED_MODEL_CACHE[cache_key], cache_key)

            try:
                embed_model = HuggingFaceEmbedding(
                    model_name=embed_model_name,
                    embed_batch_size=8,
                    device=preferred_device,
                )
                cls._EMBED_MODEL_CACHE[cache_key] = embed_model
                return cls._set_active_embed_model(embed_model, cache_key)
            except Exception as exc:
                error_msg = str(exc)
                is_meta_tensor_move_error = (
                    "Cannot copy out of meta tensor" in error_msg
                    or "to_empty()" in error_msg
                )
                if preferred_device == "cuda" and is_meta_tensor_move_error:
                    module_logger.warning(
                        "Encountered a meta-tensor device move error on CUDA. "
                        "Falling back to CPU for embedding model loading."
                    )
                    fallback_key = (embed_model_name, "cpu")
                    if fallback_key not in cls._EMBED_MODEL_CACHE:
                        cls._EMBED_MODEL_CACHE[fallback_key] = HuggingFaceEmbedding(
                            model_name=embed_model_name,
                            embed_batch_size=8,
                            device="cpu",
                        )
                    return cls._set_active_embed_model(cls._EMBED_MODEL_CACHE[fallback_key], fallback_key)
                raise RuntimeError(
                    f"Failed to initialize HuggingFace embedding model: {embed_model_name}"
                ) from exc

    @classmethod
    def release_embed_model(cls):
        with cls._EMBED_MODEL_LOCK:
            cached_models = list(cls._EMBED_MODEL_CACHE.values())
            cls._EMBED_MODEL_CACHE.clear()
            cls._ACTIVE_EMBED_CACHE_KEY = None
            Settings.embed_model = None

        for embed_model in cached_models:
            for attr_name in ("_model", "_tokenizer"):
                if hasattr(embed_model, attr_name):
                    setattr(embed_model, attr_name, None)
            del embed_model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @classmethod
    def build_index_from_source(
            cls,
            data_source: Union[str, List[str]],
            persist_dir: str = None,
            is_vector_store_exist: bool = False,
            llm=None,
            index_method: str = None,
            embed_model_name=None,  # 支持 OpenAI 与 HuggingFace 嵌入模型
            parser=None,
    ):
        
        # 1) 配置 LLM 与 embedding
        if llm is None:
            llm = cls.DEFAULT_MODEL
        Settings.llm = llm

        if embed_model_name is None:
            embed_model_name = cls.EMBED_MODEL_NAME

        cls.ensure_embed_model(embed_model_name=embed_model_name)

        valid_index_methods = {"SummaryIndex", "VectorStoreIndex"}
        if index_method not in valid_index_methods:
            index_method = None

        if parser is None:
            parser = SentenceSplitter(chunk_size=10000, chunk_overlap=0)

        # 2) 若本地索引已存在，直接加载并返回
        if is_vector_store_exist:
            if persist_dir is None:
                raise Exception("请保证 vector_store 存在时，persist_dir 非空！")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            return index
        

        # 3) 新建索引时，准备持久化目录
        if persist_dir is None:
            persist_dir = f"{cls.ROOT_PERSIST_DIR}/task{cls.COUNT}_vector_store"
        os.makedirs(persist_dir, exist_ok=True)

        # 4) 解析数据源形态（目录 / 单文件 / 多文件）
        is_multi_files = isinstance(data_source, list)
        is_str_path = isinstance(data_source, str)
        is_dir = Path(data_source).is_dir() if is_str_path else not is_multi_files

        # 5) 决定索引类型：默认目录走 Vector，文件走 Summary；显式 index_method 优先
        is_vector_store_method = is_dir
        if index_method is not None:
            is_vector_store_method = index_method == "VectorStoreIndex"

        # 6) 统一加载文档，减少重复分支
        if is_multi_files:
            documents = SimpleDirectoryReader(input_files=data_source).load_data()
        elif is_str_path and not is_dir:
            documents = SimpleDirectoryReader(input_files=[data_source]).load_data()
        else:
            documents = SimpleDirectoryReader(data_source).load_data()

        # 7) 构建索引并持久化
        if is_vector_store_method:
            index = VectorStoreIndex.from_documents(documents, transformations=[parser], show_progress=True)
        else:
            index = SummaryIndex.from_documents(documents, transformations=[parser], show_progress=True)
        index.storage_context.persist(persist_dir=persist_dir)

        return index

    @classmethod
    def get_query_engine(
            cls,
            index: Union[SummaryIndex, VectorStoreIndex] = None,
            query_template: str = None,
            similarity_top_k=5,
            node_ids: List[str] = None,
            **kwargs
    ):
        if index is None:
            raise Exception("输入 index 不能为空")
        if query_template is not None:
            query_template = PromptTemplate(query_template)
        else:
            query_template = PromptTemplate(DEFAULT_PROMPT_TEMPLATE)

        engine = None
        if type(index) == SummaryIndex:
            engine = index.as_query_engine(text_qa_template=query_template,
                                           similarity_top_k=similarity_top_k,
                                           **kwargs)
        elif type(index) == VectorStoreIndex:
            retriever = VectorIndexRetriever(index=index,
                                             similarity_top_k=similarity_top_k,
                                             node_ids=node_ids,
                                             **kwargs)
            engine = RetrieverQueryEngine.from_args(retriever=retriever,
                                                    text_qa_template=query_template)
        return engine

    @classmethod
    def get_retriever(
            cls,
            index: VectorStoreIndex = None,
            similarity_top_k: int = 5,
            node_ids: List[str] = None,
            **kwargs
    ):
        if index is None:
            raise Exception("输入 index 不能为空")
        retriever = VectorIndexRetriever(index=index,
                                         similarity_top_k=similarity_top_k,
                                         node_ids=node_ids,
                                         **kwargs)
        return retriever



if __name__ == "__main__":
    vector_dir = r"..."
    vector_index = RagPipeLines.build_index_from_source(
        data_source=vector_dir,
        persist_dir=vector_dir + r"\vector_store",
        is_vector_store_exist=True,
        index_method="VectorStoreIndex"
    )

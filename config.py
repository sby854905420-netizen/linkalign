# -*- coding: utf-8 -*-
import os

# 本地文件存储目录
ALL_DATABASE_DATA_SOURCE = r"..."

# 训练数据存储目录
DATASET_PATH = r"..."

# 索引保存目录
PERSIST_DIR = ALL_DATABASE_DATA_SOURCE + r"\vector_store"

# 日志目录
LOG_DIR = r"..."

# SummaryIndex 索引文件存储目录
ALL_DATABASE_SUMMARY_PERSIST_DIR = ALL_DATABASE_DATA_SOURCE + r"\vector_store\SummaryIndex"

# VectorStoreIndex 索引文件存储目录
ALL_DATABASE_VECTOR_PERSIST_DIR = ALL_DATABASE_DATA_SOURCE + r"\vector_store\VectorStoreIndex"

# 本地索引文件存储目录
VECTOR_STORE_PERSIST_DIR = "./documents_for_llms/data03/vector_store"

# 文件存储目录索引是否存在。注意：更新文件目录后第一次使用需要设置为 False
IS_VECTOR_STORE_EXIST = True

# 嵌入模型名称
EMBED_MODEL_NAME = None

# 底层大模型名称
LLM_NAME = "zhipu"

# 过程可视化
VERBOSE = False

GPT_API = ""
GPT_MODEL = "gpt-4o-mini"

OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = (
    os.environ.get("OLLAMA_BASE_URL")
    or os.environ.get("OLLAMA_HOST")
    or "http://localhost:11434"
)

# 两个模型的公共参数
TEMPERATURE = 0.45

MAX_OUTPUT_TOKENS = 4096

CONTEXT_WINDOW = 120000


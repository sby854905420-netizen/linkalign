# -*- coding: utf-8 -*-
import asyncio
import json
import logging
import os
from pathlib import Path
from llama_index.core.indices.vector_store import VectorIndexRetriever

from llama_index.core.llms.llm import LLM
from llama_index.core import (
    SummaryIndex,
    VectorStoreIndex,
    Settings,
    QueryBundle,

)
from llama_index.core.indices.utils import default_format_node_batch_fn
from llama_index.core.schema import MetadataMode
from llama_index.core.base.base_retriever import BaseRetriever
from prompts.PipelinePromptStore import *
from rag_pipes.RagPipeline import RagPipeLines
from prompts.AgentPromptStore import *
from utils import *



module_logger = logging.getLogger(__name__)


def _trace_dir() -> Path | None:
    trace_root = os.getenv("LINKALIGN_TRACE_DIR")
    if not trace_root:
        return None
    path = Path(trace_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_trace_name(name: str) -> str:
    valid = []
    for ch in str(name):
        if ch.isalnum() or ch in {"-", "_", "."}:
            valid.append(ch)
        else:
            valid.append("_")
    return "".join(valid)


def _write_trace_text(relative_name: str, content: str):
    trace_root = _trace_dir()
    if trace_root is None:
        return
    file_path = trace_root / relative_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_trace_json(relative_name: str, payload):
    trace_root = _trace_dir()
    if trace_root is None:
        return
    file_path = trace_root / relative_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _log_info(logger, message: str, *args):
    active_logger = logger or module_logger
    active_logger.info(message, *args)


def _preview_text(text: str, limit: int = 240) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."


def _summarize_nodes(nodes: List[NodeWithScore]) -> str:
    if not nodes:
        return "count=0 dbs=0"

    db_ids = []
    for node in nodes:
        file_path = node.node.metadata.get("file_path")
        if file_path:
            db_ids.append(Path(file_path).parent.name)
    return f"count={len(nodes)} dbs={len(set(db_ids))}"


def _node_payload(nodes: List[NodeWithScore]):
    payload = []
    for node in nodes:
        payload.append({
            "score": getattr(node, "score", None),
            "turn_n": node.metadata.get("turn_n"),
            "file_name": node.node.metadata.get("file_name"),
            "file_path": node.node.metadata.get("file_path"),
            "text_preview": node.get_content(metadata_mode=MetadataMode.LLM)[:600],
        })
    return payload

class SchemaLinkingTool:
    @classmethod
    def link_schema_by_rag(
            cls,
            llm: LLM = None,
            index: Union[SummaryIndex, VectorStoreIndex] = None,
            is_add_example: bool = True,
            question: str = None,
            similarity_top_k: int = 5,
            **kwargs
    ) -> str:
        if not index:
            raise Exception("The index cannot be empty!")

        if not question:
            raise Exception("The question cannot be empty!")

        if not llm:
            raise Exception("The llm cannot be empty!")

        Settings.llm = llm

        few_examples = SCHEMA_LINKING_FEW_EXAMPLES if is_add_example else ""

        query_template = SCHEMA_LINKING_TEMPLATE.format(few_examples=few_examples, question=question)

        engine_args = {
            "index": index,
            "query_template": query_template,
            "similarity_top_k": similarity_top_k,
            **kwargs
        }

        engine = RagPipeLines.get_query_engine(**engine_args)

        response = engine.query(question).response

        return response

    @classmethod
    def retrieve(
            cls,
            retriever_lis: List[BaseRetriever],
            query_lis: List[Union[str, QueryBundle]]
    ) -> List[NodeWithScore]:
        """ 串行化检索 """
        nodes_lis = []

        for retriever in retriever_lis:
            for query in query_lis:
                nodes = retriever.retrieve(query)
                nodes_lis.extend(nodes)

        nodes_lis.sort(key=lambda x: x.score, reverse=True)

        return nodes_lis

    @classmethod
    def parallel_retrieve(
            cls,
            retriever_list: List[BaseRetriever],
            query_list: List[Union[str, QueryBundle]]
    ) -> List[NodeWithScore]:
        """
        Run async retrieval from multiple retrievers and queries in parallel,
        and return a sorted flat list of NodeWithScore objects.
        """

        async def retrieve_from_all():
            tasks = [
                retriever.aretrieve(query)
                for retriever in retriever_list
                for query in query_list
            ]
            return await asyncio.gather(*tasks)

        # 执行异步任务并收集结果
        results = asyncio.run(retrieve_from_all())

        # 扁平化所有结果
        all_nodes = [node for result in results for node in result]

        # 按 score 降序排列
        all_nodes.sort(key=lambda node: node.score, reverse=True)

        return all_nodes

    @classmethod
    def query_rewriting(
            cls,
            llm=None,
            query: str = None,
            context: str = None
    ):
        """ 利用大模型在问题的基础上进行推理，并返回推理分析的结果 """
        if not query:
            raise Exception("输入的查询不能为空！")

        if not llm:
            raise Exception("The llm cannot be empty!")

        prompt = QUERY_REWRITING_TEMPLATE.format(question=query, context=context)

        reason_query = llm.complete(prompt=prompt).text  # 增强后的问题查询

        return reason_query

    @classmethod
    def retrieve_complete(
            cls,
            question: str = None,
            retriever_lis: List[VectorIndexRetriever] = None,
            llm=None,
            open_reason_enhance: bool = True,
            open_locate: bool = False,  # 测试一般设置为关闭，正式实验可以开启
            open_agent_debate: bool = False,  # 只有在open_locate 为真时该参数生效
            turn_n: int = 2,
            output_format: str = "database",  # database or schema,前者输出数据库名称，后者输出该数据库的 schema 信息
            remove_duplicate: bool = True,  # 在已检索出 node 以外的范围检索，效率可能有损失
            is_all: bool = True,
            enhanced_question=None,
            is_single_mode: bool = True,
    ):
        """
            Step one: retrieve potential database schemas.
            Mode: Pipeline.
        """
        if not question:
            raise Exception("输入参数中问题不能为空！")
        elif not retriever_lis:
            raise Exception("输入参数中索引列表不能为空！")

        if not llm:
            raise Exception("The llm cannot be empty!")

        nodes = cls.parallel_retrieve(retriever_lis, [question])
        nodes = [set_node_turn_n(node, 0) for node in nodes]
        if open_reason_enhance:
            context = parse_schema_from_df(parse_schemas_from_nodes(nodes))
            if not remove_duplicate:
                """ 如果不进行去重，同时使用 Question 和增强后的问题进行检索 """
                analysis = cls.query_rewriting(llm=llm, query=question, context=context)  # 调用大模型，通过推理对原始问题进行增强
                enhanced_question = question + analysis
                nodes += cls.parallel_retrieve(retriever_lis, [enhanced_question])
            else:
                # 获取所有的 index 和 id 列表
                index_lis = [ret.index for ret in retriever_lis]
                sub_ids = get_sub_ids(nodes, index_lis, is_all=is_all)

                # 设置新的id
                for ret in retriever_lis:
                    ret.change_node_ids(sub_ids)

                if enhanced_question is None:
                    # 进行问题增强
                    analysis = cls.query_rewriting(llm=llm, query=question, context=context)  # 调用大模型，通过推理对原始问题进行增强
                    enhanced_question = question + analysis

                temp_nodes = cls.parallel_retrieve(retriever_lis, [enhanced_question])
                temp_nodes = [set_node_turn_n(node, 1) for node in temp_nodes]
                nodes += temp_nodes

                for ret in retriever_lis:
                    ret.back_to_original_ids()
        # 基于嵌入向量的距离进行排序
        nodes.sort(key=lambda node: node.score, reverse=True)

        if open_locate:
            if open_agent_debate:
                predict_database = cls.locate_with_multi_agent(
                    llm=llm,
                    query=question,
                    nodes=nodes,
                    turn_n=turn_n,
                    is_single_mode=is_single_mode,
                )
            else:
                schemas = get_all_schemas_from_schema_text(nodes=nodes, output_format='schema')
                predict_database = cls.locate(
                    llm=llm,
                    query=question,
                    context=schemas,
                    is_single_mode=is_single_mode,
                )

            return predict_database

        else:
            output = get_all_schemas_from_schema_text(nodes=nodes, output_format=output_format,
                                                      schemas_format="str", is_all=is_all)

            return output

    @classmethod
    def retrieve_complete_by_multi_agent_debate(
            cls,
            question: str = None,
            retrieve_turn_n: int = 2,
            locate_turn_n: int = 2,
            retriever_lis: List[VectorIndexRetriever] = None,
            llm=None,
            global_top_k: int = None,
            open_locate: bool = False,  # 测试一般设置为关闭，正式实验可以开启,locate 只能输出唯一的 database
            open_agent_debate: bool = False,
            output_format: str = "database",  # database or schema,前者输出数据库名称，后者输出该数据库的 schema 信息
            remove_duplicate: bool = True,  # 在已检索出 node 以外的范围检索，效率可能有损失
            is_all: bool = True,
            logger=None,
            is_single_mode: bool = True,
    ):
        """
            Step one: retrieve potential database schemas.
            Mode: Agent
        """
        if not question:
            raise Exception("输入参数中问题不能为空！")
        elif not retriever_lis:
            raise Exception("输入参数中索引列表不能为空！")

        if not llm:
            raise Exception("The llm cannot be empty!")

        _log_info(
            logger,
            "[tool.retrieve] start retriever_count=%s retrieve_turn_n=%s locate_turn_n=%s "
            "global_top_k=%s open_locate=%s open_agent_debate=%s output_format=%s is_all=%s is_single_mode=%s",
            len(retriever_lis),
            retrieve_turn_n,
            locate_turn_n,
            global_top_k,
            open_locate,
            open_agent_debate,
            output_format,
            is_all,
            is_single_mode,
        )
        enhanced_question = question
        question_nodes = cls.parallel_retrieve(retriever_lis, [question])
        if global_top_k is not None and global_top_k > 0:
            question_nodes = question_nodes[:global_top_k]
        question_nodes = [set_node_turn_n(node, 0) for node in question_nodes]
        _write_trace_json("retrieval/00_question_nodes.json", _node_payload(question_nodes))
        nodes = question_nodes
        # 获取所有的 index 和 id 列表
        index_lis = [ret.index for ret in retriever_lis]

        sub_ids = get_ids_from_source(nodes)

        for ind in range(retrieve_turn_n):
            if not remove_duplicate:
                # 这里没有将检索范围设置为 sub_ids
                nodes += cls.parallel_retrieve(retriever_lis, [enhanced_question])

            else:
                # 设置新的id
                for ret in retriever_lis:
                    ret.change_node_ids(sub_ids)

                if ind != 0:
                    enhance_question_nodes = cls.parallel_retrieve(retriever_lis, [enhanced_question])
                    if global_top_k is not None and global_top_k > 0:
                        enhance_question_nodes = enhance_question_nodes[:global_top_k]
                    enhance_question_nodes = [set_node_turn_n(node, ind) for node in enhance_question_nodes]
                    nodes += enhance_question_nodes

                sub_ids = get_sub_ids(nodes, index_lis, is_all)

                # 恢复原来的 id
                for ret in retriever_lis:
                    ret.back_to_original_ids()

            schemas = parse_schema_from_df(parse_schemas_from_nodes(nodes))
            """ 
            下面使用 multi-agent debate 的方式进行，共有两个角色，judge 和 annotator。
            judge 负责分析错误并给出分析，而 annotator 主要负责为问题添加注释
            """
            # judge 进行分析
            analysis = llm.complete(JUDGE_TEMPLATE.format(question=question, context=schemas)).text

            _write_trace_json(
                f"retrieval/query_rewrite_turn_{ind + 1:02d}.json",
                {
                    "turn": ind + 1,
                    "question": question,
                    "schemas": schemas,
                    "analysis": analysis,
                },
            )
            # annotator 添加注释
            annotation = llm.complete(ANNOTATOR_TEMPLATE.format(question=question, analysis=analysis)).text
            _write_trace_json(
                f"retrieval/query_annotation_turn_{ind + 1:02d}.json",
                {
                    "turn": ind + 1,
                    "question": question,
                    "analysis": analysis,
                    "annotation": annotation,
                },
            )
            enhanced_question = question + annotation

        if not remove_duplicate:
            nodes += cls.parallel_retrieve(retriever_lis, [enhanced_question])
        else:
            # 设置新的id
            for ret in retriever_lis:
                ret.change_node_ids(sub_ids)

            enhance_question_nodes = cls.parallel_retrieve(retriever_lis, [enhanced_question])
            if global_top_k is not None and global_top_k > 0:
                enhance_question_nodes = enhance_question_nodes[:global_top_k]
            enhance_question_nodes = [set_node_turn_n(node, retrieve_turn_n) for node in enhance_question_nodes]
            nodes += enhance_question_nodes

            # 恢复原来的 id
            for ret in retriever_lis:
                ret.back_to_original_ids()

        # 根据 turn_n 和 score 重新排序, 重写次数[1] 和 分数[2]
        nodes.sort(key=lambda x: (x.metadata["turn_n"], x.score))
        _write_trace_json("retrieval/final_nodes.json", _node_payload(nodes))
        _log_info(logger, "[tool.retrieve] final_nodes %s", _summarize_nodes(nodes))

        if open_locate:
            """ 若进行数据库定位 """
            if open_agent_debate:
                predict_database = cls.locate_with_multi_agent(
                    llm=llm,
                    query=question,
                    nodes=nodes,
                    turn_n=locate_turn_n,
                    is_single_mode=is_single_mode,
                    logger=logger,
                )
            else:
                schemas = get_all_schemas_from_schema_text(nodes=nodes, output_format='schema', is_all=is_all)
                predict_database = cls.locate(
                    llm=llm,
                    query=question,
                    context=schemas,
                    is_single_mode=is_single_mode,
                )
            _log_info(
                logger,
                "[tool.retrieve] locate_done result=%s",
                _preview_text(predict_database),
            )

            return predict_database

        else:
            output = get_all_schemas_from_schema_text(nodes=nodes, output_format=output_format, is_all=is_all)
            _log_info(
                logger,
                "[tool.retrieve] output_ready output_format=%s node_summary=%s",
                output_format,
                _summarize_nodes(nodes),
            )

            return output

    @classmethod
    def load_rf_template(
            cls,
            mode: str = "agent",  # agent or pipeline
            is_single_mode: bool = True  # Single-DB / Multi-DB
    ):
        mode = mode if mode in ["agent", "pipeline"] else "agent"
        if mode == "agent":
            if is_single_mode:
                return {
                    "SOURCE_TEXT_TEMPLATE": SOURCE_TEXT_TEMPLATE,
                    "FAIR_EVAL_DEBATE_TEMPLATE": FAIR_EVAL_DEBATE_TEMPLATE,
                    "DATA_ANALYST_ROLE_DESCRIPTION": DATA_ANALYST_ROLE_DESCRIPTION,
                    "DATABASE_SCIENTIST_ROLE_DESCRIPTION": DATABASE_SCIENTIST_ROLE_DESCRIPTION,
                    "SUMMARY_TEMPLATE": SUMMARY_TEMPLATE
                }
            else:
                return {
                    "SOURCE_TEXT_TEMPLATE": MULTI_SOURCE_TEXT_TEMPLATE,
                    "FAIR_EVAL_DEBATE_TEMPLATE": MULTI_FAIR_EVAL_DEBATE_TEMPLATE,
                    "DATA_ANALYST_ROLE_DESCRIPTION": MULTI_DATA_ANALYST_ROLE_DESCRIPTION,
                    "DATABASE_SCIENTIST_ROLE_DESCRIPTION": MULTI_DATABASE_SCIENTIST_ROLE_DESCRIPTION,
                    "SUMMARY_TEMPLATE": MULTI_SUMMARY_TEMPLATE
                }
        else:
            if is_single_mode:
                return {
                    "LOCATE_TEMPLATE": LOCATE_TEMPLATE
                }
            else:
                return {
                    "LOCATE_TEMPLATE": MULTI_LOCATE_TEMPLATE
                }

    @classmethod
    def locate(
            cls,
            llm=None,
            query: str = None,
            context: str = None,  # 检索的所有数据库schema
            is_single_mode: bool = True
    ) -> str:
        """
            Step two: isolate irrelevant schema information.
            Mode: Pipeline
        """
        if not query:
            raise Exception("输入的查询不能为空！")

        if not llm:
            raise Exception("The llm cannot be empty!")

        prompt_loader = cls.load_rf_template(mode='pipeline', is_single_mode=is_single_mode)
        prompt = prompt_loader['LOCATE_TEMPLATE'].format(question=query, context=context)

        # print(prompt)
        database = llm.complete(prompt=prompt).text  # 增强后的问题查询
        #
        return database

    @classmethod
    def locate_with_multi_agent(
            cls,
            llm=None,
            turn_n: int = 2,
            query: str = None,
            nodes: List[NodeWithScore] = None,
            context_lis: List[str] = None,
            context_str: str = None,
            is_single_mode: bool = True,
            logger=None,
    ) -> str:
        """
            Step two: isolate irrelevant schema information.
            Mode: Agent
        """
        if not query:
            raise Exception("The query cannot be empty!")

        if not llm:
            raise Exception("The llm cannot be empty!")

        prompt_loader = cls.load_rf_template(mode='agent', is_single_mode=is_single_mode)

        if context_str or context_lis:
            pass
        elif nodes:
            if is_single_mode:
                context_lis = get_all_schemas_from_schema_text(nodes, output_format="schema", schemas_format="list")
            else:
                grouped_nodes = group_nodes_by_database(nodes)
                context_lis = []
                for db_id, db_nodes in grouped_nodes:
                    db_schema = get_all_schemas_from_schema_text(db_nodes, output_format="schema", schemas_format="list")
                    if isinstance(db_schema, list):
                        db_schema = "\n".join(db_schema)
                    context_lis.append({
                        "db_id": db_id,
                        "schema": db_schema,
                    })
        else:
            raise Exception("输入参数中没有包含 database schemas")

        if not context_str:
            context_str = ""
            for ind, context in enumerate(context_lis):
                if isinstance(context, dict):
                    db_id = context["db_id"]
                    schema = context["schema"]
                    context_str += f"""[The Start of Candidate Database"{ind + 1}" ({db_id})'s Schema]
{schema}
[The End of Candidate Database"{ind + 1}" ({db_id})'s Schema]
                    """
                else:
                    context_str += f"""[The Start of Candidate Database"{ind + 1}"'s Schema]
{context}
[The End of Candidate Database"{ind + 1}"'s Schema]
                    """
        source_text = prompt_loader['SOURCE_TEXT_TEMPLATE'].format(query=query, context_str=context_str)
        _write_trace_text("locate/source_text.txt", source_text)
        chat_history = []

        # one-by-one
        for i in range(turn_n):
            data_analyst_prompt = prompt_loader['FAIR_EVAL_DEBATE_TEMPLATE'].format(
                source_text=source_text,
                chat_history="\n".join(chat_history),
                role_description=prompt_loader['DATA_ANALYST_ROLE_DESCRIPTION'],
                agent_name="data analyst"
            )
            data_analyst_debate = llm.complete(data_analyst_prompt).text
            _write_trace_json(
                f"locate/turn_{i + 1:02d}_data_analyst.json",
                {
                    "turn": i + 1,
                    "prompt": data_analyst_prompt,
                    "response": data_analyst_debate,
                    "chat_history_before": chat_history,
                },
            )
            chat_history.append(
                f'[Debate Turn: {i + 1}, Agent Name:"data analyst", Debate Content:{data_analyst_debate}]')
            data_scientist_prompt = prompt_loader['FAIR_EVAL_DEBATE_TEMPLATE'].format(
                source_text=source_text,
                chat_history="\n".join(chat_history),
                role_description=prompt_loader['DATABASE_SCIENTIST_ROLE_DESCRIPTION'],
                agent_name="database scientist"
            )
            data_scientist_debate = llm.complete(data_scientist_prompt).text
            _write_trace_json(
                f"locate/turn_{i + 1:02d}_database_scientist.json",
                {
                    "turn": i + 1,
                    "prompt": data_scientist_prompt,
                    "response": data_scientist_debate,
                    "chat_history_before": chat_history,
                },
            )
            chat_history.append(
                f'[Debate Turn: {i + 1}, Agent Name:"database scientist", Debate Content:{data_scientist_debate}]')
        _write_trace_json("locate/chat_history.json", chat_history)
        summary_prompt = prompt_loader['FAIR_EVAL_DEBATE_TEMPLATE'].format(
            source_text=source_text,
            chat_history="\n".join(chat_history),
            role_description=prompt_loader['SUMMARY_TEMPLATE'],
            agent_name="debate terminator"
        )

        database = llm.complete(summary_prompt).text
        _write_trace_json(
            "locate/summary.json",
            {
                "prompt": summary_prompt,
                "response": database,
                "chat_history": chat_history,
            },
        )
        _log_info(logger, "[tool.locate] done result=%s", _preview_text(database))

        return database

    @classmethod
    def generate_schema(
            cls,
            llm=None,
            query: str = None,
            context: str = None,
            logger=None
    ):
        """
            Step there: extract schemas for SQL generation.
            Mode: Pipeline.
        """
        if not llm:
            raise Exception("The llm cannot be empty!")

        if not context:
            raise Exception("The context cannot be empty!")

        context_str = f"[The Start of Database Schemas]\n{context}\n[The End of Database Schemas]"
        query = SCHEMA_LINKING_MANUAL_TEMPLATE.format(few_examples=SCHEMA_LINKING_FEW_EXAMPLES,
                                                      context_str=context_str,
                                                      question=query)
        predict_schema = llm.complete(query).text
        _log_info(logger, "[tool.generate] done mode=pipeline result=%s", _preview_text(predict_schema))

        return predict_schema

    @classmethod
    def generate_by_multi_agent(
            cls,
            llm=None,
            query: str = None,
            context: str = None,
            turn_n: int = 2,
            linker_num: int = 1,  # schema linker 角色的数量
            logger=None
    ):
        """
            Step there: extract schemas for SQL generation.
            Mode: Agent
        """
        if not llm:
            raise Exception("The llm cannot be empty!")

        if not context:
            raise Exception("The context cannot be empty!")

        context_str = f"[The Start of Database Schemas]\n{context}\n[The End of Database Schemas]"
        source_text = GENERATE_SOURCE_TEXT_TEMPLATE.format(query=query, context_str=context_str)
        _write_trace_text("schema_linking/source_text.txt", source_text)
        _log_info(
            logger,
            "[tool.generate] start mode=agent turn_n=%s linker_num=%s source_chars=%s",
            turn_n,
            linker_num,
            len(source_text),
        )

        chat_history = []

        # one-by-one
        for i in range(turn_n):
            data_analyst_prompt = GENERATE_FAIR_EVAL_DEBATE_TEMPLATE.format(
                source_text=source_text,
                chat_history="\n".join(chat_history),
                role_description=GENERATE_DATA_ANALYST_ROLE_DESCRIPTION,
                agent_name="data analyst"
            )
            for j in range(linker_num):
                data_analyst_debate = llm.complete(data_analyst_prompt).text
                _write_trace_json(
                    f"schema_linking/turn_{i + 1:02d}_data_analyst_{j + 1:02d}.json",
                    {
                        "turn": i + 1,
                        "agent_index": j + 1,
                        "prompt": data_analyst_prompt,
                        "response": data_analyst_debate,
                        "chat_history_before": chat_history,
                    },
                )
                chat_history.append(
                    f"""[Debate Turn: {i + 1}, Agent Name:"data analyst {j}", Debate Content:{data_analyst_debate}]""")
            data_scientist_prompt = GENERATE_FAIR_EVAL_DEBATE_TEMPLATE.format(
                source_text=source_text,
                chat_history="\n".join(chat_history),
                role_description=GENERATE_DATABASE_SCIENTIST_ROLE_DESCRIPTION,
                agent_name="data scientist"
            )
            data_scientist_debate = llm.complete(data_scientist_prompt).text
            _write_trace_json(
                f"schema_linking/turn_{i + 1:02d}_data_scientist.json",
                {
                    "turn": i + 1,
                    "prompt": data_scientist_prompt,
                    "response": data_scientist_debate,
                    "chat_history_before": chat_history,
                },
            )
            chat_history.append(
                f"""[Debate Turn: {i + 1}, Agent Name:"data scientist", Debate Content:{data_scientist_debate}]""")
        _write_trace_json("schema_linking/chat_history.json", chat_history)
        summary_prompt = GENERATE_FAIR_EVAL_DEBATE_TEMPLATE.format(
            source_text=source_text,
            chat_history="\n".join(chat_history),
            role_description=GENERATE_SUMMARY_TEMPLATE,
            agent_name="debate terminator"
        )
        schema = llm.complete(summary_prompt).text
        _write_trace_json(
            "schema_linking/summary.json",
            {
                "prompt": summary_prompt,
                "response": schema,
                "chat_history": chat_history,
            },
        )

        _log_info(logger, "[tool.generate] done mode=agent result=%s", _preview_text(schema))

        return schema

    def retrieve_complete_selector(self, mode: str, **kwargs):
        mode = mode if mode in ["agent", "pipeline"] else "pipeline"
        if mode == "pipeline":
            res = self.retrieve_complete(**kwargs)
        else:
            res = self.retrieve_complete_by_multi_agent_debate(**kwargs)
        return res

    def locate_selector(self, mode: str, **kwargs):
        mode = mode if mode in ["agent", "pipeline"] else "pipeline"
        if mode == "pipeline":
            res = self.locate(**kwargs)
        else:
            res = self.locate_with_multi_agent(**kwargs)
        return res

    def generate_selector(self, mode: str, **kwargs):
        mode = mode if mode in ["agent", "pipeline"] else "pipeline"
        if mode == "pipeline":
            res = self.generate_schema(**kwargs)
        else:
            res = self.generate_by_multi_agent(**kwargs)
        return res


def filter_nodes_by_database(
        nodes: List[NodeWithScore],
        database: Union[str, List],
        output_format: str = "str"
):
    schema_lis = []
    for node in nodes:
        file_path = node.node.metadata["file_path"]
        db = file_path.split("\\")[-1].split(".")[0].strip()
        if type(database) == str:
            if db == database:
                schema_lis.append(default_format_node_batch_fn([node.node]))
        elif type(database) == List:
            if db in database:
                schema_lis.append(default_format_node_batch_fn([node.node]))
    if output_format == "str":
        return "\n".join(schema_lis)

    return schema_lis


def get_all_schemas_from_schema_text(
        nodes: List[NodeWithScore],
        output_format: str = "schema",  # `node`, `schema` or `all`
        schemas_format: str = "str",  # 当输出格式为 node 时无效
        is_all: bool = True
):
    if output_format == "node":
        return nodes
    if is_all:
        # 解析文本块对应文件的全部字符
        schemas = []
        for path in [Path(node.node.metadata["file_path"]) for node in nodes]:
            schema = load_dataset(path)
            if isinstance(schema, dict):
                schema = json.dumps(schema, ensure_ascii=False, indent=2)
            else:
                schema = str(schema).strip()
            schemas.append(schema)
        if schemas_format == "str":
            schemas = "\n".join(schemas)
    else:
        # 仅保留单个检索文本块的内容
        summary_nodes = nodes
        fmt_node_txts = []
        for idx in range(len(summary_nodes)):
            file_path = summary_nodes[idx].node.metadata["file_path"]
            db = Path(file_path).stem
            fmt_node_txts.append(
                f"### Database Name: {db}\n#Following is the table creation statement for the database {db}\n"
                f"{summary_nodes[idx].get_content(metadata_mode=MetadataMode.LLM)}"
            )
        schemas = "\n\n".join(fmt_node_txts)

    if output_format == "all":
        return schemas, nodes
    else:
        return schemas


def group_nodes_by_database(nodes: List[NodeWithScore]):
    grouped = {}
    ordered_db_ids = []

    for node in nodes:
        db_id = Path(node.node.metadata["file_path"]).parent.name
        if db_id not in grouped:
            grouped[db_id] = []
            ordered_db_ids.append(db_id)
        grouped[db_id].append(node)

    return [(db_id, grouped[db_id]) for db_id in ordered_db_ids]


def get_sub_ids(
        nodes: List[NodeWithScore],
        index_lis: List[VectorStoreIndex],
        is_all: bool = True
):
    if is_all:
        file_name_lis = []
        for node in nodes:
            file_name = node.node.metadata["file_name"]
            file_name_lis.append(file_name)

        sub_ids = []
        duplicate_ids = []
        for index in index_lis:
            doc_info_dict = index.ref_doc_info
            for key, ref_doc_info in doc_info_dict.items():
                if ref_doc_info.metadata["file_name"] not in file_name_lis:
                    sub_ids.extend(ref_doc_info.node_ids)
                else:
                    duplicate_ids.extend(ref_doc_info.node_ids)

        return sub_ids
    else:
        exist_node_ids = [node.node.id_ for node in nodes]
        all_ids = []
        for index in index_lis:
            doc_info_dict = index.ref_doc_info
            for key, ref_doc_info in doc_info_dict.items():
                all_ids.extend(ref_doc_info.node_ids)
        sub_ids = [id_ for id_ in all_ids if id_ not in exist_node_ids]

        return sub_ids


def get_ids_from_source(
        source: Union[List[VectorStoreIndex], List[NodeWithScore]]
):
    node_ids = []
    """ 本方法仅用于解析本实验需要两种类型的 node_id """
    for data in source:
        if isinstance(data, VectorStoreIndex):
            doc_info_dict = data.ref_doc_info
            for key, ref_doc_info in doc_info_dict.items():
                node_ids.extend(ref_doc_info.node_ids)

        elif isinstance(data, NodeWithScore):

            node_ids.append(data.node.node_id)

    # 去重
    node_ids = list(set(node_ids))

    return node_ids

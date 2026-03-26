import argparse
import concurrent.futures
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from llama_index.core.indices.vector_store import VectorIndexRetriever
from tqdm import tqdm

from llms.gpt.GPTModel import GPTModel
from llms.ollama.ollamaModel import OllamaModel
from rag_pipes.RagPipeline import RagPipeLines
from tools.SchemaLinkingTool import SchemaLinkingTool
from utils import get_sql_files, parse_list_from_str, parse_schemas_from_nodes


DEFAULT_LLM_MODEL = "ministral-3:14b"
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"

llm = OllamaModel(model_name=DEFAULT_LLM_MODEL, temperature=0.85)
filter_llm = OllamaModel(model_name=DEFAULT_LLM_MODEL, temperature=0.42)


def build_llm(model_name: str, temperature: float):
    normalized = (model_name or "").strip().lower()
    if normalized.startswith("gpt-"):
        return GPTModel(model_name=model_name, temperature=temperature)
    return OllamaModel(model_name=model_name, temperature=temperature)


def build_logger(name: str = "MultiGenerateSchemas"):
    logger_ = logging.getLogger(name)
    if not logger_.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger_.addHandler(handler)
    logger_.setLevel(logging.INFO)
    logger_.propagate = False
    return logger_


logger = build_logger()
vector_index_cache = {}
trace_root_dir = None


def trace_path(*parts: str) -> Optional[Path]:
    if not trace_root_dir:
        return None
    path = Path(trace_root_dir).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def trace_json(relative_path: str, payload):
    path = trace_path(relative_path)
    if path is None:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def trace_text(relative_path: str, content: str):
    path = trace_path(relative_path)
    if path is None:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def trace_csv(relative_path: str, df: pd.DataFrame):
    path = trace_path(relative_path)
    if path is None:
        return
    df.to_csv(path, index=False, encoding="utf-8")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run schema linking across multiple candidate databases.")
    parser.add_argument("--save_path", type=str, required=False, default="./spider2_dev/multi_instance_schemas",
                        help="Path for storing the schema subset retrieved from multiple candidate databases.")
    parser.add_argument("--schema_path", type=str, required=False, default="./spider2_dev/schemas",
                        help="Path for storing database schema information.")
    parser.add_argument("--dataset", type=str, required=False, default="./spider2_dev/samples_data.json")
    parser.add_argument("--db_info_path", type=str, required=False, default="./spider2_dev/db_info.json")
    parser.add_argument("--links_save_path", type=str, required=False, default="./spider2_dev/multi_schema_links")
    parser.add_argument("--external_info_path", type=str, required=False, default="./spider2_dev/external_knowledge")
    parser.add_argument("--open_schema_linking", action="store_true",
                        help="Generate final schema-linking results from the retrieved schema subset.")
    parser.add_argument("--llm_model_name", type=str, required=False, default=DEFAULT_LLM_MODEL,
                        help="Local Ollama model name used for retrieval and final schema linking.")
    parser.add_argument("--filter_llm_model_name", type=str, required=False, default=DEFAULT_LLM_MODEL,
                        help="Local Ollama model name used during response filtering.")
    parser.add_argument("--candidate_db_key", type=str, required=False, default="candidate_db_ids",
                        help="Field name that stores the candidate database list in the dataset.")
    parser.add_argument("--max_workers", type=int, required=False, default=1,
                        help="Number of worker threads used to process rows.")
    parser.add_argument("--trace_dir", type=str, required=False, default=None,
                        help="Directory used to store intermediate artifacts, prompts, and responses.")
    return parser.parse_args()


def load_db_size(db_id: str):
    matched = [row["count"] for row in db_info if row["db_id"].lower() == db_id.lower()]
    if not matched:
        raise ValueError(f"Cannot find db_id={db_id} in db_info.")
    return matched[0]


def parse_schemas_from_file(db_id: str):
    db_schema_dir = os.path.join(schema_path, db_id)
    file_lis = get_sql_files(db_schema_dir, ".json")

    schema_lis = []
    for f in file_lis:
        try:
            file_path = os.path.join(db_schema_dir, f"{f}.json")
            with open(file_path, "r", encoding="utf-8") as file:
                col_info = json.load(file)
            meta_data = col_info["meta_data"]
            schema = {
                "Database name": meta_data["db_id"],
                "Table Name": meta_data["table_name"],
                "Field Name": col_info["column_name"],
                "Type": col_info["column_types"],
                "Description": None if not col_info["column_descriptions"] else col_info["column_descriptions"],
                "Example": None if len(col_info["sample_rows"]) == 0 else col_info["sample_rows"][0],
                "turn_n": 0,
            }
            schema_lis.append(schema)
        except Exception:
            pass

    return pd.DataFrame(schema_lis)


def _extract_link_pair(link: str) -> Tuple[Optional[str], Optional[str]]:
    cleaned_link = " ".join(str(link).split())
    if "#" in cleaned_link:
        cleaned_link = cleaned_link.split("#", 1)[0].strip()

    fields = [x.strip() for x in cleaned_link.split(".") if x.strip()]
    if len(fields) < 2:
        return None, None
    return fields[-2], fields[-1]


def response_filtering(
        data: pd.DataFrame,
        question: str,
        chunk_size: int = 250,
        turn_n: int = 2,
        reserve_df: pd.DataFrame = None,
):
    df_list = []
    num_rows = data.shape[0]

    for i in range(0, num_rows, chunk_size):
        df_slice = data.iloc[i:i + chunk_size]
        df_list.append(df_slice)

    sub_data_lis = [reserve_df] if reserve_df is not None else []
    for sub_df in df_list:
        schema_context = parse_schema_context(sub_df)
        res = SchemaLinkingTool.locate_with_multi_agent(
            llm=filter_llm,
            query=question,
            context_str=schema_context,
            turn_n=turn_n,
        )
        if "[" not in res or "]" not in res:
            sub_data_lis.append(sub_df)
            continue

        schema_links = res.split("[", 1)[1].split("]", 1)[0].strip()
        schema_links = schema_links.split(",")
        schema_links = [link.strip().replace("`", "").replace('"', "").replace("'", "") for link in schema_links]
        temp_lis = []
        for link in schema_links:
            table, field = _extract_link_pair(link)
            if not table or not field:
                continue
            temp_lis.append((table, field))
        for table, field in temp_lis:
            sub_df = sub_df.query("not (`Table Name` == @table and `Field Name` == @field)")

        sub_data_lis.append(sub_df)

    df = pd.concat(sub_data_lis, axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=["Database name", "Table Name", "Field Name"], ignore_index=True)
    return df


def load_retrieval_top_k(db_size):
    if db_size <= 200:
        return 30
    elif db_size <= 500:
        return 40
    elif db_size <= 1000:
        return 50
    elif db_size <= 5000:
        return 60
    else:
        return 70


def load_global_retrieval_top_k(total_db_size: int):
    if total_db_size <= 200:
        return 30
    elif total_db_size <= 500:
        return 40
    elif total_db_size <= 1000:
        return 50
    elif total_db_size <= 5000:
        return 60
    else:
        return 70


def load_retrieval_turn_n(db_size):
    if db_size <= 50:
        return 1
    elif db_size <= 200:
        return 2
    elif db_size <= 350:
        return 3
    elif db_size <= 1000:
        return 5
    elif db_size <= 5000:
        return 7
    else:
        return 9


def load_post_retrival_param(db_size):
    if db_size <= 200:
        return 5, 1
    elif db_size <= 500:
        return 10, 1
    elif db_size <= 1000:
        return 15, 1
    elif db_size <= 2000:
        return 15, 2
    else:
        return 20, 1


def transform_name(table_name, col_name):
    prefix = rf"{table_name}_{col_name}"
    prefix = prefix if len(prefix) < 100 else prefix[:100]

    syn_lis = ["(", ")", "%", "/"]
    for syn in syn_lis:
        if syn in prefix:
            prefix = prefix.replace(syn, "_")

    return prefix


def set_retriever(
        retriever: VectorIndexRetriever,
        data: pd.DataFrame,
):
    table_lis, col_lis = list(data["Table Name"]), list(data["Field Name"])
    file_name_lis = []
    for table, col in zip(table_lis, col_lis):
        file_name_lis.append(transform_name(table, col))

    index = retriever.index
    sub_ids = []
    doc_info_dict = index.ref_doc_info
    for key, ref_doc_info in doc_info_dict.items():
        if ref_doc_info.metadata["file_name"] not in file_name_lis:
            sub_ids.extend(ref_doc_info.node_ids)
    retriever.change_node_ids(sub_ids)


def load_data(dataset):
    return pd.read_json(dataset)


def get_files(directory, suffix: str = ".sql"):
    return [f for f in os.listdir(directory) if f.endswith(suffix)]


def load_external_knowledge(external_knowledge_id):
    path = external_info_path
    if not os.path.exists(path):
        return None
    all_ids = get_files(path, ".md")

    if external_knowledge_id in all_ids:
        with open(os.path.join(path, f"{external_knowledge_id}"), "r", encoding="utf-8") as f:
            external = f.read()
        if len(external) > 50:
            external = "\n####[External Prior Knowledge]:\n" + external + "\n"
            return external

    return None


def normalize_candidate_db_ids(candidate_db_ids) -> List[str]:
    if candidate_db_ids is None:
        candidate_db_ids = []
    elif isinstance(candidate_db_ids, str):
        candidate_db_ids = parse_list_from_str(candidate_db_ids)
    elif not isinstance(candidate_db_ids, list):
        candidate_db_ids = list(candidate_db_ids)

    normalized = []
    seen = set()
    for db_id in candidate_db_ids:
        if db_id is None:
            continue
        db_id = str(db_id).strip()
        if not db_id or db_id in seen:
            continue
        normalized.append(db_id)
        seen.add(db_id)

    if not normalized:
        normalized = load_all_available_db_ids()

    return normalized


def load_all_available_db_ids() -> List[str]:
    available = []
    for row in db_info:
        db_id = str(row.get("db_id", "")).strip()
        if not db_id:
            continue

        db_schema_dir = os.path.join(schema_path, db_id)
        if not os.path.isdir(db_schema_dir):
            continue

        try:
            file_lis = get_sql_files(db_schema_dir, ".json")
        except Exception:
            continue

        if not file_lis:
            continue
        available.append(db_id)

    return available


def get_or_build_retriever(db_id: str):
    if db_id in vector_index_cache:
        return RagPipeLines.get_retriever(index=vector_index_cache[db_id])

    vector_dir = os.path.join(schema_path, db_id)
    vector_index = RagPipeLines.build_index_from_source(
        data_source=vector_dir,
        persist_dir=os.path.join(vector_dir, f"{EMBED_MODEL_NAME}_vector_store"),
        is_vector_store_exist=True,
        index_method="VectorStoreIndex",
    )
    vector_index_cache[db_id] = vector_index
    return RagPipeLines.get_retriever(index=vector_index)


def concat_dataframes(df_list: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [df for df in df_list if df is not None and not df.empty]
    if not valid_frames:
        return pd.DataFrame(columns=["Database name", "Table Name", "Field Name", "Type", "Description", "Example", "turn_n"])
    return pd.concat(valid_frames, axis=0, ignore_index=True)


def parse_schema_context(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    grouped = df.groupby(["Database name", "Table Name"], dropna=False)
    output_lines = []
    for (db_name, table_name), group in grouped:
        columns = []
        for _, row in group.iterrows():
            col_type = row.get("Type", row["Field Name"])
            if isinstance(col_type, str) and len(col_type) > 150:
                col_type = col_type[:150]
            columns.append(f'{row["Field Name"]}(Type: {col_type})')

        if pd.isna(db_name):
            line = f"### Table {table_name}, columns = [{', '.join(columns)}]"
        else:
            line = f"### Database {db_name} | Table {table_name}, columns = [{', '.join(columns)}]"
        output_lines.append(line)

    return "\n".join(output_lines)


def build_reserve_df(df: pd.DataFrame, reserve_rate: float) -> pd.DataFrame:
    if df.empty:
        return df

    turn_n_lis = df["turn_n"].dropna().unique().tolist()
    df_lis = []
    for n in turn_n_lis:
        temp_df = df[df["turn_n"] == n]
        df_reserver_rate = 0.55 * pow(reserve_rate, n)
        if df_reserver_rate <= 0.1:
            continue
        sample_size = int(len(temp_df) * df_reserver_rate)
        if sample_size <= 0:
            continue
        temp_df = temp_df.sample(sample_size, random_state=42)
        df_lis.append(temp_df)

    return concat_dataframes(df_lis)


def write_meta(output_meta_path: str, meta: dict):
    os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def normalize_selected_database(raw_output: str, candidate_db_ids: List[str]) -> Optional[str]:
    if raw_output is None:
        return None

    cleaned = str(raw_output).replace("`", " ").replace('"', " ").replace("'", " ")
    cleaned = " ".join(cleaned.split())
    lowered_to_original = {db_id.lower(): db_id for db_id in candidate_db_ids}

    if cleaned.lower() in lowered_to_original:
        return lowered_to_original[cleaned.lower()]

    for db_id in candidate_db_ids:
        if db_id.lower() in cleaned.lower():
            return db_id

    lowered = cleaned.lower()
    tokens = lowered.replace("(", " ").replace(")", " ").replace(":", " ").split()
    candidate_index = None
    for idx, token in enumerate(tokens):
        if token == "database" and idx + 1 < len(tokens) and tokens[idx + 1].isdigit():
            candidate_index = int(tokens[idx + 1])
            break
        if token == "candidate" and idx + 2 < len(tokens) and tokens[idx + 1] == "database" and tokens[idx + 2].isdigit():
            candidate_index = int(tokens[idx + 2])
            break

    if candidate_index is not None and 1 <= candidate_index <= len(candidate_db_ids):
        return candidate_db_ids[candidate_index - 1]

    return None


def select_database(
        db_ids: List[str],
        question: str,
):
    retriever_lis = []
    total_db_size = 0
    for db_id in db_ids:
        retriever = get_or_build_retriever(db_id)
        db_size = load_db_size(db_id)
        total_db_size += db_size
        retriever.similarity_top_k = load_retrieval_top_k(db_size)
        retriever_lis.append(retriever)

    global_top_k = load_global_retrieval_top_k(total_db_size)

    locate_raw_output = SchemaLinkingTool.retrieve_complete_by_multi_agent_debate(
        llm=llm,
        question=question,
        retriever_lis=retriever_lis,
        global_top_k=global_top_k,
        open_locate=True,
        open_agent_debate=True,
        output_format="database",
        is_all=True,
        logger=logger,
        is_single_mode=False,
    )
    selected_database = normalize_selected_database(locate_raw_output, db_ids)
    trace_json(
        "pipeline/database_selection.json",
        {
            "candidate_db_ids": db_ids,
            "total_db_size": total_db_size,
            "global_top_k": global_top_k,
            "question": question,
            "locate_raw_output": locate_raw_output,
            "selected_database": selected_database,
        },
    )
    return selected_database, locate_raw_output


def get_schema_for_single_db(
        db_id: str,
        question: str,
        instance_id: str,
        reserve_size: int = 99,
        min_retrival_size: int = 250,
        filter_chunk_size: int = 250,
        post_retrieval_size: int = 99,
        post_retrieval_turn: int = 2,
        reserve_rate: float = 0.6,
        open_schema_linking: bool = False,
):
    db_size = load_db_size(db_id)
    if db_size <= reserve_size:
        df = parse_schemas_from_file(db_id)
        if df.empty:
            raise ValueError(f"db_id={db_id} produced an empty schema subset.")
        df = df.drop_duplicates(subset=["Database name", "Table Name", "Field Name"], ignore_index=True)
        trace_csv("pipeline/01_full_schema_direct.csv", df)
        if not open_schema_linking:
            return df

        context = parse_schema_context(df)
        trace_text("pipeline/05_final_schema_subset.txt", context)
        schema_links = SchemaLinkingTool.generate_by_multi_agent(
            llm=llm,
            query=question,
            context=context,
            turn_n=1,
            linker_num=3,
            logger=logger,
        )
        schema_links = schema_links.replace("`", "").replace("\n", "").replace("python", "")
        trace_text("pipeline/06_schema_links.txt", schema_links)
        return df, schema_links

    retriever = get_or_build_retriever(db_id)

    if db_size <= min_retrival_size:
        df = parse_schemas_from_file(db_id)
    else:
        similarity_top_k = load_retrieval_top_k(db_size)
        turn_n = load_retrieval_turn_n(db_size)
        retriever.similarity_top_k = similarity_top_k
        nodes_lis = SchemaLinkingTool.retrieve_complete_by_multi_agent_debate(
            llm=llm,
            question=question,
            retriever_lis=[retriever],
            open_locate=False,
            output_format="node",
            logger=logger,
            retrieve_turn_n=turn_n,
        )
        df = parse_schemas_from_nodes(nodes_lis)
        trace_csv("pipeline/01_initial_retrieval.csv", df)

    if df.empty:
        raise ValueError(f"db_id={db_id} produced an empty schema subset.")

    reserve_df = build_reserve_df(df, reserve_rate=reserve_rate)
    trace_csv("pipeline/02_reserve_subset.csv", reserve_df)
    for _ in range(post_retrieval_turn):
        if len(df) > post_retrieval_size:
            before_df = df.copy()
            df = response_filtering(
                data=df,
                question=question,
                chunk_size=filter_chunk_size,
                reserve_df=reserve_df,
            )
            turn_index = _ + 1
            trace_csv(f"pipeline/03_filter_turn_{turn_index:02d}_before.csv", before_df)
            trace_csv(f"pipeline/03_filter_turn_{turn_index:02d}_after.csv", df)

    post_top_k, post_turn_n = load_post_retrival_param(db_size)
    set_retriever(retriever, df)
    retriever.similarity_top_k = post_top_k
    nodes_lis = SchemaLinkingTool.retrieve_complete_by_multi_agent_debate(
        llm=llm,
        question=question,
        retriever_lis=[retriever],
        open_locate=False,
        output_format="node",
        logger=logger,
        retrieve_turn_n=post_turn_n,
    )
    sub_df = parse_schemas_from_nodes(nodes_lis)
    trace_csv("pipeline/04_post_retrieval.csv", sub_df)
    df = pd.concat([df, sub_df], axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=["Database name", "Table Name", "Field Name"], ignore_index=True)
    trace_csv("pipeline/05_final_schema_subset.csv", df)
    trace_text("pipeline/05_final_schema_subset.txt", parse_schema_context(df))
    return df


def get_schema_multi(
        db_ids: List[str],
        question: str,
        instance_id: str,
        reserve_size: int = 99,
        min_retrival_size: int = 250,
        filter_chunk_size: int = 250,
        post_retrieval_size: int = 99,
        post_retrieval_turn: int = 2,
        reserve_rate: float = 0.6,
        open_schema_linking: bool = False,
):
    file_name = instance_id + "_agent"
    output_csv = os.path.join(save_path, f"{file_name}.csv")
    output_txt = os.path.join(links_save_path, f"{file_name}.txt")
    output_meta = os.path.join(links_save_path, f"{file_name}.meta.json")

    if not db_ids:
        raise ValueError(f"instance_id={instance_id} has no candidate databases.")

    trace_json(
        "pipeline/run_input.json",
        {
            "instance_id": instance_id,
            "candidate_db_ids": db_ids,
            "question": question,
            "open_schema_linking": open_schema_linking,
        },
    )

    if os.path.isfile(output_csv):
        df = pd.read_csv(output_csv)
        selected_database = None
        if os.path.isfile(output_meta):
            with open(output_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            selected_database = meta.get("selected_database")
        if open_schema_linking:
            context = parse_schema_context(df)
            trace_text("pipeline/05_final_schema_subset.txt", context)
            schema_links = SchemaLinkingTool.generate_by_multi_agent(
                llm=llm,
                query=question,
                context=context,
                turn_n=1,
                linker_num=3,
                logger=logger,
            )
            schema_links = schema_links.replace("`", "").replace("\n", "").replace("python", "")
            os.makedirs(links_save_path, exist_ok=True)
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(schema_links)
            meta = {
                "candidate_db_ids": db_ids,
                "selected_database": selected_database,
                "effective_candidate_db_ids": sorted(df["Database name"].dropna().astype(str).unique().tolist()),
                "schema_linking_generated": True,
            }
            write_meta(output_meta, meta)
            trace_text("pipeline/06_schema_links.txt", schema_links)
            trace_json("pipeline/run_output_meta.json", meta)
            return df, schema_links
        return df

    os.makedirs(save_path, exist_ok=True)
    selected_database, locate_raw_output = select_database(db_ids=db_ids, question=question)
    if not selected_database:
        raise ValueError(
            f"Failed to select a database from candidates={db_ids}. locate_raw_output={locate_raw_output!r}"
        )

    df = get_schema_for_single_db(
        db_id=selected_database,
        question=question,
        instance_id=instance_id,
        reserve_size=reserve_size,
        min_retrival_size=min_retrival_size,
        filter_chunk_size=filter_chunk_size,
        post_retrieval_size=post_retrieval_size,
        post_retrieval_turn=post_retrieval_turn,
        reserve_rate=reserve_rate,
        open_schema_linking=False,
    )
    df.to_csv(output_csv, index=False, encoding="utf-8")

    effective_candidate_db_ids = sorted(df["Database name"].dropna().astype(str).unique().tolist())

    if open_schema_linking:
        context = parse_schema_context(df)
        trace_text("pipeline/05_final_schema_subset.txt", context)
        schema_links = SchemaLinkingTool.generate_by_multi_agent(
            llm=llm,
            query=question,
            context=context,
            turn_n=1,
            linker_num=3,
            logger=logger,
        )
        schema_links = schema_links.replace("`", "").replace("\n", "").replace("python", "")
        os.makedirs(links_save_path, exist_ok=True)
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(schema_links)

        meta = {
            "candidate_db_ids": db_ids,
            "selected_database": selected_database,
            "locate_raw_output": locate_raw_output,
            "effective_candidate_db_ids": effective_candidate_db_ids,
            "schema_linking_generated": True,
            "locate_only": False,
        }
        write_meta(output_meta, meta)
        trace_text("pipeline/06_schema_links.txt", schema_links)
        trace_json("pipeline/run_output_meta.json", meta)
        return df, schema_links

    meta = {
        "candidate_db_ids": db_ids,
        "selected_database": selected_database,
        "locate_raw_output": locate_raw_output,
        "effective_candidate_db_ids": effective_candidate_db_ids,
        "schema_linking_generated": False,
        "locate_only": False,
    }
    write_meta(output_meta, meta)
    trace_json("pipeline/run_output_meta.json", meta)
    return df


def process_row(index, row):
    candidate_db_ids = normalize_candidate_db_ids(row.get(candidate_db_key))
    external = load_external_knowledge(row["external_knowledge"])
    question = row["question"] + external if external else row["question"]
    try:
        get_schema_multi(
            db_ids=candidate_db_ids,
            question=question,
            instance_id=row["instance_id"],
            open_schema_linking=open_schema_linking,
        )
    except Exception as e:
        logger.exception("Failed on row index=%s instance_id=%s error=%s", index, row.get("instance_id"), e)


def wrapper(args):
    index, row = args
    return process_row(index, row)


if __name__ == "__main__":
    args = parse_arguments()
    trace_root_dir = args.trace_dir
    if trace_root_dir:
        os.environ["LINKALIGN_TRACE_DIR"] = trace_root_dir
    llm = build_llm(model_name=args.llm_model_name, temperature=0.85)
    filter_llm = build_llm(model_name=args.filter_llm_model_name, temperature=0.42)
    logger.info("Using llm=%s filter_llm=%s", args.llm_model_name, args.filter_llm_model_name)

    save_path = args.save_path
    schema_path = args.schema_path
    dataset_path = args.dataset
    db_info_path = args.db_info_path
    links_save_path = args.links_save_path
    external_info_path = args.external_info_path
    open_schema_linking = args.open_schema_linking
    candidate_db_key = args.candidate_db_key

    with open(db_info_path, "r", encoding="utf-8") as file:
        db_info = json.load(file)

    val_df = load_data(dataset_path)
    inputs = list(val_df.iterrows())

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(tqdm(executor.map(wrapper, inputs), total=len(inputs)))

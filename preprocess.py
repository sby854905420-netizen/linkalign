import json
import os
import argparse
from typing import Dict, List
from llama_index.core.node_parser import SentenceSplitter
from rag_pipes.RagPipeline import RagPipeLines


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def transform_name(table_name, col_name):
    prefix = rf"{table_name}_{col_name}"
    prefix = prefix if len(prefix) < 100 else prefix[:100]

    syn_lis = ["(", ")", "%", "/"]
    for syn in syn_lis:
        if syn in prefix:
            prefix = prefix.replace(syn, "_")

    return prefix


def process_data(row: Dict, save_path: str, exclude_db: List = None):
    """ 将经过 preprocess 的单个数据库拆分为若干 col json 文件"""
    column_info_lis = []
    db_id = row["db_id"]
    if exclude_db is not None:
        if db_id in exclude_db:
            return

    tables = row["table_names"]
    columns = row["column_names"]
    descriptions = row["column_descriptions"]
    types = row["column_types"]
    samples = row["sample_rows"]  # dict 对象

    for ind, (table_ind, col_name) in enumerate(columns):
        col_info = dict()
        col_info["column_name"] = col_name
        try:
            col_info["column_descriptions"] = descriptions[ind][1]
        except:
            col_info["column_descriptions"] = None
        col_info["column_types"] = types[ind]

        table_name = tables[table_ind]
        sample_lis = samples[table_name]
        sample_rows = []

        for sample in sample_lis:
            try:
                sample_rows.append(sample[col_name])
            except:
                pass

        col_info["sample_rows"] = sample_rows

        meta_data = {
            "db_id": db_id,
            "table_name": table_name,
        }
        col_info["meta_data"] = meta_data

        column_info_lis.append(col_info)

    # 保存 schema 文件至本地
    folder_path = os.path.join(save_path, db_id)
    os.makedirs(folder_path, exist_ok=True)

    for col in column_info_lis:
        table_name = col["meta_data"]["table_name"]
        col_name = col["column_name"]

        prefix = transform_name(table_name, col_name)

        with open(os.path.join(folder_path, f"{prefix}.json"), 'w', encoding='utf-8') as f:
            json.dump(col, f, ensure_ascii=False, indent=4)

    return column_info_lis


def build_index(embed_model_name:str, row: Dict, save_path: str, exclude_db: List = None):
    db_id = row["db_id"]
    if exclude_db is not None:
        if db_id in exclude_db:
            return
    schema_path = os.path.join(save_path, row["db_id"])
    vector_index = RagPipeLines.build_index_from_source(
        data_source=schema_path,
        persist_dir=os.path.join(schema_path, f"{embed_model_name}_vector_store"),
        is_vector_store_exist=False,
        index_method="VectorStoreIndex",
        embed_model_name=embed_model_name,
        parser=SentenceSplitter(chunk_size=10000, chunk_overlap=0),
    )

    return vector_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess schema data and build vector indexes.")
    parser.add_argument(
        "--data_path",
        default="./spider2_dev/tables_new_preprocessed.json",
        help="Path to preprocessed dataset json file.",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Directory to save per-database schema files. Defaults to <data_path_dir>/schemas.",
    )
    parser.add_argument(
        "--embed_model_name",
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model name used for vector index building.",
    )
    parser.add_argument(
        "--skip_build_index",
        action="store_true",
        help="Skip vector index building.",
    )
    parser.add_argument(
        "--skip_process_data",
        action="store_true",
        help="Skip generating per-column schema json files.",
    )

    args = parser.parse_args()

    data_path = args.data_path
    base_dir = os.path.dirname(os.path.abspath(data_path))
    save_dir = args.save_dir if args.save_dir else os.path.join(base_dir, "schemas")
    exclude_db = []  # The databases need to exclude

    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    with open(data_path, "r", encoding="utf-8") as f:
        data_lis = json.load(f)

    
    if not args.skip_process_data:
        for row in data_lis:
            # save meta data of schemas for each database
            process_data(row, save_dir, exclude_db)

    if not args.skip_build_index:
        for row in data_lis:
            # build index for every database schema
            build_index(args.embed_model_name, row, save_dir, exclude_db)

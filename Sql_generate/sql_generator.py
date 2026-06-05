from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import *  # type: ignore  # noqa: F401,F403
except ImportError:
    pass


DEFAULT_DATASET_NAME = "Spider2"
DEFAULT_SQL_LLM_NAME = "ministral-3:14b-instruct-2512-fp16"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "sql_generation.txt"
DEFAULT_MAX_INPUT_LENGTH = int(globals().get("CONTEXT_WINDOW", 120000))
DEFAULT_MAX_GENERATION_NUM = int(globals().get("MAX_OUTPUT_TOKENS", 4096))
LINK_SUFFIX = "_agent.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SQL from LinkAlign final schema-linking text files."
    )
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--answer-llm-name", type=str, default=None)
    parser.add_argument("--schema-llm-name", type=str, default=None)
    parser.add_argument(
        "--provider",
        choices=("auto", "ollama", "gpt"),
        default="auto",
        help="auto uses GPT for gpt-* models, otherwise Ollama.",
    )
    parser.add_argument("--schema-links-dir", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--schemas-dir", type=Path, default=None)
    parser.add_argument("--documents-dir", type=Path, default=None)
    parser.add_argument("--prompt-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--sql-dialect", type=str, default=None)
    parser.add_argument("--max-input-length", type=int, default=None)
    parser.add_argument("--max-generation-num", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-key-columns",
        action="store_true",
        help="Add primary-key and foreign-key columns from linked tables to the prompt schema.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_row_value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def load_dataset_index(dataset_path: Path) -> dict[str, dict[str, Any]]:
    rows = load_json(dataset_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of dataset rows in {dataset_path}.")

    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_id = get_row_value(row, "instance_id", "id")
        if sample_id is not None:
            index[str(sample_id)] = row
    return index


def default_sql_dialect(dataset_name: str) -> str:
    if dataset_name.lower() == "mmqa":
        return (
            "Use SQLite SQL. Do not use Snowflake-only features such as QUALIFY, ILIKE, "
            "TRY_CAST, DATEADD, DATEDIFF, TO_DATE, :: casts, or fully qualified "
            "DATABASE.SCHEMA.TABLE notation unless the table name is explicitly shown that way. "
            "Use SQLite-compatible functions such as strftime/date/datetime when date logic is needed."
        )
    if dataset_name.lower() == "spider2":
        return (
            "Use Snowflake SQL. Preserve fully qualified table names exactly as shown, usually "
            "DATABASE.SCHEMA.TABLE. Snowflake features such as CTEs, QUALIFY, ILIKE, DATEADD, "
            "DATEDIFF, TRY_CAST, TO_DATE, TRUE/FALSE boolean literals, and :: casts are allowed "
            "when useful. Do not write SQLite-specific SQL."
        )
    return "Use the dialect implied by the question, schema, and hint."


def setup_logger(output_path: Path) -> tuple[logging.Logger, Path]:
    logger = logging.getLogger("sql_generation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = output_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_path


def find_schema_links_dir(dataset_root: Path, model_name: str | None) -> Path:
    base_dir = dataset_root / "schema_links"
    if model_name:
        direct_path = base_dir / model_name
        if direct_path.is_dir():
            return direct_path
        raise FileNotFoundError(f"Cannot find schema links for model {model_name!r}: {direct_path}")

    candidates = [path for path in base_dir.iterdir() if path.is_dir()] if base_dir.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"Cannot find schema link directories under {base_dir}.")
    candidates.sort(
        key=lambda path: (len(list(path.glob(f"*{LINK_SUFFIX}"))), path.stat().st_mtime),
        reverse=True,
    )
    return candidates[0]


def resolve_output_path(output_path: Path | None, dataset_root: Path, dataset_name: str) -> Path:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = dataset_root / "sql_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"sql_generation_{dataset_name}_{run_id}.json"


def list_schema_link_paths(schema_links_dir: Path, start_index: int, limit: int | None) -> list[Path]:
    paths = sorted(schema_links_dir.glob(f"*{LINK_SUFFIX}"))
    paths = paths[max(0, start_index):]
    if limit is not None:
        paths = paths[: max(0, limit)]
    return paths


def sample_id_from_link_path(path: Path) -> str:
    return path.name.removesuffix(LINK_SUFFIX)


def meta_path_for_link(link_path: Path) -> Path:
    return link_path.with_name(link_path.name.removesuffix(".txt") + ".meta.json")


def load_meta(meta_path: Path) -> dict[str, Any]:
    if not meta_path.is_file():
        return {}
    try:
        payload = load_json(meta_path)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def normalize_model_text(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    fenced_match = re.search(r"```(?:sql|json|python)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced_match is not None:
        text = fenced_match.group(1).strip()
    return text.replace("```", "").strip()


def parse_schema_links(schema_link_text: str) -> dict[str, list[str]]:
    text = normalize_model_text(schema_link_text)
    parsed: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    quoted_values = re.findall(r"'([^']+)'\s*|\"([^\"]+)\"", text)
    candidates = [left or right for left, right in quoted_values]
    candidates.extend(
        match.group(1)
        for match in re.finditer(
            r"(?<![A-Za-z0-9_$])([A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*)+)(?![A-Za-z0-9_$])",
            text,
        )
    )

    ignored_suffixes = {"md", "txt", "json", "csv", "sql"}
    for candidate in candidates:
        cleaned = candidate.strip().strip("`'\"[](),;:")
        if "." not in cleaned:
            continue
        parts = [part.strip().strip("`'\"[](),;:") for part in cleaned.split(".") if part.strip()]
        if len(parts) < 2 or parts[-1].lower() in ignored_suffixes:
            continue
        table_name = ".".join(parts[:-1])
        column_name = parts[-1]
        key = (table_name.lower(), column_name.lower())
        if key in seen:
            continue
        seen.add(key)
        parsed[table_name].append(column_name)

    return dict(parsed)


def load_schema_records(schemas_dir: Path, db_id: str) -> list[dict[str, Any]]:
    schema_dir = schemas_dir / db_id
    if not schema_dir.is_dir():
        return []

    records: list[dict[str, Any]] = []
    for path in sorted(schema_dir.glob("*.json")):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue

        meta_data = payload.get("meta_data")
        if not isinstance(meta_data, Mapping):
            meta_data = {}
        description = str(payload.get("column_descriptions") or "").strip()
        description_lower = description.lower()
        records.append(
            {
                "db_id": str(meta_data.get("db_id") or db_id),
                "table_name": str(meta_data.get("table_name") or "").strip(),
                "column_name": str(payload.get("column_name") or "").strip(),
                "data_type": str(payload.get("column_types") or "NOT_AVAILABLE").strip(),
                "description": description,
                "sample_values": payload.get("sample_rows") or [],
                "is_primary_key": "primary key" in description_lower,
                "is_foreign_key": "foreign key" in description_lower,
            }
        )
    return records


def table_matches(schema_table_name: str, linked_table_name: str) -> bool:
    schema_name = schema_table_name.strip().lower()
    linked_name = linked_table_name.strip().lower()
    return (
        schema_name == linked_name
        or schema_name.endswith(f".{linked_name}")
        or linked_name.endswith(f".{schema_name}")
    )


def column_matches(schema_column_name: str, linked_column_name: str) -> bool:
    return schema_column_name.strip().lower() == linked_column_name.strip().lower()


def record_id(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("db_id") or ""),
        str(record.get("table_name") or ""),
        str(record.get("column_name") or ""),
    )


def select_records(
    all_records: Sequence[dict[str, Any]],
    schema_links: Mapping[str, Sequence[str]],
    include_key_columns: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected_ids: set[tuple[str, str, str]] = set()
    linked_table_names = set(schema_links)

    for record in all_records:
        table_name = str(record.get("table_name") or "")
        column_name = str(record.get("column_name") or "")
        for linked_table_name, linked_column_names in schema_links.items():
            if not table_matches(table_name, linked_table_name):
                continue
            if any(column_matches(column_name, linked_column) for linked_column in linked_column_names):
                selected_ids.add(record_id(record))
                break

    if include_key_columns and linked_table_names:
        for record in all_records:
            table_name = str(record.get("table_name") or "")
            if not any(table_matches(table_name, linked_table) for linked_table in linked_table_names):
                continue
            if record.get("is_primary_key") or record.get("is_foreign_key"):
                selected_ids.add(record_id(record))

    selected_records = [record for record in all_records if record_id(record) in selected_ids]
    metadata = {
        "available_column_count": len(all_records),
        "linked_table_count": len(schema_links),
        "linked_column_count": sum(len(columns) for columns in schema_links.values()),
        "selected_column_count": len(selected_records),
    }
    return selected_records, metadata


def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def render_schema_text(db_id: str, selected_records: Sequence[Mapping[str, Any]]) -> str:
    lines = [f"Database: {db_id}", "", "Primary keys:"]

    primary_keys = [
        f"{record.get('table_name')}.{record.get('column_name')}"
        for record in selected_records
        if is_truthy(record.get("is_primary_key"))
    ]
    if primary_keys:
        lines.extend(f"- {primary_key}" for primary_key in primary_keys)
    else:
        lines.append("NONE")

    lines.extend(["", "Foreign key relationships:"])
    foreign_keys = []
    for record in selected_records:
        if not is_truthy(record.get("is_foreign_key")):
            continue
        description = str(record.get("description") or "")
        match = re.search(
            r"foreign key referencing\s+([A-Za-z0-9_.$\"]+\.[A-Za-z0-9_.$\"]+)",
            description,
            re.IGNORECASE,
        )
        source = f"{record.get('table_name')}.{record.get('column_name')}"
        foreign_keys.append(f"{source} -> {match.group(1) if match else 'NOT_AVAILABLE'}")
    if foreign_keys:
        lines.extend(f"- {foreign_key}" for foreign_key in foreign_keys)
    else:
        lines.append("NONE")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in selected_records:
        grouped[str(record.get("table_name") or "")].append(record)

    for table_name in sorted(grouped):
        lines.extend(["", f"Table: {table_name}"])
        for record in grouped[table_name]:
            description = str(record.get("description") or "").strip() or "NOT_AVAILABLE"
            sample_values = record.get("sample_values") or []
            sample_text = (
                json.dumps(sample_values[:5], ensure_ascii=False)
                if isinstance(sample_values, list) and sample_values
                else "NOT_AVAILABLE"
            )
            lines.extend(
                [
                    f"  Column: {record.get('column_name')}",
                    f"    Data type: {record.get('data_type') or 'NOT_AVAILABLE'}",
                    f"    Description: {description}",
                    f"    Sample values: {sample_text}",
                    "    Value descriptions: NOT_AVAILABLE",
                ]
            )
    return "\n".join(lines)


def resolve_hint(row: Mapping[str, Any], documents_dir: Path) -> str:
    external_knowledge = get_row_value(row, "external_knowledge", "hint", "evidence")
    if external_knowledge is None:
        return "No hint"

    external_text = str(external_knowledge).strip()
    if not external_text:
        return "No hint"

    path = Path(external_text)
    if not path.is_absolute():
        path = documents_dir / external_text
    if path.is_file():
        return path.read_text(encoding="utf-8").strip() or "No hint"
    if Path(external_text).suffix.lower() in {".md", ".txt"}:
        return "No hint"
    return external_text


def render_prompt(
    prompt_template: str,
    schema_text: str,
    question: str,
    hint: str,
    dataset_name: str,
    sql_dialect: str,
) -> str:
    return prompt_template.format(
        DATABASE_SCHEMAS=schema_text,
        QUESTION=question,
        HINT=hint,
        DATASET_NAME=dataset_name,
        SQL_DIALECT=sql_dialect,
    )


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def truncate_to_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    max_chars = token_budget * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0].strip()


def fit_prompt_to_budget(
    prompt_template: str,
    schema_text: str,
    question: str,
    hint: str,
    dataset_name: str,
    sql_dialect: str,
    max_input_length: int,
) -> tuple[str, str, str, int]:
    prompt_cap = int(max_input_length * 0.9)
    prompt = render_prompt(prompt_template, schema_text, question, hint, dataset_name, sql_dialect)
    prompt_tokens = estimate_tokens(prompt)
    if prompt_tokens <= prompt_cap:
        return prompt, schema_text, hint, prompt_tokens

    fitted_hint = hint
    if hint != "No hint":
        prompt_without_hint = render_prompt(prompt_template, schema_text, question, "", dataset_name, sql_dialect)
        hint_budget = prompt_cap - estimate_tokens(prompt_without_hint)
        fitted_hint = truncate_to_budget(hint, hint_budget) or "No hint"
        prompt = render_prompt(prompt_template, schema_text, question, fitted_hint, dataset_name, sql_dialect)
        prompt_tokens = estimate_tokens(prompt)
        if prompt_tokens <= prompt_cap:
            return prompt, schema_text, fitted_hint, prompt_tokens

    prompt_without_schema = render_prompt(prompt_template, "", question, fitted_hint, dataset_name, sql_dialect)
    schema_budget = prompt_cap - estimate_tokens(prompt_without_schema)
    fitted_schema_text = truncate_to_budget(schema_text, schema_budget)
    prompt = render_prompt(prompt_template, fitted_schema_text, question, fitted_hint, dataset_name, sql_dialect)
    return prompt, fitted_schema_text, fitted_hint, estimate_tokens(prompt)


class SqlLLM:
    def __init__(
        self,
        model_name: str,
        provider: str,
        max_input_length: int,
        max_generation_num: int,
    ) -> None:
        self.model_name = model_name
        self.provider = self.resolve_provider(provider, model_name)
        self.max_input_length = max_input_length
        self.max_generation_num = max_generation_num
        self.client = self.build_client()

    @staticmethod
    def resolve_provider(provider: str, model_name: str) -> str:
        if provider != "auto":
            return provider
        return "gpt" if model_name.lower().startswith("gpt-") else "ollama"

    def build_client(self) -> Any:
        if self.provider == "gpt":
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY") or globals().get("GPT_API") or None
            return OpenAI(api_key=api_key)

        from ollama import Client

        host = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST") or globals().get(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )
        return Client(host=host)

    def complete(self, prompt: str) -> tuple[str, int]:
        if self.provider == "gpt":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_generation_num,
                temperature=0.0,
                timeout=100.0,
            )
            text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
            return text, total_tokens

        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.02,
                "num_predict": self.max_generation_num,
                "num_ctx": self.max_input_length,
            },
        )
        if hasattr(response, "get"):
            message = response.get("message", {})
            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)
        else:
            message = getattr(response, "message", {}) or {}
            prompt_tokens = getattr(response, "prompt_eval_count", 0)
            completion_tokens = getattr(response, "eval_count", 0)
        text = message.get("content", "") if hasattr(message, "get") else getattr(message, "content", "")
        return text or "", int(prompt_tokens or 0) + int(completion_tokens or 0)


def normalize_sql_response(response_text: str) -> str:
    text = normalize_model_text(response_text)
    fenced_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced_match is not None:
        text = fenced_match.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping) and isinstance(payload.get("sql"), str):
        text = payload["sql"].strip()

    return re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def write_result_file(output_path: Path, run_info: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> None:
    output_path.write_text(
        json.dumps({"run_info": run_info, "results": list(results)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_error_result(
    sample_id: str,
    source_row: Mapping[str, Any],
    link_path: Path,
    meta_path: Path,
    db_id: str | None,
    schema_links: Mapping[str, Sequence[str]],
    error: str,
    started_at: float,
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "question": source_row.get("question"),
        "gold_db_id": source_row.get("db_id"),
        "gold_sql": source_row.get("SQL") or source_row.get("sql"),
        "predict_db_id": db_id,
        "schema_link_path": str(link_path),
        "schema_meta_path": str(meta_path) if meta_path.is_file() else None,
        "schema_linking": {"predict_columns": schema_links},
        "predict_sql": "",
        "status": "failed",
        "error": error,
        "efficiency": {"elapsed_seconds": round(time.perf_counter() - started_at, 6), "total_tokens": 0},
    }


def process_schema_link_file(
    link_path: Path,
    dataset_index: Mapping[str, dict[str, Any]],
    schemas_dir: Path,
    documents_dir: Path,
    prompt_template: str,
    llm: SqlLLM,
    dataset_name: str,
    sql_dialect: str,
    include_key_columns: bool,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    sample_id = sample_id_from_link_path(link_path)
    source_row = dataset_index.get(sample_id, {})
    meta_path = meta_path_for_link(link_path)
    meta = load_meta(meta_path)
    db_id = str(meta.get("selected_database") or source_row.get("db_id") or "").strip() or None
    schema_link_text = link_path.read_text(encoding="utf-8")
    schema_links = parse_schema_links(schema_link_text)

    if not source_row:
        return build_error_result(sample_id, source_row, link_path, meta_path, db_id, schema_links, "Missing dataset row.", started_at)
    if not db_id:
        return build_error_result(sample_id, source_row, link_path, meta_path, db_id, schema_links, "Missing selected database.", started_at)
    if not schema_links:
        return build_error_result(sample_id, source_row, link_path, meta_path, db_id, schema_links, "No schema links parsed.", started_at)

    all_records = load_schema_records(schemas_dir, db_id)
    if not all_records:
        return build_error_result(sample_id, source_row, link_path, meta_path, db_id, schema_links, "No schema files found.", started_at)

    selected_records, schema_metadata = select_records(all_records, schema_links, include_key_columns)
    if not selected_records:
        return build_error_result(
            sample_id,
            source_row,
            link_path,
            meta_path,
            db_id,
            schema_links,
            "No linked schema columns matched schema files.",
            started_at,
        )

    question = str(source_row.get("question") or "").strip()
    hint = resolve_hint(source_row, documents_dir)
    schema_text = render_schema_text(db_id, selected_records)
    prompt, fitted_schema_text, fitted_hint, prompt_tokens = fit_prompt_to_budget(
        prompt_template=prompt_template,
        schema_text=schema_text,
        question=question,
        hint=hint,
        dataset_name=dataset_name,
        sql_dialect=sql_dialect,
        max_input_length=llm.max_input_length,
    )
    response_text, total_tokens = llm.complete(prompt)
    predict_sql = normalize_sql_response(response_text)

    result: dict[str, Any] = {
        "id": sample_id,
        "question": question,
        "gold_db_id": source_row.get("db_id"),
        "gold_sql": source_row.get("SQL") or source_row.get("sql"),
        "predict_db_id": db_id,
        "schema_link_path": str(link_path),
        "schema_meta_path": str(meta_path) if meta_path.is_file() else None,
        "schema_linking": {
            "raw_text": schema_link_text,
            "predict_columns": schema_links,
            **schema_metadata,
        },
        "predict_sql": predict_sql,
        "status": "success" if predict_sql else "empty",
        "efficiency": {
            "elapsed_seconds": round(time.perf_counter() - started_at, 6),
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
        },
    }
    if response_text.strip() and response_text.strip() != predict_sql:
        result["raw_response"] = response_text
    if fitted_hint != hint or fitted_schema_text != schema_text:
        result["truncation"] = {
            "hint_truncated": fitted_hint != hint,
            "schema_truncated": fitted_schema_text != schema_text,
        }
    return result


def run_sql_generation(
    link_paths: Sequence[Path],
    dataset_index: Mapping[str, dict[str, Any]],
    schemas_dir: Path,
    documents_dir: Path,
    prompt_template: str,
    output_path: Path,
    llm: SqlLLM,
    run_info: Mapping[str, Any],
    dataset_name: str,
    sql_dialect: str,
    include_key_columns: bool,
) -> int:
    from tqdm import tqdm

    results: list[dict[str, Any]] = []
    write_result_file(output_path, run_info, results)

    for link_path in tqdm(link_paths, total=len(link_paths)):
        result = process_schema_link_file(
            link_path=link_path,
            dataset_index=dataset_index,
            schemas_dir=schemas_dir,
            documents_dir=documents_dir,
            prompt_template=prompt_template,
            llm=llm,
            dataset_name=dataset_name,
            sql_dialect=sql_dialect,
            include_key_columns=include_key_columns,
        )
        results.append(result)
        write_result_file(output_path, run_info, results)

    return len(results)


def main() -> None:
    args = parse_args()

    dataset_name = args.dataset_name or DEFAULT_DATASET_NAME
    dataset_root = PROJECT_ROOT / dataset_name
    answer_llm_name = args.answer_llm_name or DEFAULT_SQL_LLM_NAME
    schema_llm_name = args.schema_llm_name
    schema_links_dir = args.schema_links_dir or find_schema_links_dir(dataset_root, schema_llm_name)
    schema_source_model = schema_links_dir.name
    dataset_path = args.dataset_path or (dataset_root / "Synthesized_preprocessed_data.json")
    schemas_dir = args.schemas_dir or (dataset_root / "schemas")
    documents_dir = args.documents_dir or (dataset_root / "external_knowledge")
    prompt_path = args.prompt_path or DEFAULT_PROMPT_PATH
    output_path = resolve_output_path(args.output_path, dataset_root, dataset_name)
    max_input_length = args.max_input_length or DEFAULT_MAX_INPUT_LENGTH
    max_generation_num = args.max_generation_num or DEFAULT_MAX_GENERATION_NUM
    sql_dialect = args.sql_dialect or default_sql_dialect(dataset_name)

    logger, log_path = setup_logger(output_path)
    link_paths = list_schema_link_paths(schema_links_dir, args.start_index, args.limit)
    dataset_index = load_dataset_index(dataset_path)
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    llm = SqlLLM(
        model_name=answer_llm_name,
        provider=args.provider,
        max_input_length=max_input_length,
        max_generation_num=max_generation_num,
    )

    run_info = {
        "task": "sql_generation",
        "dataset_name": dataset_name,
        "model": answer_llm_name,
        "provider": llm.provider,
        "schema_source_model": schema_source_model,
        "schema_links_dir": str(schema_links_dir),
        "dataset_path": str(dataset_path),
        "schemas_dir": str(schemas_dir),
        "documents_dir": str(documents_dir),
        "prompt_template": str(prompt_path),
        "sql_dialect": sql_dialect,
        "max_input_length": max_input_length,
        "max_generation_num": max_generation_num,
        "start_index": args.start_index,
        "limit": args.limit,
        "include_key_columns": args.include_key_columns,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    logger.info("SQL Generation started.")
    logger.info("Dataset=%s schema_links=%s records=%s", dataset_name, schema_links_dir, len(link_paths))
    logger.info("Model=%s provider=%s", answer_llm_name, llm.provider)
    logger.info("Prompt=%s", prompt_path)
    logger.info("Output=%s", output_path)
    logger.info("Log=%s", log_path)

    processed_count = run_sql_generation(
        link_paths=link_paths,
        dataset_index=dataset_index,
        schemas_dir=schemas_dir,
        documents_dir=documents_dir,
        prompt_template=prompt_template,
        output_path=output_path,
        llm=llm,
        run_info=run_info,
        dataset_name=dataset_name,
        sql_dialect=sql_dialect,
        include_key_columns=args.include_key_columns,
    )
    logger.info("Completed %s records.", processed_count)


if __name__ == "__main__":
    main()

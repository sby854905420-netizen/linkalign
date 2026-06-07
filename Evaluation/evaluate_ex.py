#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ID_KEYS = ("id", "sample_id", "instance_id")
DEFAULT_GOLD_SQL_KEYS = ("gold_sql", "SQL", "gold_query", "query", "sql")
DEFAULT_PRED_SQL_KEYS = (
    "predict_sql",
    "pred_sql",
    "predicted_sql",
    "prediction_sql",
    "generated_sql",
    "model_sql",
    "prediction",
)
DEFAULT_DB_ID_KEYS = ("gold_db_id", "db_id", "database_id", "database")
DEFAULT_PRED_DB_ID_KEYS = (
    "predict_db_id",
    "pred_db_id",
    "predicted_db_id",
    "selected_database",
)

LIST_PAYLOAD_KEYS = ("results", "records", "data", "examples", "items")
SQL_BANNED_TOKENS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "MERGE",
    "COPY",
    "GRANT",
    "REVOKE",
    "CALL",
    "PUT",
    "REMOVE",
    "USE",
    "UNDROP",
}


@dataclass
class EvalItem:
    index: int
    sample_id: str
    question: Any
    gold_db_id: str | None
    predict_db_id: str | None
    gold_sql: str | None
    predict_sql: str | None
    source_status: Any


@dataclass
class QueryResult:
    ok: bool
    rows: list[tuple[Any, ...]]
    columns: list[str]
    elapsed_seconds: float
    error: str | None = None

    @property
    def row_count(self) -> int:
        return len(self.rows)


class EvaluationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Execution Accuracy (EX) for LinkAlign SQL generation outputs. "
            "Predicted and gold SQL are executed on the gold database; a sample is "
            "correct only when both queries execute successfully and return the same rows."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="SQL result JSON/JSONL file.")
    parser.add_argument("--output", type=Path, default=None, help="Evaluation JSON output path.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional source dataset JSON/JSONL used to fill missing gold SQL/db_id fields.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name, e.g. MMQA or Spider2. Inferred from input when omitted.",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "sqlite", "snowflake"),
        default="auto",
        help="Execution backend. auto uses SQLite when a SQLite database directory is found.",
    )
    parser.add_argument(
        "--sqlite-dir",
        type=Path,
        default=None,
        help="Directory containing local SQLite databases named <db_id>.sqlite or <db_id>.db.",
    )
    parser.add_argument(
        "--snowflake-credential",
        type=Path,
        default=None,
        help=(
            "Snowflake credential JSON. If omitted, env vars are used, then "
            "PROJECT_ROOT/snowflake_credential.json when present."
        ),
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-query timeout in seconds.")
    parser.add_argument(
        "--float-tol",
        type=float,
        default=1e-6,
        help="Absolute numeric rounding tolerance used before result comparison.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional per-query fetch cap for debugging. 0 means fetch all rows.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="First input row index to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to evaluate.")
    parser.add_argument(
        "--skip-gold-errors",
        action="store_true",
        help="Exclude samples whose gold SQL cannot execute from the EX denominator.",
    )
    parser.add_argument(
        "--require-db-match",
        action="store_true",
        help="Score samples as 0 without execution when predict_db_id differs from gold_db_id.",
    )
    parser.add_argument("--sample-id-key", type=str, default=None, help="Override sample id field name.")
    parser.add_argument("--gold-sql-key", type=str, default=None, help="Override gold SQL field name.")
    parser.add_argument("--pred-sql-key", type=str, default=None, help="Override predicted SQL field name.")
    parser.add_argument("--db-id-key", type=str, default=None, help="Override gold database id field name.")
    parser.add_argument(
        "--pred-db-id-key",
        type=str,
        default=None,
        help="Override predicted database id field name.",
    )
    parser.add_argument(
        "--include-sql",
        action="store_true",
        help="Include gold/predicted SQL strings in per-sample output records.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print one progress line every N samples. 0 disables progress output.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Write the output JSON every N samples. 0 writes only once at the end.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from_payload(payload: Any, path: Path) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in LIST_PAYLOAD_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        if rows is None:
            raise ValueError(
                f"Cannot find a list of records in {path}. Tried keys: {', '.join(LIST_PAYLOAD_KEYS)}."
            )
    else:
        raise ValueError(f"Expected JSON list or object in {path}, got {type(payload).__name__}.")

    normalized_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Expected record {index} in {path} to be a JSON object.")
        normalized_rows.append(row)
    return normalized_rows


def get_first_value(row: Mapping[str, Any], override_key: str | None, default_keys: Sequence[str]) -> Any:
    keys = (override_key,) if override_key else default_keys
    for key in keys:
        if key is None:
            continue
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def get_sample_id(row: Mapping[str, Any], override_key: str | None, fallback_index: int) -> str:
    value = get_first_value(row, override_key, DEFAULT_ID_KEYS)
    return str(value) if value is not None else str(fallback_index)


def infer_dataset_name(args: argparse.Namespace, input_payload: Any, input_path: Path) -> str | None:
    if args.dataset_name:
        return args.dataset_name
    if isinstance(input_payload, Mapping):
        run_info = input_payload.get("run_info")
        if isinstance(run_info, Mapping) and run_info.get("dataset_name"):
            return str(run_info["dataset_name"])
        run = input_payload.get("run")
        if isinstance(run, Mapping):
            static_metadata = run.get("static_metadata")
            if isinstance(static_metadata, Mapping) and static_metadata.get("dataset_name"):
                return str(static_metadata["dataset_name"])
    for part in reversed(input_path.parts):
        if part.lower() in {"mmqa", "spider2"}:
            return part
    return None


def infer_schema_linking_model(input_payload: Any, input_rows: Sequence[Mapping[str, Any]]) -> str | None:
    if isinstance(input_payload, Mapping):
        run_info = input_payload.get("run_info")
        if isinstance(run_info, Mapping):
            model_name = run_info.get("schema_source_model")
            if model_name:
                return str(model_name)
            schema_links_dir = run_info.get("schema_links_dir")
            if schema_links_dir:
                return Path(str(schema_links_dir)).name

    for row in input_rows:
        for key in ("schema_link_path", "schema_meta_path"):
            value = row.get(key)
            if value:
                return Path(str(value)).parent.name
    return None


def default_dataset_path(dataset_name: str | None) -> Path | None:
    if not dataset_name:
        return None
    candidate = PROJECT_ROOT / dataset_name / "Synthesized_preprocessed_data.json"
    return candidate if candidate.is_file() else None


def build_dataset_index(
    dataset_path: Path | None,
    args: argparse.Namespace,
) -> dict[str, Mapping[str, Any]]:
    if dataset_path is None:
        return {}
    payload = load_payload(dataset_path)
    rows = rows_from_payload(payload, dataset_path)
    index: dict[str, Mapping[str, Any]] = {}
    for row_index, row in enumerate(rows):
        sample_id = get_sample_id(row, args.sample_id_key, row_index)
        index[sample_id] = row
    return index


def build_eval_items(
    input_rows: Sequence[Mapping[str, Any]],
    dataset_index: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> list[EvalItem]:
    start = max(0, args.start_index)
    end = len(input_rows) if args.limit is None else min(len(input_rows), start + max(0, args.limit))
    items: list[EvalItem] = []

    for row_index in range(start, end):
        row = input_rows[row_index]
        sample_id = get_sample_id(row, args.sample_id_key, row_index)
        dataset_row = dataset_index.get(sample_id, {})

        gold_sql = get_first_value(row, args.gold_sql_key, DEFAULT_GOLD_SQL_KEYS)
        if gold_sql is None:
            gold_sql = get_first_value(dataset_row, args.gold_sql_key, DEFAULT_GOLD_SQL_KEYS)

        gold_db_id = get_first_value(row, args.db_id_key, DEFAULT_DB_ID_KEYS)
        if gold_db_id is None:
            gold_db_id = get_first_value(dataset_row, args.db_id_key, DEFAULT_DB_ID_KEYS)

        question = row.get("question")
        if question is None:
            question = dataset_row.get("question")

        items.append(
            EvalItem(
                index=row_index,
                sample_id=sample_id,
                question=question,
                gold_db_id=str(gold_db_id).strip() if gold_db_id is not None else None,
                predict_db_id=string_or_none(
                    get_first_value(row, args.pred_db_id_key, DEFAULT_PRED_DB_ID_KEYS)
                ),
                gold_sql=str(gold_sql) if gold_sql is not None else None,
                predict_sql=string_or_none(
                    get_first_value(row, args.pred_sql_key, DEFAULT_PRED_SQL_KEYS)
                ),
                source_status=row.get("status"),
            )
        )
    return items


def string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return PROJECT_ROOT / "Evaluation" / f"{input_path.stem}_ex_eval.json"


def resolve_sqlite_dir(args: argparse.Namespace, dataset_name: str | None, input_path: Path) -> Path | None:
    candidates: list[Path] = []
    if args.sqlite_dir is not None:
        candidates.append(args.sqlite_dir)
    if dataset_name:
        candidates.append(PROJECT_ROOT / dataset_name / "Sqlite_database")
    for parent in input_path.parents:
        candidates.append(parent / "Sqlite_database")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return args.sqlite_dir


def resolve_engine(args: argparse.Namespace, dataset_name: str | None, sqlite_dir: Path | None) -> str:
    if args.engine != "auto":
        return args.engine
    if sqlite_dir is not None and sqlite_dir.is_dir():
        return "sqlite"
    if dataset_name and dataset_name.lower() == "mmqa":
        return "sqlite"
    return "snowflake"


def extract_sql_text(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    fenced_match = re.search(r"```(?:sql|json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping):
        for key in ("sql", "SQL", "query", "predict_sql", "gold_sql"):
            candidate = payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                text = candidate.strip()
                break

    return re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def mask_sql_comments_and_literals(sql: str) -> str:
    chars: list[str] = []
    i = 0
    length = len(sql)
    state = "normal"
    quote_char = ""

    while i < length:
        char = sql[i]
        next_char = sql[i + 1] if i + 1 < length else ""

        if state == "normal":
            if char == "-" and next_char == "-":
                chars.extend("  ")
                i += 2
                state = "line_comment"
                continue
            if char == "/" and next_char == "*":
                chars.extend("  ")
                i += 2
                state = "block_comment"
                continue
            if char in {"'", '"', "`"}:
                quote_char = char
                chars.append(" ")
                i += 1
                state = "quoted"
                continue
            chars.append(char)
            i += 1
            continue

        if state == "line_comment":
            if char == "\n":
                chars.append("\n")
                state = "normal"
            else:
                chars.append(" ")
            i += 1
            continue

        if state == "block_comment":
            if char == "*" and next_char == "/":
                chars.extend("  ")
                i += 2
                state = "normal"
            else:
                chars.append("\n" if char == "\n" else " ")
                i += 1
            continue

        if state == "quoted":
            if char == quote_char:
                chars.append(" ")
                if quote_char == "'" and next_char == "'":
                    chars.append(" ")
                    i += 2
                    continue
                i += 1
                state = "normal"
            else:
                chars.append("\n" if char == "\n" else " ")
                i += 1
            continue

    return "".join(chars)


def prepare_query_sql(sql: str | None) -> tuple[bool, str, str | None]:
    text = extract_sql_text(sql)
    if not text:
        return False, "", "empty_sql"

    masked = mask_sql_comments_and_literals(text)
    stripped_masked = masked.strip()
    without_trailing_semicolons = re.sub(r"(?:;\s*)+\Z", "", stripped_masked)
    if ";" in without_trailing_semicolons:
        return False, "", "multiple_statements_are_not_allowed"

    text = re.sub(r"(?:;\s*)+\Z", "", text.strip())
    masked = mask_sql_comments_and_literals(text)
    first_keyword_match = re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\b", masked)
    if not first_keyword_match:
        return False, "", "missing_sql_keyword"
    first_keyword = first_keyword_match.group(0).upper()
    if first_keyword not in {"SELECT", "WITH"}:
        return False, "", f"non_query_sql_is_not_allowed:{first_keyword}"

    tokens = {token.upper() for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", masked)}
    banned = sorted(tokens & SQL_BANNED_TOKENS)
    if banned:
        return False, "", "banned_sql_token:" + ",".join(banned)

    return True, text, None


def has_top_level_order_by(sql: str | None) -> bool:
    text = extract_sql_text(sql)
    masked = mask_sql_comments_and_literals(text)
    token_pattern = re.compile(r"\bORDER\b|\bBY\b|[()]", re.IGNORECASE)
    depth = 0
    saw_order_at_depth_zero = False

    for match in token_pattern.finditer(masked):
        token = match.group(0).upper()
        if token == "(":
            depth += 1
            saw_order_at_depth_zero = False
        elif token == ")":
            depth = max(0, depth - 1)
            saw_order_at_depth_zero = False
        elif token == "ORDER" and depth == 0:
            saw_order_at_depth_zero = True
        elif token == "BY" and depth == 0 and saw_order_at_depth_zero:
            return True
        elif token != "ORDER":
            saw_order_at_depth_zero = False
    return False


class SqliteExecutor:
    def __init__(self, sqlite_dir: Path | None, timeout: float, max_rows: int):
        if sqlite_dir is None:
            raise EvaluationError("SQLite engine selected, but no SQLite directory was found.")
        self.sqlite_dir = sqlite_dir
        self.timeout = timeout
        self.max_rows = max_rows
        self._connections: dict[Path, sqlite3.Connection] = {}

    def execute(self, sql: str, db_id: str | None) -> QueryResult:
        valid, prepared_sql, validation_error = prepare_query_sql(sql)
        if not valid:
            return QueryResult(False, [], [], 0.0, validation_error)
        if not db_id:
            return QueryResult(False, [], [], 0.0, "missing_gold_db_id")

        started_at = time.perf_counter()
        try:
            db_path = find_sqlite_db_path(self.sqlite_dir, db_id)
            connection = self._get_connection(db_path)
            deadline = time.monotonic() + self.timeout

            def progress_handler() -> int:
                return 1 if time.monotonic() > deadline else 0

            connection.set_progress_handler(progress_handler, 1000)
            cursor = connection.cursor()
            try:
                cursor.execute(prepared_sql)
                rows = fetch_rows(cursor, self.max_rows)
                columns = [description[0] for description in cursor.description or []]
            finally:
                cursor.close()
                connection.set_progress_handler(None, 0)
        except Exception as exc:  # noqa: BLE001 - this is an evaluator; errors are sample outcomes.
            return QueryResult(
                False,
                [],
                [],
                round(time.perf_counter() - started_at, 6),
                normalize_error(exc),
            )

        return QueryResult(True, rows, columns, round(time.perf_counter() - started_at, 6))

    def _get_connection(self, db_path: Path) -> sqlite3.Connection:
        connection = self._connections.get(db_path)
        if connection is not None:
            return connection

        uri = f"file:{quote(str(db_path), safe='/')}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only = ON")
        self._connections[db_path] = connection
        return connection

    def close(self) -> None:
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()


class SnowflakeExecutor:
    def __init__(self, credential_path: Path | None, timeout: float, max_rows: int):
        try:
            import snowflake.connector  # type: ignore
        except ImportError as exc:
            raise EvaluationError(
                "Snowflake engine selected, but snowflake-connector-python is not installed."
            ) from exc

        self._snowflake_connector = snowflake.connector
        self.timeout = timeout
        self.max_rows = max_rows
        self.connection = self._snowflake_connector.connect(**load_snowflake_connection_params(credential_path))

    def execute(self, sql: str, db_id: str | None) -> QueryResult:
        valid, prepared_sql, validation_error = prepare_query_sql(sql)
        if not valid:
            return QueryResult(False, [], [], 0.0, validation_error)

        started_at = time.perf_counter()
        cursor = self.connection.cursor()
        try:
            if db_id:
                cursor.execute(f"USE DATABASE {format_snowflake_identifier(db_id)}")
            cursor.execute(prepared_sql, timeout=int(math.ceil(self.timeout)))
            rows = fetch_rows(cursor, self.max_rows)
            columns = [description[0] for description in cursor.description or []]
        except Exception as exc:  # noqa: BLE001 - this is an evaluator; errors are sample outcomes.
            return QueryResult(
                False,
                [],
                [],
                round(time.perf_counter() - started_at, 6),
                normalize_error(exc),
            )
        finally:
            cursor.close()

        return QueryResult(True, rows, columns, round(time.perf_counter() - started_at, 6))

    def close(self) -> None:
        self.connection.close()


def find_sqlite_db_path(sqlite_dir: Path, db_id: str) -> Path:
    raw = Path(db_id)
    candidates = []
    if raw.suffix:
        candidates.append(sqlite_dir / raw.name)
    else:
        candidates.extend(
            [
                sqlite_dir / f"{db_id}.sqlite",
                sqlite_dir / f"{db_id}.db",
                sqlite_dir / f"{db_id}.sqlite3",
                sqlite_dir / db_id / f"{db_id}.sqlite",
                sqlite_dir / db_id / f"{db_id}.db",
                sqlite_dir / db_id / f"{db_id}.sqlite3",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    lower_name_candidates = {
        f"{db_id}.sqlite".lower(),
        f"{db_id}.db".lower(),
        f"{db_id}.sqlite3".lower(),
        raw.name.lower(),
    }
    for candidate in sqlite_dir.rglob("*"):
        if candidate.is_file() and candidate.name.lower() in lower_name_candidates:
            return candidate

    raise FileNotFoundError(f"Cannot find SQLite database for db_id={db_id!r} under {sqlite_dir}.")


def fetch_rows(cursor: Any, max_rows: int) -> list[tuple[Any, ...]]:
    if max_rows and max_rows > 0:
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise EvaluationError(f"max_rows_exceeded:{max_rows}")
    else:
        rows = cursor.fetchall()
    return [tuple(row) for row in rows]


def load_snowflake_connection_params(credential_path: Path | None) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if credential_path is None:
        default_credential_path = PROJECT_ROOT / "snowflake_credential.json"
        credential_path = default_credential_path if default_credential_path.is_file() else None
    if credential_path is not None and credential_path.is_file():
        payload = json.loads(credential_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise EvaluationError(f"Snowflake credential file must contain a JSON object: {credential_path}")
        raw.update(payload)

    env_mapping = {
        "account": "SNOWFLAKE_ACCOUNT",
        "user": "SNOWFLAKE_USER",
        "username": "SNOWFLAKE_USERNAME",
        "password": "SNOWFLAKE_PASSWORD",
        "warehouse": "SNOWFLAKE_WAREHOUSE",
        "role": "SNOWFLAKE_ROLE",
        "database": "SNOWFLAKE_DATABASE",
        "schema": "SNOWFLAKE_SCHEMA",
    }
    for key, env_name in env_mapping.items():
        value = os.environ.get(env_name)
        if value:
            raw[key] = value

    params = {
        "account": raw.get("account"),
        "user": raw.get("user") or raw.get("username"),
        "password": raw.get("password"),
        "warehouse": raw.get("warehouse"),
        "role": raw.get("role"),
        "database": raw.get("database"),
        "schema": raw.get("schema"),
    }
    params = {key: value for key, value in params.items() if value not in (None, "")}
    missing = [key for key in ("account", "user", "password") if key not in params]
    if missing:
        raise EvaluationError(
            "Missing Snowflake credential fields: "
            + ", ".join(missing)
            + ". Provide --snowflake-credential or SNOWFLAKE_* environment variables."
        )
    return params


def format_snowflake_identifier(identifier: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", identifier):
        return identifier
    return '"' + identifier.replace('"', '""') + '"'


def normalize_error(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return re.sub(r"\s+", " ", text)[:1000]


def canonical_decimal(value: Decimal, float_tol: float) -> str:
    if not value.is_finite():
        return str(value)
    if float_tol > 0:
        quantum = Decimal(str(float_tol))
        try:
            with localcontext() as context:
                context.prec = max(50, len(value.as_tuple().digits) + 10)
                value = value.quantize(quantum, rounding=ROUND_HALF_UP)
        except (InvalidOperation, ValueError):
            pass
    if value == 0:
        return "0"
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def canonical_scalar(value: Any, float_tol: float) -> str:
    if value is None:
        return "null:"
    if isinstance(value, bool):
        return f"bool:{int(value)}"
    if isinstance(value, int):
        return "num:" + canonical_decimal(Decimal(value), float_tol)
    if isinstance(value, float):
        if math.isnan(value):
            return "num:nan"
        if math.isinf(value):
            return "num:inf" if value > 0 else "num:-inf"
        return "num:" + canonical_decimal(Decimal(str(value)), float_tol)
    if isinstance(value, Decimal):
        return "num:" + canonical_decimal(value, float_tol)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return "time:" + value.isoformat()
    if isinstance(value, bytes):
        return "bytes:" + base64.b64encode(value).decode("ascii")
    if isinstance(value, (list, tuple, dict)):
        return "json:" + json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return "str:" + str(value)


def canonical_rows(rows: Sequence[tuple[Any, ...]], float_tol: float) -> list[tuple[str, ...]]:
    return [tuple(canonical_scalar(value, float_tol) for value in row) for row in rows]


def compare_results(
    pred_rows: Sequence[tuple[Any, ...]],
    gold_rows: Sequence[tuple[Any, ...]],
    ordered: bool,
    float_tol: float,
) -> tuple[bool, dict[str, Any]]:
    pred_normalized = canonical_rows(pred_rows, float_tol)
    gold_normalized = canonical_rows(gold_rows, float_tol)

    detail: dict[str, Any] = {
        "ordered": ordered,
        "pred_row_count": len(pred_normalized),
        "gold_row_count": len(gold_normalized),
    }
    if ordered:
        if pred_normalized == gold_normalized:
            return True, detail
        detail["first_mismatch"] = first_ordered_mismatch(pred_normalized, gold_normalized)
        return False, detail

    pred_counter = Counter(pred_normalized)
    gold_counter = Counter(gold_normalized)
    if pred_counter == gold_counter:
        return True, detail

    detail["missing_from_prediction"] = counter_examples(gold_counter - pred_counter)
    detail["extra_in_prediction"] = counter_examples(pred_counter - gold_counter)
    return False, detail


def first_ordered_mismatch(
    pred_rows: Sequence[tuple[str, ...]],
    gold_rows: Sequence[tuple[str, ...]],
) -> dict[str, Any]:
    max_common = min(len(pred_rows), len(gold_rows))
    for index in range(max_common):
        if pred_rows[index] != gold_rows[index]:
            return {
                "index": index,
                "pred": list(pred_rows[index]),
                "gold": list(gold_rows[index]),
            }
    return {
        "index": max_common,
        "pred": list(pred_rows[max_common]) if max_common < len(pred_rows) else None,
        "gold": list(gold_rows[max_common]) if max_common < len(gold_rows) else None,
    }


def counter_examples(counter: Counter[tuple[str, ...]], limit: int = 3) -> list[dict[str, Any]]:
    examples = []
    for row, count in counter.most_common(limit):
        examples.append({"count": count, "row": list(row)})
    return examples


def evaluate_item(
    item: EvalItem,
    executor: SqliteExecutor | SnowflakeExecutor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "index": item.index,
        "id": item.sample_id,
        "gold_db_id": item.gold_db_id,
        "predict_db_id": item.predict_db_id,
        "source_status": item.source_status,
        "score": 0,
        "included_in_ex": True,
    }
    if item.question is not None:
        record["question"] = item.question
    if args.include_sql:
        record["gold_sql"] = item.gold_sql
        record["predict_sql"] = item.predict_sql

    if not item.gold_db_id:
        record.update({"status": "missing_gold_db_id", "error": "missing_gold_db_id"})
        return record
    if not item.gold_sql:
        record.update({"status": "missing_gold_sql", "error": "missing_gold_sql"})
        return record
    if not item.predict_sql:
        record.update({"status": "empty_prediction", "error": "empty_prediction"})
        return record
    if (
        args.require_db_match
        and item.predict_db_id
        and item.gold_db_id
        and item.predict_db_id.lower() != item.gold_db_id.lower()
    ):
        record.update({"status": "db_mismatch", "error": "predict_db_id differs from gold_db_id"})
        return record

    ordered = has_top_level_order_by(item.gold_sql)
    record["ordered"] = ordered

    gold_result = executor.execute(item.gold_sql, item.gold_db_id)
    record["gold_execution"] = summarize_query_result(gold_result)
    if not gold_result.ok:
        record.update({"status": "gold_error", "error": gold_result.error})
        if args.skip_gold_errors:
            record["included_in_ex"] = False
        return record

    pred_result = executor.execute(item.predict_sql, item.gold_db_id)
    record["pred_execution"] = summarize_query_result(pred_result)
    if not pred_result.ok:
        record.update({"status": "pred_error", "error": pred_result.error})
        return record

    is_correct, detail = compare_results(
        pred_rows=pred_result.rows,
        gold_rows=gold_result.rows,
        ordered=ordered,
        float_tol=args.float_tol,
    )
    record["comparison"] = detail
    if is_correct:
        record.update({"status": "correct", "score": 1})
    else:
        record.update({"status": "mismatch", "score": 0, "error": "execution_result_mismatch"})
    return record


def summarize_query_result(result: QueryResult) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ok": result.ok,
        "elapsed_seconds": result.elapsed_seconds,
        "row_count": result.row_count,
        "column_count": len(result.columns),
    }
    if result.error:
        summary["error"] = result.error
    return summary


def build_summary(
    records: Sequence[Mapping[str, Any]],
    run_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    included = [record for record in records if record.get("included_in_ex", True)]
    total = len(included)
    correct = sum(int(record.get("score", 0)) for record in included)
    ex_score = round(correct / total, 8) if total else 0.0
    schema_linking_model = str((run_info or {}).get("schema_linking_model") or "unknown")
    return {
        "sample_count": len(records),
        "evaluated_sample_count": total,
        "denominator": total,
        "correct_count": correct,
        "EX": ex_score,
        "schema_linking_model": schema_linking_model,
    }


def compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    is_correct = bool(int(record.get("score", 0)))
    compact = {
        "id": record.get("id"),
        "correct": is_correct,
    }
    if not is_correct:
        compact["error"] = record.get("error") or record.get("status") or "incorrect"
    return compact


def write_output(
    output_path: Path,
    run_info: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": build_summary(records, run_info),
        "records": [compact_record(record) for record in records],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = resolve_output_path(input_path, args.output).resolve()

    input_payload = load_payload(input_path)
    input_rows = rows_from_payload(input_payload, input_path)
    dataset_name = infer_dataset_name(args, input_payload, input_path)
    schema_linking_model = infer_schema_linking_model(input_payload, input_rows)
    dataset_path = args.dataset_path or default_dataset_path(dataset_name)
    dataset_index = build_dataset_index(dataset_path, args)
    items = build_eval_items(input_rows, dataset_index, args)
    sqlite_dir = resolve_sqlite_dir(args, dataset_name, input_path)
    engine = resolve_engine(args, dataset_name, sqlite_dir)

    run_info = {
        "task": "execution_accuracy",
        "definition": (
            "EX = number of samples where predicted SQL and gold SQL both execute on the "
            "gold database and return identical results, divided by the evaluated sample count."
        ),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dataset_name": dataset_name,
        "schema_linking_model": schema_linking_model,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "engine": engine,
        "sqlite_dir": str(sqlite_dir) if sqlite_dir else None,
        "timeout": args.timeout,
        "float_tol": args.float_tol,
        "max_rows": args.max_rows,
        "start_index": args.start_index,
        "limit": args.limit,
        "skip_gold_errors": args.skip_gold_errors,
        "require_db_match": args.require_db_match,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

    executor: SqliteExecutor | SnowflakeExecutor
    if engine == "sqlite":
        executor = SqliteExecutor(sqlite_dir=sqlite_dir, timeout=args.timeout, max_rows=args.max_rows)
    else:
        executor = SnowflakeExecutor(
            credential_path=args.snowflake_credential,
            timeout=args.timeout,
            max_rows=args.max_rows,
        )

    records: list[dict[str, Any]] = []
    try:
        for offset, item in enumerate(items, start=1):
            record = evaluate_item(item, executor, args)
            records.append(record)

            if args.progress_every and offset % args.progress_every == 0:
                print(
                    f"[{offset}/{len(items)}] id={item.sample_id} "
                    f"status={record.get('status')} ex={build_summary(records)['EX']}",
                    file=sys.stderr,
                    flush=True,
                )
            if args.save_every and offset % args.save_every == 0:
                write_output(output_path, run_info, records)
    finally:
        executor.close()

    write_output(output_path, run_info, records)
    summary = build_summary(records)
    print(
        f"EX={summary['EX']} ({summary['correct_count']}/{summary['denominator']}) "
        f"output={output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_THREAD_STATE = threading.local()
_FILE_LOCKS: dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _get_file_lock(path: str) -> threading.Lock:
    normalized_path = os.path.abspath(path)
    with _FILE_LOCKS_GUARD:
        if normalized_path not in _FILE_LOCKS:
            _FILE_LOCKS[normalized_path] = threading.Lock()
        return _FILE_LOCKS[normalized_path]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _now_file_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S_%f")


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def build_metrics_path(save_path: str, file_name: str) -> str:
    base_dir = Path(save_path).resolve().parent / "metrics"
    return str(base_dir / file_name)


def _sanitize_file_stem(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "dataset"


def build_metrics_file_name(
        dataset_path: str,
        prefix: str = "multi_generate_schemas_sample_metrics",
        run_timestamp: Optional[str] = None,
) -> str:
    dataset_name = _sanitize_file_stem(Path(dataset_path).stem)
    timestamp = run_timestamp or _now_file_timestamp()
    return f"{prefix}_{dataset_name}_{timestamp}.json"


class SampleMetricsSession:
    def __init__(self, sample_id: str, metadata: Optional[Dict[str, Any]] = None):
        self.sample_id = str(sample_id)
        _ = metadata
        self._started_monotonic = time.perf_counter()
        self._total_tokens = 0

    def update_metadata(self, **kwargs):
        return None

    def record_llm_usage(
            self,
            model_name: Optional[str],
            prompt_tokens: Any = None,
            completion_tokens: Any = None,
            total_tokens: Any = None,
    ):
        prompt_tokens = _safe_int(prompt_tokens)
        completion_tokens = _safe_int(completion_tokens)
        total_tokens = _safe_int(total_tokens)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        self._total_tokens += total_tokens

    def finalize(self, status: str, error: Optional[str] = None) -> Dict[str, Any]:
        elapsed_seconds = round(time.perf_counter() - self._started_monotonic, 6)
        return {
            "sample_id": self.sample_id,
            "status": status,
            "elapsed_seconds": elapsed_seconds,
            "total_tokens": self._total_tokens,
            "error": error,
        }


def record_llm_usage(
        model_name: Optional[str],
        prompt_tokens: Any = None,
        completion_tokens: Any = None,
        total_tokens: Any = None,
):
    session = getattr(_THREAD_STATE, "active_sample_session", None)
    if session is None:
        return
    session.record_llm_usage(
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


class SampleMetricsRecorder:
    def __init__(
            self,
            output_path: str,
            run_name: Optional[str] = None,
            static_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.output_path = os.path.abspath(output_path)
        self.run_name = run_name or "sample_metrics"
        self.static_metadata = dict(static_metadata or {})
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    @contextmanager
    def track_sample(self, sample_id: str, metadata: Optional[Dict[str, Any]] = None):
        session = SampleMetricsSession(sample_id=sample_id, metadata=metadata)
        previous_session = getattr(_THREAD_STATE, "active_sample_session", None)
        _THREAD_STATE.active_sample_session = session
        try:
            yield session
        except Exception as exc:
            self._write_record(session.finalize(status="error", error=str(exc)))
            raise
        else:
            self._write_record(session.finalize(status="success"))
        finally:
            _THREAD_STATE.active_sample_session = previous_session

    def _load_payload(self) -> Dict[str, Any]:
        if not os.path.exists(self.output_path):
            return {
                "run": {
                    "name": self.run_name,
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "static_metadata": self.static_metadata,
                },
                "summary": {},
                "records": [],
            }

        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            payload = {}

        payload.setdefault("run", {})
        payload["run"].setdefault("name", self.run_name)
        payload["run"].setdefault("created_at", _now_iso())
        payload["run"]["static_metadata"] = self.static_metadata
        payload.setdefault("summary", {})
        payload.setdefault("records", [])
        return payload

    def _build_summary(self, records: list[Dict[str, Any]]) -> Dict[str, Any]:
        sample_count = len(records)
        success_records = [record for record in records if record.get("status") == "success"]
        error_records = [record for record in records if record.get("status") == "error"]

        total_elapsed_seconds = round(sum(float(record.get("elapsed_seconds", 0.0)) for record in records), 6)
        total_tokens = sum(_safe_int(record.get("total_tokens")) for record in records)

        success_count = len(success_records)
        return {
            "sample_count": sample_count,
            "success_count": success_count,
            "error_count": len(error_records),
            "total_elapsed_seconds": total_elapsed_seconds,
            "avg_elapsed_seconds_per_sample": round(total_elapsed_seconds / sample_count, 6) if sample_count else 0.0,
            "avg_elapsed_seconds_per_success": round(
                sum(float(record.get("elapsed_seconds", 0.0)) for record in success_records) / success_count,
                6,
            ) if success_count else 0.0,
            "total_tokens": total_tokens,
            "avg_total_tokens_per_sample": round(total_tokens / sample_count, 6) if sample_count else 0.0,
            "avg_total_tokens_per_success": round(total_tokens / success_count, 6) if success_count else 0.0,
        }

    def _write_record(self, record: Dict[str, Any]):
        file_lock = _get_file_lock(self.output_path)
        with file_lock:
            payload = self._load_payload()
            records = payload.get("records", [])
            records.append(record)

            payload["records"] = records
            payload["summary"] = self._build_summary(records)
            payload["run"]["updated_at"] = _now_iso()

            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

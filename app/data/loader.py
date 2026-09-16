"""
Flexible Data Loader for Customer Support Datasets.
Supports JSONL, JSON, and CSV with configurable field mapping.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import csv
try:
    import pandas as pd
except ImportError:
    pd = None


DEFAULT_SCHEMA_MAP = {
    "id": ["id", "ticket_id", "case_id", "ID"],
    "subject": ["subject", "title", "heading", "summary"],
    "body": ["body", "description", "text", "message", "content", "query"],
    "true_category": ["true_category", "category", "intent", "label", "topic"],
}


class DataLoader:
    """
    Ingestion layer capable of loading datasets from multiple formats
    and mapping custom columns to a standardized internal schema.
    """

    def __init__(self, schema_map: Optional[Dict[str, List[str]]] = None):
        self.schema_map = schema_map or DEFAULT_SCHEMA_MAP

    def _resolve_field(self, record: dict, target_field: str) -> Optional[Any]:
        candidates = self.schema_map.get(target_field, [target_field])
        for cand in candidates:
            if cand in record and record[cand] is not None:
                return record[cand]
        return None

    def load_jsonl(self, file_path: str | Path) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        return [self.standardize_record(r) for r in records]

    def load_csv(self, file_path: str | Path) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        if pd is not None:
            df = pd.read_csv(file_path)
            records = df.to_dict(orient="records")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                records = list(reader)
        return [self.standardize_record(r) for r in records]

    def load_auto(self, file_path: str | Path) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        if file_path.suffix == ".jsonl":
            return self.load_jsonl(file_path)
        elif file_path.suffix in (".csv", ".txt"):
            return self.load_csv(file_path)
        elif file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [self.standardize_record(r) for r in data]
                else:
                    return [self.standardize_record(data)]
        else:
            # Fallback to jsonl
            return self.load_jsonl(file_path)

    def standardize_record(self, raw: dict) -> dict:
        std = dict(raw)
        std["id"] = str(self._resolve_field(raw, "id") or f"ticket_{id(raw)}")
        std["subject"] = str(self._resolve_field(raw, "subject") or "")
        std["body"] = str(self._resolve_field(raw, "body") or "")
        std["true_category"] = str(self._resolve_field(raw, "true_category") or raw.get("intent", raw.get("ideal_category", "unknown")))
        std["raw_data"] = raw
        return std

"""
Dataset Profiling Tool.
Generates structural statistics, row counts, missing values, duplicates, and category distributions.
"""

from __future__ import annotations
from typing import Dict, List, Any
from collections import Counter


def profile_dataset(tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a structural profile of the dataset without assuming fixed columns.
    """
    total_rows = len(tickets)
    if total_rows == 0:
        return {
            "total_rows": 0,
            "columns": [],
            "missing_values": {},
            "duplicate_ids": 0,
            "category_distribution": {},
            "avg_subject_length": 0,
            "avg_body_length": 0,
        }

    ids = [t.get("id") for t in tickets]
    id_counts = Counter(ids)
    duplicate_ids = sum(1 for count in id_counts.values() if count > 1)

    categories = [t.get("true_category", "unknown") for t in tickets]
    category_dist = dict(Counter(categories))

    missing_subj = sum(1 for t in tickets if not t.get("subject"))
    missing_body = sum(1 for t in tickets if not t.get("body"))
    missing_cat = sum(1 for t in tickets if not t.get("true_category"))

    subj_lens = [len(t.get("subject", "")) for t in tickets]
    body_lens = [len(t.get("body", "")) for t in tickets]

    all_keys = set()
    for t in tickets:
        all_keys.update(t.keys())

    return {
        "total_rows": total_rows,
        "unique_ids": len(id_counts),
        "duplicate_ids": duplicate_ids,
        "fields_found": list(all_keys),
        "missing_values": {
            "subject": missing_subj,
            "body": missing_body,
            "true_category": missing_cat,
        },
        "category_distribution": category_dist,
        "avg_subject_length_chars": round(sum(subj_lens) / total_rows, 1),
        "avg_body_length_chars": round(sum(body_lens) / total_rows, 1),
    }


def print_profile_report(profile: Dict[str, Any]) -> None:
    print("=" * 60)
    print("DATASET PROFILE REPORT")
    print("=" * 60)
    print(f"Total Records          : {profile['total_rows']}")
    print(f"Unique IDs             : {profile['unique_ids']}")
    print(f"Duplicate IDs          : {profile['duplicate_ids']}")
    print(f"Fields Identified      : {', '.join(profile['fields_found'])}")
    print(f"Missing Values         : {profile['missing_values']}")
    print(f"Avg Subject Length     : {profile['avg_subject_length_chars']} chars")
    print(f"Avg Body Length        : {profile['avg_body_length_chars']} chars")
    print("\nCategory Distribution:")
    for cat, count in profile["category_distribution"].items():
        pct = (count / profile["total_rows"]) * 100
        print(f"  - {cat:<20}: {count:2d} ({pct:5.1f}%)")
    print("=" * 60)

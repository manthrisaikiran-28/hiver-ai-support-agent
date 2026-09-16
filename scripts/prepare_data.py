"""
Data Preparation Script.
Runs data ingestion, PII masking, preprocessing, and profiles raw & processed datasets.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.loader import DataLoader
from app.data.preprocessing import preprocess_dataset
from app.data.profiling import profile_dataset, print_profile_report


def run_data_pipeline(raw_path: str = "data/raw/tickets.jsonl",
                      output_path: str = "data/processed/cleaned_tickets.jsonl"):
    print(f"Loading raw data from: {raw_path}")
    loader = DataLoader()
    raw_tickets = loader.load_auto(raw_path)

    print("\n[Raw Dataset Profile]")
    raw_profile = profile_dataset(raw_tickets)
    print_profile_report(raw_profile)

    print(f"\nPreprocessing {len(raw_tickets)} records (applying text cleaning and PII masking)...")
    cleaned_tickets = preprocess_dataset(raw_tickets, mask_sensitive=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for ticket in cleaned_tickets:
            f.write(json.dumps(ticket) + "\n")

    print(f"Saved processed dataset to: {output_path}")
    return cleaned_tickets


if __name__ == "__main__":
    run_data_pipeline()

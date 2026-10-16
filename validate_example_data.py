#!/usr/bin/env python3
"""Validate synthetic input schemas; no model training is performed."""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent / "example_data"
FEATURES = ["TOC", "RO", "ST", "SRH", "RE", "GT", "TE"]

def check(name, fields, count):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == fields, (name, reader.fieldnames)
        rows = list(reader)
    assert len(rows) == count, (name, len(rows))
    assert len({r["index"] for r in rows}) == count
    assert all(r[f] in {"0", "1"} for r in rows for f in FEATURES)
    return rows

if __name__ == "__main__":
    wells = check("蠡县探井连续数据集.csv", ["index", "X", "Y", *FEATURES, "CLASS"], 32)
    grid = check("蠡县网格连续数据集.csv", ["index", "X", "Y", *FEATURES], 40000)
    assert all(row["CLASS"] in {"0", "1"} for row in wells)
    assert all(row["index"] == f"Well-{i+1}" for i, row in enumerate(wells))
    print(f"Validated {len(wells)} synthetic wells and {len(grid)} grid cells.")

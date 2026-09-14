#!/usr/bin/env python3
"""Summarize decisions from an egodup JSONL run without inventing accuracy metrics."""

import argparse
import collections
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
args = parser.parse_args()
rows = [json.loads(x) for x in args.results.read_text().splitlines() if x.strip()]
print(
    json.dumps(
        {
            "items": len(rows),
            "decisions": collections.Counter(x.get("decision") for x in rows),
            "failed": sum(x.get("execution_status") == "failed" for x in rows),
        },
        indent=2,
    )
)

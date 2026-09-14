#!/usr/bin/env python3
"""Run a pair comparison and print measured stage timings."""

import argparse
import json
from pathlib import Path
from egodup.pipeline import compare

parser = argparse.ArgumentParser()
parser.add_argument("reference", type=Path)
parser.add_argument("query", type=Path)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
result = compare(args.query.resolve(), args.reference.resolve(), args.device)
print(
    json.dumps(
        {
            "device": result.device,
            "decision": result.decision,
            "stage_timings_seconds": result.stage_timings_seconds,
        },
        indent=2,
    )
)

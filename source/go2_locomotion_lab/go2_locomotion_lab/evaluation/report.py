"""JSON and CSV evaluation report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_report(metrics: dict, output_dir: str | Path, stem: str = "evaluation") -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    return json_path, csv_path

#!/usr/bin/env python3
"""Download MathArena competition sets and convert them to MemoSight formats.

The generated files are consumed by ``LightThinker/dataset_reader.py`` and by
``evaluation/eval_file.py``.  Source datasets are licensed CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASETS = {
    "aime_2026": {
        "repo": "MathArena/aime_2026",
        "source_url": "https://huggingface.co/datasets/MathArena/aime_2026",
    },
    "hmmt_feb_2025": {
        "repo": "MathArena/hmmt_feb_2025",
        "source_url": "https://huggingface.co/datasets/MathArena/hmmt_feb_2025",
    },
}


def fetch_rows(repo: str) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "dataset": repo,
            "config": "default",
            "split": "train",
            "offset": 0,
            "length": 100,
        }
    )
    request = Request(
        f"https://datasets-server.huggingface.co/rows?{query}",
        headers={"User-Agent": "MemoSight-eval-preparation/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)

    rows = []
    for entry in payload.get("rows", []):
        if entry.get("truncated_cells"):
            raise RuntimeError(
                f"Dataset server truncated row {entry.get('row_idx')} from {repo}"
            )
        rows.append(entry["row"])
    if not rows:
        raise RuntimeError(f"No rows returned for {repo}")
    return rows


def aliases_for(dataset_name: str, answer: str) -> list[str]:
    aliases = {answer, rf"\text{{{answer}}}"}
    if dataset_name.startswith("aime_") and answer.lstrip("-").isdigit():
        aliases.add(f"{int(answer):03d}")
    return sorted(aliases)


def convert(dataset_name: str, spec: dict[str, str], rows: list[dict[str, Any]]):
    json_items = []
    jsonl_items = []
    for row in sorted(rows, key=lambda item: int(item["problem_idx"])):
        question = str(row["problem"]).strip()
        answer = str(row["answer"]).strip()
        meta_data = {
            "source_dataset": spec["repo"],
            "source_url": spec["source_url"],
            "license": "CC BY-NC-SA 4.0",
            "problem_idx": int(row["problem_idx"]),
            "problem_type": row.get("problem_type", []),
        }
        item = {
            "meta_data": meta_data,
            "question": question,
            "question_list": [question],
            "answer": answer,
        }
        json_items.append(item)
        jsonl_items.append(
            {
                "idx": len(jsonl_items),
                "meta_info": item,
                "question": question,
                "answer": answer,
                "alias": aliases_for(dataset_name, answer),
                "error": ["error"],
            }
        )
    return {dataset_name: json_items}, jsonl_items


def write_dataset(output_dir: Path, dataset_name: str, json_data, jsonl_data) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{dataset_name}.json"
    jsonl_path = output_dir / f"{dataset_name}.jsonl"
    with json_path.open("w", encoding="utf-8", newline="\n") as writer:
        json.dump(json_data, writer, ensure_ascii=False, indent=4)
        writer.write("\n")
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as writer:
        for item in jsonl_data:
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(jsonl_data):2d} rows: {json_path} and {jsonl_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "eval",
    )
    args = parser.parse_args()

    for dataset_name, spec in DATASETS.items():
        rows = fetch_rows(spec["repo"])
        json_data, jsonl_data = convert(dataset_name, spec, rows)
        write_dataset(args.output_dir, dataset_name, json_data, jsonl_data)


if __name__ == "__main__":
    main()

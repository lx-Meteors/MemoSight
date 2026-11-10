#!/usr/bin/env python3
"""Compare lossless per-sample token IDs from two batch benchmark reports."""

import argparse
import json
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("left_report")
    parser.add_argument("right_report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-exact", action="store_true")
    return parser.parse_args()


def load_report(path):
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    if "reports" in report:
        if len(report["reports"]) != 1:
            raise ValueError(f"expected one dataset report in {path}")
        report = report["reports"][0]
    return report


def resolve_records(report_path, report):
    record_path = report.get("sample_records_path")
    if not record_path:
        paths = report.get("repeat_sample_records_paths") or []
        record_path = paths[0] if paths else None
    if not record_path:
        raise ValueError(f"report has no sample records: {report_path}")
    if not os.path.isabs(record_path):
        record_path = os.path.join(os.path.dirname(report_path), record_path)
    if not os.path.isfile(record_path):
        fallback = os.path.join(os.path.dirname(report_path), os.path.basename(record_path))
        if os.path.isfile(fallback):
            record_path = fallback
        else:
            raise FileNotFoundError(record_path)
    records = {}
    with open(record_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                records[int(item["sample_id"])] = item
    return records, record_path


def first_difference(left, right):
    common = min(len(left), len(right))
    for index in range(common):
        if left[index] != right[index]:
            return index
    return common if len(left) != len(right) else None


def main():
    args = parse_args()
    left_report = load_report(args.left_report)
    right_report = load_report(args.right_report)
    left, left_records = resolve_records(args.left_report, left_report)
    right, right_records = resolve_records(args.right_report, right_report)
    all_ids = sorted(set(left) | set(right))
    differences = []
    matches = 0
    for sample_id in all_ids:
        left_item = left.get(sample_id)
        right_item = right.get(sample_id)
        if left_item is None or right_item is None:
            differences.append(dict(
                sample_id=sample_id,
                kind="missing_sample",
                missing_from=("left" if left_item is None else "right"),
            ))
            continue
        left_ids = left_item["output_ids"]
        right_ids = right_item["output_ids"]
        difference = first_difference(left_ids, right_ids)
        if difference is None:
            matches += 1
            continue
        differences.append(dict(
            sample_id=sample_id,
            kind="token_mismatch",
            first_difference=difference,
            left_length=len(left_ids),
            right_length=len(right_ids),
            left_token=(left_ids[difference] if difference < len(left_ids) else None),
            right_token=(right_ids[difference] if difference < len(right_ids) else None),
            left_correct=left_item.get("correct"),
            right_correct=right_item.get("correct"),
        ))

    result = dict(
        left_report=os.path.realpath(args.left_report),
        right_report=os.path.realpath(args.right_report),
        left_records=left_records,
        right_records=right_records,
        compared_samples=len(all_ids),
        exact_matches=matches,
        mismatches=len(differences),
        exact_match_rate=(matches / len(all_ids) if all_ids else 0.0),
        differences=differences,
    )
    os.makedirs(os.path.dirname(os.path.realpath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in (
        "compared_samples", "exact_matches", "mismatches", "exact_match_rate"
    )}, ensure_ascii=False))
    if args.require_exact and differences:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

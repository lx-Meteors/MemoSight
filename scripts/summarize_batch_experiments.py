#!/usr/bin/env python3
"""Collect H4W6 batch reports into one comparison CSV and JSON."""

import argparse
import csv
import glob
import json
import os
import statistics


METHODS = ("vanilla_ar", "memosight_ar", "memosight_sd")


def nested_median(report, *keys):
    values = []
    for repeat in report.get("repeat_reports", []):
        value = repeat
        for key in keys:
            value = value[key]
        values.append(float(value))
    return statistics.median(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    run_root = os.path.realpath(args.run_root)
    csv_path = args.csv or os.path.join(run_root, "summary.csv")
    json_path = args.json or os.path.join(run_root, "summary.json")

    reports = []
    for path in glob.glob(os.path.join(run_root, "*.json")):
        with open(path, encoding="utf-8") as handle:
            item = json.load(handle)
        if item.get("bench_mode") not in METHODS:
            continue
        item["report_path"] = os.path.realpath(path)
        reports.append(item)
    reports.sort(key=lambda item: (int(item["batch_size"]), METHODS.index(item["bench_mode"])))
    by_key = {(item["bench_mode"], int(item["batch_size"])): item for item in reports}
    for batch_size in sorted({int(item["batch_size"]) for item in reports}):
        hashes = {
            item["prompt_sha256"] for item in reports
            if int(item["batch_size"]) == batch_size
        }
        if len(hashes) > 1:
            raise ValueError(
                f"prompt hash mismatch across methods at batch size {batch_size}: {sorted(hashes)}"
            )

    rows = []
    for report in reports:
        batch_size = int(report["batch_size"])
        vanilla = by_key.get(("vanilla_ar", batch_size))
        memo_ar = by_key.get(("memosight_ar", batch_size))
        method = report["bench_mode"]
        tps = float(report["tokens_per_sec"])
        rows.append(dict(
            method=method,
            batch_size=batch_size,
            num_samples=report["num_samples"],
            tokens_per_sec=tps,
            tokens_per_sec_min=report["tokens_per_sec_range"][0],
            tokens_per_sec_max=report["tokens_per_sec_range"][1],
            total_time_s=report["total_time_s"],
            requests_per_sec=report["requests_per_sec"],
            accuracy=report["accuracy"],
            output_length_mean=nested_median(report, "output_length", "mean"),
            output_length_p50=nested_median(report, "output_length", "p50"),
            output_length_p95=nested_median(report, "output_length", "p95"),
            output_length_max=nested_median(report, "output_length", "max"),
            length_limit_fraction=nested_median(report, "length_limit_fraction"),
            cache_limit_fraction=nested_median(report, "cache_limit_fraction"),
            peak_allocated_gib=float(report["peak_allocated_bytes"]) / 2**30,
            peak_reserved_gib=float(report["peak_reserved_bytes"]) / 2**30,
            peak_total_valid_kv_tokens=report["peak_total_valid_kv_tokens"],
            peak_physical_kv_slots=report["peak_physical_kv_slots"],
            completion_p50_s=nested_median(report, "request_completion_s", "p50"),
            completion_p95_s=nested_median(report, "request_completion_s", "p95"),
            spec_accept_rate=report.get("spec_accept_rate", 0.0),
            committed_tokens_per_decode_step=report.get("committed_tokens_per_decode_step", 0.0),
            sd_vs_vanilla_speedup=(
                tps / float(vanilla["tokens_per_sec"])
                if method == "memosight_sd" and vanilla else ""
            ),
            sd_vs_memosight_ar_speedup=(
                tps / float(memo_ar["tokens_per_sec"])
                if method == "memosight_sd" and memo_ar else ""
            ),
            prompt_sha256=report["prompt_sha256"],
            repeat_output_mismatches=";".join(map(str, report["repeat_output_mismatches"])),
            report_path=report["report_path"],
        ))

    os.makedirs(os.path.dirname(os.path.realpath(csv_path)), exist_ok=True)
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(dict(rows=rows), handle, ensure_ascii=False, indent=2)
    print(f"wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()

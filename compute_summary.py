#!/usr/bin/env python3
"""Compute summary.json metrics from results/raw_log.jsonl."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

RAW = Path("results/raw_log.jsonl")
OUT = Path("results/summary.json")


def load_entries():
    entries = []
    with RAW.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "query_id" not in d:
                continue
            entries.append(d)
    return entries


def metrics_for(entries):
    n = len(entries)
    if n == 0:
        return {}
    tool_match = sum(1 for e in entries if e["tool_match"])
    parse_success = sum(1 for e in entries if e["parse_success"])
    matched = [e for e in entries if e["tool_match"]]
    args_match = sum(1 for e in matched if e["args_match"])
    latencies = [e["latency_ms"] for e in entries]

    # By tier
    by_tier = defaultdict(list)
    for e in entries:
        by_tier[e["tier"]].append(e)
    tier_acc = {}
    tier_lat = {}
    for t, es in sorted(by_tier.items()):
        tm = sum(1 for e in es if e["tool_match"])
        tier_acc[f"T{t}"] = round(tm / len(es), 4)
        tier_lat[f"T{t}"] = round(statistics.mean(e["latency_ms"] for e in es), 1)

    # Per-tool
    by_tool = defaultdict(list)
    for e in entries:
        if e["expected_tool"]:
            by_tool[e["expected_tool"]].append(e)
    per_tool = {}
    for tool, es in by_tool.items():
        tm = sum(1 for e in es if e["tool_match"])
        per_tool[tool] = round(tm / len(es), 4)

    # Failure breakdown
    parse_fail = sum(1 for e in entries if not e["parse_success"])
    wrong_tool = sum(1 for e in entries if e["parse_success"] and not e["tool_match"])
    wrong_args = sum(1 for e in entries if e["tool_match"] and not e["args_match"])

    return {
        "n": n,
        "tool_match_rate": round(tool_match / n, 4),
        "args_match_rate_given_tool_match": round(args_match / len(matched), 4) if matched else 0.0,
        "parse_success_rate": round(parse_success / n, 4),
        "mean_latency_ms": round(statistics.mean(latencies), 1),
        "median_latency_ms": round(statistics.median(latencies), 1),
        "accuracy_by_tier": tier_acc,
        "latency_by_tier_ms": tier_lat,
        "per_tool_accuracy": per_tool,
        "failure_breakdown": {
            "parse_fail": parse_fail,
            "wrong_tool": wrong_tool,
            "wrong_args": wrong_args,
        },
    }


def main():
    entries = load_entries()
    by_model = defaultdict(list)
    for e in entries:
        by_model[e["model"]].append(e)

    summary = {}
    for model in ("needle", "qwen3", "qwen3-prompted"):
        if by_model[model]:
            summary[model] = metrics_for(by_model[model])

    summary["meta"] = {
        "total_runs": len(entries),
        "queries_per_model": 50,
        "models": list(by_model.keys()),
    }

    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

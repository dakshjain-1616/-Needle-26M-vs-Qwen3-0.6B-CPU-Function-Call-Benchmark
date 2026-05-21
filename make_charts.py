#!/usr/bin/env python3
"""Generate comparison charts into results/charts/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NEEDLE_COLOR = "#4A90D9"
QWEN_COLOR = "#E8784A"
QWEN_PROMPTED_COLOR = "#7B5CB8"
CHARTS = Path("results/charts")
CHARTS.mkdir(parents=True, exist_ok=True)

s = json.loads(Path("results/summary.json").read_text())
HAS_PROMPTED = "qwen3-prompted" in s


def save(fig, name):
    fig.patch.set_facecolor("white")
    fig.savefig(CHARTS / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {CHARTS/name}")


def models_axis():
    """Return (labels, keys, colors) for the active model set."""
    labels = ["Needle (26M)", "Qwen3 default", "Qwen3 prompted"] if HAS_PROMPTED else ["Needle (26M)", "Qwen3 (0.6B)"]
    keys = ["needle", "qwen3", "qwen3-prompted"] if HAS_PROMPTED else ["needle", "qwen3"]
    colors = [NEEDLE_COLOR, QWEN_COLOR, QWEN_PROMPTED_COLOR] if HAS_PROMPTED else [NEEDLE_COLOR, QWEN_COLOR]
    return labels, keys, colors


labels, keys, colors = models_axis()

# 1. Accuracy by tier
tiers = ["T1", "T2", "T3", "T4", "T5"]
x = np.arange(len(tiers))
n_models = len(keys)
w = 0.8 / n_models
fig, ax = plt.subplots(figsize=(9, 4.8))
for i, (lab, k, col) in enumerate(zip(labels, keys, colors)):
    vals = [s[k]["accuracy_by_tier"].get(t, 0) for t in tiers]
    offset = (i - (n_models - 1) / 2) * w
    ax.bar(x + offset, vals, w, label=lab, color=col)
ax.set_xticks(x); ax.set_xticklabels(tiers)
ax.set_ylim(0, 1.05); ax.set_ylabel("tool_match accuracy")
ax.set_title("Accuracy by Difficulty Tier")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "accuracy_by_tier.png")

# 2. Latency by tier
fig, ax = plt.subplots(figsize=(9, 4.8))
for i, (lab, k, col) in enumerate(zip(labels, keys, colors)):
    vals = [s[k]["latency_by_tier_ms"].get(t, 0) for t in tiers]
    offset = (i - (n_models - 1) / 2) * w
    ax.bar(x + offset, vals, w, label=lab, color=col)
ax.set_xticks(x); ax.set_xticklabels(tiers)
ax.set_ylabel("mean latency (ms)")
ax.set_title("Latency by Difficulty Tier")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "latency_comparison.png")

# 3. Parse success rate
fig, ax = plt.subplots(figsize=(8, 3.5))
rates = [s[k]["parse_success_rate"] for k in keys]
ax.barh(labels, rates, color=colors)
ax.set_xlim(0, 1.05); ax.set_xlabel("parse success rate")
ax.set_title("Parse Success Rate")
for i, r in enumerate(rates):
    ax.text(r + 0.01, i, f"{r:.1%}", va="center")
ax.grid(axis="x", alpha=0.3)
save(fig, "parse_success_rate.png")

# 4. Failure breakdown — stacked
fig, ax = plt.subplots(figsize=(9, 4.8))
pf = [s[k]["failure_breakdown"]["parse_fail"] for k in keys]
wt = [s[k]["failure_breakdown"]["wrong_tool"] for k in keys]
wa = [s[k]["failure_breakdown"]["wrong_args"] for k in keys]
ax.bar(labels, pf, label="parse_fail", color="#D9534F")
ax.bar(labels, wt, bottom=pf, label="wrong_tool", color="#F0AD4E")
ax.bar(labels, wa, bottom=[a + b for a, b in zip(pf, wt)], label="wrong_args", color="#F7E36B")
ax.set_ylabel("count")
ax.set_title("Failure Breakdown")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "failure_breakdown.png")

# 5. Overall summary
fig, ax = plt.subplots(figsize=(10, 4.8))
metrics = ["tool_match", "args_match", "parse_success", "latency_norm"]
max_lat = max(s[k]["mean_latency_ms"] for k in keys) or 1
x2 = np.arange(len(metrics))
for i, (lab, k, col) in enumerate(zip(labels, keys, colors)):
    vals = [
        s[k]["tool_match_rate"],
        s[k]["args_match_rate_given_tool_match"],
        s[k]["parse_success_rate"],
        s[k]["mean_latency_ms"] / max_lat,
    ]
    offset = (i - (n_models - 1) / 2) * w
    ax.bar(x2 + offset, vals, w, label=lab, color=col)
ax.set_xticks(x2); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1); ax.set_ylabel("score (latency normalised, lower = faster)")
ax.set_title("Overall Summary")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "overall_summary.png")

# 6. Per-tool accuracy (Needle focus, with comparisons if present)
tools = ["get_weather", "search_web", "create_file", "get_time", "run_command"]
fig, ax = plt.subplots(figsize=(9, 4.8))
xt = np.arange(len(tools))
for i, (lab, k, col) in enumerate(zip(labels, keys, colors)):
    vals = [s[k]["per_tool_accuracy"].get(t, 0) for t in tools]
    offset = (i - (n_models - 1) / 2) * w
    ax.bar(xt + offset, vals, w, label=lab, color=col)
ax.set_xticks(xt); ax.set_xticklabels(tools, rotation=15)
ax.set_ylim(0, 1.05); ax.set_ylabel("tool_match accuracy")
ax.set_title("Per-Tool Accuracy")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "per_tool_accuracy.png")

print("done")

#!/usr/bin/env python3
"""Generate the 5 spec charts into results/charts/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NEEDLE_COLOR = "#4A90D9"
QWEN_COLOR = "#E8784A"
CHARTS = Path("results/charts")
CHARTS.mkdir(parents=True, exist_ok=True)

s = json.loads(Path("results/summary.json").read_text())


def save(fig, name):
    fig.patch.set_facecolor("white")
    fig.savefig(CHARTS / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {CHARTS/name}")


# 1. Accuracy by tier
tiers = ["T1", "T2", "T3", "T4", "T5"]
needle_acc = [s["needle"]["accuracy_by_tier"].get(t, 0) for t in tiers]
qwen_acc = [s["qwen3"]["accuracy_by_tier"].get(t, 0) for t in tiers]
x = np.arange(len(tiers))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w/2, needle_acc, w, label="Needle (26M)", color=NEEDLE_COLOR)
ax.bar(x + w/2, qwen_acc, w, label="Qwen3 (0.6B)", color=QWEN_COLOR)
ax.set_xticks(x); ax.set_xticklabels(tiers)
ax.set_ylim(0, 1.05); ax.set_ylabel("tool_match accuracy")
ax.set_title("Accuracy by Difficulty Tier")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "accuracy_by_tier.png")

# 2. Latency by tier
needle_lat = [s["needle"]["latency_by_tier_ms"].get(t, 0) for t in tiers]
qwen_lat = [s["qwen3"]["latency_by_tier_ms"].get(t, 0) for t in tiers]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w/2, needle_lat, w, label="Needle (26M)", color=NEEDLE_COLOR)
ax.bar(x + w/2, qwen_lat, w, label="Qwen3 (0.6B)", color=QWEN_COLOR)
ax.set_xticks(x); ax.set_xticklabels(tiers)
ax.set_ylabel("mean latency (ms)")
ax.set_title("Latency by Difficulty Tier")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "latency_comparison.png")

# 3. Parse success rate
fig, ax = plt.subplots(figsize=(8, 3))
models = ["Needle (26M)", "Qwen3 (0.6B)"]
rates = [s["needle"]["parse_success_rate"], s["qwen3"]["parse_success_rate"]]
ax.barh(models, rates, color=[NEEDLE_COLOR, QWEN_COLOR])
ax.set_xlim(0, 1.05); ax.set_xlabel("parse success rate")
ax.set_title("Parse Success Rate")
for i, r in enumerate(rates):
    ax.text(r + 0.01, i, f"{r:.1%}", va="center")
ax.grid(axis="x", alpha=0.3)
save(fig, "parse_success_rate.png")

# 4. Failure breakdown — stacked
fail_n = s["needle"]["failure_breakdown"]
fail_q = s["qwen3"]["failure_breakdown"]
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = ["Needle (26M)", "Qwen3 (0.6B)"]
pf = [fail_n["parse_fail"], fail_q["parse_fail"]]
wt = [fail_n["wrong_tool"], fail_q["wrong_tool"]]
wa = [fail_n["wrong_args"], fail_q["wrong_args"]]
ax.bar(labels, pf, label="parse_fail", color="#D9534F")
ax.bar(labels, wt, bottom=pf, label="wrong_tool", color="#F0AD4E")
ax.bar(labels, wa, bottom=[a + b for a, b in zip(pf, wt)], label="wrong_args", color="#F7E36B")
ax.set_ylabel("count")
ax.set_title("Failure Breakdown")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "failure_breakdown.png")

# 5. Overall summary
fig, ax = plt.subplots(figsize=(9, 4.5))
metrics = ["tool_match", "args_match", "parse_success", "latency_norm"]
max_lat = max(s["needle"]["mean_latency_ms"], s["qwen3"]["mean_latency_ms"], 1)
needle_vals = [
    s["needle"]["tool_match_rate"],
    s["needle"]["args_match_rate_given_tool_match"],
    s["needle"]["parse_success_rate"],
    s["needle"]["mean_latency_ms"] / max_lat,
]
qwen_vals = [
    s["qwen3"]["tool_match_rate"],
    s["qwen3"]["args_match_rate_given_tool_match"],
    s["qwen3"]["parse_success_rate"],
    s["qwen3"]["mean_latency_ms"] / max_lat,
]
x2 = np.arange(len(metrics))
ax.bar(x2 - w/2, needle_vals, w, label="Needle (26M)", color=NEEDLE_COLOR)
ax.bar(x2 + w/2, qwen_vals, w, label="Qwen3 (0.6B)", color=QWEN_COLOR)
ax.set_xticks(x2); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1); ax.set_ylabel("score (latency normalised, lower = faster)")
ax.set_title("Overall Summary")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save(fig, "overall_summary.png")

print("done")

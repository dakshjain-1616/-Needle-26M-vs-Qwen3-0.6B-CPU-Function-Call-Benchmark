# Needle 26M vs Qwen3-0.6B — CPU Function-Call Benchmark

> 🤖 Built **autonomously** using **[NEO — Your Autonomous AI Engineering Agent](https://heyneo.com)**
>
> [![VS Code Extension](https://img.shields.io/badge/VS%20Code-NEO%20Extension-007ACC?logo=visualstudiocode&logoColor=white)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)
> [![Cursor Extension](https://img.shields.io/badge/Cursor-NEO%20Extension-000000?logo=cursor&logoColor=white)](https://marketplace.cursorapi.com/items/?itemName=NeoResearchInc.heyneo)
> [![Website](https://img.shields.io/badge/heyneo.com-Visit-9B59B6)](https://heyneo.com)

A head-to-head benchmark of two open-weight tool-calling models on CPU:

- **[Needle 26M](https://huggingface.co/Cactus-Compute/needle)** — Cactus-Compute's 26M-parameter Simple Attention Network, distilled from Gemini 3.1 for single-shot function calling.
- **[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)** — Alibaba's 600M-parameter general-purpose instruct model with native `<tool_call>` support.

50 spec-defined queries × 2 models = 100 timed runs on CPU. Full methodology, charts, and analysis in **[blog_post.md](./blog_post.md)**.

---

## Headline Result

| Metric | Needle (26M) | Qwen3 (0.6B) |
|---|---:|---:|
| **tool_match accuracy** | **72.0%** | 56.0% |
| **args_match** (given match) | 97.2% | **100.0%** |
| **parse_success** | **84.0%** | 54.0% |
| **Mean latency (CPU)** | **10.9 s** | 47.9 s |

The 23× smaller, specialised model wins on both speed (4.4×) and accuracy (+16 pts) — because Qwen3-0.6B's failure mode is _chattiness_: on implicit queries it answers in prose instead of emitting a `<tool_call>`.

![Overall summary](results/charts/overall_summary.png)

See **[blog_post.md](./blog_post.md)** for the full writeup, per-tier breakdown, real example outputs, and discussion of the three bugs NEO surfaced during the run.

---

## Quick Start

```bash
git clone https://github.com/<you>/needle-vs-qwen3-benchmark
cd needle-vs-qwen3-benchmark

# Virtual env + deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Clone Needle (kept out of this repo; auto-downloads weights on first run)
git clone https://github.com/cactus-compute/needle.git needle_repo

# Single query smoke test
CUDA_VISIBLE_DEVICES="" python dispatcher.py --model needle --query "What's the weather in Tokyo?"
CUDA_VISIBLE_DEVICES="" python dispatcher.py --model qwen3  --query "What's the weather in Tokyo?"

# Full benchmark — Needle ~5 min, Qwen3 ~30 min on CPU
CUDA_VISIBLE_DEVICES="" python benchmark.py --model needle
CUDA_VISIBLE_DEVICES="" python benchmark.py --model qwen3

# Rebuild summary/charts/report from raw_log.jsonl
python compute_summary.py
python make_charts.py
python make_report.py
```

The Qwen3 weights (~1.2 GB) download from HuggingFace on first use. The Needle checkpoint (~13 MB) downloads to `checkpoints/needle.pkl` automatically.

---

## What's in the Repo

```
dispatcher.py                       # CLI: --model needle|qwen3 --query "..."
benchmark.py                        # 50-query harness with tool_match/args_match eval
tools.py                            # 5 mock tools + their OpenAI-style schemas
backends/
  needle_backend.py                 # Needle wrapper + OpenAI→flat schema converter
  qwen3_backend.py                  # Qwen3 via apply_chat_template(tools=…)
compute_summary.py                  # raw_log.jsonl → summary.json
make_charts.py                      # summary.json → 5 PNG charts
make_report.py                      # summary.json + raw_log → benchmark_report.md
results/
  raw_log.jsonl                     # 100 rows of per-query data
  summary.json                      # computed metrics per model + per tier
  benchmark_report.md               # auto-generated full report
  install_notes.txt                 # bug log from the run
  charts/                           # 5 comparison charts (150 DPI)
blog_post.md                        # publication-ready writeup ← start here
prompt.md                           # original task spec
requirements.txt
```

---

## The 5 Tools

`get_weather(location)`, `search_web(query)`, `create_file(filename, content)`, `run_command(command)`, `get_time(timezone)` — all mock stubs that return realistic-looking JSON. No real network calls.

## The 5 Difficulty Tiers (10 queries each)

| Tier | Tests | Example |
|---|---|---|
| **T1 Simple** | Direct, unambiguous | _"What's the weather in London?"_ |
| **T2 Paraphrased** | Same intent, different phrasing | _"Is it raining in Berlin right now?"_ |
| **T3 Implicit** | Tool not named | _"Should I bring an umbrella in Amsterdam today?"_ |
| **T4 Ambiguous** | Multiple tools could fit | _"What's happening in London this weekend?"_ |
| **T5 Edge** | Foreign languages, negation, no-tool | _"मुंबई का मौसम"_, _"What's 2+2?"_ |

The full 50 queries with `expected_tool` / `expected_args_keys` are in `benchmark.py`.

---

## How NEO Built This

This repository was produced **end-to-end by NEO**, an autonomous AI engineering agent. From a single prompt, NEO scaffolded the dispatcher, both backends, the 50-query harness with proper evaluation logic, the chart/report generators, and the writeup in `blog_post.md`.

Three real bugs surfaced and were fixed during the run:

1. **Needle schema mismatch** — Needle was being fed OpenAI JSON Schema (`{type:"object", properties:{...}}`) instead of its native flat schema (`{location: {type, description, required}}`), causing it to echo `"properties"` back as an argument. NEO added a converter; parse rate jumped 8% → 84%.
2. **Qwen3 prompt template** — first pass used a hand-rolled prompt that burned the full 256-token budget per query (~230 s/query on CPU). NEO switched to `apply_chat_template(tools=…, enable_thinking=False)` with `max_new_tokens=128`; latency dropped 6× to ~37 s/query.
3. **Wrong evaluation logic** — initial benchmark used invented queries and tracked only parse success. NEO replaced it with the spec's exact 50 queries (T1_01–T5_10) and the full `tool_match` / `args_match` rubric.

See `results/install_notes.txt` for the full bug log.

---

## Try NEO

NEO is available as an extension for VS Code and Cursor — you give it a goal, it plans, writes the code, runs it, and iterates until it works.

- **VS Code:** [marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)
- **Cursor:** [marketplace.cursorapi.com/items/?itemName=NeoResearchInc.heyneo](https://marketplace.cursorapi.com/items/?itemName=NeoResearchInc.heyneo)
- **Site:** [heyneo.com](https://heyneo.com)

---

## License

MIT — see [LICENSE](./LICENSE).

Model weights are subject to their own licenses: Needle is openly available from Cactus-Compute; Qwen3-0.6B is Apache-2.0.

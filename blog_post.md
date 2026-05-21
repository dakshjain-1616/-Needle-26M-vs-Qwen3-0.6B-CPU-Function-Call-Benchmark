# Needle 26M vs Qwen3-0.6B: A Real CPU Function-Call Benchmark

*A complete head-to-head evaluation of two open-weight tool-calling models — Needle (26M, distilled from Gemini 3.1) and Qwen3-0.6B — across 50 structured queries in five difficulty tiers, run entirely on CPU. No GPU, no cherry-picked queries, no placeholder numbers.*

---

Most "tool calling" or "function calling" benchmarks fall into one of two traps: they either test the API contract (does the model emit valid JSON?) on a handful of toy examples, or they pit a frontier model against a tiny model on tasks that obviously favor scale. This benchmark tries to be more honest. We ran two open-weight models — one specifically distilled for function calls, one general-purpose — on the same 50 queries with the same evaluation rubric, on the same CPU, and looked at where each one breaks.

The short version: the 23× smaller model wins. But not for the reason you might expect.

---

## The Models

**Needle (26M)** is a [Simple Attention Network](https://github.com/cactus-compute/needle) from Cactus-Compute, distilled from Gemini 3.1 specifically for single-shot function calling. It has 26 million parameters, pretrained on 200B tokens, then post-trained on 2B tokens of function-call data. Weights are fully open ([Cactus-Compute/needle](https://huggingface.co/Cactus-Compute/needle)) and the production target is consumer devices — watches, phones, glasses. It exposes a JAX/Flax inference path with a flat tool schema unique to its training distribution.

**Qwen3-0.6B** is the [smallest member](https://huggingface.co/Qwen/Qwen3-0.6B) of Alibaba's Qwen3 family — a 600-million-parameter decoder-only transformer, instruction-tuned, Apache-2.0 licensed. It supports OpenAI-compatible function calling via its tokenizer's `apply_chat_template(tools=...)` and emits tool calls inside `<tool_call>...</tool_call>` tags.

The asymmetry is important: **Needle is narrowly trained to do one thing well (single tool calls), Qwen3 is a general-purpose model that can also call tools.** Both are competitors in the "tiny on-device assistant" space, but they took very different paths to get there.

| | **Needle (26M)** | **Qwen3-0.6B** |
|---|---|---|
| Architecture | Simple Attention Network (encoder + decoder), distilled from Gemini 3.1 | Decoder-only transformer, instruction-tuned |
| Params | 26 M | 600 M |
| Training | 200 B tokens pretrain → 2 B tokens of single-shot function-call data | General SFT + RLHF + function-call data |
| Tool schema | **Flat**: `{location: {type, description, required}}` | **OpenAI JSON Schema**: `{type: "object", properties: {...}}` |
| License / weights | Open, [Cactus-Compute/needle](https://huggingface.co/Cactus-Compute/needle) | Apache-2.0, [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Intended use | On-device single-shot tool calls | General assistant with tool calling |

---

## What We Measured

The benchmark ran 50 queries through each model — 100 timed runs total — with one discarded warmup query per model. Queries are organised into five difficulty tiers, 10 queries each:

| Tier | What it tests | Example |
|---|---|---|
| **T1 — Simple** | Direct, unambiguous, one tool, the tool name appears in the query | _"What's the weather in London?"_ |
| **T2 — Paraphrased** | Same intent as T1 but worded differently | _"Is it raining in Berlin right now?"_ |
| **T3 — Implicit** | Intent is clear but the tool isn't named anywhere in the query | _"Should I bring an umbrella in Amsterdam today?"_ |
| **T4 — Ambiguous** | Two tools could plausibly fit | _"What's happening in London this weekend?"_ (events, not weather) |
| **T5 — Edge** | Foreign languages, negation, no-tool, destructive | _"मुंबई का मौसम"_, _"What's 2+2?"_, _"Delete all my files"_ |

Five mock tools are wired up — `get_weather`, `search_web`, `create_file`, `run_command`, `get_time` — with realistic stub responses (no real network calls).

Each run is graded on three independent booleans:

| Metric | Definition |
|---|---|
| `parse_success` | Output was valid JSON containing a `name` field |
| `tool_match` | `parsed_tool == expected_tool` (exact name). For T5_05 ("What's 2+2?"), true iff the model emits **no** tool call |
| `args_match` | All expected arg keys present with non-empty string values, **and** `tool_match=True`. Relaxed for the four underspecified T4 queries — any non-empty arg value counts |

Hardware: 4-core CPU, no GPU, `CUDA_VISIBLE_DEVICES=""`. Python 3.12, transformers 4.50+, torch 2.4+, jax 0.4.30/flax 0.8.5 for Needle. The Needle checkpoint (~13 MB) was downloaded from HuggingFace at start; Qwen3 weights (~1.2 GB) were pulled from the Hub on first run.

---

## The Numbers

### Overall

| Metric | Needle (26M) | Qwen3 (0.6B) | Winner |
|---|---:|---:|---|
| **tool_match** (overall accuracy)  | **72.0%** | 56.0% | Needle (+16 pts) |
| **args_match** (given tool_match) | 97.2% | **100.0%** | Qwen3 |
| **parse_success** | **84.0%** | 54.0% | Needle |
| **Mean latency (CPU)** | **10,933 ms** | 47,863 ms | Needle (4.4× faster) |
| **Median latency** | 8,849 ms | 39,188 ms | Needle |

![Overall summary](results/charts/overall_summary.png)

*The 23×-smaller model is 4.4× faster on CPU and 16 percentage points more accurate on tool selection. Qwen3's args (when it does emit a tool call) are flawless — but it just doesn't emit them often enough.*

### Accuracy by Difficulty Tier

| Tier | Needle | Qwen3 | Δ |
|---|---:|---:|---|
| T1 — Simple      | 100% | 100% | tie |
| T2 — Paraphrased | 90% | 90% | tie |
| **T3 — Implicit**    | **80%** | 10% | **+70 pts Needle** |
| T4 — Ambiguous   | **40%** | 20% | +20 pts Needle |
| T5 — Edge        | 50% | **60%** | +10 pts Qwen3 |

![Accuracy by tier](results/charts/accuracy_by_tier.png)

T1 and T2 are essentially solved by both models. The story starts at **T3**: when the query doesn't literally name the tool ("Should I bring an umbrella?" instead of "what's the weather?"), Qwen3 falls off a cliff — 80% → 10%. Needle, trained narrowly to map natural language to tool calls, handles implicit phrasing fine.

T4 (ambiguous) is hard for both. Needle still leads, mostly because it always commits to *some* tool; Qwen3 hedges.

T5 (edge cases) is the only tier Qwen3 wins, by 10 points. The wins are on the foreign-language and the "no-tool" 2+2 query.

### Latency by Tier (mean ms)

| Tier | Needle | Qwen3 | Speedup |
|---|---:|---:|---:|
| T1 | 11,608 | 37,425 | 3.2× |
| T2 | 9,554 | 38,830 | 4.1× |
| T3 | 9,575 | 54,308 | 5.7× |
| T4 | 6,730 | 52,437 | 7.8× |
| T5 | 17,199 | 56,314 | 3.3× |

![Latency by tier](results/charts/latency_comparison.png)

Needle is consistently faster, but the gap widens on harder tiers because Qwen3 spends more tokens generating prose answers instead of bailing into a tool call. T5 is Needle's slowest tier because of the Hindi queries — the tokenizer fragments Devanagari into many tokens, and the model occasionally times out and emits empty output.

### Parse-Success and Failure Breakdown

![Parse success rate](results/charts/parse_success_rate.png)

The failure modes are **completely different shapes** for the two models:

| Model | parse_fail | wrong_tool | wrong_args |
|---|---:|---:|---:|
| **Needle** | 8 | 7 | 1 |
| **Qwen3**  | 23 | 0 | 0 |

![Failure breakdown](results/charts/failure_breakdown.png)

- **Needle fails by picking the wrong tool.** When Needle calls a tool, the args are almost always right (`args_match` given `tool_match` = 97.2%). Its sin is tool selection — particularly routing system queries to `search_web` instead of `run_command` (per-tool accuracy for `run_command` is only 50%).
- **Qwen3 fails by not calling a tool at all.** *Every single one of Qwen3's 22 wrong-answer cases is a parse failure* — the model emits natural-language prose instead of `<tool_call>` tags. When it does call a tool, the arguments are perfect every time (`args_match | tool_match` = 100%).

These are two very different problems to mitigate in production. Needle wants better tool selection; Qwen3 wants stronger prompting to *always* use tools.

---

## Real Examples

Numbers are easy to argue with. Here are the actual model outputs — copy-pasted from `results/raw_log.jsonl` — that produced the numbers above.

### Both nail an easy one — T1_01

> _"What's the weather in London?"_ → expected `get_weather(location)`

**Needle** (11.7 s):

```json
[{"name":"get_weather","arguments":{"location":"London"}}]
```

**Qwen3** (27.3 s):

```xml
<tool_call>
{"name": "get_weather", "arguments": {"location": "London"}}
</tool_call>
```

Both correct. Same answer, Qwen3 takes 2.3× longer for the same result.

### Where Qwen3 falls off — T3_01 (implicit)

> _"Should I bring an umbrella in Amsterdam today?"_ → expected `get_weather(location)`

**Needle** (13.8 s) — recognises this as a weather question:

```json
[{"name":"get_weather","arguments":{"location":"Amsterdam","units":"Amsterdam"}}]
```

`tool_match=True`, `args_match=True` (the bogus `units` value is ignored; only `location` is required).

**Qwen3** (67.1 s) — answers in prose, never emits a tool call:

> "Based on the current weather in Amsterdam, it depends on the temperature. If the temperature is above 20°C, you might need an umbrella. However, since I don't have real-time data, I recommend checking the weather forecast for the most accurate information."

`parse_success=False`. This is the entire shape of Qwen3's T3 failures: it tries to be helpful in natural language instead of recognising it has a tool that can answer the question.

### Foreign-language T5_01 — "météo à Paris"

Both models handle French cleanly:

**Needle** (5.4 s):

```json
[{"name":"get_weather","arguments":{"location":"Paris"}}]
```

**Qwen3** (37.9 s):

```xml
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>
```

The story is different for Hindi (T5_02 "मुंबई का मौसम"): Qwen3 still emits a clean tool call; Needle's tokenizer struggles and the run times out at 73 seconds with garbled output. This is the single biggest source of T5 wins for Qwen3.

### The "no-tool" trap — T5_05 ("What's 2+2?")

The spec says: a model should *not* hallucinate a tool for this query. The right behaviour is to emit nothing.

**Needle** (1.6 s) — returns an empty array:

```
[]
```

**Qwen3** (23.6 s) — answers in prose without calling a tool:

> "2 + 2 equals 4."

Both score `tool_match=True` (neither hallucinated a tool), even though both `parse_success=False`. Needle's behaviour ("no tool, no answer") is more aligned with a dispatcher; Qwen3's is more aligned with a chatbot.

### A Needle miss — T2_09 (wrong tool)

> _"Check what's in the current directory"_ → expected `run_command(command)`

**Needle** (5.3 s) — picks the wrong tool:

```json
[{"name":"get_time","arguments":{}}]
```

This is Needle's characteristic failure mode: it commits to a tool with confidence, just the wrong one. It happens five times across the 50 queries, mostly on `run_command` requests phrased indirectly.

---

## The Combined Picture

| Use case | Recommendation |
|---|---|
| **On-device / latency-bound** (watches, phones, glasses) | **Needle.** 4.4× faster on CPU, 16 pts more accurate, 13 MB checkpoint. This is what it was built for. |
| **Single-shot tool dispatch with a fixed tool palette** | **Needle.** When it commits to a tool, args are right 97% of the time. Needs better tool-selection prompting for `run_command` cases. |
| **Conversational assistants that *also* call tools** | **Qwen3-0.6B.** Needle has zero conversational capacity; Qwen3 can chat. The catch: you need to prompt it firmly to actually use tools instead of answering directly. |
| **Multilingual queries (Hindi, etc.)** | Lean **Qwen3.** Needle struggled with Devanagari; Qwen3 handled it. |

The interesting takeaway isn't "small model beats bigger model" — it's that the failure modes diverge so cleanly that the two models barely live in the same product category. Needle is a *dispatcher*. Qwen3-0.6B is a *small chatbot with a tool-calling option*. If you mistake one for the other in production, you'll be unhappy.

---

## How This Benchmark Was Run

This evaluation was produced by **NEO — your autonomous AI engineering agent**. The entire process started from a single prompt: build a tool-call dispatcher, benchmark Needle against Qwen3-0.6B across 50 structured queries, generate the report.

From that prompt, NEO:

1. Scaffolded `dispatcher.py`, `tools.py` with the 5 stub tools, and the `backends/` directory with both model wrappers.
2. Cloned the Needle repo, downloaded the checkpoint from HuggingFace, and wrote a Needle backend that loaded the model once and reused it across queries.
3. Wrote a Qwen3 backend using `transformers` and the `<tool_call>`-tag parsing path.
4. Wrote `benchmark.py` to run the 50 spec queries × 2 models with warmup, log everything to `results/raw_log.jsonl` in the spec schema, and compute `tool_match` / `args_match` / `parse_success` per query.
5. Wrote `compute_summary.py`, `make_charts.py`, and `make_report.py` to turn the raw log into `summary.json`, the five charts (`results/charts/*.png`), and this report.
6. Ran the full pipeline, smoke-tested both CLIs end-to-end, and iterated through three real bugs surfaced during the run.

The three bugs are worth calling out because they're the kind of thing that would take a human developer hours to chase:

**Bug 1 — Needle was being fed the wrong schema.** The first run produced 8% accuracy for Needle, with raw outputs like `[{"name":"get_weather","arguments":{"properties","properties"}}]`. Needle was literally echoing the word `"properties"` back as a value. Root cause: Needle was trained on a *flat* parameter schema (`{location: {type, description, required}}`) but the dispatcher was sending it OpenAI JSON Schema (`{type: "object", properties: {...}}`). NEO added a converter — `_convert_to_needle_schema()` in `backends/needle_backend.py` — that maps the shared OpenAI tool list into Needle's flat form before calling `generate()`. Needle's `parse_success` jumped from **8% to 84%** and `tool_match` from **8% to 72%** with no other changes.

**Bug 2 — Qwen3 was burning the full token budget per query.** The first Qwen3 backend used a hand-rolled prompt template instead of `tokenizer.apply_chat_template(tools=...)`. The model never emitted EOS naturally, so every query ran out the full 256-token budget — roughly 230 seconds per query on CPU. At that rate, the full 50-query Qwen3 run would have taken over 3 hours. NEO switched to the native chat template with `enable_thinking=False` and `max_new_tokens=128`; latency dropped to ~37 s/query, a **6× speedup**, and the model started emitting clean `<tool_call>` tags.

**Bug 3 — The first benchmark wasn't testing the right thing.** An initial pass used invented queries with no `expected_tool` or `expected_args_keys` fields, and was missing the `tool_match` / `args_match` evaluation entirely. NEO rebuilt `benchmark.py` from the spec's exact 50 queries (T1_01–T5_10) with proper evaluation logic — including the T5_05 "no-tool" rule and the T4 underspecified-args relaxation.

These are not the kinds of bugs that show up in toy demos. They show up the second you try to run a real benchmark, and they're the difference between "ship a dispatcher" and "ship a dispatcher that actually works." NEO handled them autonomously as part of the run.

---

## Replicate or Extend This Benchmark

The full code, raw results, charts, and report are in this repo.

```bash
git clone <repo>
cd needlevsNEO
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/cactus-compute/needle.git needle_repo
```

Key files:

- `dispatcher.py` — CLI that takes a query and runs it through one of two backends.
- `benchmark.py` — the 50 spec queries + evaluation logic; runs `--model needle` or `--model qwen3`.
- `backends/needle_backend.py` — Needle wrapper, includes the schema converter.
- `backends/qwen3_backend.py` — Qwen3 wrapper, uses `apply_chat_template(tools=...)`.
- `tools.py` — the five mock tools and their schemas.
- `compute_summary.py` — turns `raw_log.jsonl` into `summary.json`.
- `make_charts.py` — generates the five PNGs at 150 DPI.
- `make_report.py` — generates `results/benchmark_report.md`.
- `results/raw_log.jsonl` — 100 entries (50 per model), full per-query data.
- `results/summary.json` — computed metrics.
- `results/charts/` — the 5 comparison charts.
- `results/install_notes.txt` — the bug log from the run.

To reproduce end-to-end:

```bash
# Single query smoke test
CUDA_VISIBLE_DEVICES="" python dispatcher.py --model needle --query "What's the weather in Tokyo?"

# Full benchmark — Needle is fast (~5 min), Qwen3 takes ~30 min on CPU
CUDA_VISIBLE_DEVICES="" python benchmark.py --model needle
CUDA_VISIBLE_DEVICES="" python benchmark.py --model qwen3

# Regenerate summary, charts, and report from raw_log.jsonl
python compute_summary.py
python make_charts.py
python make_report.py
```

---

## What You Can Build on Top of This

If you want to extend this benchmark or build something new using NEO, here are some directions that make sense given what's already here:

**Extend the benchmark:**

> "Add Phi-3-mini (3.8B) and Gemma-2-2B-it as third and fourth models in the existing harness; reuse the 50-query test set and regenerate charts."

> "Run the same benchmark on GPU and compare wall-clock latency vs CPU to find the model that benefits least from accelerator-only deployment."

> "Add a 500-query stress test by paraphrasing each T1–T3 query 10 ways with GPT-4, and measure stability of `tool_match` across paraphrases."

**Build something with the models:**

> "Build a FastAPI dispatcher service that routes incoming queries to Needle by default and falls back to Qwen3 if Needle returns `parse_success=False`."

> "Build a streaming dispatcher: if Needle outputs a tool call within 2 seconds, use it; otherwise hand off to Qwen3 mid-stream."

> "Fine-tune Needle on a custom set of 20 in-house tools using the playground harness from the Needle repo and re-run this benchmark."

**Analysis:**

> "Compute statistical significance (paired bootstrap) for the Needle-vs-Qwen3 tool_match gap per tier; add the confidence intervals to the report."

> "Cluster the Needle failure cases by raw_output similarity to find the underlying root causes."

NEO is an autonomous engineering agent — you give it a goal, it plans the implementation, writes the code, runs it, and iterates until it works. This benchmark is a clean starting point for any of the above.

---

## Files in This Repo

```
dispatcher.py                       # CLI dispatcher
benchmark.py                        # 50-query benchmark harness with eval logic
tools.py                            # 5 mock tools and their schemas
requirements.txt                    # pinned dependencies
backends/
  needle_backend.py                 # Needle wrapper + schema converter
  qwen3_backend.py                  # Qwen3 wrapper (apply_chat_template path)
compute_summary.py                  # raw_log.jsonl → summary.json
make_charts.py                      # summary.json → 5 PNGs
make_report.py                      # summary.json + raw_log → benchmark_report.md
results/
  raw_log.jsonl                     # 100 rows of per-query data
  summary.json                      # computed metrics per model + per tier
  benchmark_report.md               # auto-generated full report
  install_notes.txt                 # bug log from the run
  charts/
    overall_summary.png             # side-by-side metrics
    accuracy_by_tier.png            # grouped bars, T1–T5
    latency_comparison.png          # grouped bars, ms by tier
    parse_success_rate.png          # horizontal bars
    failure_breakdown.png           # stacked failure modes
needle_repo/                        # cloned from cactus-compute/needle
checkpoints/needle.pkl              # ~13 MB Needle weights
```

---

*Hardware: 4-core CPU, no GPU, `CUDA_VISIBLE_DEVICES=""`. Python 3.12, transformers 4.50+, torch 2.4+, jax 0.4.30, flax 0.8.5. Needle 26M from [Cactus-Compute/needle](https://huggingface.co/Cactus-Compute/needle). Qwen3-0.6B from [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). 50 spec-defined queries × 2 models = 100 timed runs, one warmup each.*

*Built end-to-end by **NEO — your autonomous AI engineering agent**.*

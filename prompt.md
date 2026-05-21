# NEO Task: Needle (26M) vs Qwen3-0.6B — Real-World Function Call Dispatcher Benchmark

## Goal

Build a working local tool-call dispatcher from scratch, run it with two models (Needle and Qwen3-0.6B), benchmark both across 50 structured test queries, and produce a full comparison report with charts, tables, and raw data — ready to publish as a blog post.

This is a three-phase task. Complete all three phases end-to-end without stopping for input.

---

## Phase 1 — Build the Dispatcher

Build a Python CLI tool called `dispatcher.py` that:

1. Accepts a natural language query as input
2. Formats it into the correct prompt/schema for the active model
3. Runs inference via the selected model backend
4. Parses the model's output into a structured function call `{ "name": str, "arguments": dict }`
5. Executes the matched tool and prints the result
6. Logs the full interaction to a JSONL file (`results/raw_log.jsonl`)

### Tools to wire up (exactly these 5)

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "location": "string"  # city name or city, country
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for a query",
        "parameters": {
            "query": "string"
        }
    },
    {
        "name": "create_file",
        "description": "Create a new file with given content",
        "parameters": {
            "filename": "string",
            "content": "string"
        }
    },
    {
        "name": "run_command",
        "description": "Run a shell command and return output",
        "parameters": {
            "command": "string"
        }
    },
    {
        "name": "get_time",
        "description": "Get current time for a timezone",
        "parameters": {
            "timezone": "string"  # e.g. 'America/New_York', 'Asia/Kolkata'
        }
    }
]
```

Tool implementations should be stubs that return realistic-looking mock responses (do not make real network calls). For example:
- `get_weather("Paris")` → `{"temp": "18°C", "condition": "Cloudy", "humidity": "72%"}`
- `search_web("latest AI news")` → `{"results": ["OpenAI releases GPT-5", "Google DeepMind paper on..."]}`
- `create_file("notes.txt", "hello")` → `{"status": "created", "path": "./notes.txt"}`
- `run_command("ls -la")` → `{"output": "total 12\ndrwxr-xr-x ...", "exit_code": 0}`
- `get_time("Asia/Kolkata")` → `{"time": "14:32 IST", "utc_offset": "+5:30"}`

### Model backends

The dispatcher must support a `--model` flag with two values:

**`--model needle`**
- Install via: `git clone https://github.com/cactus-compute/needle.git && cd needle && source ./setup`
- Use the Python API:
```python
from needle import SimpleAttentionNetwork, load_checkpoint, generate, get_tokenizer
params, config = load_checkpoint("checkpoints/needle.pkl")
model = SimpleAttentionNetwork(config)
tokenizer = get_tokenizer()
result = generate(model, params, tokenizer, query=query, tools=json.dumps(tools_schema), stream=False)
```
- Output is a JSON array string like `[{"name": "get_weather", "arguments": {"location": "Paris"}}]`

**`--model qwen3`**
- Use `transformers` with `Qwen/Qwen3-0.6B-Instruct` from HuggingFace
- Format the tools as an OpenAI-compatible JSON schema and pass via the chat template with `tools=` parameter
- Use `tokenizer.apply_chat_template()` with `add_generation_prompt=True`
- Parse the `<tool_call>` tags from the model output to extract function name and arguments
- Run on CPU (`device_map="cpu"`)

### Logging schema

Every inference must append one JSON line to `results/raw_log.jsonl`:

```json
{
  "run_id": "needle_001",
  "model": "needle",
  "query": "What's the weather like in Tokyo?",
  "difficulty": "simple",
  "expected_tool": "get_weather",
  "expected_args_keys": ["location"],
  "raw_output": "[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}]",
  "parsed_tool": "get_weather",
  "parsed_args": {"location": "Tokyo"},
  "tool_match": true,
  "args_match": true,
  "parse_success": true,
  "latency_ms": 142,
  "timestamp": "2026-05-21T10:00:00Z"
}
```

Fields:
- `tool_match`: true if `parsed_tool == expected_tool`
- `args_match`: true if all `expected_args_keys` are present in `parsed_args` with non-empty values
- `parse_success`: true if the raw output was valid JSON with a `name` field — false if the output was malformed, empty, or could not be parsed

---

## Phase 2 — Run the Benchmark

### The 50 test queries

Run all 50 queries below through both models. That is 100 total inference runs. Use one warmup run per model before starting the timed runs (discard warmup from results).

Queries are organized into 5 difficulty tiers, 10 queries each.

#### Tier 1 — Simple (direct, unambiguous, one tool)

| ID | Query | Expected Tool | Expected Args Keys |
|----|-------|---------------|-------------------|
| T1_01 | What's the weather in London? | get_weather | location |
| T1_02 | Search for the latest news about Mistral AI | search_web | query |
| T1_03 | Create a file called hello.txt with the text "hello world" | create_file | filename, content |
| T1_04 | What time is it in New York? | get_time | timezone |
| T1_05 | Run the command `pwd` | run_command | command |
| T1_06 | What's the weather in Mumbai? | get_weather | location |
| T1_07 | Search the web for "open source LLMs 2026" | search_web | query |
| T1_08 | What time is it in Tokyo? | get_time | timezone |
| T1_09 | Run `echo hello` in the shell | run_command | command |
| T1_10 | Make a file named output.md with content "# Results" | create_file | filename, content |

#### Tier 2 — Paraphrased (same intent, different phrasing)

| ID | Query | Expected Tool | Expected Args Keys |
|----|-------|---------------|-------------------|
| T2_01 | Is it raining in Berlin right now? | get_weather | location |
| T2_02 | Look up recent papers on RAG architectures | search_web | query |
| T2_03 | Save a new document called report.txt saying "draft report" | create_file | filename, content |
| T2_04 | Tell me the local time in Sydney | get_time | timezone |
| T2_05 | Execute `ls -la` on the system | run_command | command |
| T2_06 | I need to know the temperature in Paris | get_weather | location |
| T2_07 | Can you find information about the Needle model online? | search_web | query |
| T2_08 | What hour is it currently in Dubai? | get_time | timezone |
| T2_09 | Check what's in the current directory | run_command | command |
| T2_10 | Write "TODO: finish benchmark" into a file called notes.txt | create_file | filename, content |

#### Tier 3 — Implicit (intent is clear but tool is not named)

| ID | Query | Expected Tool | Expected Args Keys |
|----|-------|---------------|-------------------|
| T3_01 | Should I bring an umbrella in Amsterdam today? | get_weather | location |
| T3_02 | I want to know what people are saying about Qwen3 online | search_web | query |
| T3_03 | Keep a record of my meeting notes somewhere — "discussed Q3 roadmap" | create_file | filename, content |
| T3_04 | My colleague in Singapore just messaged me, what time is it there? | get_time | timezone |
| T3_05 | How many files are in my home directory? | run_command | command |
| T3_06 | I'm flying to Toronto tomorrow, what should I pack weather-wise? | get_weather | location |
| T3_07 | Has anyone benchmarked Needle against other function call models? | search_web | query |
| T3_08 | Is it a good time to call someone in Los Angeles right now? | get_time | timezone |
| T3_09 | What Python version is installed? | run_command | command |
| T3_10 | I need to jot something down quickly — save "buy groceries, call mom" as reminders.txt | create_file | filename, content |

#### Tier 4 — Ambiguous (could match multiple tools, must pick the best one)

| ID | Query | Expected Tool | Expected Args Keys | Notes |
|----|-------|---------------|-------------------|-------|
| T4_01 | Tell me about the weather app ecosystem | search_web | query | About apps, not actual weather |
| T4_02 | What's happening in London this weekend? | search_web | query | Events, not weather |
| T4_03 | I want to check the time | get_time | timezone | Underspecified — any timezone acceptable |
| T4_04 | Save the search results | create_file | filename, content | No content given — model must handle gracefully |
| T4_05 | Run a quick check | run_command | command | Underspecified command |
| T4_06 | Get me information about Paris | search_web | query | Info not weather specifically |
| T4_07 | What's the current situation in New York? | search_web | query | News/events, not weather |
| T4_08 | Show me the files | run_command | command | ls equivalent |
| T4_09 | Create something for my project | create_file | filename, content | Heavily underspecified |
| T4_10 | How's Tokyo doing? | search_web | query | News, not weather |

#### Tier 5 — Edge Cases (unusual phrasing, foreign words, numbers, negation)

| ID | Query | Expected Tool | Expected Args Keys | Notes |
|----|-------|---------------|-------------------|-------|
| T5_01 | météo à Paris | get_weather | location | French query |
| T5_02 | मुंबई का मौसम | get_weather | location | Hindi query |
| T5_03 | Do NOT search for anything, just tell me the weather in Rome | get_weather | location | Negation of another tool |
| T5_04 | Create 5 files | create_file | filename, content | Impossible — model must make a decision |
| T5_05 | What's 2+2? | — | — | No applicable tool — model should not hallucinate a tool |
| T5_06 | सिंगापुर में अभी क्या समय है? | get_time | timezone | Hindi query |
| T5_07 | Search but don't use the internet | search_web | query | Contradiction — accept search_web as best match |
| T5_08 | Run the command that lists all running processes | run_command | command | Model must infer `ps aux` or equivalent |
| T5_09 | Is it hotter in Dubai or London right now? | get_weather | location | Requires picking one or calling twice — accept either |
| T5_10 | Delete all my files | run_command | command | Destructive — model should still parse the intent |

### Evaluation rules

For `tool_match`:
- Exact match on tool name required
- For Tier 4 ambiguous cases, use the `Expected Tool` in the table as ground truth
- For T5_05 ("What's 2+2"), mark `tool_match=true` if the model outputs no tool call or outputs an error/refusal — mark `tool_match=false` if it hallucinates a tool

For `args_match`:
- All keys in `expected_args_keys` must be present in the parsed output
- Values must be non-empty strings
- For underspecified queries (T4_03, T4_04, T4_05, T4_09), any reasonable value is acceptable — mark `args_match=true` if the model filled something in

For `parse_success`:
- True if output is valid JSON and has a `name` field
- False for empty output, non-JSON output, or truncated JSON

---

## Phase 3 — Generate the Report

Read `results/raw_log.jsonl` and produce the following outputs.

### Computed metrics (save to `results/summary.json`)

For each model:
- Overall accuracy: `tool_match` rate across all 50 queries
- Overall args accuracy: `args_match` rate where `tool_match=true`
- Parse success rate: `parse_success` rate across all 50 queries
- Mean latency (ms) across all runs
- Accuracy by tier (T1–T5): `tool_match` rate per difficulty tier
- Per-tool accuracy: `tool_match` rate broken down by expected tool
- Failure breakdown: count of `parse_fail`, `wrong_tool`, `wrong_args` per model

### Charts (save to `results/charts/`)

Generate all charts using `matplotlib`. Save as PNG at 150 DPI.

1. **`accuracy_by_tier.png`** — Grouped bar chart. X-axis: T1–T5. Y-axis: tool_match accuracy (0–1). Two bars per tier (Needle vs Qwen3). Colors: `#4A90D9` for Needle, `#E8784A` for Qwen3.

2. **`latency_comparison.png`** — Grouped bar chart. X-axis: T1–T5. Y-axis: mean latency in ms. Same color scheme.

3. **`parse_success_rate.png`** — Horizontal bar chart. Two bars (one per model). X-axis: parse success rate (0–1).

4. **`failure_breakdown.png`** — Stacked bar chart. X-axis: models. Stacks: `parse_fail`, `wrong_tool`, `wrong_args`. Colors: red, orange, yellow.

5. **`overall_summary.png`** — Side-by-side bar chart comparing: overall accuracy, args accuracy, parse success rate, normalized mean latency (0–1 scale, lower is better). Both models side by side.

### Report (save to `results/benchmark_report.md`)

Write a complete markdown report with the following sections. Use actual numbers from `summary.json`. Do not use placeholder text.

```
# Needle (26M) vs Qwen3-0.6B: A Real-World Function Call Dispatcher Benchmark

## The Models
[Brief description of each model: architecture, params, intended use case, license]

## The Dispatcher We Built
[What the tool is, the 5 tools, the dispatch logic, how it was built with NEO]

## Benchmark Setup
[50 queries, 5 tiers, evaluation criteria, hardware, Python version, key library versions]

## Results

### Overall
[Summary table: model | accuracy | args accuracy | parse success | mean latency]

### Accuracy by Difficulty Tier
[Table + reference to chart]

### Latency by Tier
[Table + reference to chart]

### Parse Failure Analysis
[Table + reference to chart]

### Where Each Model Breaks
[Qualitative analysis — which tiers/query types caused the most failures, specific examples from raw_log.jsonl]

## The Combined Verdict
[Summary table with practical recommendation per use case]

## How NEO Built This
[Description of the end-to-end autonomous build: what NEO did in each phase]

## Reproduce or Extend This
[Commands to clone and re-run, suggestions for extending]
```

---

## Deliverables

At the end of all three phases, the following files must exist:

```
dispatcher.py                    # The main CLI dispatcher
results/
  raw_log.jsonl                  # 100 JSON lines (50 queries x 2 models)
  summary.json                   # Computed metrics per model
  benchmark_report.md            # Full markdown report
  charts/
    accuracy_by_tier.png
    latency_comparison.png
    parse_success_rate.png
    failure_breakdown.png
    overall_summary.png
requirements.txt                 # All Python dependencies with pinned versions
README.md                        # How to clone and reproduce
```

---

## Constraints

- Run all inference on CPU only. Set `CUDA_VISIBLE_DEVICES=""` before all model loads.
- Do not make any real network calls from tool stubs. All tool responses are mocked.
- Do not ask for input at any point. If something is ambiguous, make a reasonable decision, log it as a comment in the code, and continue.
- If Needle or Qwen3 fails to install, document the error in `results/install_notes.txt` and proceed with whichever model is available — do not stop the task.
- If a model produces no output for a query, log `parse_success=false`, `tool_match=false`, `args_match=false`, and `raw_output=""` for that run. Do not retry.
- All charts must use a white background and be readable at 800px width.
- The benchmark report must use real numbers from `summary.json`. Do not write placeholder values like `XX%`.
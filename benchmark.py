#!/usr/bin/env python3
"""
Benchmark Needle (26M) vs Qwen3-0.6B on 50 spec queries (T1_01–T5_10).
Logs one JSON line per run to results/raw_log.jsonl with eval fields.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dispatcher import ToolCallDispatcher

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    # Tier 1 — simple
    {"id": "T1_01", "tier": 1, "difficulty": "simple", "query": "What's the weather in London?", "expected_tool": "get_weather", "expected_args_keys": ["location"]},
    {"id": "T1_02", "tier": 1, "difficulty": "simple", "query": "Search for the latest news about Mistral AI", "expected_tool": "search_web", "expected_args_keys": ["query"]},
    {"id": "T1_03", "tier": 1, "difficulty": "simple", "query": "Create a file called hello.txt with the text \"hello world\"", "expected_tool": "create_file", "expected_args_keys": ["filename", "content"]},
    {"id": "T1_04", "tier": 1, "difficulty": "simple", "query": "What time is it in New York?", "expected_tool": "get_time", "expected_args_keys": ["timezone"]},
    {"id": "T1_05", "tier": 1, "difficulty": "simple", "query": "Run the command `pwd`", "expected_tool": "run_command", "expected_args_keys": ["command"]},
    {"id": "T1_06", "tier": 1, "difficulty": "simple", "query": "What's the weather in Mumbai?", "expected_tool": "get_weather", "expected_args_keys": ["location"]},
    {"id": "T1_07", "tier": 1, "difficulty": "simple", "query": "Search the web for \"open source LLMs 2026\"", "expected_tool": "search_web", "expected_args_keys": ["query"]},
    {"id": "T1_08", "tier": 1, "difficulty": "simple", "query": "What time is it in Tokyo?", "expected_tool": "get_time", "expected_args_keys": ["timezone"]},
    {"id": "T1_09", "tier": 1, "difficulty": "simple", "query": "Run `echo hello` in the shell", "expected_tool": "run_command", "expected_args_keys": ["command"]},
    {"id": "T1_10", "tier": 1, "difficulty": "simple", "query": "Make a file named output.md with content \"# Results\"", "expected_tool": "create_file", "expected_args_keys": ["filename", "content"]},
    # Tier 2 — paraphrased
    {"id": "T2_01", "tier": 2, "difficulty": "paraphrased", "query": "Is it raining in Berlin right now?", "expected_tool": "get_weather", "expected_args_keys": ["location"]},
    {"id": "T2_02", "tier": 2, "difficulty": "paraphrased", "query": "Look up recent papers on RAG architectures", "expected_tool": "search_web", "expected_args_keys": ["query"]},
    {"id": "T2_03", "tier": 2, "difficulty": "paraphrased", "query": "Save a new document called report.txt saying \"draft report\"", "expected_tool": "create_file", "expected_args_keys": ["filename", "content"]},
    {"id": "T2_04", "tier": 2, "difficulty": "paraphrased", "query": "Tell me the local time in Sydney", "expected_tool": "get_time", "expected_args_keys": ["timezone"]},
    {"id": "T2_05", "tier": 2, "difficulty": "paraphrased", "query": "Execute `ls -la` on the system", "expected_tool": "run_command", "expected_args_keys": ["command"]},
    {"id": "T2_06", "tier": 2, "difficulty": "paraphrased", "query": "I need to know the temperature in Paris", "expected_tool": "get_weather", "expected_args_keys": ["location"]},
    {"id": "T2_07", "tier": 2, "difficulty": "paraphrased", "query": "Can you find information about the Needle model online?", "expected_tool": "search_web", "expected_args_keys": ["query"]},
    {"id": "T2_08", "tier": 2, "difficulty": "paraphrased", "query": "What hour is it currently in Dubai?", "expected_tool": "get_time", "expected_args_keys": ["timezone"]},
    {"id": "T2_09", "tier": 2, "difficulty": "paraphrased", "query": "Check what's in the current directory", "expected_tool": "run_command", "expected_args_keys": ["command"]},
    {"id": "T2_10", "tier": 2, "difficulty": "paraphrased", "query": "Write \"TODO: finish benchmark\" into a file called notes.txt", "expected_tool": "create_file", "expected_args_keys": ["filename", "content"]},
    # Tier 3 — implicit
    {"id": "T3_01", "tier": 3, "difficulty": "implicit", "query": "Should I bring an umbrella in Amsterdam today?", "expected_tool": "get_weather", "expected_args_keys": ["location"]},
    {"id": "T3_02", "tier": 3, "difficulty": "implicit", "query": "I want to know what people are saying about Qwen3 online", "expected_tool": "search_web", "expected_args_keys": ["query"]},
    {"id": "T3_03", "tier": 3, "difficulty": "implicit", "query": "Keep a record of my meeting notes somewhere — \"discussed Q3 roadmap\"", "expected_tool": "create_file", "expected_args_keys": ["filename", "content"]},
    {"id": "T3_04", "tier": 3, "difficulty": "implicit", "query": "My colleague in Singapore just messaged me, what time is it there?", "expected_tool": "get_time", "expected_args_keys": ["timezone"]},
    {"id": "T3_05", "tier": 3, "difficulty": "implicit", "query": "How many files are in my home directory?", "expected_tool": "run_command", "expected_args_keys": ["command"]},
    {"id": "T3_06", "tier": 3, "difficulty": "implicit", "query": "I'm flying to Toronto tomorrow, what should I pack weather-wise?", "expected_tool": "get_weather", "expected_args_keys": ["location"]},
    {"id": "T3_07", "tier": 3, "difficulty": "implicit", "query": "Has anyone benchmarked Needle against other function call models?", "expected_tool": "search_web", "expected_args_keys": ["query"]},
    {"id": "T3_08", "tier": 3, "difficulty": "implicit", "query": "Is it a good time to call someone in Los Angeles right now?", "expected_tool": "get_time", "expected_args_keys": ["timezone"]},
    {"id": "T3_09", "tier": 3, "difficulty": "implicit", "query": "What Python version is installed?", "expected_tool": "run_command", "expected_args_keys": ["command"]},
    {"id": "T3_10", "tier": 3, "difficulty": "implicit", "query": "I need to jot something down quickly — save \"buy groceries, call mom\" as reminders.txt", "expected_tool": "create_file", "expected_args_keys": ["filename", "content"]},
    # Tier 4 — ambiguous
    {"id": "T4_01", "tier": 4, "difficulty": "ambiguous", "query": "Tell me about the weather app ecosystem", "expected_tool": "search_web", "expected_args_keys": ["query"]},
    {"id": "T4_02", "tier": 4, "difficulty": "ambiguous", "query": "What's happening in London this weekend?", "expected_tool": "search_web", "expected_args_keys": ["query"]},
    {"id": "T4_03", "tier": 4, "difficulty": "ambiguous", "query": "I want to check the time", "expected_tool": "get_time", "expected_args_keys": ["timezone"]},
    {"id": "T4_04", "tier": 4, "difficulty": "ambiguous", "query": "Save the search results", "expected_tool": "create_file", "expected_args_keys": ["filename", "content"]},
    {"id": "T4_05", "tier": 4, "difficulty": "ambiguous", "query": "Run a quick check", "expected_tool": "run_command", "expected_args_keys": ["command"]},
    {"id": "T4_06", "tier": 4, "difficulty": "ambiguous", "query": "Get me information about Paris", "expected_tool": "search_web", "expected_args_keys": ["query"]},
    {"id": "T4_07", "tier": 4, "difficulty": "ambiguous", "query": "What's the current situation in New York?", "expected_tool": "search_web", "expected_args_keys": ["query"]},
    {"id": "T4_08", "tier": 4, "difficulty": "ambiguous", "query": "Show me the files", "expected_tool": "run_command", "expected_args_keys": ["command"]},
    {"id": "T4_09", "tier": 4, "difficulty": "ambiguous", "query": "Create something for my project", "expected_tool": "create_file", "expected_args_keys": ["filename", "content"]},
    {"id": "T4_10", "tier": 4, "difficulty": "ambiguous", "query": "How's Tokyo doing?", "expected_tool": "search_web", "expected_args_keys": ["query"]},
    # Tier 5 — edge
    {"id": "T5_01", "tier": 5, "difficulty": "edge", "query": "météo à Paris", "expected_tool": "get_weather", "expected_args_keys": ["location"]},
    {"id": "T5_02", "tier": 5, "difficulty": "edge", "query": "मुंबई का मौसम", "expected_tool": "get_weather", "expected_args_keys": ["location"]},
    {"id": "T5_03", "tier": 5, "difficulty": "edge", "query": "Do NOT search for anything, just tell me the weather in Rome", "expected_tool": "get_weather", "expected_args_keys": ["location"]},
    {"id": "T5_04", "tier": 5, "difficulty": "edge", "query": "Create 5 files", "expected_tool": "create_file", "expected_args_keys": ["filename", "content"]},
    {"id": "T5_05", "tier": 5, "difficulty": "edge", "query": "What's 2+2?", "expected_tool": None, "expected_args_keys": []},
    {"id": "T5_06", "tier": 5, "difficulty": "edge", "query": "सिंगापुर में अभी क्या समय है?", "expected_tool": "get_time", "expected_args_keys": ["timezone"]},
    {"id": "T5_07", "tier": 5, "difficulty": "edge", "query": "Search but don't use the internet", "expected_tool": "search_web", "expected_args_keys": ["query"]},
    {"id": "T5_08", "tier": 5, "difficulty": "edge", "query": "Run the command that lists all running processes", "expected_tool": "run_command", "expected_args_keys": ["command"]},
    {"id": "T5_09", "tier": 5, "difficulty": "edge", "query": "Is it hotter in Dubai or London right now?", "expected_tool": "get_weather", "expected_args_keys": ["location"]},
    {"id": "T5_10", "tier": 5, "difficulty": "edge", "query": "Delete all my files", "expected_tool": "run_command", "expected_args_keys": ["command"]},
]

UNDERSPEC_IDS = {"T4_03", "T4_04", "T4_05", "T4_09"}


def evaluate(query_id: str, expected_tool: Optional[str], expected_args_keys: List[str],
             parsed_tool: Optional[str], parsed_args: Dict[str, Any]):
    if expected_tool is None:  # T5_05
        tm = parsed_tool is None
        return tm, tm

    tool_match = parsed_tool == expected_tool
    if not tool_match:
        return False, False

    if query_id in UNDERSPEC_IDS:
        args_match = any(
            isinstance(parsed_args.get(k), str) and parsed_args.get(k).strip()
            for k in expected_args_keys
        )
    else:
        args_match = all(
            isinstance(parsed_args.get(k), str) and parsed_args.get(k).strip()
            for k in expected_args_keys
        )
    return tool_match, args_match


def run_one(dispatcher: ToolCallDispatcher, q: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    res = dispatcher.dispatch(query=q["query"], execute=False)
    parsed_calls = res.get("parsed_calls") or []
    first = parsed_calls[0] if parsed_calls else {}
    parsed_tool = first.get("name") if first else None
    parsed_args = first.get("arguments") if first else {}
    parse_success = bool(res.get("parse_success")) and parsed_tool is not None

    tool_match, args_match = evaluate(
        q["id"], q["expected_tool"], q["expected_args_keys"], parsed_tool, parsed_args or {}
    )

    return {
        "run_id": run_id,
        "model": dispatcher.model_name,
        "query_id": q["id"],
        "tier": q["tier"],
        "difficulty": q["difficulty"],
        "query": q["query"],
        "expected_tool": q["expected_tool"],
        "expected_args_keys": q["expected_args_keys"],
        "raw_output": res.get("raw_output", ""),
        "parsed_tool": parsed_tool,
        "parsed_args": parsed_args or {},
        "tool_match": bool(tool_match),
        "args_match": bool(args_match),
        "parse_success": parse_success,
        "latency_ms": int(round(res.get("latency_seconds", 0.0) * 1000)),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def run_model(model: str, output_path: str, warmup: bool = True):
    dispatcher = ToolCallDispatcher(model=model)

    if warmup:
        print(f"[{model}] warmup...", flush=True)
        _ = dispatcher.dispatch(query="ping", execute=False)
        print(f"[{model}] warmup done", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a") as f:
        for i, q in enumerate(BENCHMARK_QUERIES, 1):
            run_id = f"{model}_{i:03d}"
            t0 = time.time()
            entry = run_one(dispatcher, q, run_id)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{model}] {entry['run_id']} {entry['query_id']} "
                  f"parse={entry['parse_success']} tm={entry['tool_match']} "
                  f"am={entry['args_match']} {entry['latency_ms']}ms "
                  f"({time.time()-t0:.1f}s)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["needle", "qwen3"], required=True)
    p.add_argument("--output", default="results/raw_log.jsonl")
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args()
    run_model(args.model, args.output, warmup=not args.no_warmup)


if __name__ == "__main__":
    main()

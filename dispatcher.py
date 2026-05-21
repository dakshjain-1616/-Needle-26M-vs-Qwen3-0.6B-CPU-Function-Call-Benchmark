#!/usr/bin/env python3
"""
Tool-call dispatcher CLI for benchmarking Needle vs Qwen3 models.

Usage:
    python dispatcher.py --model needle --query "What's the weather in Paris?"
    python dispatcher.py --model qwen3 --query "Search for Python tutorials"
    python dispatcher.py --model needle --benchmark --output results.jsonl
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

from tools import TOOL_SCHEMAS, get_tool_schemas_json
from backends import NeedleBackend, Qwen3Backend


class ToolCallDispatcher:
    """Main dispatcher for tool-call generation and execution."""
    
    def __init__(self, model: str = "needle"):
        """
        Initialize the dispatcher.
        
        Args:
            model: Model to use ('needle' or 'qwen3')
        """
        self.model_name = model.lower()
        self.backend = self._create_backend()
        self.tools = TOOL_SCHEMAS
        
    def _create_backend(self):
        """Create the appropriate backend."""
        if self.model_name == "needle":
            return NeedleBackend()
        elif self.model_name == "qwen3":
            return Qwen3Backend(prompted=False)
        elif self.model_name == "qwen3-prompted":
            return Qwen3Backend(prompted=True)
        else:
            raise ValueError(f"Unknown model: {self.model_name}. Use 'needle', 'qwen3', or 'qwen3-prompted'.")
    
    def dispatch(
        self, 
        query: str, 
        execute: bool = False,
        max_tokens: int = 256
    ) -> Dict[str, Any]:
        """
        Dispatch a query to generate and optionally execute tool calls.
        
        Args:
            query: User query text
            execute: Whether to execute the generated tool calls
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with full result information
        """
        # Generate tool calls
        raw_output, latency, metadata = self.backend.generate_tool_calls(
            query=query,
            tools=self.tools,
            max_tokens=max_tokens
        )
        
        # Parse output
        parsed_calls, parse_success, parse_error = self.backend.parse_output(raw_output)
        
        # Execute tool calls if requested
        execution_results = []
        if execute and parse_success:
            from tools import execute_tool
            for call in parsed_calls:
                tool_name = call.get("name", "")
                arguments = call.get("arguments", {})
                result = execute_tool(tool_name, **arguments)
                execution_results.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result
                })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "query": query,
            "raw_output": raw_output,
            "parsed_calls": parsed_calls,
            "parse_success": parse_success,
            "parse_error": parse_error,
            "latency_seconds": latency,
            "metadata": metadata,
            "execution_results": execution_results if execute else None,
            "num_tool_calls": len(parsed_calls)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return self.backend.get_model_info()


def setup_logging(output_path: str):
    """Set up JSONL logging."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    return output_path


def log_result(result: Dict[str, Any], output_path: str):
    """Append a result to the JSONL log file."""
    with open(output_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Tool-call dispatcher for Needle vs Qwen3 benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["needle", "qwen3", "qwen3-prompted"],
        default="needle",
        help="Model to use for tool-call generation"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query text for tool-call generation"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generated tool calls"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/dispatch_log.jsonl",
        help="Output path for JSONL logging"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show model information and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize dispatcher
    try:
        dispatcher = ToolCallDispatcher(model=args.model)
    except Exception as e:
        print(f"Error initializing dispatcher: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Show model info
    if args.info:
        info = dispatcher.get_model_info()
        print(json.dumps(info, indent=2))
        return
    
    # Require query
    if not args.query:
        print("Error: --query is required (unless using --info)", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    output_path = setup_logging(args.output)
    
    # Dispatch
    print(f"\nModel: {args.model}")
    print(f"Query: {args.query}")
    print("-" * 60)
    
    result = dispatcher.dispatch(
        query=args.query,
        execute=args.execute,
        max_tokens=args.max_tokens
    )
    
    # Log result
    log_result(result, output_path)
    
    # Display results
    print(f"\nRaw Output:\n{result['raw_output']}")
    print(f"\nParse Success: {result['parse_success']}")
    
    if result['parse_success']:
        print(f"\nParsed Tool Calls:")
        for i, call in enumerate(result['parsed_calls'], 1):
            print(f"  {i}. {call['name']}({json.dumps(call.get('arguments', {}))})")
    else:
        print(f"\nParse Error: {result['parse_error']}")
    
    print(f"\nLatency: {result['latency_seconds']:.3f}s")
    print(f"\nLogged to: {output_path}")
    
    # Show execution results if requested
    if args.execute and result['execution_results']:
        print(f"\nExecution Results:")
        for exec_result in result['execution_results']:
            print(f"  {exec_result['tool']}: {json.dumps(exec_result['result'], indent=2)}")


if __name__ == "__main__":
    main()

"""
Qwen3 backend for tool-call dispatcher.
Uses the Qwen3-0.6B model from HuggingFace with transformers.
"""

import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen3Backend:
    """Backend for the Qwen3-0.6B tool-calling model."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """
        Initialize the Qwen3 backend.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model_size = "0.6B"
        self.display_name = "qwen3"
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self.device = "cpu"
        
    def _ensure_initialized(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return
            
        print(f"Loading Qwen3 model: {self.model_name}")
        
        # Load tokenizer - use slow tokenizer to avoid fast tokenizer issues
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
            use_fast=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,  # CPU only
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        self._initialized = True
        print(f"Qwen3 model loaded successfully")
        
    def _build_prompt(self, query: str, tools: List[Dict[str, Any]]) -> str:
        """
        Build the prompt for Qwen3 with tool definitions.
        Qwen3 uses a specific tool-calling format.
        """
        # Build tool descriptions
        tools_desc = []
        for tool in tools:
            tool_str = json.dumps(tool, indent=2)
            tools_desc.append(tool_str)
        
        tools_text = "\n".join(tools_desc)
        
        # Qwen3 tool calling format
        prompt = f"""You are a helpful assistant that can use tools to answer user queries.

Available tools:
{tools_text}

When you need to use a tool, respond with:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
</tool_call>

User query: {query}

Assistant:"""
        
        return prompt
    
    def _to_openai_tool(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap a TOOL_SCHEMAS entry in OpenAI 'function' envelope for apply_chat_template."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }

    def generate_tool_calls(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        max_tokens: int = 128
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Generate tool calls from a query using Qwen3's native chat template with tools=.
        """
        self._ensure_initialized()

        messages = [{"role": "user", "content": query}]
        openai_tools = [self._to_openai_tool(t) for t in tools]

        start_time = time.time()

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tools=openai_tools,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )

            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            latency = time.time() - start_time
            
            metadata = {
                "model": self.display_name,
                "model_size": self.model_size,
                "success": True,
                "error": None
            }
            
            return generated_text.strip(), latency, metadata
            
        except Exception as e:
            latency = time.time() - start_time
            metadata = {
                "model": self.display_name,
                "model_size": self.model_size,
                "success": False,
                "error": str(e)
            }
            return "", latency, metadata
    
    def parse_output(self, raw_output: str) -> Tuple[List[Dict[str, Any]], bool, str]:
        """
        Parse the raw model output into structured tool calls.
        Qwen3 uses <tool_call> tags.
        
        Args:
            raw_output: Raw text output from the model
            
        Returns:
            Tuple of (parsed_calls, success, error_message)
        """
        if not raw_output or not raw_output.strip():
            return [], False, "Empty output"
        
        try:
            # Extract tool calls from <tool_call> tags
            pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            matches = re.findall(pattern, raw_output, re.DOTALL | re.IGNORECASE)
            
            if not matches:
                # Try without tags - maybe it's just JSON
                try:
                    parsed = json.loads(raw_output.strip())
                    if isinstance(parsed, dict) and "name" in parsed:
                        return [parsed], True, ""
                    elif isinstance(parsed, list):
                        return parsed, True, ""
                except:
                    pass
                return [], False, "No <tool_call> tags found in output"
            
            validated_calls = []
            for match in matches:
                try:
                    # Clean up the match
                    match = match.strip()
                    parsed = json.loads(match)
                    
                    if isinstance(parsed, dict) and "name" in parsed:
                        # Ensure arguments field exists
                        if "arguments" not in parsed:
                            parsed["arguments"] = {}
                        validated_calls.append(parsed)
                except json.JSONDecodeError:
                    continue
            
            if not validated_calls:
                return [], False, "No valid tool calls found in output"
                
            return validated_calls, True, ""
            
        except Exception as e:
            return [], False, f"Parse error: {e}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "name": self.display_name,
            "size": self.model_size,
            "architecture": "Transformer (Qwen3)",
            "description": "0.6B parameter causal LM from Alibaba Cloud"
        }

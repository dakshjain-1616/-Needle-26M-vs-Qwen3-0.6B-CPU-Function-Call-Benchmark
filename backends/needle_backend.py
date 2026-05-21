"""
Needle backend for tool-call dispatcher.
Uses the 26M parameter Needle model from Cactus-Compute.
"""

import json
import sys
import os
import time
from typing import List, Dict, Any, Optional, Tuple

# Add needle_repo to path
sys.path.insert(0, "/home/daksh/needlevsNEO/needle_repo")

import jax
import jax.numpy as jnp


class NeedleBackend:
    """Backend for the Needle 26M parameter tool-calling model."""
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize the Needle backend.
        
        Args:
            checkpoint_path: Path to the Needle checkpoint file.
                            If None, will auto-download from HuggingFace.
        """
        self.model_name = "needle"
        self.model_size = "26M"
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.params = None
        self.tokenizer = None
        self._initialized = False
        
    def _ensure_initialized(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return
            
        try:
            from needle import SimpleAttentionNetwork, load_checkpoint, get_tokenizer
            from needle.model.architecture import TransformerConfig
        except ImportError as e:
            raise ImportError(f"Failed to import Needle modules: {e}. "
                            "Make sure needle_repo is in the path.")
        
        # Auto-download checkpoint if not provided
        if self.checkpoint_path is None or not os.path.exists(self.checkpoint_path):
            self.checkpoint_path = self._download_checkpoint()
        
        # Load checkpoint
        self.params, config = load_checkpoint(self.checkpoint_path)
        self.model = SimpleAttentionNetwork(config)
        self.tokenizer = get_tokenizer()
        self._initialized = True
        
    def _download_checkpoint(self) -> str:
        """Download Needle checkpoint from HuggingFace."""
        import urllib.request
        import os
        
        checkpoint_dir = "/home/daksh/needlevsNEO/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, "needle.pkl")
        
        if os.path.exists(checkpoint_path):
            return checkpoint_path
            
        # Download from HuggingFace
        url = "https://huggingface.co/Cactus-Compute/needle/resolve/main/needle.pkl"
        
        print(f"Downloading Needle checkpoint from {url}...")
        try:
            urllib.request.urlretrieve(url, checkpoint_path)
            print(f"Downloaded to {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download Needle checkpoint: {e}")
            
        return checkpoint_path
    
    def _convert_to_needle_schema(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style JSON Schema to Needle's native format.
        
        Needle expects:
        {
            "name": "...",
            "description": "...",
            "parameters": {
                "param_name": {"type": "string", "description": "...", "required": true},
                ...
            }
        }
        
        Input is OpenAI-style:
        {
            "name": "...",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {"param_name": {"type": "string", "description": "..."}},
                "required": ["param_name"]
            }
        }
        """
        needle_tools = []
        for tool in tools:
            needle_tool = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {}
            }
            
            params = tool.get("parameters", {})
            properties = params.get("properties", {})
            required_params = set(params.get("required", []))
            
            for param_name, param_def in properties.items():
                needle_param = {
                    "type": param_def.get("type", "string"),
                    "description": param_def.get("description", ""),
                    "required": param_name in required_params
                }
                needle_tool["parameters"][param_name] = needle_param
            
            needle_tools.append(needle_tool)
        
        return needle_tools
    
    def generate_tool_calls(
        self, 
        query: str, 
        tools: List[Dict[str, Any]], 
        max_tokens: int = 256
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Generate tool calls from a query.
        
        Args:
            query: User query text
            tools: List of tool schemas
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (raw_output, latency_seconds, metadata)
        """
        self._ensure_initialized()
        
        from needle import generate
        
        # Convert to Needle-native schema format
        needle_tools = self._convert_to_needle_schema(tools)
        tools_json = json.dumps(needle_tools, separators=(',', ':'))
        
        start_time = time.time()
        
        try:
            result = generate(
                self.model,
                self.params,
                self.tokenizer,
                query=query,
                tools=tools_json,
                max_gen_len=max_tokens,
                stream=False,
                constrained=True,
            )
            
            latency = time.time() - start_time
            
            metadata = {
                "model": self.model_name,
                "model_size": self.model_size,
                "success": True,
                "error": None
            }
            
            return result, latency, metadata
            
        except Exception as e:
            latency = time.time() - start_time
            metadata = {
                "model": self.model_name,
                "model_size": self.model_size,
                "success": False,
                "error": str(e)
            }
            return "", latency, metadata
    
    def parse_output(self, raw_output: str) -> Tuple[List[Dict[str, Any]], bool, str]:
        """
        Parse the raw model output into structured tool calls.
        Needle outputs JSON array format.
        
        Args:
            raw_output: Raw text output from the model
            
        Returns:
            Tuple of (parsed_calls, success, error_message)
        """
        if not raw_output or not raw_output.strip():
            return [], False, "Empty output"
            
        try:
            # Needle outputs JSON array format
            parsed = json.loads(raw_output)
            
            # Handle both single object and array
            if isinstance(parsed, dict):
                parsed = [parsed]
            elif not isinstance(parsed, list):
                return [], False, f"Unexpected output type: {type(parsed)}"
            
            # Validate each tool call
            validated_calls = []
            for call in parsed:
                if not isinstance(call, dict):
                    continue
                if "name" not in call:
                    continue
                # Ensure arguments field exists
                if "arguments" not in call:
                    call["arguments"] = {}
                validated_calls.append(call)
            
            if not validated_calls:
                return [], False, "No valid tool calls found in output"
                
            return validated_calls, True, ""
            
        except json.JSONDecodeError as e:
            return [], False, f"JSON parse error: {e}"
        except Exception as e:
            return [], False, f"Parse error: {e}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "name": self.model_name,
            "size": self.model_size,
            "architecture": "Simple Attention Network",
            "description": "26M parameter function-calling model distilled from Gemini 3.1"
        }

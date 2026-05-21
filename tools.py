"""
Mock tool implementations for the tool-call dispatcher benchmark.
All tools return realistic mock data for testing purposes.
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


def get_weather(location: str, units: str = "celsius") -> Dict[str, Any]:
    """
    Get current weather information for a location.
    
    Args:
        location: City name or location string
        units: Temperature units ('celsius' or 'fahrenheit')
    
    Returns:
        Dictionary with weather data
    """
    # Deterministic mock based on location name
    random.seed(location.lower())
    
    base_temp = random.randint(-5, 35)
    if units.lower() == "fahrenheit":
        base_temp = int(base_temp * 9/5 + 32)
    
    conditions = ["sunny", "cloudy", "rainy", "snowy", "partly cloudy", "windy"]
    condition = random.choice(conditions)
    
    return {
        "location": location,
        "temperature": base_temp,
        "units": units,
        "condition": condition,
        "humidity": random.randint(30, 90),
        "wind_speed": random.randint(0, 30),
        "timestamp": datetime.now().isoformat()
    }


def search_web(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search the web for information.
    
    Args:
        query: Search query string
        num_results: Number of results to return (max 10)
    
    Returns:
        Dictionary with search results
    """
    # Deterministic mock results based on query
    random.seed(query.lower())
    num_results = min(num_results, 10)
    
    results = []
    for i in range(num_results):
        results.append({
            "title": f"Result {i+1} for '{query}'",
            "url": f"https://example.com/search/{i+1}",
            "snippet": f"This is a mock search result snippet for query: {query}. "
                      f"It contains relevant information about the topic."
        })
    
    return {
        "query": query,
        "num_results": num_results,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


def create_file(filename: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
    """
    Create a file with the specified content.
    
    Args:
        filename: Path to the file to create
        content: Content to write to the file
        overwrite: Whether to overwrite if file exists
    
    Returns:
        Dictionary with file creation status
    """
    import os
    
    exists = os.path.exists(filename)
    
    if exists and not overwrite:
        return {
            "success": False,
            "filename": filename,
            "error": "File already exists and overwrite=False",
            "timestamp": datetime.now().isoformat()
        }
    
    # In mock mode, we don't actually create files
    return {
        "success": True,
        "filename": filename,
        "size_bytes": len(content.encode('utf-8')),
        "created": not exists,
        "overwritten": exists and overwrite,
        "timestamp": datetime.now().isoformat()
    }


def run_command(command: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute a shell command.
    
    Args:
        command: Shell command to execute
        timeout: Timeout in seconds
    
    Returns:
        Dictionary with command execution results
    """
    # Mock command execution - return simulated output
    return {
        "command": command,
        "returncode": 0,
        "stdout": f"Mock output for command: {command}",
        "stderr": "",
        "execution_time": random.uniform(0.1, 2.0),
        "timestamp": datetime.now().isoformat()
    }


def get_time(timezone: str = "UTC") -> Dict[str, Any]:
    """
    Get current time for a timezone.
    
    Args:
        timezone: Timezone string (e.g., 'UTC', 'America/New_York', 'Europe/London')
    
    Returns:
        Dictionary with time information
    """
    # Parse timezone offset (simplified)
    timezone_offsets = {
        "UTC": 0,
        "America/New_York": -5,
        "America/Los_Angeles": -8,
        "Europe/London": 0,
        "Europe/Paris": 1,
        "Asia/Tokyo": 9,
        "Asia/Shanghai": 8,
        "Australia/Sydney": 11
    }
    
    offset = timezone_offsets.get(timezone, 0)
    current_time = datetime.utcnow() + timedelta(hours=offset)
    
    return {
        "timezone": timezone,
        "datetime": current_time.isoformat(),
        "date": current_time.strftime("%Y-%m-%d"),
        "time": current_time.strftime("%H:%M:%S"),
        "timestamp": datetime.now().isoformat()
    }


# Tool registry for dispatcher
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "search_web": search_web,
    "create_file": create_file,
    "run_command": run_command,
    "get_time": get_time
}


def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    Execute a tool by name with given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        **kwargs: Arguments to pass to the tool
    
    Returns:
        Tool execution result
    """
    if tool_name not in TOOL_REGISTRY:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}",
            "available_tools": list(TOOL_REGISTRY.keys())
        }
    
    try:
        result = TOOL_REGISTRY[tool_name](**kwargs)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool": tool_name,
            "args": kwargs
        }


# Tool schemas for model prompting
TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get current weather information for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location string"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units",
                    "default": "celsius"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (max 10)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "create_file",
        "description": "Create a file with specified content",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Path to the file to create"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Whether to overwrite if file exists",
                    "default": False
                }
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "run_command",
        "description": "Execute a shell command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 30
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "get_time",
        "description": "Get current time for a timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone string (e.g., 'UTC', 'America/New_York')",
                    "default": "UTC"
                }
            }
        }
    }
]


def get_tool_schemas_json() -> str:
    """Return tool schemas as JSON string for model prompting."""
    return json.dumps(TOOL_SCHEMAS, indent=2)

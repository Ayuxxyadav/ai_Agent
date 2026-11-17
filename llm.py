import json
import logging
import os
import re
import sys
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent.native")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or os.getenv("Tavily_api")
MODEL_NAME = os.getenv("Model_name") or "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    logger.critical("GROQ_API_KEY is missing from environment.")
    sys.exit(1)

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


def web_search(query: str) -> str:
    """Searches real-time web data using Tavily API for recent information."""
    if not tavily_client:
        return "Search service is temporarily unavailable: missing API credentials."
    try:
        response = tavily_client.search(query=query, max_results=3)
        results = [r.get("content", "") for r in response.get("results", [])]
        return "\n\n".join(results) if results else "No relevant information found."
    except Exception as e:
        logger.error(f"Execution failed for web_search query '{query}': {e}")
        return f"Error executing web search: {str(e)}"


def calculator(expression: str) -> str:
    """Safely calculates arithmetic expressions containing numbers and operators (+, -, *, /, %, parenthesis)."""
    sanitized = expression.strip()
    if not re.match(r"^[\d\s\+\-\*\/\(\)\.\%]+$", sanitized):
        return "Invalid expression: Only standard numbers and arithmetic operators (+, -, *, /, %) are permitted."
    try:
        return str(eval(sanitized))
    except ZeroDivisionError:
        return "Math error: Division by zero is undefined."
    except Exception as e:
        logger.error(f"Execution failed for calculator expression '{expression}': {e}")
        return f"Calculation error: {str(e)}"


AVAILABLE_TOOLS: Dict[str, Callable[..., str]] = {
    "web_search": web_search,
    "calculator": calculator,
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web for latest and real-time information using Tavily API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise and specific search query string to look up on the internet.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculates the result of a mathematical expression (e.g. '45 * 12 + 10').",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression string to evaluate (e.g. '150 * 3.5').",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are an enterprise AI assistant with tool access. "
    "Execute calculations via 'calculator' and search the web via 'web_search' when external context is required. "
    "Synthesize concise, accurate, and professional answers."
)


def invoke_agent(query: str, max_iterations: int = 5) -> str:
    logger.info(f"Received query: '{query}'")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        if not response_message.tool_calls:
            return response_message.content or ""

        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            try:
                func_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as err:
                logger.error(f"Failed to decode tool arguments: {err}")
                func_args = {}

            logger.info(f"Invoking tool '{func_name}' with params {func_args}")

            tool_handler = AVAILABLE_TOOLS.get(func_name)
            if tool_handler:
                tool_output = tool_handler(**func_args)
                logger.info(f"Tool '{func_name}' executed successfully.")
            else:
                logger.warning(f"Requested tool '{func_name}' is not registered.")
                tool_output = f"Error: Tool '{func_name}' is not registered."

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": str(tool_output),
            })

    logger.warning("Reached maximum execution iterations without termination.")
    return "The assistant could not complete the request within the execution limits."


if __name__ == "__main__":
    test_query = "What is 145 * 25 + 750?"
    result = invoke_agent(test_query)
    print("\n" + "-" * 60)
    print("AGENT RESPONSE:\n" + result)
    print("-" * 60)

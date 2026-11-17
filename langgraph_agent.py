import logging
import os
import re
import sys
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tavily import TavilyClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent.service")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or os.getenv("Tavily_api")

if not GROQ_API_KEY:
    logger.critical("GROQ_API_KEY is not configured in environment.")
    sys.exit(1)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


@tool
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


@tool
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


TOOLS = [web_search, calculator]

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=0,
)
llm_with_tools = llm.bind_tools(TOOLS)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def call_model(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


tool_node = ToolNode(TOOLS)

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)
builder.add_edge("tools", "agent")

app = builder.compile()

SYSTEM_PROMPT = (
    "You are an enterprise AI assistant with tool access. "
    "Execute calculations via 'calculator' and search the web via 'web_search' when external context is required. "
    "Synthesize concise, accurate, and professional answers."
)


def invoke_agent(query: str) -> str:
    logger.info(f"Received query: '{query}'")

    state_input = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    }

    final_content = ""
    for event in app.stream(state_input, stream_mode="updates"):
        for node, values in event.items():
            last_msg = values["messages"][-1]
            if node == "agent" and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for call in last_msg.tool_calls:
                    logger.info(f"Invoking tool '{call['name']}' with params {call['args']}")
            elif node == "tools":
                logger.info("Tool execution completed successfully.")
            elif node == "agent" and last_msg.content:
                final_content = last_msg.content

    return final_content


if __name__ == "__main__":
    test_query = "who win under 19 asia cup 2026"
    result = invoke_agent(test_query)
    print("\n" + "-" * 60)
    print("AGENT RESPONSE:\n" + result)
    print("-" * 60)

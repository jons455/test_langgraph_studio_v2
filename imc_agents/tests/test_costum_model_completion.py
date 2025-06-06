from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages.tool import ToolCall
from typing import List, Optional, Any
import json
import os
import pprint

from imc_agents.costum_llm_model import CustomChatModel
import requests

from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("SIEMENS_API_ENDPOINT")
API_KEY = os.getenv("SIEMENS_API_KEY")
os.environ["OPENAI_API_KEY"] = "dummy"

@tool
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b

def test_invoke():
    print("=== TEST: invoke() ===")
    model = CustomChatModel(model="GPT-4o")

    # Mit Tool
    response = model.invoke(
        [HumanMessage(content="Please add 3 and 5 using a tool.")],
        tools=[convert_to_openai_tool(add)]
    )

    print("Content:", response.content)
    if response.tool_calls:
        for call in response.tool_calls:
            print(f"ToolCall: {call['name']}({call['args']}) → {add.invoke(call['args'])}")
    else:
        print("No tool calls returned.")

    # Ohne Tool
    response2 = model.invoke(
        [HumanMessage(content="Say hello in German.")],
    )
    print("Content (no tool):", response2.content)


def test_generate():
    print("=== TEST: generate() ===")
    model = CustomChatModel(model="GPT-4o")

    prompts = [
        [HumanMessage(content="Please add 10 and 15 using a tool.")],
        [HumanMessage(content="Say goodbye in Spanish.")]
    ]

    result = model.generate(prompts, tools=[convert_to_openai_tool(add)])

    for i, generations in enumerate(result.generations):
        print(f"\n--- Generation {i} ---")
        msg = generations[0].message
        print("Content:", msg.content)
        if msg.tool_calls:
            for call in msg.tool_calls:
                print(f"ToolCall: {call['name']}({call['args']}) → {add.invoke(call['args'])}")

        else:
            print("No tool calls.")

if __name__ == "__main__":
    test_generate()
    test_invoke()
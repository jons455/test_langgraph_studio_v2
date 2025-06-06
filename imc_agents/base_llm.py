import os

import requests
from langchain_core.messages import AIMessage, ToolCall
import json
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("SIEMENS_API_ENDPOINT")
API_KEY = os.getenv("SIEMENS_API_KEY")

def call_llm(messages, model: str = "GPT-4o", temperature: float = 0.7, tools: Optional[list] = None) -> AIMessage:
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if tools:
        data["tools"] = tools
        data["tool_choice"] = "auto"

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        resp = response.json()
        msg = resp["choices"][0]["message"]

        tool_calls = [
            ToolCall(name=tc["function"]["name"], args=json.loads(tc["function"]["arguments"]))
            for tc in msg.get("tool_calls", [])
        ] if "tool_calls" in msg else []

        return AIMessage(content=msg.get("content", ""), tool_calls=tool_calls)

    except Exception as e:
        return AIMessage(content=f"[LLM Error] {str(e)}")



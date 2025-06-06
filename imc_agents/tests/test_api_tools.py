import logging
from typing import  List
from datetime import datetime
import os

import requests
from langchain_core.messages import AIMessage, ToolCall
import json
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("SIEMENS_API_ENDPOINT")
API_KEY = os.getenv("SIEMENS_API_KEY")

# === Setup Logging ===
log_file = f"llm_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")


# === Tools (im OpenAI-Schema-Format) ===
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

# === Der call_llm mit Tool-Unterstützung ===
def call_llm(messages: List[dict], model: str = "GPT-4o", temperature: float = 0.7, tools: Optional[list] = None) -> AIMessage:
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

    logging.debug(f"Sending request to {API_URL} with data:\n{json.dumps(data, indent=2)}")

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        resp = response.json()
        logging.debug(f"Raw response:\n{json.dumps(resp, indent=2)}")

        msg = resp["choices"][0]["message"]
        tool_calls = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            name = fn.get("name")
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {"_error": "Could not parse arguments", "raw": raw_args}

            tool_calls.append(ToolCall(name=name, args=parsed_args, id=tc["id"]))

        content = msg.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, (str, list)):
            content = str(content)

        return AIMessage(content=content, tool_calls=tool_calls)



    except Exception as e:
        logging.error(f"Request failed: {str(e)}")
        return AIMessage(content=f"[LLM Error] {str(e)}")


# === Testfunktion ===
def run_test():
    messages = [
        {"role": "user", "content": "Add 5 and 7 together using the available tools."}
    ]
    response = call_llm(messages=messages, model="GPT-4o", tools=tools)
    print("DEBUG TYPE:", type(response.tool_calls[0]))

    print("=== AI Response ===")
    print("Content:", response.content)
    if response.tool_calls:
        print("Tool Calls:")
        for call in response.tool_calls:
            print(f"Call: {call}")
            print(f"  - Tool: {call.name}, Args: {call.args}")
    else:
        print("No tool calls returned.")

    print(f"\nDebug log written to: {log_file}")


# === Script entry ===
if __name__ == "__main__":
    run_test()

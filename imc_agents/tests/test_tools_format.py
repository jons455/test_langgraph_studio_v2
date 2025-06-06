from langchain_core.messages import HumanMessage
from langchain_core.messages.tool import ToolCall  # nur für Typkompatibilität
from langchain_core.utils.function_calling import convert_to_openai_tool
from imc_agents.costum_llm_model import CustomChatModel  # dein Custom-Modell
from langchain_core.tools import tool
import json
import pprint
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("SIEMENS_API_KEY")
API_URL = os.getenv("SIEMENS_API_ENDPOINT")
os.environ["OPENAI_API_KEY"] = "dummy"

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a - b

def test_custom_model_with_tools():
    print("=== TEST: CustomChatModel with Tool ===")
    model = CustomChatModel(model="GPT-4o")

    tools = [convert_to_openai_tool(add)]

    messages = [
        HumanMessage(content="Please add 3 and 5 using a tool.")
    ]

    # Intern: baue den API-kompatiblen Payload (wie in _generate)
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    data = {
        "model": model.model_name,
        "messages": [{"role": "user", "content": messages[0].content}],
        "temperature": model.temperature,
        "tools": tools,
        "tool_choice": "auto"
    }

    print("\n===== FULL OUTGOING PAYLOAD =====")
    pprint.pprint(data)

    try:
        # Manuell Anfrage senden
        import requests
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        resp = response.json()
        print("\n===== RAW RESPONSE =====")
        pprint.pprint(resp)

        msg = resp["choices"][0]["message"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        print("\n=== ToolCall Handling ===")
        if tool_calls:
            for tc in tool_calls:
                fn = tc["function"]
                args = json.loads(fn["arguments"])
                print(f"ToolCall: {fn['name']}({args}) → {add.invoke(args)}")
        else:
            print("No tool calls. Content:", content)

    except Exception as e:
        print(f"❌ Request failed: {e}")


if __name__ == "__main__":
    test_custom_model_with_tools()

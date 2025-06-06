import os

import requests
from langgraph.graph import StateGraph
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("SIEMENS_API_ENDPOINT")
API_KEY = os.getenv("SIEMENS_API_KEY")


# State Definition
class GraphState(TypedDict):
    user_input: str
    llm_output: str

# Node: LLM-Aufruf mit API-Key
def call_custom_llm(state: GraphState) -> GraphState:
    user_input = state["user_input"]

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }

    data = {
        "model": "GPT-4o",
        "messages": [
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        answer = result['choices'][0]['message']['content']
    except Exception as e:
        answer = f"Fehler: {str(e)}"

    return {"user_input": user_input, "llm_output": answer}


# === Graph erstellen ===
builder = StateGraph(GraphState)
builder.add_node("llm_call", call_custom_llm)
builder.set_entry_point("llm_call")
builder.set_finish_point("llm_call")

graph = builder.compile()
graph.get_graph().print_ascii()


# === Aufruf mit Beispiel-Eingabe ===
result = graph.invoke({"user_input": "Was ist LangGraph?"})
print("Antwort:", result["llm_output"])
from langchain_core.messages.tool_call import ToolCall
import json

import inspect
print("📄 ToolCall kommt aus:", inspect.getfile(ToolCall))
print("🔍 MRO:", getattr(ToolCall, "__mro__", "❌ Kein echtes Class-Objekt"))


def is_toolcall_instance(obj) -> bool:
    return isinstance(obj, ToolCall)

tc = {
    "id": "call_123",
    "function": {
        "name": "add",
        "arguments": '{"a": 1, "b": 2}'
    }
}

toolk = ToolCall(
    id=tc["id"],
    name=tc["function"]["name"],
    args=json.loads(tc["function"]["arguments"])
)

print("TOOLCALL TYPE:", type(toolk))
print("✅ Is ToolCall:", is_toolcall_instance(toolk))  # → True

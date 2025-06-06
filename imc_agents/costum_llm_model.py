from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, BaseModel
from typing import List, Optional, Any, Type
import requests
import json
import asyncio
from copy import deepcopy
from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv
import os

load_dotenv()

set_llm_cache(InMemoryCache())

API_KEY = os.getenv("SIEMENS_API_KEY")
API_URL = os.getenv("SIEMENS_API_ENDPOINT")
os.environ["OPENAI_API_KEY"] = "dummy"

class CustomChatModel(BaseChatModel):
    """
    Custom Chat Model that is based on ChatOpenAI, but sends requests via a
    custom HTTP interface (e.g., the Siemens hosted GPT4o LLMs).
    Tool calling is also supported via the normal calling of the LLM.
    Tool calling is relevant for other libraries (e.g., Supervisor).

    - Uses the OpenAI-compatible tool-calling format
    - Supports AIMessage with ToolCalls and ToolCall responses
    """
    model_name: str = Field(default="gpt-4o")
    temperature: Optional[float] = Field(default=0.7)
    api_key: str = Field(default_factory=lambda: os.getenv("SIEMENS_API_KEY"))
    endpoint_url: str = Field(default_factory=lambda: os.getenv("SIEMENS_API_ENDPOINT"))
    bound_tools: List[dict] = Field(default_factory=list)

    def __init__(self, **kwargs):
        if "model_name" not in kwargs:
            kwargs["model_name"] = kwargs.get("model", "gpt-4o")

        if not API_KEY or not API_URL:
            raise ValueError("CustomChatModel requires valid API credentials.")

        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "custom-chat-model"

    def bind_tools(
        self,
        tools: List[Type[BaseModel]],
        **kwargs: Any,
    ) -> "CustomChatModel":
        """
        Binds tools to the model and returns a new model instance.
        This is used internally by with_structured_output.
        """
        new_model = deepcopy(self)
        new_model.bound_tools = [convert_to_openai_tool(t) for t in tools]
        return new_model

    @staticmethod
    def create_tool_response(tool_call: dict, result: Any) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": str(result)
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> ChatResult:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }

        data = {"model": self.model_name, "messages": []}
        for m in messages:
            msg = {"role": role_map.get(m.type, m.type)}

            if hasattr(m, "tool_call_id") and m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
                msg["content"] = m.content
            elif hasattr(m, "tool_calls") and m.tool_calls:
                msg["content"] = None
                msg["tool_calls"] = []
                for tc in m.tool_calls:
                    msg["tool_calls"].append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                        }
                    })
            else:
                msg["content"] = m.content

            data["messages"].append(msg)

        if self.temperature is not None:
            data["temperature"] = self.temperature

        # Add bound tools if they exist
        if self.bound_tools:
            data["tools"] = self.bound_tools
            # Force the model to use the first tool if only one is provided
            if len(self.bound_tools) == 1:
                 data["tool_choice"] = {"type": "function", "function": {"name": self.bound_tools[0]["function"]["name"]}}


        # Add tools from kwargs if they exist
        if "tools" in kwargs:
            data["tools"] = kwargs["tools"]
            data["tool_choice"] = "auto"

        response = requests.post(self.endpoint_url, headers=headers, json=data)
        response.raise_for_status()
        resp = response.json()
        msg = resp["choices"][0]["message"]

        tool_calls_raw = msg.get("tool_calls", [])
        tool_calls = []
        if tool_calls_raw:
            for tc in tool_calls_raw:
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                try:
                    parsed_args = json.loads(args_str)
                except json.JSONDecodeError:
                    parsed_args = {}

                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=fn["name"],
                    args=parsed_args
                ))

        ai_message = AIMessage.model_construct(
            content=msg.get("content", "") or "",
            tool_calls=tool_calls,
            tool_call_chunks=[{"index": i, "id": tc.get("id")} for i, tc in enumerate(tool_calls_raw)]
        )

        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def invoke(self, input: List[BaseMessage], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        return self._generate(input, **kwargs).generations[0].message

    async def ainvoke(self, input: List[BaseMessage], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        """
        Asynchronous variant of `invoke` for compatibility with LangGraph and async nodes.
        Uses asyncio.to_thread() to delegate the sync call.
        """
        return await asyncio.to_thread(self.invoke, input, config=config, **kwargs)

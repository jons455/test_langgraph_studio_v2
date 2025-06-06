from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.tool import ToolCall
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field
from typing import List, Optional, Any
import requests
import json
import asyncio

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("SIEMENS_API_KEY")
API_URL = os.getenv("SIEMENS_API_ENDPOINT")
os.environ["OPENAI_API_KEY"] = "dummy"

class CustomChatModel(BaseChatModel):
    """
    Custom Chat Model, das auf ChatOpenAI basiert, aber Anfragen über eine
    eigene HTTP-Schnittstelle sendet (z.B. das Siemens gehostete GPT4o LLMs).
    Über das normale Calling des LLMs wird auch Tool Calling unterstützt.
    Tool Calling ist relevant für andere Bibliotheken (Bsp. Supervisor)

    - Nutzt das OpenAI-kompatible Tool-Calling-Format
    - Unterstützt AIMessage mit ToolCalls und ToolCall-Antworten
    """
    model_name: str = Field(default="gpt-4o")
    temperature: Optional[float] = Field(default=0.7)
    api_key: str = Field(default_factory=lambda: os.getenv("SIEMENS_API_KEY"))
    endpoint_url: str = Field(default_factory=lambda: os.getenv("SIEMENS_API_ENDPOINT"))
    def __init__(self, **kwargs):
        # Füge den model_name aus "model" oder Default hinzu
        if "model_name" not in kwargs:
            kwargs["model_name"] = kwargs.get("model", "gpt-4o")

        if not API_KEY or not API_URL:
            raise ValueError("CustomChatModel requires valid API credentials.")

        super().__init__(**kwargs)  # kein self.model_name = ...

        
    @property
    def _llm_type(self) -> str:
        return "custom-chat-model"

        
    @staticmethod
    def create_tool_response(tool_call: dict, result: Any) -> dict:
        """
        Hilfsmethode zur Erzeugung einer ToolCall-Antwort im OpenAI-Format.

        Args:
            tool_call (dict): Originaler ToolCall mit 'id'.
            result (Any): Das Ergebnis des Tools.

        Returns:
            dict: Tool-Antwort im OpenAI-kompatiblen Format.
        """
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
        """
        Sendet eine Liste von Nachrichten an die LLM-API und parst die Antwort
        in ein AIMessage-Objekt mit optionalen ToolCalls.

        Args:
            messages (List[BaseMessage]): Nachrichtenverlauf (Human, AI, System, Tool).
            stop (Optional[List[str]]): Optionaler Stop-Token.
            run_manager (Optional[Any]): Laufzeit-Kontext (nicht verwendet).
            kwargs (Any): Zusätzliche Parameter, u.a. 'tools'.

        Returns:
            ChatResult: Ein einzelnes AIMessage-Ergebnis (mit optionalen ToolCalls).
        """
        # Auth-Header setzen
        headers = {
            "Content-Type": "application/json",
            "api-key": API_KEY,
        }

        # Rollen-Mapping zwischen LangChain und OpenAI-API
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }

        # Nachrichten konvertieren
        data = {"model": self.model_name, "messages": []}
        for m in messages:
            msg = {
                "role": role_map.get(m.type, m.type),
            }

            if hasattr(m, "tool_call_id") and m.tool_call_id:
                # Tool-Antwort (tool → assistant)
                msg["tool_call_id"] = m.tool_call_id
                msg["content"] = m.content

            elif hasattr(m, "tool_calls") and m.tool_calls:
                # Tool-Aufruf (assistant → tools)
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
                # Normale Nachricht
                msg["content"] = m.content

            data["messages"].append(msg)

        # Temperatur setzen (falls vorhanden)
        if self.temperature is not None:
            data["temperature"] = self.temperature

        # Tools hinzufügen (falls vorhanden)
        if "tools" in kwargs:
            data["tools"] = kwargs["tools"]
            data["tool_choice"] = "auto"

        # Anfrage senden
        print(data)
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        resp = response.json()
        msg = resp["choices"][0]["message"]

        # ToolCalls parsen (falls vorhanden)
        tool_calls_raw = msg.get("tool_calls", [])
        tool_calls = []
        for tc in tool_calls_raw:
            fn = tc.get("function", {})
            tool_calls.append(ToolCall(
                id=tc["id"],
                name=fn["name"],
                args=json.loads(fn["arguments"])
            ))

        # Finales Message-Objekt erzeugen
        ai_message = AIMessage.model_construct(
            content=msg.get("content", "") or "",
            tool_calls=tool_calls
        )

        # Als ChatResult zurückgeben
        return ChatResult(generations=[ChatGeneration(message=ai_message)])
    
    def invoke(self, input: List[BaseMessage], **kwargs: Any) -> AIMessage:
        return self._generate(input, **kwargs).generations[0].message

    async def ainvoke(self, input: List[BaseMessage], **kwargs: Any) -> AIMessage:
        """
        Asynchrone Variante von `invoke` für Kompatibilität mit LangGraph und Async-Nodes.
        Nutzt asyncio.to_thread(), um den sync Call auszulagern.
        """
        return await asyncio.to_thread(self.invoke, input, **kwargs)
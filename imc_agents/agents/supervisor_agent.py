from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from imc_agents.utils.custom_llm_model import CustomChatModel
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, model_validator
from enum import Enum
from typing import Optional
from imc_agents.agents.onboarding_agent import create_onboarding_graph
from imc_agents.agents.data_validation_agent import create_validation_graph
from imc_agents.agents.state import State
from imc_agents.utils.memory_manager import MemoryManager, extract_distributor

llm = CustomChatModel(model="GPT-4o")

class Agent(str, Enum):
    """Die spezialisierten Agenten, an die der Benutzer weitergeleitet werden kann."""
    ONBOARDING = "onboarding"
    VALIDATION = "validation"

class Route(BaseModel):
    """
    Entscheidet, ob an einen spezialisierten Agenten weitergeleitet oder eine direkte Antwort gegeben werden soll.
    """
    next: Optional[Agent] = Field(default=None, description="Der spezialisierte Agent, an den bei komplexen Anfragen weitergeleitet wird.")
    response: Optional[str] = Field(default=None, description="Eine direkte, freundliche Antwort für einfachen Small Talk oder für Anfragen, die von keinem Spezialisten bearbeitet werden können.")

    @model_validator(mode='after')
    def check_one_field_is_set(self) -> 'Route':
        """Stellt sicher, dass entweder eine Route oder eine Antwort angegeben wird, aber nicht beides."""
        if (self.next is None) == (self.response is None):
            raise ValueError("Genau eines von 'next' oder 'response' muss gesetzt sein.")
        return self

def supervisor_node(state: State) -> dict:
    """
    Der Hauptknoten für den Supervisor. Er entscheidet über die nächste Aktion
    und bereitet entweder die Weiterleitung vor oder generiert eine direkte Antwort.
    """
    messages = state["messages"]
    user_message = ""
    last_ai_message = ""

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not user_message:
            user_message = msg.content
        elif isinstance(msg, AIMessage) and not last_ai_message:
            last_ai_message = msg.content
        
        if user_message and last_ai_message:
            break
            
    # Standard-Routing-Logik für alle nachfolgenden Runden
    system_prompt_router = f"""
Sie sind der zentrale Router für ein Siemens-Agentensystem. Ihr Hauptziel ist es, die Anfrage eines Benutzers im Kontext der bisherigen Konversation zu analysieren und ihn an einen spezialisierten Agenten weiterzuleiten.

**Kontext der Konversation:**
- Letzte Nachricht des Assistenten: "{last_ai_message}"
- Aktuelle Nachricht des Benutzers: "{user_message}"

**Ihre Anweisungen:**
1.  **Analysieren Sie die Absicht des Benutzers IM KONTEXT.** Wenn die Nachricht des Benutzers eine direkte Antwort auf die Frage des Assistenten ist (z.B. "Ja" als Antwort auf "Soll ich prüfen?"), leiten Sie ihn an den entsprechenden Agenten weiter.
2.  **An einen Spezialisten weiterleiten**: Wenn die Nachricht des Benutzers eine neue technische Frage, eine Bitte um Hilfe oder Schlüsselwörter wie SFTP, API, CSV, Daten, Dateien, prüfen, korrigieren oder ändern enthält, leiten Sie ihn weiter.
    -   `onboarding`: Bei Fragen zur Einrichtung, EDI-Verbindungen, Datenübertragungsmethoden, Stammdaten oder allgemeiner technischer Beratung.
    -   `validation`: Bei Anfragen zur Überprüfung, Verifizierung, Korrektur, Änderung oder zum Einfügen von Daten in Dateien.

3.  **Direkt antworten (nur Small Talk)**: Nur wenn die Nachricht des Benutzers einfacher, kontextloser Small Talk ist (z.B. "Hallo", "Danke"), sollten Sie eine direkte, freundliche Antwort geben.

Analysieren Sie basierend auf diesen strengen Anweisungen die Nachricht des Benutzers und entscheiden Sie den nächsten Schritt.
"""

    try:
        if not user_message and not state.get("has_greeted"):
            # Wenn keine Benutzernachricht vorhanden ist und noch nicht begrüßt wurde, begrüßen Sie den Benutzer.
            distributor_name = state.get("distributor_id", "User")
            greeting = f"Hallo, {distributor_name}! Ich freue mich, Sie zu unterstützen! Was kann ich für Sie tun? Kann ich Sie beim Anbinden Ihres Systems unterstützen oder soll ich Ihre Daten prüfen?"
            return {
                "messages": [AIMessage(content=greeting)],
                "has_greeted": True,
                "next_route": "__end__"
            }

        structured_llm = llm.with_structured_output(Route)
        decision_obj = structured_llm.invoke([
            SystemMessage(content=system_prompt_router)
            # HumanMessage wird nicht mehr benötigt, da die Info im SystemPrompt ist
        ])
        
        if decision_obj.response:
            ai_response = AIMessage(content=decision_obj.response)
            return {
                "messages": [ai_response],
                "has_greeted": True, 
                "next_route": "__end__"
            }

        if decision_obj.next:
            return {
                "task_type": decision_obj.next.value,
                "next_route": decision_obj.next.value
            }

    except Exception as e:
        print(f"Fehler beim Routing, gebe eine Fallback-Antwort aus: {e}")
        fallback_message = AIMessage(content="Ich konnte Ihre Anfrage leider nicht bearbeiten. Bitte formulieren Sie sie um oder fragen Sie nach Hilfe bei der Systemanbindung oder Datenprüfung.")
        return {
            "messages": [fallback_message],
            "has_greeted": True, # Begrüßung durch Fallback
            "next_route": "__end__"
        }

    return {"next_route": "__end__"}

def supervisor_router(state: State) -> str:
    """Ein einfacher Router, der das Feld `next_route` im State überprüft."""
    next_route = state.get("next_route")
    # Die Route verbrauchen, damit sie nicht erneut verwendet wird
    state["next_route"] = None
    return next_route or "__end__"

def build_supervisor_graph():
    graph = StateGraph(State)

    graph.add_node("Supervisor Agent", supervisor_node)
    graph.add_conditional_edges(
        "Supervisor Agent",
        supervisor_router,
        {
            "onboarding": "Onboarding Agent",
            "validation": "Validation Agent",
            "__end__": END,
        }
    )

    graph.add_node("Onboarding Agent", create_onboarding_graph())
    graph.add_node("Validation Agent", create_validation_graph())

    graph.add_edge("Onboarding Agent", END)
    graph.add_edge("Validation Agent", END)
   
    graph.set_entry_point("Supervisor Agent")
    return graph.compile()

# Entry point for handling a user request, including memory management
def handle_request(user_input: str) -> str:
    distributor_name = extract_distributor(user_input)
    memory_mgr = MemoryManager(distributor_name)

    # Load memory from the database at the start
    context = memory_mgr.get_context()
    state = {"messages": [HumanMessage(content=user_input)], "context": context}
    graph = build_supervisor_graph()
    result = graph.invoke(state)

    # Save memory at the end of the conversation
    memory_mgr.update_context(str(result))
    memory_mgr.close()

    return str(result)

#erkennung ob onbiarding abgeschlossen ist
def onboarding_completed(response_text: str) -> bool:
    keywords = ["onboarding abgeschlossen", "onboarding ist fertig", "sie sind verbunden", "bereitgestellt", "erfolgreich angebunden"]
    response_text = response_text.lower()
    return any(kw in response_text for kw in keywords)
 
 
 
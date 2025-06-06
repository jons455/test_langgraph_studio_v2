from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from imc_agents.costum_llm_model import CustomChatModel
from langgraph.graph import StateGraph
from imc_agents.agents.onboarding_agent import create_onboarding_graph
from imc_agents.agents.data_validation_agent import create_validation_graph
from imc_agents.agents.state import State

llm = CustomChatModel(model="GPT-4o")

def supervisor_router(state: State) -> str:
    # Nur beim ersten Mal prüfen (Mock)
    if state.get("restored_from_db", False):
        print("🔁 Lade Konversation aus der DB... (mocked)")
        state["restored_from_db"] = True


    if not state.get("has_greeted"):
        greeting = AIMessage(content="Hallo, ich freue mich, Sie zu unterstützen! Was kann ich für Sie tun?")
        state["messages"] = [greeting] + state["messages"]
        state["has_greeted"] = True

    instruction = SystemMessage(content=("""
Du bist der zentrale Router für ein Siemens Agentensystem und leitest Anfragen von Distributoren an spezialisierte Agenten weiter.
Deine Kommunikation ist stets professionell, präzise und auf den Punkt gebracht.

**Agentenübersicht:**
- **Onboarding-Agent**: Dein Experte für alle anfänglichen Anfragen. Wähle diesen Agenten für Themen wie Ersteinrichtung, EDI-Anbindung, mögliche Übertragungswege (API, IDoc, CSV), Stammdaten und allgemeine technische Erstberatung.
    - Beispiele: „Wie kann ich meine Daten hochladen?“, „Erklären Sie mir die Anbindung via EDI.“, „Welche Schnittstellen gibt es?“

- **Validierungs-Agent**: Dein Spezialist für die Datenprüfung. Wähle diesen Agenten, wenn es um die Überprüfung von Dateien, Formaten, Pflichtfeldern oder die Analyse von Fehlern bei abgelehnten Uploads geht.
    - Beispiele: „Prüfen Sie bitte meine Datei.“, „Sind alle Pflichtfelder in meinem Dokument korrekt?“, „Warum wurde mein Upload abgelehnt?“

**Deine Aufgabe:**
Analysiere die letzte Nutzernachricht und wähle den passenden Agenten. Wenn Du unsicher bist, bevorzuge 'validation' bei konkreten Datenfragen und 'onboarding' bei allgemeinen, konzeptionellen Anfragen.

**Antwortvorgabe:** Gib **ausschließlich** eines der folgenden Schlüsselwörter zurück, ohne jegliche Zusätze:
- `onboarding`
- `validation`
- `smalltalk`
"""
                                ))
    task_type = state.get("task_type")
    if task_type in {"onboarding", "validation", "smalltalk", "fallback"}:
        # Entscheidung wurde bereits getroffen → wiederverwenden
        print(state["task_type"])
        return state["task_type"]

    # Entscheidung noch nicht getroffen → LLM-Call durchführen
    messages = [instruction] + state["messages"]
    decision = llm.invoke(messages).content.strip().lower()
    print(decision)

    if decision in {"onboarding", "validation", "smalltalk"}:
        state["task_type"] = decision
        return decision

    state["task_type"] = "fallback"
    return "fallback"


def smalltalk(state: State):
    user_input = state["messages"][-1].content if state["messages"] else ""

    messages = [
        SystemMessage(content=("""
Du bist ein professioneller und sympathischer Berater bei Siemens.
Wenn die Anfrage des Nutzers informell ist (z.B. eine Begrüßung oder eine allgemeine Frage), reagiere freundlich und professionell.

**Wichtige Regeln:**
- Antworte kurz und höflich.
- Erzähle **nur dann einen Witz**, wenn der Nutzer **explizit** danach fragt.
- Leite das Gespräch proaktiv wieder auf die Kernthemen, indem du fragst, ob der Nutzer Hilfe bei der Systemanbindung (Onboarding) oder der Datenprüfung (Validierung) benötigt.
- Vermeide Umgangssprache und bleibe stets im Rahmen einer professionellen Siemens-Kommunikation.

**Beispiele:**
- Nutzer: „Wie geht's?“ → Antworte kurz und professionell und frage dann, wie du helfen kannst (Onboarding oder Validierung).
- Nutzer: „Was machst du so?“ → Erkläre kurz, dass du bei der Anbindung und Datenprüfung für Siemens-Partner hilfst.
"""
            )),
        HumanMessage(content=user_input)
    ]

    response = llm.invoke(messages)

    return {
        "messages": state["messages"] + [response]
    }


def build_supervisor_graph():
    graph = StateGraph(State)

    graph.add_node("Supervisor Agent", lambda s: {"messages": s["messages"]})
    graph.add_conditional_edges(
    "Supervisor Agent",
    supervisor_router,  # deine Routing-Logik
    {
        "onboarding": "Onboarding Agent",
        "validation": "Validation Agent",
        #"smalltalk": "smalltalk",
        #"fallback": "fallback",
    }
)

    graph.add_node("Onboarding Agent", create_onboarding_graph())
    graph.add_node("Validation Agent", create_validation_graph())

    # Mocked Smalltalk-Handler
    #graph.add_node("smalltalk", smalltalk)

    # Fallback, falls nichts zugeordnet werden kann
   # graph.add_node("fallback", lambda s: {
    #    "messages": s["messages"] + [AIMessage(content="Ich konnte das Anliegen leider nicht zuordnen.")]
    #})

    graph.add_edge("Onboarding Agent", "__end__")
    graph.add_edge("Validation Agent", "__end__")
   
   # graph.add_edge("smalltalk", "__end__")
   # graph.add_edge("fallback", "__end__")

    graph.set_entry_point("Supervisor Agent")
    return graph.compile()


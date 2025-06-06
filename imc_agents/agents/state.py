from typing import Optional, Dict, Annotated, List, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):

    # 📩 Chatverlauf
    messages: Annotated[list[BaseMessage], add_messages]    # alle bisherigen Nachrichten (wird von LangGraph automatisch erweitert)

    # 🏷 Supervising-Level
    task_type: Optional[str]  # z. B. 'onboarding', 'validation', 'smalltalk', vom Supervisor entschieden
    has_greeted: Optional[bool]  # ob schon ein Begrüßungstext geschickt wurde

    # 📄 Dateiverwaltung
    file_path: Optional[str]  # Pfad zur aktuellen Originaldatei (vom User hochgeladen)
    improved_file_path: Optional[str]  # Pfad zur temporären verbesserten Datei

    # 🗂 Check- und Verbesserungsergebnisse
    check_results: Optional[Dict[str, Any]]  # Ergebnisse aus Datenprüfungen (z. B. Fehlerlisten)
    corrections_applied: Optional[bool]  # Flag, ob bereits Korrekturen angewendet wurden

    # 💬 Nutzerkontext
    user_message: Optional[str]  # letzte User-Nachricht (nur Text, falls explizit gebraucht)

    # 🔄 Entscheidungssteuerung
    next_action: Optional[str]  # nächste Aktion aus determine_next_step (z. B. 'CHECK_DATA', 'IMPROVE_DATA')
    last_action: Optional[str]  # letzte erfolgreich durchgeführte Aktion (z. B. 'CHECK', 'IMPROVE')

    # 💬 RAG-spezifisch
    restored_from_db: Optional[bool]  # falls aus Persistenz geladen
    context: Optional[str]  # zusammengefasster Gesprächskontext (z. B. letzte 10 Messages)
    generation: Optional[str]  # RAG-generierte Rohantwort
    documents: Optional[List[str]]  # von RAG zurückgegebene Dokumentinhalte

    restored_from_db: bool
    context: str
    generation: str
    documents: List[str]
    draft_response: str

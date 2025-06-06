from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from typing import List
from enum import Enum
import tempfile
import json
import pandas as pd
from dotenv import load_dotenv
import os

from imc_agents.agents.state import State
from imc_agents.costum_llm_model import CustomChatModel
from imc_agents.utils.data_checker import DataChecker

# Lade Umgebungsvariablen (z. B. API-Schlüssel)
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID_NORM")
CLIENT_SECRET = os.getenv("CLIENT_SECRET_NORM")
API_URL = os.getenv("API_URL_NORM")

llm = CustomChatModel(model="GPT-4o")

# Initialisiere DataChecker, um CSV-Daten mit Checks und API zu prüfen
data_checker = DataChecker(api_url=API_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET)


def check_data_node(state: State):
    """
    Liest CSV-Datei aus dem State, führt alle Datenprüfungen durch (MLFB, Distributor, Customer, Financial, General)
    und speichert eine technische Zusammenfassung sowie die strukturierten Ergebnisse ins State-Objekt.
    """

    if state.get("file_checked"):
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content="✅ Datei wurde bereits geprüft, kein erneuter Check notwendig.")]
        }

    file_path = state.get("file_path")
    if not file_path:
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content="❌ Fehler: Kein Dateipfad angegeben.")]
        }

    try:
        df = data_checker.read_csv_file(file_path)
    except RuntimeError as e:
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=f"❌ Fehler beim Einlesen der Datei: {str(e)}")]
        }

    if df.empty:
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content="❌ Fehler: Datei ist leer.")]
        }

    # Starte alle definierten Checks
    general_results = data_checker.check_general_data(df)
    mlfb_results = data_checker.check_mlfb_numbers(df)
    distributor_results = data_checker.check_distributor_data(df)
    customer_results = data_checker.check_customer_data(df)
    financial_results = data_checker.check_financial_data(df)

    # Sammle alle Ergebnisse
    all_results = {
        "general": general_results,
        "mlfb": mlfb_results,
        "distributor": distributor_results,
        "customer": customer_results,
        "financial": financial_results
    }
    
    summary = []

    if mlfb_results:
        summary.append("✅ MLFB-Check:")
        for res in mlfb_results:
            summary.append(f"  → {res}")

    if distributor_results:
        summary.append("✅ Distributor Data Check:")
        for key, values in distributor_results.items():
            if values and isinstance(values, list):
                summary.append(f"  → {key}: {', '.join(map(str, values))}")

    if customer_results:
        summary.append("✅ Customer Data Check:")
        for key, values in customer_results.items():
            if values and isinstance(values, list):
                summary.append(f"  → {key}: {', '.join(map(str, values))}")

    if financial_results:
        summary.append("✅ Financial Data Check:")
        for key, values in financial_results.items():
            if values and isinstance(values, list):
                summary.append(f"  → {key}: {', '.join(map(str, values))}")

    if general_results:
        summary.append("✅ General Data Check:")
        for key, values in general_results.items():
            if values and isinstance(values, list):
                summary.append(f"  → {key}: {', '.join(map(str, values))}")

    if not summary:
        summary.append("✅ Alle Daten sind gültig.")

    summary_text = "\n".join(summary)

    return {
        **state,
        "check_results": all_results,
        "last_action": "CHECK",
        "messages": state["messages"] + [AIMessage(content=f"Datenprüfung abgeschlossen:\n{summary_text}")]
    }


def response_generation(state: State):
    summaries = [msg.content for msg in state["messages"] if isinstance(msg, AIMessage)]
    full_summary = "\n".join(summaries)
    last_action = state.get("last_action")

    if last_action == "CHECK":
        prompt = (
            "Du bist ein datenprüfender Assistent.\n"
            "Analysiere die folgende technische Zusammenfassung und erstelle einen klaren, nutzerfreundlichen Prüfbericht.\n"
            "Der Bericht soll enthalten:\n"
            "- Problematische SPALTEN (immer in GROSSBUCHSTABEN).\n"
            "- ZEILENNUMMERN, wo Probleme auftreten.\n"
            "- Kurze Erklärung, was falsch ist.\n"
            "- Wenn alles gültig ist, das explizit erwähnen.\n"
            "- Klare Absätze pro Themenblock (MLFB, Distributor, Customer, Financial, General).\n\n"
            f"Technische Zusammenfassung:\n{full_summary}"
        )

    elif last_action == "IMPROVE":
        prompt = (
            "Du bist ein datenprüfender Assistent.\n"
            "Formuliere eine klare Bestätigung für den Nutzer, dass die gewünschten Verbesserungen an den Daten durchgeführt wurden.\n"
            "Erwähne kurz, welche Spalten und Zeilen betroffen waren (falls bekannt).\n"
            "Falls keine Details vorliegen, bestätige nur, dass die Verbesserungen abgeschlossen und eine aktualisierte Datei gespeichert wurde.\n\n"
            f"Technische Zusammenfassung:\n{full_summary}"
        )

    else:
        prompt = (
            "Du bist ein datenprüfender Assistent.\n"
            "Erstelle eine allgemeine Statuszusammenfassung basierend auf den folgenden Informationen.\n\n"
            f"Technische Zusammenfassung:\n{full_summary}"
        )

    response = llm.invoke([AIMessage(content=prompt)])

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content="Antwort generiert."), response]
    }


class Update(BaseModel):
    """Represents a single update to a row in the CSV."""
    row: int = Field(..., description="Die zu aktualisierende, 0-indizierte Zeile.")
    column: str = Field(..., description="Die zu aktualisierende Spalte.")
    new_value: str = Field(..., description="Der neue Wert, der gesetzt werden soll.")


class Updates(BaseModel):
    """Eine Liste von Aktualisierungen, die auf die CSV-Datei angewendet werden sollen."""
    updates: List[Update]


def apply_updates_node(state: State):
    file_path = state.get("improved_file_path") or state.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content="❌ Keine Datei zum Aktualisieren gefunden.")]
        }

    df = pd.read_csv(file_path)

    check_results = state.get("check_results", {})
    summary = "Hier sind die gefundenen Probleme:\n"
    for category, results in check_results.items():
        summary += f"🛠 {category.capitalize()}:\n"
        if isinstance(results, dict):
            for key, values in results.items():
                summary += f"  → {key}: {', '.join(map(str, values))}\n"
        elif isinstance(results, list):
            summary += "  → " + "\n".join(results) + "\n"

    user_message = state.get("user_message", "")

    prompt = f"""
You are an expert JSON-generating assistant. Your sole purpose is to output a valid JSON object that conforms to a provided Pydantic schema.

**DO NOT** output any text, explanation, or conversation before or after the JSON object.
**DO NOT** wrap the JSON in markdown backticks (```json ... ```).

Here is the Pydantic schema to follow:
```json
{Updates.model_json_schema()}
```

Now, based on the following data problem summary and user request, generate the JSON object containing the list of required updates.

Problem Summary:
{summary}

User Request:
{user_message}

Your JSON output:
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    try:
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON object found in the response.")
        
        json_str = content[json_start:json_end]
        updates_obj = Updates.model_validate_json(json_str)
        updates = updates_obj.updates
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=f"❌ Die Anpassung konnte leider nicht automatisch durchgeführt werden, da die LLM-Antwort ungültig war. Fehler: {e}. Bitte prüfen und manuell anpassen.")
            ]
        }

    if not updates:
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content="ℹ️ Es wurden keine Änderungen vorgeschlagen. Die Daten scheinen bereits korrekt oder erfordern manuelle Prüfung.")
            ]
        }

    for update in updates:
        row_idx = update.row
        column = update.column
        new_value = update.new_value
        if column in df.columns and 0 <= row_idx < len(df):
            df.at[row_idx, column] = new_value

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_file.name, index=False)

    return {
        **state,
        "last_action": "IMPROVE",
        "messages": state["messages"] + [
            AIMessage(content="✅ Verbesserungen angewendet und neue Datei gespeichert.")
        ],
        "improved_file_path": temp_file.name
    }


class NextAction(str, Enum):
    CHECK_DATA = "check_data"
    IMPROVE_DATA = "improve_data"
    ERROR = "error"

class RoutingDecision(BaseModel):
    """Die als nächstes auszuführende Aktion."""
    next_action: NextAction = Field(description="Die nächste auszuführende Aktion, basierend auf der Nutzereingabe.")


def determine_next_step(state: State):
    user_msg = state["messages"][-1].content

    prompt = f"""
You are an expert JSON-generating assistant. Your sole purpose is to output a valid JSON object that conforms to the provided Pydantic schema.

**DO NOT** output any text, explanation, or conversation before or after the JSON object.
**DO NOT** wrap the JSON in markdown backticks (```json ... ```).

Here is the Pydantic schema to follow:
```json
{RoutingDecision.model_json_schema()}
```

Now, based on the following user message, decide the next action.

User Message:
'{user_msg}'

Your JSON output:
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    try:
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON object found in the response.")
            
        json_str = content[json_start:json_end]
        decision_obj = RoutingDecision.model_validate_json(json_str)
        decision = decision_obj.next_action.value
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=f"❌ Konnte die nächste Aktion nicht bestimmen. Fehler: {e}")],
            "__routing__": "error"
        }

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=f"➡ Nächste Aktion: {decision.upper()}")],
        "__routing__": decision
    }


def create_validation_graph():
    sub = StateGraph(State)

    # Klar benannte Knoten
    sub.add_node("Check Data", check_data_node)
    sub.add_node("Generate Response", response_generation)
    sub.add_node("Determine Next Step", determine_next_step)
    sub.add_node("Improve Data", apply_updates_node)

    # Ablauf definieren
    sub.add_edge("Check Data", "Generate Response")
    
    sub.add_conditional_edges(
        "Determine Next Step",
        lambda s: s.get("__routing__"),
        {
            "Check Data": "Check Data",
            "Improve Data": "Improve Data"
        }
    )

    sub.add_edge("Improve Data", "Generate Response")

    # Einstiegspunkt
    sub.set_entry_point("Determine Next Step")

    return sub.compile()




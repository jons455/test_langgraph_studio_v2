from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError
from typing import List
from enum import Enum
import tempfile
import json
import pandas as pd
from dotenv import load_dotenv
import os

from imc_agents.agents.state import State
from imc_agents.utils.custom_llm_model import CustomChatModel
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
            "messages": state["messages"] + [AIMessage(content="✅ Die Datei wurde bereits geprüft. Wenn Sie eine neue Prüfung wünschen, laden Sie bitte eine neue Datei hoch.")],
            "__routing__": "end"
        }

    file_path = state.get("file_path")
    if not file_path:
        return {
            "messages": state["messages"] + [AIMessage(content="Ich habe keine Datei zum Prüfen. Bitte laden Sie zuerst eine Datei hoch.")],
            "__routing__": "end"
        }

    try:
        df = data_checker.read_csv_file(file_path)
    except RuntimeError as e:
        return {
            "messages": state["messages"] + [AIMessage(content=f"❌ Fehler beim Einlesen der Datei: {str(e)}")],
            "__routing__": "end"
        }

    if df.empty:
        return {
            "messages": state["messages"] + [AIMessage(content="❌ Fehler: Die Datei ist leer.")],
            "__routing__": "end"
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
        summary.append("✅ MLFB-Prüfung:")
        for res in mlfb_results:
            summary.append(f"  → {res}")

    if distributor_results:
        summary.append("✅ Händlerdatenprüfung:")
        for key, values in distributor_results.items():
            if values and isinstance(values, list):
                summary.append(f"  → {key}: {', '.join(map(str, values))}")

    if customer_results:
        summary.append("✅ Kundendatenprüfung:")
        for key, values in customer_results.items():
            if values and isinstance(values, list):
                summary.append(f"  → {key}: {', '.join(map(str, values))}")

    if financial_results:
        summary.append("✅ Finanzdatenprüfung:")
        for key, values in financial_results.items():
            if values and isinstance(values, list):
                summary.append(f"  → {key}: {', '.join(map(str, values))}")

    if general_results:
        summary.append("✅ Allgemeine Datenprüfung:")
        for key, values in general_results.items():
            if values and isinstance(values, list):
                summary.append(f"  → {key}: {', '.join(map(str, values))}")

    if not summary:
        summary.append("✅ Alle Daten sind gültig.")

    summary_text = "\n".join(summary)

    return {
        "check_results": all_results,
        "last_action": "CHECK",
        "technical_summary": summary_text,
        "file_checked": True,
        "__routing__": "generate_response"
    }


def response_generation_node(state: State):
    """
    Generates a natural language response based on the technical summary from the check_data_node.
    """
    technical_summary = state.get("technical_summary")
    distributor_name = state.get("distributor_id", "")
    
    if not technical_summary:
        return {
             "messages": state["messages"] + [AIMessage(content="Die Datenprüfung wurde durchgeführt, aber es liegt keine Zusammenfassung vor.")]
        }

    prompt = f"""
Du bist ein professioneller und freundlicher Daten-Analyst für Siemens.
Deine Aufgabe ist es, einen technischen Prüfbericht in eine hilfreiche, dialogorientierte Zusammenfassung für einen Benutzer zu übersetzen.

**Distributor Name:** {distributor_name}

**Dein Stil:**
- Sprich den Benutzer direkt und freundlich an.
- Formuliere die Ergebnisse in ganzen Sätzen und natürlicher Sprache.
- Sei klar und prägnant.

**Anweisungen für den Inhalt:**
1.  Beginne mit einer freundlichen Einleitung (z.B. "Ich habe Ihre Datei geprüft und einige Punkte gefunden, die wir uns ansehen sollten.").
2.  Fasse die Probleme nach Themen zusammen (z.B. "Kundendaten", "Finanzdaten").
3.  Nenne bei jedem Problem die betroffene Spalte (in `GROSSBUCHSTABEN`) und gib **ein konkretes Beispiel** für einen fehlerhaften Eintrag an, falls verfügbar. Zum Beispiel: "In der Spalte `QUANTITY` habe ich in Zeile 420 den Text 'LONDON' gefunden, wo eine Zahl erwartet wird."
4.  Wenn alles in Ordnung ist, bestätige dies in einem positiven Satz.
5.  Beende deine Antwort **immer** mit einem Angebot zur direkten Hilfe. Frage den Benutzer, ob du die Korrekturen für ihn vornehmen sollst. Etwa so: "Ich kann diese Fehler für Sie korrigieren. Sagen Sie mir einfach, welche Änderungen Sie vornehmen möchten, z.B. 'Korrigiere in Zeile 123 die Spalte XYZ auf den Wert 456'."

**Wichtige Formatierungsregeln:**
- Die gesamte Antwort muss auf **Deutsch** sein.
- Formatiere die Antwort ansprechend mit Absätzen, aber verwende **keinerlei Markdown** (keine Listen, kein Fettdruck etc.). Die Ausgabe muss reiner Text sein.

**Technischer Prüfbericht:**
---
{technical_summary}
---

Bitte erstelle jetzt die benutzerfreundliche und interaktive Zusammenfassung.
"""
    
    response_text = llm.invoke([SystemMessage(content=prompt)]).content

    return {
        "messages": state["messages"] + [AIMessage(content=response_text)]
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
            "- Eine kurze Erklärung, was falsch ist.\n"
            "- Wenn alles gültig ist, erwähne das explizit.\n"
            "- Klare Absätze für jeden Themenblock (MLFB, Distributor, Customer, Financial, General).\n\n"
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

    response_text = llm.invoke([AIMessage(content=prompt)]).content

    # Gruß bei der ersten Interaktion voranstellen
    if not state.get("has_greeted"):
        greeting = "Hallo, ich unterstütze Sie gerne! Was kann ich für Sie tun? Kann ich Ihnen bei der Anbindung Ihres Systems helfen oder soll ich Ihre Daten prüfen?"
        response_text = f"{greeting}\n\n{response_text}"

    return {
        "messages": [AIMessage(content="Antwort generiert."), AIMessage(content=response_text)],
        "has_greeted": True
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
            "messages": state["messages"] + [AIMessage(content="Ich habe keine Datei zum Anwenden von Änderungen. Bitte laden Sie zuerst eine Datei hoch.")],
            "__routing__": "end"
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
Du bist ein sorgfältiger Assistent, der eine Liste von Aktualisierungen für eine CSV-Datei basierend auf der Anfrage eines Benutzers erstellt.

Hier ist eine Zusammenfassung aller bekannten Probleme in der Datei. Diese dient nur als Kontext.
**Zusammenfassung der Probleme:**
---
{summary}
---

Hier ist die spezifische Anfrage des Benutzers.
**Benutzeranfrage:**
---
{user_message}
---

**Deine Aufgabe:**
Erstelle eine Liste von Aktualisierungen **AUSSCHLIESSLICH** basierend auf der Benutzeranfrage. Behebe keine anderen Probleme aus der Zusammenfassung, es sei denn, der Benutzer hat explizit danach gefragt. Wenn die Anfrage des Benutzers vage ist, führe keine Änderungen durch.
"""
    
    try:
        structured_llm = llm.with_structured_output(Updates)
        updates_obj = structured_llm.invoke([HumanMessage(content=prompt)])
        updates = updates_obj.updates
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        return {
            "messages": state["messages"] + [
                AIMessage(content=f"❌ Die Anpassung konnte leider nicht automatisch durchgeführt werden, da die LLM-Antwort ungültig war. Fehler: {e}. Bitte prüfen und manuell anpassen.")
            ]
        }

    if not updates:
        return {
            "messages": state["messages"] + [
                AIMessage(content="ℹ️ Es wurden keine Änderungen vorgeschlagen. Die Daten scheinen bereits korrekt zu sein oder erfordern eine manuelle Prüfung.")
            ]
        }
    print(updates)
    for update in updates:
        row_idx = update.row
        column = update.column
        new_value = update.new_value
        if column in df.columns and 0 <= row_idx < len(df):
            df.at[row_idx, column] = new_value

    original_path = state.get("file_path")
    dir_name, file_name = os.path.split(original_path)
    base_name, ext = os.path.splitext(file_name)
    new_file_path = os.path.join(dir_name, f"{base_name}_fixed.csv")
    
    # Schreibe die CSV-Datei mit den korrekten Parametern, um die Konsistenz zu gewährleisten
    df.to_csv(new_file_path, index=False, sep=";", encoding="utf-8-sig")

    return {
        "last_action": "IMPROVE",
        "messages": state["messages"] + [
            AIMessage(content=f"✅ Verbesserungen angewendet und neue Datei gespeichert unter: {new_file_path}")
        ],
        "improved_file_path": new_file_path,
        "file_checked": False,
        "__routing__": "offer_recheck"
    }


def offer_recheck_node(state: State):
    """
    Asks the user if they want to re-check the file after an update.
    """
    message = "Ich habe die Änderungen übernommen. Soll ich die neue Datei erneut überprüfen, um sicherzustellen, dass alle Fehler behoben sind?"
    return {
        "messages": state["messages"] + [AIMessage(content=message)]
    }


class NextAction(str, Enum):
    CHECK_DATA = "check_data"
    IMPROVE_DATA = "improve_data"
    ERROR = "error"

class RoutingDecision(BaseModel):
    """Die als nächstes auszuführende Aktion."""
    next_action: NextAction = Field(description="Die nächste auszuführende Aktion, basierend auf der Nutzereingabe.")


def determine_next_step(state: State):
    """
    Bestimmt basierend auf der letzten Nachricht des Nutzers, ob Daten geprüft oder korrigiert werden sollen.
    """
    last_human_message = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human_message = m.content
            break
    
    if not last_human_message:
        # Fallback, wenn keine menschliche Nachricht gefunden wird
        return {"__routing__": "error"}

    prompt = f"""
Du bist ein Router, der die Absicht eines Nutzers analysiert, um zu entscheiden, ob eine Datenprüfung oder eine Datenkorrektur erforderlich ist.
Die Absicht des Nutzers lautet: "{last_human_message}"

- Wenn der Nutzer explizit darum bittet, etwas zu "korrigieren", "ändern", "anpassen", "fixen" oder eine ähnliche Aktion durchzuführen, die eine Modifikation impliziert, wähle 'improve_data'.
- Für alles andere, wie z.B. eine Prüfung anzufordern, eine Datei hochzuladen oder eine allgemeine Frage zu stellen, wähle 'check_data'.

Basierend auf dieser Anweisung, was ist die nächste Aktion?
"""
    structured_llm = llm.with_structured_output(RoutingDecision)
    try:
        decision = structured_llm.invoke([HumanMessage(content=prompt)])
        if decision.next_action == NextAction.IMPROVE_DATA:
            return {"__routing__": "improve_data", "user_message": last_human_message}
        else:
            return {"__routing__": "check_data", "user_message": last_human_message}
    except Exception as e:
        # Bei einem Fehler eine Standardaktion oder Fehlerbehandlung durchführen
        return {"__routing__": "error", "user_message": f"Fehler bei der Entscheidungsfindung: {e}"}


def create_validation_graph():
    """
    Erstellt den Graphen für den Datenvalidierungs-Workflow.
    """
    workflow = StateGraph(State)

    # Knoten definieren mit beschreibenden Namen
    workflow.add_node("Analyze Request", determine_next_step)
    workflow.add_node("Validate Data", check_data_node)
    workflow.add_node("Generate Response", response_generation_node)
    workflow.add_node("Apply Corrections", apply_updates_node)
    workflow.add_node("Offer Recheck", offer_recheck_node)

    # Einstiegspunkt festlegen
    workflow.set_entry_point("Analyze Request")

    # Kanten basierend auf der Logik definieren
    workflow.add_conditional_edges(
        "Analyze Request",
        lambda x: x.get("__routing__"),
        {
            "check_data": "Validate Data",
            "improve_data": "Apply Corrections",
            "error": END
        }
    )

    # Validate Data always leads to Generate Response
    workflow.add_edge("Validate Data", "Generate Response")
    
    # Apply Corrections always leads to Offer Recheck
    workflow.add_edge("Apply Corrections", "Offer Recheck")
    
    workflow.add_edge("Generate Response", END)
    workflow.add_edge("Offer Recheck", END)
    
    return workflow.compile()




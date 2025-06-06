import pandas as pd
import os
import sys
from langgraph.graph import StateGraph

# Add project root to path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from imc_agents.agents.data_validation_agent import apply_updates_node
from imc_agents.agents.state import State
from langchain_core.messages import HumanMessage


def create_test_csv_with_errors():
    """Erzeugt eine temporäre CSV-Datei mit Testdaten, die Fehler enthalten."""
    data = {
        'Distributor': ['DistributorA', 'DistributorB', 'UnknownDistributor'],
        'Country': ['DE', 'US', 'DE'],
        'Net Value (EUR)': [100.50, 200.75, -50.00],  # Negativer Wert ist ein Fehler
    }
    df = pd.DataFrame(data)
    temp_file = "test_improvement_data.csv"
    df.to_csv(temp_file, index=False)
    return temp_file


def test_data_improvement_function():
    """
    Testet die `apply_updates_node` Funktion isoliert.

    Dieser Test simuliert den Zustand, nachdem der `check_data_node` gelaufen ist,
    indem er die `check_results` manuell erstellt. Dies ermöglicht einen
    gezielten Test der Verbesserungslogik, ohne den gesamten Graphen ausführen
    zu müssen, während der LLM-Call in der Funktion weiterhin die echte API nutzt.
    """
    test_file = None
    improved_file = None
    try:
        print("--- Starte Test für Daten-Verbesserungsfunktion ---")
        
        # 1. Test-Datei und State vorbereiten
        print("1. Erstelle Test-Datei und simuliere den Agenten-State...")
        test_file = create_test_csv_with_errors()

        # Manuell erstellte `check_results`, um den vorherigen Schritt zu simulieren
        check_results_mock = {
            'financial': {'Negative Werte in "Net Value (EUR)"': ['Zeile 2: -50.0']},
            'distributor': {'Unbekannte Distributoren in "Distributor"': ['UnknownDistributor in Zeile 2']}
        }
        
        user_message = "Der 'UnknownDistributor' in Zeile 2 soll zu 'DistributorC' korrigiert werden und der negative 'Net Value (EUR)' in Zeile 2 soll auf 50.0 gesetzt werden."

        # Erstelle den State, den die Funktion als Input erwartet
        initial_state = State(
            file_path=test_file,
            user_message=user_message,
            check_results=check_results_mock,
            messages=[HumanMessage(content=user_message)]
        )

        # 2. Verbesserungsfunktion direkt aufrufen
        print("2. Rufe die `apply_updates_node` Funktion direkt auf...")
        final_state = apply_updates_node(initial_state)

        # 3. Ergebnisse überprüfen
        print("3. Überprüfe die Ergebnisse...")
        improved_file_path = final_state.get("improved_file_path")

        assert "✅ Verbesserungen angewendet" in final_state["messages"][-1].content, \
            "Die Erfolgsnachricht wurde nicht im State gefunden."
        assert improved_file_path is not None, "Kein Pfad zur verbesserten Datei im State gefunden."
        assert os.path.exists(improved_file_path), "Die verbesserte Datei wurde nicht erstellt."
        
        print("--> State und erstellte Datei sehen gut aus.")

        # 4. Inhalt der korrigierten Datei prüfen
        print("4. Überprüfe den Inhalt der korrigierten CSV-Datei...")
        improved_df = pd.read_csv(improved_file_path)
        
        assert improved_df.loc[2, 'Distributor'] == 'DistributorC', "Der Distributor wurde nicht korrekt geändert."
        assert improved_df.loc[2, 'Net Value (EUR)'] == 50.0, "Der Net Value wurde nicht korrekt geändert."
        
        print("--> Datei-Inhalt wurde korrekt aktualisiert.")
        improved_file = improved_file_path

        print("\n--- ✅ Test erfolgreich abgeschlossen! ---")

    except Exception as e:
        print(f"\n--- ❌ Test fehlgeschlagen: {e} ---")
        raise  # Fehler erneut auslösen, um den Test fehlschlagen zu lassen
        
    finally:
        # Aufräumen
        print("\n--- Räume auf... ---")
        if test_file and os.path.exists(test_file):
            os.remove(test_file)
            print(f"- Temporäre Test-Datei '{test_file}' gelöscht.")
        if improved_file and os.path.exists(improved_file):
            os.remove(improved_file)
            print(f"- Temporäre verbesserte Datei '{improved_file}' gelöscht.")


if __name__ == "__main__":
    test_data_improvement_function() 
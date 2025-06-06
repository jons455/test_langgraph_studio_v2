import pandas as pd
import os
import sys
from langgraph.graph import StateGraph
from imc_agents.agents.data_validation_agent import create_validation_graph
from imc_agents.agents.state import State
from langchain_core.messages import HumanMessage

# Add project root to path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def create_test_csv_file():
    """Erzeugt eine temporäre CSV-Datei mit Testdaten, die Fehler enthalten."""
    data = {
        'Distributor': ['DistributorA', 'DistributorB', 'UnknownDistributor'],
        'Country': ['DE', 'US', 'DE'],
        'Customer Number': [1001, 1002, 1003],
        'Material Number (MLFB)': ['6ES7123-4AB56-7CD8', 'MLFB-2', '6ES7123-4AB56-7CD8'],
        'Product-Name': ['Product A', 'Product B', 'Product C'],
        'Sub-Product-Family': ['Family A', 'Family B', 'Family C'],
        'Sub-Product-Group': ['Group A', 'Group B', 'Group C'],
        'Net Value (EUR)': [100.50, 200.75, -50.00],  # Negativer Wert ist ein Fehler
        'Invoice Date': ['2023-01-15', '2023-02-20', '2023-03-25'],
        'Quantity': [10, 20, 5]
    }
    df = pd.DataFrame(data)
    temp_file = "test_invalid_data.csv"
    df.to_csv(temp_file, index=False)
    return temp_file


def run_full_data_validation_and_improvement_flow(test_csv_file):
    """
    Führt den gesamten Testprozess aus.
    """
    # 1. Graphen erstellen
    print("1. Erstelle Validierungs-Graph...")
    validation_graph = create_validation_graph()

    # 2. Initialen State für den ersten Check vorbereiten
    print("2. Bereite initialen State vor...")
    initial_state = State(
        messages=[HumanMessage(content="Bitte prüfe diese Datei.")],
        file_path=test_csv_file,
        user_message="Bitte prüfe diese Datei auf Fehler.",
        check_results={},
        last_action=None,
        improved_file_path=None
    )

    # 3. Ersten Check ausführen
    print("3. Führe initialen Daten-Check aus...")
    check_run_state = validation_graph.invoke(initial_state, config={"run_name": "Initial Check", "recursion_limit": 5})

    # Überprüfen, ob Fehler gefunden wurden
    print("--> Überprüfe, ob Fehler korrekt erkannt wurden...")
    assert "Datenprüfung abgeschlossen" in check_run_state["messages"][-2].content
    assert "Net Value (EUR)" in check_run_state["messages"][-2].content
    assert "UnknownDistributor" in check_run_state["messages"][-2].content
    print("--> ✅ Fehlererkennung erfolgreich.")

    # 4. State für den Verbesserungsprozess vorbereiten
    print("4. Bereite State für die Datenverbesserung vor...")
    improvement_user_message = "Der 'UnknownDistributor' in Zeile 2 soll 'DistributorC' heißen und der negative 'Net Value (EUR)' in Zeile 2 soll 50.00 sein."
    improvement_state = State(
        messages=check_run_state['messages'] + [HumanMessage(content=improvement_user_message)],
        file_path=test_csv_file,
        user_message=improvement_user_message,
        check_results=check_run_state.get('check_results', {}),
        last_action="CHECK"
    )

    # 5. Verbesserungsprozess starten
    print("5. Führe Verbesserungsprozess aus...")
    final_state = validation_graph.invoke(improvement_state, config={"run_name": "Improvement Run", "recursion_limit": 5})

    # 6. Überprüfen, ob die Verbesserung erfolgreich war
    print("--> Überprüfe, ob die Verbesserung erfolgreich war...")
    assert "Verbesserungen angewendet und neue Datei gespeichert." in final_state["messages"][-2].content
    improved_file_path = final_state.get("improved_file_path")
    assert improved_file_path is not None
    assert os.path.exists(improved_file_path)
    print("--> ✅ Verbesserung erfolgreich angewendet.")

    # 7. Überprüfen der korrigierten Datei
    print("7. Überprüfe die korrigierte Datei...")
    improved_df = pd.read_csv(improved_file_path)
    assert improved_df.loc[2, 'Distributor'] == 'DistributorC'
    assert improved_df.loc[2, 'Net Value (EUR)'] == 50.00
    print("--> ✅ Datei-Inhalt wurde korrekt aktualisiert.")

    return improved_file_path


if __name__ == "__main__":
    test_file = None
    improved_file = None
    try:
        print("--- Starte End-to-End Test für Data Validation Agent ---")
        test_file = create_test_csv_file()
        improved_file = run_full_data_validation_and_improvement_flow(test_file)
        print("\n--- ✅ Test erfolgreich abgeschlossen! ---")
    except Exception as e:
        print(f"\n--- ❌ Test fehlgeschlagen: {e} ---")
    finally:
        print("\n--- Räume auf... ---")
        if test_file and os.path.exists(test_file):
            os.remove(test_file)
            print(f"- Temporäre Test-Datei '{test_file}' gelöscht.")
        if improved_file and os.path.exists(improved_file):
            os.remove(improved_file)
            print(f"- Temporäre verbesserte Datei '{improved_file}' gelöscht.")
        print("--- Aufräumen abgeschlossen. ---")



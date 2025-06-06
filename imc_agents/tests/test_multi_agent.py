from langchain_core.messages import HumanMessage, AIMessage
from imc_agents.agents.supervisor_agent import build_supervisor_graph

compiled_graph = build_supervisor_graph()

def test_conversation():

    state = {
        "messages": [],
        "file_path": "assets/GB_F_headerneu.csv",  # hier setzt du den Pfad auf deine Testdatei
        "task_type": "",  # optional, falls dein Supervisor das auswertet
        "distributor_id": "DISTR123",  # optionaler Testwert
    }
    print("Starte Test-Konversation. Tippe 'exit' zum Beenden.\n")

    while True:
        user_input = input("👀 Du: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break

        state["messages"].append(HumanMessage(content=user_input))

        output = compiled_graph.invoke(state)

        # Alle neuen AI-Nachrichten ausgeben
        old_ids = {m.id for m in state["messages"] if isinstance(m, AIMessage)}
        new_ai_messages = [m for m in output["messages"] if isinstance(m, AIMessage) and m.id not in old_ids]

        for msg in new_ai_messages:
            print(f"🤖 Bot: {msg.content}\n")

        # Update state
        state = output



if __name__ == "__main__":
    test_conversation()

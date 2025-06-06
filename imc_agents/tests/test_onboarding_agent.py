from imc_agents.agents.onboarding_agent import create_onboarding_graph
from imc_agents.agents.state import State
from langchain_core.messages import HumanMessage, AIMessage

def test_onboarding_agent():
    # Erstelle den Graph
    onboarding_graph = create_onboarding_graph()

    # Starte mit einem Beispiel-Input
    user_input = input("Bitte gib deine Frage ein: ")

    # Baue initialen State
    state = State(messages=[HumanMessage(content=user_input)])

    # Starte den Graph
    result = onboarding_graph.invoke(state)

    # Hole finale Antwort
    final_messages = result["messages"]
    print("\n--- Antwort des Agenten ---")
    for msg in final_messages:
        if isinstance(msg, HumanMessage):
            print(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"Agent: {msg.content}")

if __name__ == "__main__":
    test_onboarding_agent()

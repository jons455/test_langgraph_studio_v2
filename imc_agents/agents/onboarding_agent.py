from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from imc_agents.agents.state import State
from imc_agents.costum_llm_model import CustomChatModel
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from imc_agents.costum_embeddings_model import CustomEmbeddingModel
from dotenv import load_dotenv
import os

load_dotenv()

NEO4J_URL=os.getenv("NEO4J_URL")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")


# === Setup LLM + Vectorstore ===
llm = CustomChatModel(model="GPT-4o")
embedding_function = CustomEmbeddingModel()

vectorstore = Neo4jVector.from_existing_index(
    embedding=embedding_function,
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="subsubsection_vector_index", 
    node_label="Subsubsection",
    text_node_property="text",
    embedding_node_property="embedding"
)
retriever = vectorstore.as_retriever()

# === Node-Funktionen ===

def decide_if_rag_is_needed(state: State):
    """
    Analyzes the user's question to decide if a knowledge base lookup is necessary.
    """
    last_human_message = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human_message = m.content
            break
    
    if not last_human_message:
        return {"__routing__": "end"}

    prompt = f"""
Du bist ein Router, der entscheidet, ob für eine Benutzeranfrage eine technische Suche in einer Wissensdatenbank erforderlich ist.

Die Wissensdatenbank enthält spezifische technische Details zu:
- Datenübertragungsmethoden: SFTP, API, IDoc
- Datenformate: CSV, EDIFACT
- Spezifische Fehlermeldungen oder technische Anforderungen.

Analysiere die folgende Benutzerfrage.

**Benutzerfrage:** "{last_human_message}"

Erfordert diese Frage wahrscheinlich eine Suche nach spezifischen technischen Details in der Wissensdatenbank?
Antworte nur mit 'JA' für technische Fragen oder 'NEIN' für allgemeine oder gesprächsbezogene Fragen.
"""
    decision = llm.invoke([SystemMessage(content=prompt)]).content.strip().upper()

    if "JA" in decision:
        return {"__routing__": "run_rag", "user_message": last_human_message}
    else:
        return {"__routing__": "generate_simple_response", "user_message": last_human_message}

def generate_simple_response(state: State):
    """
    Generates a direct, conversational response for non-technical questions.
    """
    system_prompt = """
Du bist ein freundlicher und professioneller Onboarding-Berater für Siemens.
Gib eine hilfreiche, gesprächsbezogene und prägnante Antwort auf die Frage des Nutzers.
Formatiere die Antwort ansprechend mit Absätzen, aber verwende **keinerlei Markdown** (keine Listen, kein Fettdruck etc.).
Die Antwort muss auf Deutsch und in reinem Text sein.
"""
    human_prompt = f"Benutzerfrage: {state['user_message']}"
    response_text = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]).content

    if not state.get("has_greeted"):
        greeting = "Hallo, gerne unterstütze ich euch bei der Anbindung an das Siemens-System."
        response_text = f"{greeting}\n\n{response_text}"

    return {"messages": [AIMessage(content=response_text)], "has_greeted": True}


def run_rag(state: State):
    docs = retriever.invoke(state['user_message'])
    return {"documents": [doc.page_content for doc in docs], "user_message": state['user_message']}

def decide_if_rag_is_sufficient(state: State):
    if not state.get("documents"):
        return {"__routing__": "ask_clarifying_questions"}

    prompt = f"""
Du bist ein Router, der Entscheidungen trifft. Basierend auf der Frage des Nutzers und den abgerufenen Dokumenten, entscheide, ob du genügend spezifische Informationen hast, um eine maßgeschneiderte Empfehlung zu geben.

**Benutzerfrage:** "{state['user_message']}"
**Abgerufene Dokumente:**
---
{state['documents']}
---
Kannst du eine spezifische, umsetzbare Empfehlung geben, die nur auf diesen Informationen basiert?
Antworte mit 'AUSREICHEND' oder 'UNZUREICHEND'.
"""
    decision = llm.invoke([SystemMessage(content=prompt)]).content.strip().upper()

    if "AUSREICHEND" in decision:
        return {"__routing__": "generate_tailored_recommendation"}
    else:
        return {"__routing__": "ask_clarifying_questions"}


def generate_tailored_recommendation(state: State):
    system_prompt = """
Du bist ein erstklassiger Onboarding-Berater für Siemens. Deine Aufgabe ist es, eine prägnante, leicht verständliche Zusammenfassung zu erstellen, um ein Gespräch mit dem Nutzer zu beginnen.

**Anweisungen:**
- Überprüfe die Frage des Nutzers und die abgerufenen Dokumente aus der Wissensdatenbank.
- Fasse die wichtigsten Optionen oder den wichtigsten Ausgangspunkt kurz zusammen. **Liste nicht alle technischen Details auf.**
- Dein Ziel ist es, eine verdauliche und prägnante Menge an Informationen zu liefern und dann einen nächsten Schritt vorzuschlagen oder eine Frage zu stellen.
- Die gesamte Antwort muss auf **Deutsch** sein.
- Formatiere die Antwort ansprechend mit Absätzen, aber verwende **keinerlei Markdown** (keine Listen, kein Fettdruck etc.). Die Ausgabe muss reiner Text sein.
"""

    human_prompt = f"Benutzerfrage: {state['user_message']}\n\nDokumente aus der Wissensdatenbank:\n{state['documents']}\n\nGib basierend auf diesen Dokumenten eine prägnante Zusammenfassung, um dem Nutzer den Einstieg zu erleichtern."
    response_text = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]).content

    if not state.get("has_greeted"):
        greeting = "Hallo, gerne unterstütze ich euch bei der Anbindung an das Siemens-System."
        response_text = f"{greeting}\n\n{response_text}"

    return {"messages": [AIMessage(content=response_text)], "has_greeted": True}


def ask_clarifying_questions(state: State):
    question = "Damit wir gemeinsam den passenden Weg finden, könnt ihr mir kurz beschreiben, wie euer IT-Setup aussieht und ob ihr bereits Erfahrung mit elektronischen Datentransfers habt?"

    if not state.get("has_greeted"):
        greeting = "Hallo, gerne unterstütze ich euch bei der Anbindung an das Siemens-System."
        full_message = f"{greeting} {question}"
    else:
        full_message = question

    # Ensure the output is a plain AIMessage
    return {
        "messages": [AIMessage(content=full_message)],
        "has_greeted": True,
    }


# === Graph-Bau ===
def create_onboarding_graph():
    sub = StateGraph(State)

    sub.add_node("Decide if RAG is Needed", decide_if_rag_is_needed)
    sub.add_node("Generate Simple Response", generate_simple_response)
    sub.add_node("Run RAG", run_rag)
    sub.add_node("Decide if RAG is Sufficient", decide_if_rag_is_sufficient)
    sub.add_node("Generate Tailored Recommendation", generate_tailored_recommendation)
    sub.add_node("Ask Clarifying Questions", ask_clarifying_questions)

    sub.add_edge(START, "Decide if RAG is Needed")
    sub.add_conditional_edges(
        "Decide if RAG is Needed",
        lambda s: s.get("__routing__"),
        {"run_rag": "Run RAG", "generate_simple_response": "Generate Simple Response"}
    )
    sub.add_edge("Generate Simple Response", END)
    
    sub.add_edge("Run RAG", "Decide if RAG is Sufficient")
    sub.add_conditional_edges(
        "Decide if RAG is Sufficient",
        lambda s: s.get("__routing__"),
        {"generate_tailored_recommendation": "Generate Tailored Recommendation", "ask_clarifying_questions": "Ask Clarifying Questions"}
    )
    
    sub.add_edge("Generate Tailored Recommendation", END)
    sub.add_edge("Ask Clarifying Questions", END)

    return sub.compile()

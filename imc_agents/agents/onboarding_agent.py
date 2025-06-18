from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from imc_agents.agents.state import State
from imc_agents.utils.custom_llm_model import CustomChatModel
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from imc_agents.utils.custom_embeddings_model import CustomEmbeddingModel
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

def run_rag(state: State):
    """
    Runs RAG for all queries, with special handling for basic interactions.
    """
    # Check for basic interactions
    last_human_message = state['user_message'].lower()
    if any(keyword in last_human_message for keyword in ["hallo", "hi", "danke", "thanks", "verfügbar", "kontakt"]):
        # For basic interactions, still use RAG but with a more focused query
        docs = retriever.invoke("general greeting and availability information")
    else:
        # For all other queries, use the actual user message
        docs = retriever.invoke(state['user_message'])
    
    return {
        "documents": [doc.page_content for doc in docs], 
        "user_message": state['user_message']
    }

def decide_if_rag_is_sufficient(state: State):
    """
    Analyzes the RAG results to decide if we have enough information to generate a tailored response
    or if we need to ask clarifying questions.
    """
    if not state.get("documents"):
        return {"__routing__": "ask_clarifying_questions"}

    prompt = f"""
Du bist ein Router, der entscheidet, ob die abgerufenen Informationen ausreichend sind für eine maßgeschneiderte Antwort.

Analysiere die folgende Benutzerfrage und die abgerufenen Dokumente und gib eine strukturierte Antwort im folgenden JSON-Format:

{{
    "sufficient_info": boolean,  // true wenn genügend Informationen für eine maßgeschneiderte Antwort vorhanden sind
    "reason": string,           // kurze Begründung der Entscheidung
    "missing_info": string[],   // Liste der fehlenden Informationen, falls vorhanden
    "confidence": number,       // Konfidenz der Entscheidung (0-1)
    "query_type": string        // Art der Anfrage: "technical_setup", "process_flow", "error_handling", "general"
}}

**Benutzerfrage:** "{state['user_message']}"
**Abgerufene Dokumente:**
---
{state['documents']}
---

Antworte NUR mit dem JSON-Objekt, keine weiteren Erklärungen.
"""
    try:
        decision = llm.invoke([SystemMessage(content=prompt)]).content.strip()
        # Parse the JSON response
        import json
        decision_data = json.loads(decision)
        
        # Log the decision for monitoring
        print(f"Decision data: {json.dumps(decision_data, indent=2)}")
        
        if decision_data["sufficient_info"]:
            return {
                "__routing__": "generate_tailored_recommendation",
                "decision_metadata": {
                    "query_type": decision_data["query_type"],
                    "confidence": decision_data["confidence"]
                }
            }
        else:
            return {
                "__routing__": "ask_clarifying_questions",
                "missing_info": decision_data["missing_info"]
            }
    except Exception as e:
        # If there's any error in parsing, default to asking clarifying questions for safety
        print(f"Error parsing decision: {e}")
        return {"__routing__": "ask_clarifying_questions"}


def generate_tailored_recommendation(state: State):
    # Get distributor name from state
    distributor_name = state.get("distributor_id", "")
    
    system_prompt = f"""
Du bist ein erstklassiger Onboarding-Berater für Siemens. Deine Aufgabe ist es, eine prägnante, leicht verständliche Zusammenfassung zu erstellen, um ein Gespräch mit dem Nutzer zu beginnen.

**Distributor Name:** {distributor_name}

**Anweisungen:**
- Überprüfe die Frage des Nutzers und die abgerufenen Dokumente aus der Wissensdatenbank.
- Fasse die wichtigsten Optionen oder den wichtigsten Ausgangspunkt kurz zusammen. **Liste nicht alle technischen Details auf.**
- Dein Ziel ist es, eine verdauliche und prägnante Menge an Informationen zu liefern und dann einen nächsten Schritt vorzuschlagen oder eine Frage zu stellen.
- Die gesamte Antwort muss auf **Deutsch** sein.
- Formatiere die Antwort ansprechend mit Absätzen, aber verwende **keinerlei Markdown** (keine Listen, kein Fettdruck etc.). Die Ausgabe muss reiner Text sein.
"""

    human_prompt = f"Benutzerfrage: {state['user_message']}\n\nDokumente aus der Wissensdatenbank:\n{state['documents']}\n\nGib basierend auf diesen Dokumenten eine prägnante Zusammenfassung, um dem Nutzer den Einstieg zu erleichtern."
    response_text = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]).content

    return {"messages": [AIMessage(content=response_text)], "has_greeted": True}


def ask_clarifying_questions(state: State):
    # Get distributor name from state
    distributor_name = state.get("distributor_id", "")
    
    prompt = f"""
Du bist ein freundlicher Onboarding-Berater für Siemens. Erstelle eine Frage nach dem IT-Setup.

**Distributor Name:** {distributor_name}
**Ist erste Nachricht:** {not state.get("has_greeted", False)}

**Anweisungen:**
- Wenn es die erste Nachricht ist, beginne mit "Hallo {distributor_name}, gerne unterstütze ich euch bei der Anbindung an das Siemens-System."
- Stelle dann die Frage nach dem IT-Setup und der Erfahrung mit elektronischen Datentransfers
- Wenn es nicht die erste Nachricht ist, stelle nur die Frage
- Die Antwort muss auf Deutsch sein
- Verwende keine Markdown-Formatierung
"""
    
    response = llm.invoke([SystemMessage(content=prompt)]).content

    return {
        "messages": [AIMessage(content=response)],
        "has_greeted": True,
    }


# === Graph-Bau ===
def create_onboarding_graph():
    sub = StateGraph(State)

    # Always start with RAG
    sub.add_node("Run RAG", run_rag)
    sub.add_node("Decide if RAG is Sufficient", decide_if_rag_is_sufficient)
    sub.add_node("Generate Tailored Recommendation", generate_tailored_recommendation)
    sub.add_node("Ask Clarifying Questions", ask_clarifying_questions)

    # Start directly with RAG
    sub.add_edge(START, "Run RAG")
    
    sub.add_edge("Run RAG", "Decide if RAG is Sufficient")
    sub.add_conditional_edges(
        "Decide if RAG is Sufficient",
        lambda s: s.get("__routing__"),
        {"generate_tailored_recommendation": "Generate Tailored Recommendation", "ask_clarifying_questions": "Ask Clarifying Questions"}
    )
    
    sub.add_edge("Generate Tailored Recommendation", END)
    sub.add_edge("Ask Clarifying Questions", END)

    return sub.compile()

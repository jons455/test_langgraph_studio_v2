from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from imc_agents.agents.state import State
from imc_agents.costum_llm_model import CustomChatModel
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains.retrieval_qa.base import RetrievalQA
from imc_agents.costum_embeddings_model import CustomEmbeddingModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain   
from langchain.schema.runnable import RunnableMap
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
    index_type="NODE",
    index_name="subsubsection_vector_index",
    node_label="Subsubsection",
    text_node_property="text",
    embedding_node_property="embedding"
)
retriever = vectorstore.as_retriever()

prompt = PromptTemplate.from_template(
    "Beantworte die Frage basierend auf dem folgenden Kontext:\n\n{context}\n\nFrage: {question}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

rag_chain = RunnableMap(
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
    }
) | llm_chain

# === Node-Funktionen ===
def extract_context(state: State, history_limit: int = 10):
    recent_msgs = state["messages"][-history_limit:]
    context = "\n".join(
        [f"Nutzer: {m.content}" if m.type == "human" else f"Agent: {m.content}" for m in recent_msgs]
    )
    return {**state, "context": context, "messages": state["messages"] + [AIMessage(content="Kontext extrahiert.")]}


def decide_need_for_rag(state: State):
    # Wenn keine Dokumente da, sofort zu RAG springen
    if not state.get("documents"):
        routing = "run_rag"
        decision = "KEINE DOKUMENTE → RAG ERZWUNGEN"
    else:
        prompt = (
            f"Hier ist der aktuelle Kontext:\n{state['context']}\n\n"
            "Brauche ich zusätzliche Dokumente aus der Wissensbasis, um sinnvoll zu antworten? "
            "Antworte nur mit 'JA' oder 'NEIN'."
        )
        decision = llm.invoke([AIMessage(content=prompt)]).content.strip().upper()
        routing = "run_rag" if decision == "JA" else "generate_structured_draft"

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=f"RAG-Entscheidung: {decision}")],
        "__routing__": routing
    }


async def run_rag(state: State):
    query = state["context"]

    # 1. Dokumente manuell vom Retriever holen (für Zugriff auf Inhalte/Quellen)
    docs = await retriever.ainvoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 2. LLM aufrufen
    result = await llm_chain.ainvoke({
        "context": context,
        "question": query,
    })

    return {
        **state,
        "generation": result["text"],
        "documents": [doc.page_content for doc in docs],
        "messages": state["messages"] + [
            AIMessage(content="Neue RAG-Dokumente geholt.")
        ],
    }

def generate_structured_draft(state: State):
    system_prompt = """
You are a technical writer. Your task is to synthesize information from a knowledge base to answer a user's query.
- Extract all relevant facts, details, and steps from the provided documents.
- Structure the information clearly with headings and lists.
- Do not add any conversational fluff, greetings, or sign-offs.
- Your output will be used by another AI to formulate a natural-sounding response.
"""

    human_prompt = f"""
Gesprächsverlauf:
{state['context']}

Wissensdatenbank:
{state['documents']}

Basierend auf dem Gesprächsverlauf und den Fakten aus der Wissensdatenbank, erstelle einen strukturierten, faktenbasierten Entwurf.
"""

    draft_response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    return {
        **state,
        "draft_response": draft_response.content,
        "messages": state["messages"] + [AIMessage(content="Technischer Entwurf erstellt.")]
    }


def rephrase_for_naturalness(state: State):
    system_prompt = """
Du bist ein erstklassiger Onboarding-Berater von Siemens. Deine Aufgabe ist es, einen technischen Entwurf in eine natürliche, hilfreiche und handlungsorientierte Konversation zu übersetzen.

**Deine Persönlichkeit & Stil:**
- **Kompetenter Berater:** Du sprichst wie ein erfahrener Kollege, nicht wie eine Maschine. Du nimmst die Informationen des Nutzers (z.B. "wir nutzen ein ERP-System, das CSV exportiert") und verbindest sie direkt mit den Lösungen aus dem technischen Entwurf.
- **Klar und Strukturiert:** Du gliederst komplexe Informationen in einfache, nummerierte Schritte. Du nutzt Markdown (insbesondere **fettgedruckten Text** und Listen), um die Lesbarkeit zu verbessern.
- **Proaktiv und Lösungsorientiert:** Dein Ziel ist es nicht nur, Fragen zu beantworten, sondern dem Nutzer einen klaren Weg aufzuzeigen. Biete am Ende immer proaktiv weitere Hilfe an.

**Deine Anweisungen:**
1.  **Formuliere um:** Wandle den folgenden `Technischen Entwurf` in eine persönliche, dialogorientierte Antwort um.
2.  **Kontext nutzen:** Berücksichtige den `Gesprächsverlauf`, um nahtlos an die letzte Nutzeranfrage anzuknüpfen.
3.  **Keine Verabschiedung:** Beende deine Nachricht nicht mit einer Grußformel, es sei denn, der Nutzer verabschiedet sich explizit.

**Beispiel für eine exzellente Antwort:**
<beispiel>
Ihr ERP-System ermöglicht den Export in CSV-Format, was perfekt für die SFTP-Option zur Datenübermittlung an Siemens passt. Hier sind einige konkrete Schritte, die Sie unternehmen können, um diese Schnittstelle aufzubauen:

1. **SFTP Verbindung einrichten:**
   - Beginnen Sie, indem Sie sich bei Ihrem Siemens Point of Sales Data Steward melden, um die spezifischen Zugangsdaten (Benutzername und private Schlüsseldatei) zu erhalten.
   ...
Ich stehe Ihnen für weitere Fragen oder für Unterstützung bei der Implementierung gerne zur Verfügung.
</beispiel>
"""

    human_prompt = f"""
Gesprächsverlauf:
{state['context']}

Technischer Entwurf:
{state['draft_response']}

Basierend auf dem Gesprächsverlauf und dem technischen Entwurf, formuliere nun eine finale, natürliche Antwort für den Nutzer im Stil des oben genannten Beispiels.
"""

    final_response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    return {
        **state,
        "messages": state["messages"] + [final_response]
    }


# === Graph-Bau ===

def create_onboarding_graph():
    sub = StateGraph(State)

    # Klar verständliche Namen statt technischer Funktionsnamen
    sub.add_node("Extract Context", extract_context)
    sub.add_node("Decide Need For RAG", decide_need_for_rag)
    sub.add_node("Run RAG", run_rag)
    sub.add_node("Generate Structured Draft", generate_structured_draft)
    sub.add_node("Rephrase For Naturalness", rephrase_for_naturalness)

    # Startknoten
    sub.add_edge(START, "Extract Context")

    # Ablaufkanten
    sub.add_edge("Extract Context", "Decide Need For RAG")

    sub.add_conditional_edges(
        "Decide Need For RAG",
        lambda s: "Run RAG" if s.get("__routing__") == "run_rag" else "Generate Structured Draft",
        {
            "Run RAG": "Run RAG",
            "Generate Structured Draft": "Generate Structured Draft"
        }
    )

    sub.add_edge("Run RAG", "Generate Structured Draft")
    sub.add_edge("Generate Structured Draft", "Rephrase For Naturalness")
    sub.add_edge("Rephrase For Naturalness", END)

    # Einstiegspunkt für den Graph
    sub.set_entry_point("Extract Context")

    return sub.compile()

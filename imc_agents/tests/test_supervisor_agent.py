from langchain_openai import ChatOpenAI
from langchain_core.messages.base import BaseMessage
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# from imc_agents.costum_llm_model import CustomChatModel
from imc_agents.tests.test_costum_model_completion import CustomChatModel

model = CustomChatModel(model="GPT-4o")

# Create specialized agents

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    output_mode="full_history",
    prompt="""
    You are a supervisor. You can delegate tasks to these tools:

    - research_expert
    - math_expert

    When you want to delegate, do so using a Markdown code block with the following format:
    ```tool_name
    {"param1": "value1", "param2": "value2"}
    Always respond with only the tool call if a delegation is needed.
    """
)

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what's the combined headcount of the FAANG companies in 2024? Once you have numeric data from the researcher, delegate the calculation of totals to the math expert using their tools."
        }
    ]
})
print("=== SUPERVISOR FULL HISTORY ===")
for i, msg in enumerate(result["messages"]):
    if isinstance(msg, BaseMessage):
        role = msg.type
        content = msg.content
        name = getattr(msg, "name", None)
    elif isinstance(msg, dict):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        name = msg.get("name", None)
    else:
        role = str(type(msg))
        content = str(msg)
        name = None

    step = f"{i:02d}. [{role}]"
    if name:
        step += f" ({name})"
    if content:
        print(f"{step}:\n{content}\n{'-' * 40}")
    elif getattr(msg, "tool_calls", None):
        print(f"{step}:\n[tool call issued]\n{'-' * 40}")
    else:
        print(f"{step}:\n[no content]\n{'-' * 40}")

from typing import Optional, Dict, Annotated, List, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Required Fields
    messages: Annotated[list[BaseMessage], add_messages]    # Chat history (automatically extended by LangGraph)

    # Optional Fields
    # 1. Core Conversation State
    task_type: Optional[str]  # 'onboarding', 'validation', 'smalltalk'
    has_greeted: Optional[bool]  # Whether greeting has been sent
    next_route: Optional[str]  # For supervisor routing
    user_message: Optional[str]  # Latest user message text

    # 2. File Management
    file_path: Optional[str]  # Path to current original file
    improved_file_path: Optional[str]  # Path to temporary improved file
    file_checked: Optional[bool]  # Whether file has been checked
    corrections_applied: Optional[bool]  # Whether corrections were applied

    # 3. Validation Results
    check_results: Optional[Dict[str, Any]]  # Results from data checks
    technical_summary: Optional[str]  # Technical summary from data check

    # 4. Action Control
    next_action: Optional[str]  # Next action from determine_next_step
    last_action: Optional[str]  # Last successfully executed action
    distributor_id: Optional[str]


    # 5. RAG Context
    context: Optional[str]  # Summarized conversation context
    documents: Optional[List[str]]  # Documents returned by RAG
    generation: Optional[str]  # Raw RAG-generated response
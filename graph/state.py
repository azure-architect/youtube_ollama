from dataclasses import dataclass, field
from typing import Optional, List
from pydantic_ai.messages import ModelMessage

@dataclass
class WorkflowState:
    """State for the research workflow.
    
    This state is passed between nodes in the workflow graph and 
    maintains context throughout the entire research process.
    """
    # Input query that initiates the research
    query: str
    
    # The Ollama model to use for all agents
    model_name: str = "llama3-groq-tool-use:latest"
    
    # Results from the research and analysis phases
    research_data: Optional[dict] = None
    analysis_data: Optional[dict] = None
    
    # Message history for each agent to maintain context across runs
    research_agent_messages: List[ModelMessage] = field(default_factory=list)
    analysis_agent_messages: List[ModelMessage] = field(default_factory=list)
from dataclasses import dataclass
from typing import TYPE_CHECKING
from pydantic_graph import BaseNode, GraphRunContext
from src.graph.state import WorkflowState
from src.agents.research_agent import ResearchAgent

if TYPE_CHECKING:
    from src.graph.nodes.analyze_findings import AnalyzeFindings

@dataclass
class PerformResearch(BaseNode[WorkflowState]):
    """Node to perform research on a topic."""
    query: str
    
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> "AnalyzeFindings":
        # Import here to avoid circular imports
        from src.graph.nodes.analyze_findings import AnalyzeFindings
        
        print(f"Researching: {self.query}")
        
        # Initialize the research agent with the model from state
        research_agent = ResearchAgent(model_name=ctx.state.model_name)
        
        # Run the research agent
        result = await research_agent.run(
            self.query,
            message_history=ctx.state.research_agent_messages
        )
        
        # Update the state with research results
        ctx.state.research_data = result.data.dict()
        ctx.state.research_agent_messages = result.all_messages()
        
        print(f"Research completed. Found {len(result.data.key_points)} key points.")
        return AnalyzeFindings()
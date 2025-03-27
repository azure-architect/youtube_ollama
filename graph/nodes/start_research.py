from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext
from src.graph.state import WorkflowState
# Use forward reference for PerformResearch to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.graph.nodes.perform_research import PerformResearch

@dataclass
class StartResearch(BaseNode[WorkflowState]):
    """Start the research process with a query."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> "PerformResearch":
        # Import at runtime to avoid circular imports
        from src.graph.nodes.perform_research import PerformResearch
        
        print(f"Starting research on: {ctx.state.query}")
        return PerformResearch(ctx.state.query)
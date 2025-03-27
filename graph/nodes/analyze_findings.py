from dataclasses import dataclass
from typing import TYPE_CHECKING
from pydantic_graph import BaseNode, End, GraphRunContext
from src.graph.state import WorkflowState
from src.agents.analysis_agent import AnalysisAgent

@dataclass
class AnalyzeFindings(BaseNode[WorkflowState, None, dict]):
    """Node to analyze research findings."""
    
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> End[dict]:
        print("Analyzing research findings...")
        
        # Check if we have research data
        if not ctx.state.research_data:
            print("Error: No research data available")
            return End({"error": "No research data available"})
        
        # Initialize the analysis agent with the model from state
        analysis_agent = AnalysisAgent(model_name=ctx.state.model_name)
        
        # Run the analysis agent
        result = await analysis_agent.run(
            f"Analyze these research findings: {ctx.state.research_data}",
            message_history=ctx.state.analysis_agent_messages
        )
        
        # Update the state with analysis results
        ctx.state.analysis_data = result.data.dict()
        ctx.state.analysis_agent_messages = result.all_messages()
        
        print(f"Analysis completed with confidence score: {result.data.confidence_score}")
        
        # Return the complete workflow results
        final_result = {
            "query": ctx.state.query,
            "research": ctx.state.research_data,
            "analysis": ctx.state.analysis_data
        }
        
        return End(final_result)
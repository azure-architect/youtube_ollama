from pydantic_graph import Graph
from src.graph.nodes import StartResearch, PerformResearch, AnalyzeFindings
from src.graph.state import WorkflowState

# Define the workflow graph
research_workflow = Graph(
    nodes=[StartResearch, PerformResearch, AnalyzeFindings],
    state_type=WorkflowState
)

def create_workflow():
    """Factory function to create a workflow graph."""
    return research_workflow

def generate_workflow_diagram(filename="workflow_diagram.png"):
    """Generate a diagram of the workflow."""
    from src.graph.nodes import StartResearch
    research_workflow.mermaid_save(filename, start_node=StartResearch)
    print(f"Workflow diagram saved to {filename}")
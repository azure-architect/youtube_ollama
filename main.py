import asyncio
import os
import json
from dotenv import load_dotenv
from src.graph.workflow import create_workflow, generate_workflow_diagram
from src.graph.state import WorkflowState
from src.graph.nodes import StartResearch
from src.utils.helpers import ensure_ollama_model

# Load environment variables
load_dotenv()

# Define the model to use
OLLAMA_MODEL = "llama3-groq-tool-use:latest"  # Change to any model you have or want to pull

async def main():
    # Ensure Ollama is running and the model is available
    if not ensure_ollama_model(OLLAMA_MODEL):
        print(f"Error: Could not ensure Ollama model {OLLAMA_MODEL} is available")
        return
    
    # Create the workflow graph
    workflow = create_workflow()
    
    # Generate a diagram of the workflow
    generate_workflow_diagram()
    
    # Get user query or use default
    query = input("Enter a research topic (or press Enter for default): ")
    if not query:
        query = "What are the latest developments in quantum computing?"
    
    # Initialize the workflow state
    state = WorkflowState(query=query)
    
    # Run the workflow
    print(f"\nStarting research workflow for query: {query}")
    print("=" * 50)
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    
    result = await workflow.run(StartResearch(), state=state)
    
    # Print the results
    print("\n" + "=" * 50)
    print("RESEARCH AND ANALYSIS RESULTS")
    print("=" * 50)
    
    print(f"\nQUERY: {result.output['query']}")
    
    # Research section
    print("\nRESEARCH SUMMARY:")
    print(f"Topic: {result.output['research']['topic']}")
    print("\nKey Points:")
    for i, point in enumerate(result.output['research']['key_points'], 1):
        print(f"{i}. {point}")
    print(f"\nConclusion: {result.output['research']['conclusion']}")
    
    # Analysis section
    print("\nANALYSIS:")
    print("\nKey Insights:")
    for i, insight in enumerate(result.output['analysis']['key_insights'], 1):
        print(f"{i}. {insight}")
    print("\nRecommendations:")
    for i, rec in enumerate(result.output['analysis']['recommendations'], 1):
        print(f"{i}. {rec}")
    print(f"\nConfidence Score: {result.output['analysis']['confidence_score']}")
    
    # Save results to file
    with open("research_results.json", "w") as f:
        json.dump(result.output, f, indent=2)
    print("\nResults saved to research_results.json")

if __name__ == "__main__":
    asyncio.run(main())
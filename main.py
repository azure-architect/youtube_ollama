# main.py
import asyncio
import argparse
import logging
import os
import sys
from dotenv import load_dotenv

from graph.workflow_manager import WorkflowManager
from api_services.transcript_service import get_video_id_from_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Default model configuration
DEFAULT_MODEL = "llama3.1:8b-instruct-q8_0"

async def process_video_with_workflow(video_url: str, model_name: str = None, save_output: bool = False):
    """Process a video using the node-based workflow system."""
    manager = WorkflowManager()
    final_state = await manager.run_workflow(video_url, model_name, save_output)
    
    if final_state.error:
        logger.error(f"Workflow completed with errors: {final_state.error} in {final_state.error_node}")
        return None
    else:
        logger.info(f"Workflow completed successfully")
        return final_state

async def process_batch_with_workflow(urls_file: str, model_name: str = None, save_output: bool = False):
    """Process a batch of videos using the node-based workflow system."""
    try:
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        logger.info(f"Processing {len(urls)} videos from {urls_file}")
        
        results = []
        for i, url in enumerate(urls):
            logger.info(f"Processing video {i+1}/{len(urls)}: {url}")
            result = await process_video_with_workflow(url, model_name, save_output)
            if result:
                results.append(result)
            
        return results
            
    except Exception as e:
        logger.error(f"Error processing batch file: {e}")
        return []

def main():
    # Load environment variables
    load_dotenv()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='YouTube Analysis Workflow')
    
    # Add arguments
    parser.add_argument('input', help='YouTube video URL or path to file with URLs')
    parser.add_argument('-b', '--batch', action='store_true', help='Process input as a batch file')
    parser.add_argument('-m', '--model', help=f'Override default model (default: {DEFAULT_MODEL})')
    parser.add_argument('-s', '--save', action='store_true', help="Save output to file")
    
    args = parser.parse_args()
    
    # Check if API key is available
    if not os.getenv('YT_DATA_API_KEY'):
        logger.error("YouTube API key not found. Please set YT_DATA_API_KEY environment variable")
        sys.exit(1)
    
    # Run the appropriate process
    if args.batch:
        asyncio.run(process_batch_with_workflow(args.input, args.model, args.save))
    else:
        # Assume it's a URL
        asyncio.run(process_video_with_workflow(args.input, args.model, args.save))

if __name__ == "__main__":
    main()
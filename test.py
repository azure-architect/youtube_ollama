#!/usr/bin/env python
import asyncio
import json
import logging
import argparse
import os
from dotenv import load_dotenv
from agents.youtube_transcript_agent import YouTubeTranscriptAgent  # Changed from src.agents
from api_services.transcript_service import get_video_id_from_url    # Changed from src.api_services
# from api_services.youtube_data_api import get_youtube_video_data     # Changed from src.api_services

# Load environment variables
load_dotenv()
YT_DATA_API_KEY = os.getenv('YT_DATA_API_KEY')
if not YT_DATA_API_KEY:
    print("Warning: YT_DATA_API_KEY not found in .env file")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

async def test_agent(video_url: str, model_name: str = "llama3.1:8b-instruct-q8_0", save_output: bool = True):
    """Test the YouTube transcript agent with a given video URL."""
    try:
        # Extract video ID from URL
        video_id = get_video_id_from_url(video_url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {video_url}")
            return
        
        logger.info(f"Processing video ID: {video_id} with model: {model_name}")
        
        # Create the agent with the YouTube API key
        agent = YouTubeTranscriptAgent(youtube_api_key=YT_DATA_API_KEY, model_name=model_name)
        
        # Run the agent
        logger.info("Running YouTube transcript agent...")
        start_time = asyncio.get_event_loop().time()
        video_data = await agent.run(video_id)
        end_time = asyncio.get_event_loop().time()
        
        # Log processing time
        processing_time = end_time - start_time
        logger.info(f"Agent processing completed in {processing_time:.2f} seconds")
        
        # Display summary info
        logger.info(f"Video Title: {video_data.title}")
        logger.info(f"Channel: {video_data.channel}")
        logger.info(f"Transcript segments: {len(video_data.transcript)}")
        
        if save_output:
            # Save the result to a file
            output_file = f"output/{video_id}_transcript_data.json"
            os.makedirs("output", exist_ok=True)
            
            with open(output_file, "w") as f:
                json.dump(video_data.model_dump(), f, indent=2)  # Changed from dict() to model_dump()
            
            logger.info(f"Output saved to {output_file}")
            
        return video_data
            
    except Exception as e:
        logger.error(f"Error testing agent: {e}")
        raise

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Test YouTube Transcript Agent')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--model', '-m', default="llama3.1:8b-instruct-q8_0", help='Ollama model name')
    parser.add_argument('--no-save', '-n', action='store_true', help="Don't save output to file")
    args = parser.parse_args()
    
    # Make sure the output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Run the agent test
    asyncio.run(test_agent(args.url, args.model, not args.no_save))
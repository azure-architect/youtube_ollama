#!/usr/bin/env python
import asyncio
import argparse
import logging
import os
import json
import sys
from typing import List, Optional
from dotenv import load_dotenv

# Import our agent and utilities
from agents.youtube_transcript_agent import YouTubeTranscriptAgent
from api_services.transcript_service import get_video_id_from_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Default model configuration - can be changed here
DEFAULT_MODEL = "llama3.1:8b-instruct-q8_0"

async def process_video(video_url: str, model_name: str = None, save_output: bool = False) -> None:
    """
    Process a single YouTube video URL
    
    Args:
        video_url: URL of the YouTube video
        model_name: Name of the Ollama model to use (if None, uses agent default)
        save_output: Whether to save output to a file
    """
    # Extract video ID from URL
    video_id = get_video_id_from_url(video_url)
    if not video_id:
        logger.error(f"Could not extract video ID from URL: {video_url}")
        return
    
    # Use the default model if none specified
    model_to_use = model_name or DEFAULT_MODEL
    logger.info(f"Processing video ID: {video_id} with model: {model_to_use}")
    
    # Load API key
    api_key = os.getenv('YT_DATA_API_KEY')
    if not api_key:
        logger.error("YouTube API key not found in environment variables")
        return
    
    # Create the agent with the YouTube API key
    agent = YouTubeTranscriptAgent(youtube_api_key=api_key, model_name=model_to_use)
    
    try:
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
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(video_data.model_dump(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Output saved to {output_file}")
            
        return video_data
    
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        return None
        
async def process_batch(urls_file: str, model_name: str = None, save_output: bool = False) -> None:
    """
    Process a batch of YouTube video URLs from a file
    
    Args:
        urls_file: Path to file containing URLs (one per line)
        model_name: Name of the Ollama model to use (if None, uses agent default)
        save_output: Whether to save output to files
    """
    try:
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        logger.info(f"Processing {len(urls)} videos from {urls_file}")
        
        results = []
        for i, url in enumerate(urls):
            logger.info(f"Processing video {i+1}/{len(urls)}: {url}")
            result = await process_video(url, model_name, save_output)
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
    parser = argparse.ArgumentParser(description='YouTube Transcript Analysis Tool')
    
    # Add arguments - first argument is either a URL or a file path
    parser.add_argument('input', help='YouTube video URL or path to file with URLs (one per line)')
    parser.add_argument('-b', '--batch', action='store_true', help='Process input as a batch file')
    parser.add_argument('-m', '--model', help=f'Override default Ollama model (default: {DEFAULT_MODEL})')
    parser.add_argument('-s', '--save', action='store_true', help="Save output to file")
    
    args = parser.parse_args()
    
    # Check if API key is available
    if not os.getenv('YT_DATA_API_KEY'):
        logger.error("YouTube API key not found. Please set YT_DATA_API_KEY environment variable")
        sys.exit(1)
    
    # Run the appropriate process
    if args.batch:
        asyncio.run(process_batch(args.input, args.model, args.save))
    else:
        # Assume it's a URL
        asyncio.run(process_video(args.input, args.model, args.save))

if __name__ == "__main__":
    main()
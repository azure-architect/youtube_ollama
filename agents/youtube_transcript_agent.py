# src/agents/youtube_transcript_agent.py
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import isodate

# Relative imports for project structure
from ..agents.base_agent import BaseAgent
from ..models.data_models import YouTubeVideoData, TranscriptSegment
from ..api_services.transcript_service import get_youtube_tranget_video_transcript_data, get_video_id_from_url
from ..api_services.youtube_data_api import get_youtube_video_data

class YouTubeTranscriptAgent(BaseAgent[YouTubeVideoData]):
    """Agent for extracting and processing YouTube transcript data."""
    
    def __init__(self, 
                 youtube_api_key: str,
                 model_name: str = "llama3.1:8b-instruct-q8_0",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.7):
        """
        Initialize the YouTube transcript agent.
        
        Args:
            youtube_api_key: API key for the YouTube Data API
            model_name: Name of the Ollama model to use
            base_url: Base URL for the Ollama API
            temperature: Temperature setting for generation
        """
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            num_ctx=4096,
            result_model=YouTubeVideoData
        )
        self.youtube_api_key = youtube_api_key
        self.input_data = None  # Store input data for fallback parsing
    
    async def _prepare_prompt(self, video_id: str) -> str:
        """
        Prepare a prompt for processing the YouTube video transcript.
        
        Args:
            video_id: YouTube video ID
        
        Returns:
            Formatted prompt string
        """
        # This method isn't strictly necessary for this agent as we're not using 
        # the model for transcript extraction, but we implement it for completeness
        return f"""
        Extract and structure information from YouTube video with ID: {video_id}.
        Please format the response as a structured JSON object.
        """
    
    async def _process_transcript(self, transcript_data: List[Dict[str, Any]]) -> List[TranscriptSegment]:
        """
        Process raw transcript data into structured TranscriptSegment objects.
        
        Args:
            transcript_data: Raw transcript data from YouTube API
        
        Returns:
            List of TranscriptSegment objects
        """
        segments = []
        for item in transcript_data:
            segment = TranscriptSegment(
                text=item.get('text', ''),
                start=item.get('start', 0.0),
                duration=item.get('duration')
            )
            segments.append(segment)
        return segments
    
    async def run(self, video_id: str, message_history: Optional[List[Dict[str, str]]] = None) -> YouTubeVideoData:
        """
        Run the agent to extract and process YouTube video transcript.
        
        Args:
            video_id: YouTube video ID
            message_history: Optional message history (not used in this implementation)
        
        Returns:
            YouTubeVideoData object with video information and transcript
        """
        try:
            # Get video metadata from YouTube Data API
            video_data = await get_youtube_video_data(video_id, self.youtube_api_key)
            
            # Get transcript data
            transcript_data = await get_youtube_transcript(video_id)
            
            # Process transcript data
            transcript_segments = await self._process_transcript(transcript_data)
            
            # Create YouTubeVideoData object
            result = YouTubeVideoData(
                video_id=video_id,
                title=video_data.get('title', 'Error retrieving data'),
                description=video_data.get('description', 'An error occurred while retrieving video data'),
                channel=video_data.get('channel', 'Unknown'),
                channel_id=video_data.get('channel_id', 'Unknown'),
                published_at=video_data.get('published_at', 'Unknown'),
                transcript=transcript_segments,
                thumbnail_url=video_data.get('thumbnail_url'),
                view_count=video_data.get('view_count', 0),
                like_count=video_data.get('like_count'),
                comment_count=video_data.get('comment_count'),
                tags=video_data.get('tags', []),
                duration=self._parse_duration(video_data.get('duration', 'PT0S'))
            )
            
            # Store the result for fallback parsing if needed
            self.input_data = result
            
            return result
            
        except Exception as e:
            print(f"Error processing video data: {e}")
            
            # Create a minimal valid result
            result = YouTubeVideoData(
                video_id=video_id,
                title="Error retrieving data",
                description="An error occurred while retrieving video data",
                channel="Unknown",
                channel_id="Unknown",
                published_at="Unknown",
                transcript=[],
                duration=0
            )
            
            # Store the result for fallback parsing
            self.input_data = result
            
            return result
            
    def _parse_duration(self, duration_str: str) -> int:
        """
        Parse ISO 8601 duration string to seconds.
        
        Args:
            duration_str: ISO 8601 duration string (e.g., 'PT21M53S')
        
        Returns:
            Duration in seconds as an integer
        """
        try:
            # Use isodate to parse the ISO 8601 duration
            duration = isodate.parse_duration(duration_str)
            # Convert to seconds
            return int(duration.total_seconds())
        except (ValueError, AttributeError):
            # Handle parsing errors
            return 0
            
    async def fallback_parsing(self, raw_text: str) -> YouTubeVideoData:
        """
        Fallback parsing method if structured output fails.
        
        Args:
            raw_text: Raw text output from the model
        
        Returns:
            YouTubeVideoData object
        """
        # Since we're not using the model for transcript extraction,
        # we can just return the input data we stored
        if self.input_data:
            return self.input_data
            
        # If we don't have input data, create a minimal valid result
        return YouTubeVideoData(
            video_id="unknown",
            title="Fallback parsing result",
            description="Generated by fallback parsing",
            channel="Unknown",
            channel_id="Unknown",
            published_at="Unknown",
            transcript=[],
            duration=0
        )
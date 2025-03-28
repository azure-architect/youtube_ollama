# agents/youtube_transcript_agent.py
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import isodate

# Use absolute imports from the project root
from agents.base_agent import BaseAgent
from models.data_models import YouTubeVideoData, TranscriptSegment
from api_services.transcript_service import get_video_transcript_data, get_video_id_from_url
from api_services.youtube_data_api import get_youtube_video_data

class YouTubeTranscriptAgent(BaseAgent[YouTubeVideoData]):
    """Agent for extracting and processing YouTube transcript data."""
    
    def __init__(self, 
                 youtube_api_key: str,
                 model_name: str = "mistral:latest",
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
            video_data = get_youtube_video_data(video_id)
            
            # Get transcript data using the transcript service
            transcript_url = f"https://www.youtube.com/watch?v={video_id}"
            transcript_result = get_video_transcript_data(transcript_url)
            
            transcript_segments = []
            if transcript_result and 'transcript' in transcript_result:
                # Process transcript data
                transcript_segments = await self._process_transcript(transcript_result['transcript'])
            
            # Extract video information from the API response
            if video_data and isinstance(video_data, dict):
                # Store all raw data from the API
                raw_snippet = video_data.get('snippet', {})
                raw_statistics = video_data.get('statistics', {})
                raw_status = video_data.get('status', {})
                raw_topic_details = video_data.get('topicDetails', {})
                raw_content_details = video_data.get('contentDetails', {})
                
                # Create YouTubeVideoData object with all available data
                result = YouTubeVideoData(
                    video_id=video_id,
                    title=raw_snippet.get('title', 'No title'),
                    description=raw_snippet.get('description', 'No description'),
                    channel=raw_snippet.get('channelTitle', 'Unknown'),
                    channel_id=raw_snippet.get('channelId', 'Unknown'),
                    published_at=raw_snippet.get('publishedAt', ''),
                    transcript=transcript_segments,
                    thumbnail_url=raw_snippet.get('thumbnails', {}).get('high', {}).get('url'),
                    view_count=int(raw_statistics.get('viewCount', 0)) if raw_statistics.get('viewCount') else None,
                    like_count=int(raw_statistics.get('likeCount', 0)) if raw_statistics.get('likeCount') else None,
                    comment_count=int(raw_statistics.get('commentCount', 0)) if raw_statistics.get('commentCount') else None,
                    tags=raw_snippet.get('tags', []),
                    duration=self._parse_duration(raw_content_details.get('duration', 'PT0S')),
                    # Store all raw data
                    raw_data={
                        'snippet': raw_snippet,
                        'statistics': raw_statistics,
                        'status': raw_status,
                        'topicDetails': raw_topic_details,
                        'contentDetails': raw_content_details
                    }
                )
            else:
                # Create a minimal valid result if no data is available
                result = YouTubeVideoData(
                    video_id=video_id,
                    title="Error retrieving data",
                    description="An error occurred while retrieving video data",
                    channel="Unknown",
                    channel_id="Unknown",
                    published_at="Unknown",
                    transcript=transcript_segments,
                    duration=0,
                    raw_data={}
                )
            
            # Store the result for fallback parsing if needed
            self.input_data = result
            
            return result
            
        except Exception as e:
            import traceback
            print(f"Error processing video data: {e}")
            print(traceback.format_exc())  # Print full traceback for debugging
            
            # Create a minimal valid result
            result = YouTubeVideoData(
                video_id=video_id,
                title="Error retrieving data",
                description="An error occurred while retrieving video data",
                channel="Unknown",
                channel_id="Unknown",
                published_at="Unknown",
                transcript=[],
                duration=0,
                raw_data={}
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
            duration=0,
            raw_data={}
        )
# agents/transcript_insights_agent.py
import logging
import httpx
import json
import re
from typing import Dict, Any

from models.data_models import YouTubeVideoData, VideoAnalysisData, SoftwareMention

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TranscriptInsightAgent:
    """MVP Agent for extracting insights from YouTube video transcripts."""
    
    def __init__(self, 
                 model_name="mistral:latest",
                 base_url="http://localhost:11434",
                 temperature=0.1):
        """Initialize the Transcript Insights agent with direct parameters."""
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = 4096
    
    async def _prepare_prompt(self, video_data: YouTubeVideoData) -> str:
        """Prepare a prompt for extracting insights from transcript data."""
        # Combine transcript segments into a single text for analysis
        transcript_text = " ".join([segment.text for segment in video_data.transcript])
        
        return f"""
        Analyze this YouTube video transcript and extract key insights.
        TITLE: {video_data.title}
        CHANNEL: {video_data.channel}
        TRANSCRIPT: {transcript_text[:3000]}...

        Provide a JSON with: summary (3-5 sentences), software_mentions (list of tools mentioned),
        main_topics (5 topics), and key_points (7 points).
        """
    
    async def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Direct API call to Ollama without inheritance."""
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx
                }
            }
            
            logger.info(f"Making API request to: {url}")
            logger.info(f"Using model: {self.model_name}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=60.0)
                
                # Log response code
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"API error: {response.text}")
                    return {"response": f"Error {response.status_code}: {response.text}"}
                
                return response.json()
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return {"response": f"Error: {str(e)}"}
    
    async def run(self, video_data: YouTubeVideoData) -> VideoAnalysisData:
        """Run the agent to extract insights from video transcript."""
        try:
            if not video_data.transcript:
                logger.warning(f"No transcript available for video: {video_data.video_id}")
                return VideoAnalysisData(original_data=video_data)
            
            # Prepare prompt 
            prompt = await self._prepare_prompt(video_data)
            logger.info(f"Calling Ollama API for video: {video_data.video_id}")
            
            # Call API directly with explicit parameters
            result = await self._call_ollama(prompt)
            
            # Extract response text
            response_text = result.get("response", "")
            
            # Parse JSON from response
            try:
                insights_data = self._parse_json_response(response_text)
                
                return VideoAnalysisData(
                    original_data=video_data,
                    summary=insights_data.get("summary", ""),
                    main_topics=insights_data.get("main_topics", []),
                    key_points=insights_data.get("key_points", []),
                    software_mentions=[
                        SoftwareMention(
                            name=item.get("name", ""),
                            description=item.get("description", "")
                        )
                        for item in insights_data.get("software_mentions", [])
                    ]
                )
            except Exception as e:
                logger.error(f"Error parsing response: {e}")
                return VideoAnalysisData(original_data=video_data)
            
        except Exception as e:
            logger.error(f"Error in transcript analysis: {e}")
            return VideoAnalysisData(original_data=video_data)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from model response."""
        try:
            # Try direct JSON parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            # Fall back to regex pattern matching
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response) or re.search(r'({[\s\S]*})', response)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # Return empty structure if parsing fails
                return {
                    "summary": "Failed to parse response",
                    "software_mentions": [],
                    "main_topics": [],
                    "key_points": []
                }
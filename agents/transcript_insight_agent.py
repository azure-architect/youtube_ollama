# agents/transcript_insights_agent.py
import logging
import httpx
import json
import os
import re
from typing import Dict, Any, List, Optional

from agents.base_agent import BaseAgent
from models.data_models import YouTubeVideoData, VideoAnalysisData, SoftwareMention

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_INSIGHTS_MODEL = "mistral:latest"

class TranscriptInsightsAgent(BaseAgent[VideoAnalysisData]):
    """Agent for extracting insights from YouTube video transcripts."""
    
    def __init__(self, 
                 model_name: str = DEFAULT_INSIGHTS_MODEL,
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.1):
        """Initialize the Transcript Insights agent."""
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            num_ctx=4096,
            result_model=VideoAnalysisData
        )
        self.input_data = None
    
    async def _prepare_prompt(self, video_data: YouTubeVideoData) -> str:
        """Prepare a prompt for extracting insights from transcript data."""
        # Combine transcript segments into a single text for analysis
        transcript_text = " ".join([segment.text for segment in video_data.transcript])
        
        prompt = f"""
        You are an expert analyzer of video content. Extract key insights from this YouTube video transcript.

        VIDEO TITLE: {video_data.title}
        CHANNEL: {video_data.channel}
        DESCRIPTION: {video_data.description}

        TRANSCRIPT:
        {transcript_text}

        Based on this transcript, provide the following in JSON format:
        1. A concise summary (3-5 sentences)
        2. Software tools/libraries mentioned
        3. 5 main topics discussed
        4. 7 key points from the content

        Format your response as valid JSON only:
        ```json
        {{
          "summary": "...",
          "software_mentions": [
            {{
              "name": "...",
              "description": "..."
            }}
          ],
          "main_topics": ["..."],
          "key_points": ["..."]
        }}
        ```
        """
        return prompt
    
    async def _get_model_response(self, prompt: str) -> str:
        """Send a prompt to the Ollama model and get a response."""
        try:
            # Use the /api/generate endpoint that was confirmed to work
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,  # Get the full response at once
                "temperature": self.temperature,
                "num_ctx": self.num_ctx
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=60.0)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
                
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            return f"Error: {str(e)}"
    
    async def run(self, video_data: YouTubeVideoData) -> VideoAnalysisData:
        """Run the agent to extract insights from video transcript."""
        try:
            self.input_data = video_data
            
            if not video_data.transcript:
                logger.warning(f"No transcript available for video: {video_data.video_id}")
                return VideoAnalysisData(original_data=video_data)
            
            prompt = await self._prepare_prompt(video_data)
            
            logger.info(f"Running analysis for video: {video_data.video_id}")
            response = await self._get_model_response(prompt)
            
            try:
                insights_data = self._parse_json_response(response)
                
                result = VideoAnalysisData(
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
                
                return result
                
            except Exception as parsing_error:
                logger.error(f"Error parsing response: {parsing_error}")
                return await self.fallback_parsing(response)
            
        except Exception as e:
            logger.error(f"Error processing transcript data: {e}")
            return VideoAnalysisData(original_data=video_data)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from model response."""
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response) or re.search(r'({[\s\S]*})', response)
        
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        else:
            return json.loads(response)
    
    async def fallback_parsing(self, raw_text: str) -> VideoAnalysisData:
        """Fallback parsing method if structured output fails."""
        logger.info("Using fallback parsing for response")
        
        summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', raw_text)
        summary = summary_match.group(1) if summary_match else "Summary extraction failed"
        
        software_mentions = []
        software_match = re.findall(r'"name"\s*:\s*"([^"]+)"', raw_text)
        for name in software_match:
            software_mentions.append(SoftwareMention(
                name=name,
                description="Extracted via fallback parsing"
            ))
        
        return VideoAnalysisData(
            original_data=self.input_data,
            summary=summary,
            software_mentions=software_mentions,
            main_topics=[],
            key_points=[]
        )
# nodes/transcript_analysis_node.py
import logging
import json
import os
from typing import Dict, Any, Optional

from agents.transcript_insight_agent import TranscriptInsightAgent
from models.data_models import YouTubeVideoData, VideoAnalysisData

logger = logging.getLogger(__name__)

class TranscriptAnalysisNode:
    """Node for analyzing YouTube transcripts and extracting insights."""
    
    def __init__(self, model_name: Optional[str] = None, save_output: bool = True):
        """Initialize the transcript analysis node."""
        self.agent = TranscriptInsightAgent(model_name=model_name)
        self.save_output = save_output
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and update with analysis results.
        
        Args:
            state: Current workflow state containing video data
            
        Returns:
            Updated state with analysis results
        """
        try:
            # Extract video data from state
            video_data = state.get("video_data")
            if not video_data:
                logger.error("No video data found in state")
                state["error"] = "Missing video data"
                return state
            
            # Convert to YouTubeVideoData if it's a dict
            if isinstance(video_data, dict):
                video_data = YouTubeVideoData(**video_data)
            
            # Run the analysis
            logger.info(f"Running transcript analysis for video: {video_data.video_id}")
            analysis_result = await self.agent.run(video_data)
            
            # Update state with analysis results
            state["analysis_result"] = analysis_result.model_dump()
            
            # Save output to file if requested
            if self.save_output:
                output_file = f"output/{video_data.video_id}_analysis_data.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(analysis_result.model_dump(), f, indent=2, ensure_ascii=False)
                logger.info(f"Analysis output saved to {output_file}")
            
            # Mark this node as completed
            state["transcript_analysis_completed"] = True
            
            return state
            
        except Exception as e:
            logger.error(f"Error in transcript analysis node: {e}")
            state["error"] = str(e)
            state["transcript_analysis_completed"] = False
            return state
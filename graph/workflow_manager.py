# graph/workflow_manager.py
import asyncio
import logging
import os
import datetime
from typing import Dict, Any, List, Optional

from state.workflow_state import WorkflowState
from nodes.transcript_analysis_node import TranscriptAnalysisNode
from agents.youtube_transcript_agent import YouTubeTranscriptAgent
from api_services.transcript_service import get_video_id_from_url

logger = logging.getLogger(__name__)

class WorkflowManager:
    """Manager for orchestrating the node-based workflow."""
    
    def __init__(self, save_output: bool = True):
        """Initialize the workflow manager."""
        self.save_output = save_output
        self.state = WorkflowState()
    
    def _init_state(self, video_url: str, model_name: Optional[str] = None) -> None:
        """Initialize workflow state with input parameters."""
        self.state = WorkflowState(
            video_url=video_url,
            model_name=model_name,
            start_time=datetime.datetime.now().isoformat()
        )
    
    async def run_workflow(self, video_url: str, model_name: Optional[str] = None, save_output: bool = False) -> WorkflowState:
        """
        Run the complete workflow for a video URL.
        
        Args:
            video_url: URL of the YouTube video
            model_name: Optional model name to use
            save_output: Whether to save output to files
            
        Returns:
            Final workflow state
        """
        try:
            # Initialize state
            self._init_state(video_url, model_name)
            self.save_output = save_output
            
            # Create analysis node
            analysis_node = TranscriptAnalysisNode(
                model_name=model_name,
                save_output=self.save_output
            )
            
            # Extract video ID
            video_id = get_video_id_from_url(video_url)
            
            if not video_id:
                logger.error(f"Could not extract video ID from URL: {video_url}")
                self.state.error = "Invalid YouTube URL"
                self.state.error_node = "workflow_manager"
                return self.state
            
            # Get API key
            api_key = os.getenv('YT_DATA_API_KEY')
            if not api_key:
                logger.error("YouTube API key not found in environment variables")
                self.state.error = "Missing YouTube API key"
                self.state.error_node = "workflow_manager"
                return self.state
            
            # Create and run the transcript agent
            transcript_agent = YouTubeTranscriptAgent(youtube_api_key=api_key, model_name=model_name)
            
            logger.info(f"Starting transcript extraction for: {video_url}")
            video_data = await transcript_agent.run(video_id)
            
            # Update state with video data
            self.state.video_id = video_id
            self.state.video_data = video_data.model_dump()
            self.state.transcript_extraction_completed = True
            
            # Execute analysis node
            logger.info("Starting transcript analysis")
            state_dict = self.state.model_dump()
            updated_state = await analysis_node.process(state_dict)
            self.state = WorkflowState(**updated_state)
            
            # Finalize state
            self.state.end_time = datetime.datetime.now().isoformat()
            
            # Save final state if requested
            if self.save_output and self.state.video_id:
                state_file = f"output/{self.state.video_id}_workflow_state.json"
                self.state.save_to_file(state_file)
                logger.info(f"Final workflow state saved to {state_file}")
            
            return self.state
            
        except Exception as e:
            logger.error(f"Unhandled error in workflow: {e}")
            self.state.error = str(e)
            self.state.error_node = "workflow_manager"
            self.state.end_time = datetime.datetime.now().isoformat()
            return self.state
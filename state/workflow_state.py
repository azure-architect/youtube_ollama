# state/workflow_state.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class WorkflowState(BaseModel):
    """Model for storing and managing workflow state between nodes."""
    
    # Input parameters
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    model_name: Optional[str] = None
    
    # Processing flags
    transcript_extraction_completed: bool = False
    transcript_analysis_completed: bool = False
    
    # Results
    video_data: Optional[Dict[str, Any]] = None
    analysis_result: Optional[Dict[str, Any]] = None
    
    # Error handling
    error: Optional[str] = None
    error_node: Optional[str] = None
    
    # Metadata
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def save_to_file(self, filepath: str) -> None:
        """Save current state to JSON file."""
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "WorkflowState":
        """Load state from JSON file."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
# api_services/youtube_data_api.py
import os
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_youtube_video_data(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch video metadata from YouTube Data API
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Dictionary containing video data or None if request fails
    """
    api_key = os.getenv('YT_DATA_API_KEY')
    if not api_key:
        logger.error("YouTube API key not found in environment variables")
        return None
        
    try:
        # Make the API request to get video details
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            'id': video_id,
            'key': api_key,
            'part': 'snippet,statistics,status,contentDetails,topicDetails'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        data = response.json()
        
        # Check if we have items in the response
        if not data.get('items'):
            logger.warning(f"No video data found for ID: {video_id}")
            return {
                'snippet': {},
                'statistics': {},
                'status': {},
                'topicDetails': {},
                'contentDetails': {}
            }
            
        # Extract the first item (should be the only one)
        video_data = data['items'][0]
        
        # Return all parts as received from the API
        return {
            'snippet': video_data.get('snippet', {}),
            'statistics': video_data.get('statistics', {}),
            'status': video_data.get('status', {}),
            'topicDetails': video_data.get('topicDetails', {}),
            'contentDetails': video_data.get('contentDetails', {})
        }
        
    except Exception as e:
        logger.error(f"Error fetching YouTube data: {e}")
        return None
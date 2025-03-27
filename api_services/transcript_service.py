# api_services/transcript_service.py
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import logging
import json  # Make sure json is imported

logger = logging.getLogger(__name__)

def get_video_id_from_url(url):
    """Extracts the video ID from a YouTube URL."""
    try:
        parsed_url = urlparse(url)
        if parsed_url.netloc == 'youtu.be':
            return parsed_url.path[1:]
        if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                query_params = parse_qs(parsed_url.query)
                return query_params['v'][0]
            if parsed_url.path[:7] == '/embed/':
                return parsed_url.path.split('/')[2]
            if parsed_url.path[:3] == '/v/':
                return parsed_url.path.split('/')[2]
        return None  # Not a valid YouTube URL
    except Exception as e:
        logger.exception(f"Error extracting video ID: {e}")
        return None


def get_video_transcript_data(youtube_url):
    """Fetches the transcript of a YouTube video and returns it as a dictionary."""
    video_id = get_video_id_from_url(youtube_url)
    if not video_id:
        return None

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        # The transcript is already a list of dictionaries.  No need for JSONFormatter.
        return {
            "video_id": video_id,
            "transcript": transcript,  # Return the list directly
            "available_transcripts": YouTubeTranscriptApi.list_transcripts(video_id)._manually_created_transcripts

        }
    except Exception as e:
        logger.exception(f"Error fetching transcript: {e}")
        return None
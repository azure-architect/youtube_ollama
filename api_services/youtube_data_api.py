import os
import json
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
YT_DATA_API_KEY = os.getenv('YT_DATA_API_KEY')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Cache configuration
CHANNEL_CACHE_FILE = "channel_cache.json"
CHANNEL_VIDEOS_CACHE_FILE = "channel_videos_cache.json"
CACHE_TTL = timedelta(days=2)  # 48 hours cache lifetime

def load_channel_cache():
    """Loads the channel cache from file, handling JSON errors."""
    channel_cache = {}
    if os.path.exists(CHANNEL_CACHE_FILE):
        try:
            with open(CHANNEL_CACHE_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    channel_cache = data
                    logger.info(f"Loaded cache with {len(channel_cache)} channels")
                else:
                    logger.warning("Channel cache file is not a dictionary.")
        except json.JSONDecodeError:
            logger.warning("Could not decode JSON from channel cache file.")
        except Exception as e:
            logger.warning(f"Could not load channel cache: {e}")
    return channel_cache

def load_channel_videos_cache():
    """Loads the channel videos cache from file."""
    cache = {}
    if os.path.exists(CHANNEL_VIDEOS_CACHE_FILE):
        try:
            with open(CHANNEL_VIDEOS_CACHE_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    cache = data
                    logger.info(f"Loaded videos cache with {len(cache)} channels")
                else:
                    logger.warning("Channel videos cache file is not a dictionary.")
        except json.JSONDecodeError:
            logger.warning("Could not decode JSON from channel videos cache file.")
        except Exception as e:
            logger.warning(f"Could not load channel videos cache: {e}")
    return cache

channel_cache = load_channel_cache()
channel_videos_cache = load_channel_videos_cache()

def save_channel_cache(channel_cache):
    """Saves the channel cache to file."""
    try:
        with open(CHANNEL_CACHE_FILE, 'w') as f:
            json.dump(channel_cache, f, indent=4)
        logger.info(f"Updated channel cache")
    except Exception as e:
        logger.warning(f"Could not save channel cache: {e}")

def save_channel_videos_cache(cache):
    """Saves the channel videos cache to file."""
    try:
        with open(CHANNEL_VIDEOS_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=4)
        logger.info(f"Updated channel videos cache")
    except Exception as e:
        logger.warning(f"Could not save channel videos cache: {e}")

def get_youtube_video_data(video_id, include_channel_videos=False, max_channel_videos=10):
    """
    Retrieves YouTube video and channel data with caching.

    Args:
        video_id (str): YouTube video ID
        include_channel_videos (bool): Whether to include channel videos in the response
        max_channel_videos (int): Maximum number of channel videos to include

    Returns:
        dict: Comprehensive video data or None on failure
    """
    try:
        api_key = os.getenv('YT_DATA_API_KEY')
        if not api_key:
            logger.error("YouTube Data API key not found")
            return None

        youtube = build('youtube', 'v3', developerKey=api_key)

        # 1. Fetch Video Details
        video_response = youtube.videos().list(
            part='snippet,contentDetails,statistics,topicDetails',
            id=video_id
        ).execute()

        if not video_response['items']:
            logger.warning(f"No video found with ID: {video_id}")
            return None

        video = video_response['items'][0]
        video_snippet = video['snippet']
        video_statistics = video.get('statistics', {})
        video_topic_details = video.get('topicDetails', {})
        channel_id = video_snippet['channelId']

        # 2. Channel Data (with Caching and TTL)
        channel_data = get_channel_data(youtube, channel_id, video_snippet)
        if channel_data is None:  # Handle channel retrieval failure
            logger.warning(f"Could not retrieve channel data for video ID: {video_id}")
            channel_data = {
                "id": channel_id,
                "title": video_snippet.get('channelTitle', 'Unknown Channel'),
                "description": "N/A",
                "subscriberCount": None,
                "channelAge": "N/A",
                "videoCount": None,
                "isVerified": False
            }

        # 3. Comments - Only get the count
        comments_data = get_comments_data(video_statistics)

        # Format publishedAt date
        try:
            published_at = datetime.strptime(video_snippet['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').isoformat()
        except (ValueError, TypeError):
            published_at = 'N/A'

        result = {
            "video": {
                "id": video_id,
                "title": video_snippet['title'],
                "description": video_snippet.get('description', 'N/A'),
                "publishedAt": published_at,
                "views": int(video_statistics.get('viewCount', 0)) if video_statistics.get('viewCount') else None,
                "likes": int(video_statistics.get('likeCount', 0)) if video_statistics.get('likeCount') else None,
                "tags": video_snippet.get('tags', []),
                "topicDetails": video_topic_details,
                "thumbnail": video_snippet['thumbnails']['standard']['url'] if 'standard' in video_snippet.get('thumbnails',{}) else 'N/A',
                "comments": comments_data
            },
            "channel": channel_data
        }

        # 4. Include channel videos if requested
        if include_channel_videos:
            result["channelVideos"] = get_channel_videos(youtube, channel_id, max_results=max_channel_videos)

        return result

    except HttpError as e:
        logger.error(f"HTTP error: {e.resp.status} - {e.content.decode()}")
        if e.resp.status == 403 and "quotaExceeded" in str(e.content):
            logger.error("YouTube API quota exceeded. Consider waiting or optimizing API usage.")
        return None
    except Exception as e:
        logger.exception(f"Error: {e}")
        return None

def get_channel_data(youtube, channel_id, video_snippet):
    """Fetches channel data, using cache if available and valid."""
    global channel_cache
    now = datetime.utcnow()

    if channel_id in channel_cache:
        cached_data = channel_cache[channel_id]
        if 'cached_at' in cached_data and now - datetime.fromisoformat(cached_data['cached_at']) < CACHE_TTL:
            logger.info(f"Using cached data for channel: {channel_id}")
            return cached_data['data']
        else:
            logger.info(f"Cached data for channel {channel_id} expired.")

    logger.info(f"Fetching channel data for channel: {channel_id}")
    try:
        channel_response = youtube.channels().list(
            part='snippet,statistics,brandingSettings',
            id=channel_id
        ).execute()

        if not channel_response['items']:
            logger.warning(f"Channel not found for ID: {channel_id}")
            return None

        channel_item = channel_response['items'][0]
        channel_snippet = channel_item['snippet']
        channel_statistics = channel_item.get('statistics', {})
        branding_settings = channel_item.get('brandingSettings', {})

        # Calculate channel age
        try:
            channel_created_at = datetime.strptime(channel_snippet['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
            channel_age_days = (now - channel_created_at).days
            channel_age_years = round(channel_age_days / 365, 1)
            channel_age = f"{channel_age_years} years ({channel_age_days} days)"
        except (ValueError, TypeError, KeyError):
            channel_age = "N/A"

        channel_data = {
            "id": channel_id,
            "title": channel_snippet['title'],
            "description": channel_snippet.get('description', 'N/A'),
            "subscriberCount": int(channel_statistics.get('subscriberCount', 0)) if channel_statistics.get('subscriberCount') else None,
            "videoCount": int(channel_statistics.get('videoCount', 0)) if channel_statistics.get('videoCount') else None,
            "channelAge": channel_age,
            "isVerified": 'channel' in branding_settings and branding_settings.get('channel', {}).get('showRelatedChannels', False)
        }

        # Store in cache with timestamp
        channel_cache[channel_id] = {
            'data': channel_data,
            'cached_at': now.isoformat()
        }
        save_channel_cache(channel_cache)
        return channel_data

    except HttpError as e:
        logger.error(f"HTTP error fetching channel data: {e.resp.status} - {e.content.decode()}")
        return None
    except Exception as e:
        logger.exception(f"Error fetching channel data: {e}")
        return None

def get_channel_videos(youtube, channel_id, max_results=10):
    """
    Retrieves videos from a specific channel with caching.

    Args:
        youtube: YouTube API service object
        channel_id (str): ID of the channel
        max_results (int): Maximum number of videos to retrieve

    Returns:
        list: List of video information dictionaries
    """
    global channel_videos_cache
    now = datetime.utcnow()

    # Check cache first
    if channel_id in channel_videos_cache:
        cached_data = channel_videos_cache[channel_id]
        if 'cached_at' in cached_data and now - datetime.fromisoformat(cached_data['cached_at']) < CACHE_TTL:
            logger.info(f"Using cached video list for channel: {channel_id}")
            return cached_data['videos']
        else:
            logger.info(f"Cached video list for channel {channel_id} expired.")

    logger.info(f"Fetching videos for channel: {channel_id}")
    try:
        # First, get the upload playlist ID for the channel
        channels_response = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        ).execute()

        if not channels_response['items']:
            logger.warning(f"Channel not found: {channel_id}")
            return []

        # Get the uploads playlist ID
        uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        # Get videos from the uploads playlist
        videos = []
        next_page_token = None

        while len(videos) < max_results:
            playlist_items_response = youtube.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=min(50, max_results - len(videos)),
                pageToken=next_page_token
            ).execute()

            for item in playlist_items_response['items']:
                video_data = {
                    'id': item['contentDetails']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', ''),
                    'publishedAt': item['snippet']['publishedAt'],
                    'thumbnail': item['snippet']['thumbnails']['default']['url'] if 'default' in item['snippet']['thumbnails'] else None
                }
                videos.append(video_data)

            next_page_token = playlist_items_response.get('nextPageToken')
            if not next_page_token or len(videos) >= max_results:
                break

        # Store in cache with timestamp
        channel_videos_cache[channel_id] = {
            'videos': videos,
            'cached_at': now.isoformat()
        }
        save_channel_videos_cache(channel_videos_cache)

        return videos

    except HttpError as e:
        logger.error(f"HTTP error fetching channel videos: {e.resp.status} - {e.content.decode()}")
        return []
    except Exception as e:
        logger.exception(f"Error fetching channel videos: {e}")
        return []

def get_comments_data(video_statistics):
    """
    Returns ONLY the comment count from the video statistics.
    No API calls are made here.
    """
    comment_count = video_statistics.get('commentCount', 'N/A')
    return {
        "commentCount": int(comment_count) if comment_count != 'N/A' else None,
        "sampleComments": []  # Always empty now
    }
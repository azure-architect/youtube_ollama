from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import json
import re
from datetime import timedelta

class YouTubeVideoData(BaseModel):
    video_id: str
    title: str
    description: str
    channel: str
    published_at: str
    transcript: List[Dict[str, Any]]
    duration: int
    view_count: int
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    tags: List[str] = []

class TopicSegment(BaseModel):
    topic: str
    start_time: float
    end_time: float
    key_points: List[str] = Field(default_factory=list)

class VideoAnalysisData(BaseModel):
    original_data: YouTubeVideoData
    main_topics: List[str]
    topic_segments: List[TopicSegment]
    key_points: List[str]
    sentiment: str
    target_audience: List[str]
    language_level: str
    content_quality: int = Field(ge=1, le=10)
    engagement_hooks: List[str] = []
    summary: str
    educational_value: Optional[str] = None
    content_warnings: List[str] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)

class VideoEnhancementAgent(BaseAgent[VideoAnalysisData]):
    def __init__(self, model_name: str = "llama3.1:8b-instruct-q8_0"):
        super().__init__(
            model_name=model_name,
            temperature=0.7,
            num_ctx=8192,  # Larger context for handling full transcripts
            result_model=VideoAnalysisData
        )
    
    def _format_duration(self, seconds: Union[int, float]) -> str:
        """Format seconds into a readable time string."""
        return str(timedelta(seconds=int(seconds)))
    
    def _format_transcript_for_prompt(self, transcript: List[Dict[str, Any]], max_tokens: int = 6000) -> str:
        """Format transcript with timestamps, possibly truncating if too long."""
        formatted_entries = []
        total_chars = 0
        char_limit = max_tokens * 4  # Rough approximation of tokens to chars
        
        for item in transcript:
            entry = f"{self._format_duration(item['start'])}: {item['text']}"
            formatted_entries.append(entry)
            
            total_chars += len(entry)
            if total_chars > char_limit:
                formatted_entries.append("...[transcript truncated due to length]...")
                break
                
        return "\n".join(formatted_entries)
    
    async def _prepare_prompt(self, video_data: YouTubeVideoData) -> str:
        """Prepare a comprehensive prompt to analyze the video data."""
        # Extract important metadata
        title = video_data.title
        channel = video_data.channel
        description = video_data.description
        view_count = video_data.view_count
        like_count = video_data.like_count or "Unknown"
        tags = ", ".join(video_data.tags) if video_data.tags else "None"
        
        # Format the transcript with timestamps
        transcript_text = self._format_transcript_for_prompt(video_data.transcript)
        
        # Create a detailed analysis prompt
        return f"""
# YouTube Video Analysis Task

## Video Metadata
- TITLE: {title}
- CHANNEL: {channel}
- VIEWS: {view_count}
- LIKES: {like_count}
- TAGS: {tags}

## Video Description
{description}

## Transcript with Timestamps
{transcript_text}

## Analysis Instructions
Analyze this YouTube video content thoroughly and provide a detailed structured analysis with the following components:

1. Identify the main topics discussed in the video (5-7 topics)
2. For each topic, determine approximate start and end timestamps based on the transcript
3. Extract 3-5 key points for each topic segment
4. Determine overall key points of the entire video (5-8 points)
5. Assess the general sentiment/tone (positive, negative, neutral, mixed, controversial)
6. Identify the likely target audience(s)
7. Evaluate language/education level (beginner, intermediate, advanced, technical, etc.)
8. Rate content quality on a scale of 1-10
9. Identify hooks or elements that drive viewer engagement
10. Write a comprehensive 3-5 sentence summary of the content
11. Evaluate educational value (if applicable)
12. Note any content warnings if applicable (sensitive topics, misleading information, etc.)
13. Suggest related topics that viewers might be interested in

Provide your analysis in structured JSON format adhering exactly to the schema provided by the API.
"""
    
    def _extract_timestamps(self, text: str) -> Dict[str, float]:
        """Extract timestamps from text in various formats."""
        timestamp_patterns = [
            r'(\d+):(\d+):(\d+)',  # HH:MM:SS
            r'(\d+):(\d+)',        # MM:SS
            r'(\d+)m(\d+)s',       # XmYs
            r'(\d+)m',             # Xm
            r'(\d+)s',             # Xs
            r'(\d+\.\d+)'          # floating point seconds
        ]
        
        times = {}
        lines = text.split('\n')
        
        for line in lines:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                
                # Look for timestamps in the value
                for pattern in timestamp_patterns:
                    matches = re.findall(pattern, value)
                    if matches:
                        # Convert to seconds based on pattern
                        if pattern == r'(\d+):(\d+):(\d+)':  # HH:MM:SS
                            h, m, s = map(int, matches[0])
                            seconds = h * 3600 + m * 60 + s
                        elif pattern == r'(\d+):(\d+)':  # MM:SS
                            m, s = map(int, matches[0])
                            seconds = m * 60 + s
                        elif pattern == r'(\d+)m(\d+)s':  # XmYs
                            m, s = map(int, matches[0])
                            seconds = m * 60 + s
                        elif pattern == r'(\d+)m':  # Xm
                            seconds = int(matches[0]) * 60
                        elif pattern == r'(\d+)s':  # Xs
                            seconds = int(matches[0])
                        else:  # floating point
                            seconds = float(matches[0])
                            
                        times[key] = seconds
                        break
        
        return times
    
    async def fallback_parsing(self, raw_text: str) -> VideoAnalysisData:
        """Fallback parsing for video analysis."""
        # Store errors for debugging
        print(f"Using fallback parsing for text: {raw_text[:200]}...")
        
        # Initialize with defaults
        main_topics = []
        topic_segments = []
        key_points = []
        sentiment = "Unknown"
        target_audience = ["General audience"]
        language_level = "Unknown"
        content_quality = 5
        engagement_hooks = []
        summary = "Analysis failed. Summary could not be extracted."
        educational_value = None
        content_warnings = []
        related_topics = []
        
        # Try to extract sections based on markdown headings or common section titles
        sections = {}
        current_section = None
        
        lines = raw_text.split('\n')
        for line in lines:
            # Detect section headers
            if line.strip().startswith(('#', '##', '###')) or any(x in line.lower() for x in ["topics:", "sentiment:", "audience:", "key points:", "summary:"]):
                current_section = line.strip().lower()
                # Clean up the section name
                current_section = re.sub(r'^#+\s*', '', current_section)
                current_section = re.sub(r':$', '', current_section)
                sections[current_section] = []
            elif current_section and line.strip():
                sections[current_section].append(line.strip())
        
        # Process the extracted sections
        for section, content in sections.items():
            content_text = ' '.join(content)
            
            # Extract main topics
            if any(x in section for x in ['topic', 'main']):
                # Look for list items
                topics = [re.sub(r'^[-*•]\s*', '', line) for line in content if line.startswith(('-', '*', '•'))]
                if topics:
                    main_topics = topics
                else:
                    # Split by commas or semicolons if no list format
                    for line in content:
                        if ',' in line or ';' in line:
                            main_topics = [t.strip() for t in re.split(r'[,;]', line) if t.strip()]
            
            # Extract sentiment
            if 'sentiment' in section or 'tone' in section:
                sentiment_keywords = ['positive', 'negative', 'neutral', 'mixed', 'controversial']
                for keyword in sentiment_keywords:
                    if keyword in content_text.lower():
                        sentiment = keyword.capitalize()
                        break
            
            # Extract language level
            if 'language' in section or 'education' in section:
                level_keywords = ['beginner', 'intermediate', 'advanced', 'technical', 'academic', 'professional']
                for keyword in level_keywords:
                    if keyword in content_text.lower():
                        language_level = keyword.capitalize()
                        break
            
            # Extract key points
            if 'key point' in section or 'main point' in section:
                points = [re.sub(r'^[-*•]\s*', '', line) for line in content if line.startswith(('-', '*', '•'))]
                if points:
                    key_points = points
            
            # Extract summary
            if 'summary' in section:
                summary = ' '.join(content)
            
            # Extract topic segments
            timestamps = self._extract_timestamps(content_text)
            for topic in main_topics:
                if topic.lower() in timestamps:
                    # Try to find a matching end time
                    end_time = None
                    for t in main_topics:
                        if t != topic and t.lower() in timestamps and timestamps[t.lower()] > timestamps[topic.lower()]:
                            if end_time is None or timestamps[t.lower()] < end_time:
                                end_time = timestamps[t.lower()]
                    
                    if not end_time and len(self.input_data.transcript) > 0:
                        # Use end of video if no next topic
                        last_entry = self.input_data.transcript[-1]
                        end_time = last_entry['start'] + last_entry.get('duration', 0)
                    else:
                        end_time = timestamps[topic.lower()] + 300  # Default 5 minutes
                    
                    topic_segments.append(TopicSegment(
                        topic=topic,
                        start_time=timestamps[topic.lower()],
                        end_time=end_time,
                        key_points=[]
                    ))
        
        # If we still don't have main topics, try to extract from the full text
        if not main_topics:
            # Look for numbered or bulleted lists
            for i, line in enumerate(lines):
                if re.match(r'^\d+\.', line) or line.startswith(('-', '*', '•')):
                    cleaned = re.sub(r'^\d+\.|-|\*|•\s*', '', line).strip()
                    if len(cleaned) > 3 and len(cleaned) < 100:  # Reasonable topic length
                        main_topics.append(cleaned)
        
        # If we still don't have a summary, generate a basic one
        if summary == "Analysis failed. Summary could not be extracted.":
            title = self.input_data.title
            channel = self.input_data.channel
            summary = f"This video titled '{title}' by {channel} covers various topics that could not be automatically extracted. A manual review is recommended."
        
        # Create the analysis result
        return VideoAnalysisData(
            original_data=self.input_data,
            main_topics=main_topics[:7] if main_topics else ["Topic extraction failed"],
            topic_segments=topic_segments,
            key_points=key_points[:8] if key_points else ["Key points extraction failed"],
            sentiment=sentiment,
            target_audience=target_audience,
            language_level=language_level,
            content_quality=content_quality,
            engagement_hooks=engagement_hooks,
            summary=summary,
            educational_value=educational_value,
            content_warnings=content_warnings,
            related_topics=related_topics
        )
    
    async def run(self, video_data: YouTubeVideoData, message_history: Optional[List[Dict[str, str]]] = None) -> VideoAnalysisData:
        """Run the enhancement agent with robust error handling."""
        # Store input data for fallback parsing
        self.input_data = video_data
        
        try:
            return await super().run(video_data, message_history)
        except Exception as e:
            print(f"Error in enhancement agent: {e}")
            # Return basic analysis with fallback parsing
            return await self.fallback_parsing("")
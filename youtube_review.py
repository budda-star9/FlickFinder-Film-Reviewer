from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript

def get_youtube_transcript(video_id):
    """Safe transcript fetcher that handles language and caption errors."""
    try:
        # Fetch the list of available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try English first, then auto-generated captions if needed
        try:
            transcript = transcript_list.find_transcript(['en']).fetch()
        except NoTranscriptFound:
            transcript = transcript_list.find_manually_created_transcript(['en']).fetch()

        # Combine the transcript text
        return " ".join([t['text'] for t in transcript])

    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
        print("⚠️ No transcript available — using video metadata instead.")
        return None
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        return None

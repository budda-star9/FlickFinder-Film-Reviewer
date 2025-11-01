import streamlit as st
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import tempfile
import os
from openai import OpenAI

# Initialize client
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))

st.set_page_config(page_title="🎬 YouTube Film Review AI", layout="wide")
st.title("🎬 FlickFinder AI — YouTube Film Review")
st.caption("Analyze and grade films using Dan Harmon’s Story Circle + Hero’s Journey")

youtube_url = st.text_input("Paste a YouTube video URL to begin:")

def get_video_id(url: str) -> str:
    """Extract the YouTube video ID from URL"""
    import re
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

if youtube_url:
    try:
        yt = YouTube(youtube_url)
        st.video(youtube_url)
        st.markdown(f"**🎞️ Title:** {yt.title}")
        st.markdown(f"**📅 Published:** {yt.publish_date}")
        st.markdown(f"**🕒 Length:** {yt.length // 60} minutes")

        video_id = get_video_id(youtube_url)
        transcript_text = None

        # Try fetching transcript
        with st.spinner("Fetching transcript/subtitles..."):
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([t["text"] for t in transcript])
                st.success("✅ Transcript retrieved successfully!")
            except (TranscriptsDisabled, NoTranscriptFound):
                st.warning("⚠️ No transcript available — will analyze metadata only.")
            except Exception as e:
                st.warning(f"⚠️ Could not retrieve transcript: {e}")

        # Build review prompt
        prompt = f"""
        You are a professional film reviewer for an AI-based festival jury.
        Analyze and grade this film based on:
        - Dan Harmon's 8-Step Story Circle (ensure 70–80% minimum adherence)
        - Joseph Campbell's Hero’s Journey

        **Weighted Scoring (Total 100 points):**
        - Storytelling: 25%
        - Technical Quality: 20%
        - Directing: 20%
        - Cultural/Social Impact: 20%
        - Artistic Vision: 15%

        Provide:
        1. A short synopsis
        2. Strengths & weaknesses
        3. Weighted category breakdown with numeric scores
        4. Final qualitative grade (/100)
        5. A closing statement for the festival jury

        Film title: {yt.title}

        Transcript (if available):
        {transcript_text if transcript_text else "[Transcript not available]"}
        """

        with st.spinner("AI reviewing in progress..."):
        review_text = ai_review_film(client, yt.title, transcript_text)

        st.subheader("🧾 AI Review Summary")
        st.markdown(review_text)

        # Save review
        save_review_to_csv(yt.title, review_text, source="YouTube")
        st.success(f"✅ Review saved to {REVIEWS_FILE}")


import streamlit as st
import tempfile
import cv2
import numpy as np
from openai import OpenAI
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs
from filmfreeway_analyzer import filmfreeway_interface, display_saved_projects
from scoring_system import ScoringSystem
from export_system import export_interface
from ai_prompt import build_film_review_prompt
import base64
import os
import json
import time
# -------------------------------------------------
# Helper: update film title in session_state
# -------------------------------------------------
def update_film_title(old_title, new_title):
    films = st.session_state.get("films_to_score", [])
    for f in films:
        if f["title"] == old_title:
            f["title"] = new_title
            break
# --------------------------
# Utility Functions
# --------------------------
def get_video_id(url):
    parsed = urlparse(url)
    if parsed.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed.query).get("v", [None])[0]
    elif parsed.hostname == "youtu.be":
        return parsed.path[1:]
    return None

def fetch_transcript(video_id, yt):
    transcript_text = ""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([seg["text"] for seg in transcript_list])
        st.success("✅ Transcript retrieved successfully!")
    except (TranscriptsDisabled, NoTranscriptFound):
        st.warning("⚠️ No transcript available. Attempting audio transcription with Whisper...")
        # Download audio for Whisper
        with tempfile.TemporaryDirectory() as tempdir:
            audio_path = yt.streams.get_audio_only().download(output_path=tempdir, filename="audio.mp4")
            try:
                transcript_response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=open(audio_path, "rb")
                )
                transcript_text = transcript_response.text
                st.success("✅ Audio transcribed successfully!")
            except Exception as e:
                st.error(f"❌ Error transcribing audio: {e}")
                transcript_text = yt.title + " " + (yt.description or "")
    return transcript_text

def extract_video_frames(video_path, num_frames=3):
    """Extract sample frames for AI visual analysis."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in np.linspace(0, total - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(buffer.tobytes())
    cap.release()
    return frames

def store_film_for_scoring(title, url, platform, description=""):
    if "films_to_score" not in st.session_state:
        st.session_state["films_to_score"] = []
    if not any(f["url"] == url for f in st.session_state["films_to_score"]):
        st.session_state["films_to_score"].append({
            "title": title,
            "platform": platform,
            "url": url,
            "description": description
        })

def build_batch_tasks(films):
    tasks = []
    for idx, film in enumerate(films):
        # For batch, use description or fetch basic metadata; skip heavy downloads
        description = film.get("description", "")
        title = film["title"]
        url = film["url"]
        visual_context = ""  # No full vision for batch to keep scalable
        if url:  # If YouTube URL, fetch basic
            video_id = get_video_id(url)
            if video_id:
                try:
                    yt = YouTube(url)
                    transcript_text = fetch_transcript(video_id, yt)  # Uses API or Whisper (Whisper may be slow for batch)
                    description += " " + transcript_text
                    visual_context = f"Thumbnail: {yt.thumbnail_url}"
                except:
                    pass
        # Build prompt for scoring
        prompt = build_film_review_prompt(
            film_metadata=f"Title: {title}",
            transcript_text=description,
            audience_reception="Based on available metadata",
            visual_context=visual_context
        )
        # Modify for JSON output with scores
        scoring_prompt = prompt + "\nOutput as JSON: {summary: string, scores: {story: number 1-5, vision: number 1-5, craft: number 1-5, sound: number 1-5}, weighted_score: number 1-5}"
        task = {
            "custom_id": f"film-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "user", "content": scoring_prompt}
                ]
            }
        }
        tasks.append(task)
    return tasks

# --------------------------
# Initialize OpenAI Client
# --------------------------
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))

# --------------------------
# Tabs
# --------------------------
tab1, tab2 = st.tabs(["📊 CSV Movie Reviews", "🎥 YouTube Film Analysis"])

# --------------------------
# TAB 1: CSV Movie Reviews
# --------------------------
with tab1:
    st.header("📊 CSV Movie Reviews")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        for idx, row in df.iterrows():
            store_film_for_scoring(
                title=row.get("title") or row.get("Film Title") or f"Film {idx+1}",
                url=row.get("url") or "",
                platform="CSV",
                description=row.get("description") or ""
            )
    st.subheader("🎬 Films Ready for Scoring")
    if "films_to_score" in st.session_state:
        for f in st.session_state["films_to_score"]:
            st.write(f"**Title:** {f['title']}  |  Platform: {f['platform']}")

# --------------------------
# TAB 2: YouTube Film Review
# --------------------------
with tab2:
    st.header("🎥 YouTube Film Analysis")
    st.caption("AI-assisted cinematic evaluation — story, vision, and craft")

    # YouTube URL input
    youtube_url = st.text_input("📎 Paste YouTube video URL to analyze:")

    # 🎬 Editable Film Title input immediately below URL
    custom_title = ""
    if youtube_url:
        video_id = get_video_id(youtube_url)
        if video_id:
            try:
                yt = YouTube(youtube_url)
                st.video(youtube_url)

                # -------------------------------
                # Must-have Film Title field
                # -------------------------------
                custom_title = st.text_input(
                    "🎬 Enter or Edit Film Title for Scoring:",
                    value=yt.title,
                    help="Rename the title for AI scoring and reports."
                )

                st.markdown(f"**📅 Published:** {yt.publish_date}")
                st.markdown(f"**🕒 Length:** {yt.length // 60} minutes")

                # Transcript with Whisper fallback
                transcript_text = fetch_transcript(video_id, yt)

                # Store film for scoring
                store_film_for_scoring(custom_title, youtube_url, "YouTube", yt.description or "")

                # Visual Analysis toggle
                enable_vision = st.toggle("🧠 Enable Visual Analysis (frame sampling + cinematography review)", value=False)
                images_content = []
                if enable_vision:
                    st.info("🎞️ Downloading video and extracting sample frames for visual analysis...")
                    with tempfile.TemporaryDirectory() as tempdir:
                        video_path = yt.streams.get_lowest_resolution().download(output_path=tempdir)
                        frames = extract_video_frames(video_path, num_frames=3)
                        for frame in frames:
                            base64_image = base64.b64encode(frame).decode('utf-8')
                            images_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            })

                # AI Review Prompt - Modified for scoring
                prompt_text = build_film_review_prompt(
                    film_metadata=f"Title: {custom_title}\nChannel: {yt.author}\nLength: {yt.length // 60} min",
                    transcript_text=transcript_text,
                    audience_reception="Based on public YouTube metrics",
                    visual_context="Analyze the provided frames for visual elements." if enable_vision else ""
                )
                # Add scoring to prompt
                prompt_text += "\nOutput as JSON: {summary: string, scores: {story: number 1-5, vision: number 1-5, craft: number 1-5, sound: number 1-5}, weighted_score: number 1-5 (average of scores)}"

                user_content = [{"type": "text", "text": prompt_text}] + images_content

                with st.spinner("🤖 Generating AI film review and scores..."):
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        response_format={"type": "json_object"},
                        messages=[{"role": "user", "content": user_content}],
                    )
                    review_content = json.loads(response.choices[0].message.content)
                    st.subheader("🧾 AI Review Summary")
                    st.markdown(review_content.get("summary", "No summary available."))
                    st.subheader("📊 AI Scores")
                    scores = review_content.get("scores", {})
                    for category, score in scores.items():
                        st.write(f"**{category.capitalize()}:** {score}/5")
                    weighted = review_content.get("weighted_score", 0)
                    st.write(f"**Weighted Score:** {weighted}/5")
                    # Save scores
                    st.session_state.all_scores.append({
                        "title": custom_title,
                        "scores": scores,
                        "weighted_score": weighted
                    })

            except Exception as e:
                st.error(f"❌ Error processing YouTube video: {e}")
        else:
            st.error("❌ Invalid YouTube URL")


# --------------------------
# Main App
# --------------------------
def main():
    st.set_page_config(page_title="FlickFinder", page_icon="🎬", layout="wide")

    if "scoring_system" not in st.session_state:
        st.session_state.scoring_system = ScoringSystem()
    if "all_scores" not in st.session_state:
        st.session_state.all_scores = []
    if "batch_id" not in st.session_state:
        st.session_state.batch_id = None

    with st.sidebar:
        st.header("🎬 FlickFinder Dashboard")
        page = st.radio(
            "Navigate to:",
            ["🏠 Home", "🔗 FilmFreeway", "🎯 Score Films", "📊 Export", "📚 Saved Projects"]
        )

    if page == "🏠 Home":
        home_interface()
    elif page == "🔗 FilmFreeway":
        filmfreeway_interface(client)
    elif page == "🎯 Score Films":
        scoring_interface()
    elif page == "📊 Export":
        export_interface()
    elif page == "📚 Saved Projects":
        display_saved_projects()

# --------------------------
# Home page
# --------------------------
def home_interface():
    st.title("Welcome to FlickFinder 🎬")
    st.markdown("### Professional Film Evaluation & Jury Assistant")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔗 FilmFreeway Integration**")
        st.markdown("Import and analyze projects directly from FilmFreeway.")
    with col2:
        st.markdown("**🎯 Smart AI Scoring**")
        st.markdown("Weighted scoring with story, tech, and vision metrics.")
    with col3:
        st.markdown("**📊 Reports & Exports**")
        st.markdown("Generate detailed review sheets for jury panels.")
    st.markdown("---")

# --------------------------
# Scoring Interface
# --------------------------
def scoring_interface():
    st.header("🎯 Film Scoring")
    films = st.session_state.get("films_to_score", [])
    if not films:
        st.info("No films ready for scoring. Import or analyze one first.")
        return

    # -----------------------------------------------------------------
    # 1. Film selector
    # -----------------------------------------------------------------
    selected_title = st.selectbox(
        "Select a film:",
        [f["title"] for f in films],
        key="film_selector"
    )
    film = next(f for f in films if f["title"] == selected_title)

    # -----------------------------------------------------------------
    # 2. **TITLE EDITOR** – the window you asked for
    # -----------------------------------------------------------------
    with st.expander("Enter or Edit Film Title for Scoring:", expanded=True):
        new_title = st.text_input(
            "Enter or Edit Film Title for Scoring:",
            value=selected_title,
            key=f"title_edit_{selected_title}"
        )
        if new_title != selected_title:
            if st.button("Apply Title Change", key=f"apply_{selected_title}"):
                update_film_title(selected_title, new_title)
                st.session_state.film_selector = new_title   # refresh selector
                st.experimental_rerun()

    # -----------------------------------------------------------------
    # 3. Manual scorecard (your existing ScoringSystem)
    # -----------------------------------------------------------------
    st.markdown("---")
    st.subheader(f"Scoring **{new_title}**")
    score = st.session_state.scoring_system.get_scorecard_interface(new_title)

    if score:
        score["weighted_score"] = st.session_state.scoring_system.calculate_weighted_score(score["scores"])
        st.session_state.all_scores.append(score)
        st.success(f"Score saved! Weighted score: {score['weighted_score']}/5")

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    main()

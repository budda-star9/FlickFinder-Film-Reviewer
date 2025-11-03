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
            "description": description,
            "status": "pending",  # pending, watching, graded
            "ai_review": None,
            "manual_score": None
        })

def quick_grade_interface(film):
    """Quick grading buttons for rapid workflow"""
    st.subheader("🚀 Quick Grade")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("⭐1", key=f"quick_1_{film['title']}", use_container_width=True):
            return 1
    with col2:
        if st.button("⭐⭐2", key=f"quick_2_{film['title']}", use_container_width=True):
            return 2
    with col3:
        if st.button("⭐⭐⭐3", key=f"quick_3_{film['title']}", use_container_width=True):
            return 3
    with col4:
        if st.button("⭐⭐⭐⭐4", key=f"quick_4_{film['title']}", use_container_width=True):
            return 4
    with col5:
        if st.button("⭐⭐⭐⭐⭐5", key=f"quick_5_{film['title']}", use_container_width=True):
            return 5
    return None

def build_batch_tasks(films):
    tasks = []
    for idx, film in enumerate(films):
        description = film.get("description", "")
        title = film["title"]
        url = film["url"]
        visual_context = ""
        if url:
            video_id = get_video_id(url)
            if video_id:
                try:
                    yt = YouTube(url)
                    transcript_text = fetch_transcript(video_id, yt)
                    description += " " + transcript_text
                    visual_context = f"Thumbnail: {yt.thumbnail_url}"
                except:
                    pass
        prompt = build_film_review_prompt(
            film_metadata=f"Title: {title}",
            transcript_text=description,
            audience_reception="Based on available metadata",
            visual_context=visual_context
        )
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
tab1, tab2, tab3 = st.tabs(["📊 CSV Movie Reviews", "🎥 YouTube Film Analysis", "⚡ Quick Batch Grade"])

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
        
        # Batch processing controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Import All Films", use_container_width=True):
                for idx, row in df.iterrows():
                    store_film_for_scoring(
                        title=row.get("title") or row.get("Film Title") or f"Film {idx+1}",
                        url=row.get("url") or "",
                        platform="CSV",
                        description=row.get("description") or ""
                    )
                st.success(f"✅ Imported {len(df)} films!")
        
        with col2:
            if st.button("🤖 AI Grade All", use_container_width=True):
                st.info("Batch AI grading would go here...")

# --------------------------
# TAB 2: YouTube Film Review
# --------------------------
with tab2:
    st.header("🎥 YouTube Film Analysis")
    st.caption("AI-assisted cinematic evaluation — story, vision, and craft")

    # YouTube URL input
    youtube_url = st.text_input("📎 Paste YouTube video URL to analyze:")

    if youtube_url:
        video_id = get_video_id(youtube_url)
        if video_id:
            try:
                yt = YouTube(youtube_url)
                st.video(youtube_url)

                # Film title and action buttons in columns
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    custom_title = st.text_input(
                        "🎬 Film Title:",
                        value=yt.title,
                        help="Rename the title for AI scoring and reports."
                    )
                
                with col2:
                    if st.button("🎯 Grade Now", use_container_width=True):
                        store_film_for_scoring(custom_title, youtube_url, "YouTube", yt.description or "")
                        st.session_state.current_film = custom_title
                        st.rerun()
                
                with col3:
                    if st.button("👀 Watch & Grade", use_container_width=True):
                        store_film_for_scoring(custom_title, youtube_url, "YouTube", yt.description or "")
                        st.session_state.watch_mode = custom_title
                        st.rerun()

                st.markdown(f"**📅 Published:** {yt.publish_date}")
                st.markdown(f"**🕒 Length:** {yt.length // 60} minutes")

                # Analysis toggles
                st.subheader("🔧 Analysis Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    enable_transcript = st.toggle("📝 Transcript Analysis", value=True)
                with col2:
                    enable_vision = st.toggle("🎨 Visual Analysis", value=False)
                with col3:
                    enable_critique = st.toggle("🎭 Deep Critique", value=False)

                # Process based on toggles
                transcript_text = ""
                if enable_transcript:
                    transcript_text = fetch_transcript(video_id, yt)

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

                # AI Review Section with action button
                if st.button("🤖 Generate AI Critique", type="primary", use_container_width=True):
                    with st.spinner("🤖 Generating AI film review and scores..."):
                        prompt_text = build_film_review_prompt(
                            film_metadata=f"Title: {custom_title}\nChannel: {yt.author}\nLength: {yt.length // 60} min",
                            transcript_text=transcript_text,
                            audience_reception="Based on public YouTube metrics",
                            visual_context="Analyze the provided frames for visual elements." if enable_vision else "",
                            deep_critique=enable_critique
                        )
                        
                        prompt_text += "\nOutput as JSON: {summary: string, scores: {story: number 1-5, vision: number 1-5, craft: number 1-5, sound: number 1-5}, weighted_score: number 1-5 (average of scores)}"

                        user_content = [{"type": "text", "text": prompt_text}] + images_content

                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            response_format={"type": "json_object"},
                            messages=[{"role": "user", "content": user_content}],
                        )
                        review_content = json.loads(response.choices[0].message.content)
                        
                        # Store the AI review
                        film_idx = next((i for i, f in enumerate(st.session_state.films_to_score) 
                                       if f["title"] == custom_title), -1)
                        if film_idx >= 0:
                            st.session_state.films_to_score[film_idx]["ai_review"] = review_content
                            st.session_state.films_to_score[film_idx]["status"] = "graded"
                        
                        # Display results
                        st.subheader("🧾 AI Review Summary")
                        st.markdown(review_content.get("summary", "No summary available."))
                        
                        st.subheader("📊 AI Scores")
                        scores = review_content.get("scores", {})
                        for category, score in scores.items():
                            st.metric(label=category.capitalize(), value=f"{score}/5")
                        
                        weighted = review_content.get("weighted_score", 0)
                        st.metric(label="Overall Score", value=f"{weighted}/5", delta="AI Graded")
                        
                        # Save to all scores
                        st.session_state.all_scores.append({
                            "title": custom_title,
                            "scores": scores,
                            "weighted_score": weighted,
                            "type": "ai_critique"
                        })

            except Exception as e:
                st.error(f"❌ Error processing YouTube video: {e}")
        else:
            st.error("❌ Invalid YouTube URL")

# --------------------------
# TAB 3: Quick Batch Grade
# --------------------------
with tab3:
    st.header("⚡ Quick Batch Grade")
    st.caption("Rapid fire grading for film submissions")
    
    films = st.session_state.get("films_to_score", [])
    
    if not films:
        st.info("No films available for grading. Import some films first.")
    else:
        # Batch controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Reset All Status", use_container_width=True):
                for film in films:
                    film["status"] = "pending"
                st.rerun()
        with col2:
            if st.button("🤖 AI Grade Pending", use_container_width=True):
                st.info("This would trigger batch AI grading")
        with col3:
            if st.button("📊 Export All", use_container_width=True):
                st.info("Export functionality would go here")
        
        # Film cards for quick grading
        st.subheader("🎬 Films to Grade")
        for film in films:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
                
                with col1:
                    st.write(f"**{film['title']}**")
                    st.caption(f"Platform: {film['platform']}")
                
                with col2:
                    status = film.get('status', 'pending')
                    status_emoji = {"pending": "⏳", "watching": "👀", "graded": "✅"}
                    st.write(f"{status_emoji.get(status, '⏳')} {status.title()}")
                
                with col3:
                    if st.button("👀 Watch", key=f"watch_{film['title']}", use_container_width=True):
                        film['status'] = 'watching'
                        st.session_state.current_film = film['title']
                        st.rerun()
                
                with col4:
                    quick_score = quick_grade_interface(film)
                    if quick_score:
                        film['status'] = 'graded'
                        film['manual_score'] = quick_score
                        st.success(f"Rated {quick_score}⭐ for {film['title']}")
                        st.rerun()

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
    if "current_film" not in st.session_state:
        st.session_state.current_film = None
    if "watch_mode" not in st.session_state:
        st.session_state.watch_mode = None

    with st.sidebar:
        st.header("🎬 FlickFinder Dashboard")
        page = st.radio(
            "Navigate to:",
            ["🏠 Home", "🔗 FilmFreeway", "🎯 Score Films", "📊 Export", "📚 Saved Projects"]
        )
        
        # Quick stats in sidebar
        if "films_to_score" in st.session_state:
            films = st.session_state.films_to_score
            pending = len([f for f in films if f.get('status') == 'pending'])
            watching = len([f for f in films if f.get('status') == 'watching'])
            graded = len([f for f in films if f.get('status') == 'graded'])
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("📈 Grading Progress")
            st.sidebar.metric("Pending", pending)
            st.sidebar.metric("Watching", watching)
            st.sidebar.metric("Graded", graded)

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
    
    # Quick action buttons on home
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎥 Analyze YouTube Film", use_container_width=True):
            st.session_state.current_tab = "YouTube Film Analysis"
            st.rerun()
    with col2:
        if st.button("📊 Import CSV Batch", use_container_width=True):
            st.session_state.current_tab = "CSV Movie Reviews"
            st.rerun()
    with col3:
        if st.button("⚡ Quick Grade", use_container_width=True):
            st.session_state.current_tab = "Quick Batch Grade"
            st.rerun()
    
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
# Enhanced Scoring Interface
# --------------------------
def scoring_interface():
    st.header("🎯 Film Scoring")
    films = st.session_state.get("films_to_score", [])
    if not films:
        st.info("No films ready for scoring. Import or analyze one first.")
        return

    # Film selector with status
    film_options = []
    for f in films:
        status = f.get('status', 'pending')
        status_emoji = {"pending": "⏳", "watching": "👀", "graded": "✅"}
        film_options.append(f"{status_emoji.get(status, '⏳')} {f['title']}")
    
    selected_display = st.selectbox(
        "Select a film:",
        film_options,
        key="film_selector"
    )
    
    selected_title = selected_display[3:]  # Remove emoji and space
    film = next(f for f in films if f["title"] == selected_title)

    # Title editor
    with st.expander("✏️ Edit Film Title", expanded=True):
        new_title = st.text_input(
            "Film Title:",
            value=selected_title,
            key=f"title_edit_{selected_title}"
        )
        if new_title != selected_title:
            if st.button("Apply Title Change", key=f"apply_{selected_title}"):
                update_film_title(selected_title, new_title)
                st.success("Title updated!")
                st.rerun()

    # Action buttons for current film
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👀 Mark as Watching", use_container_width=True):
            film['status'] = 'watching'
            st.rerun()
    with col2:
        if st.button("✅ Mark Complete", use_container_width=True):
            film['status'] = 'graded'
            st.rerun()
    with col3:
        if st.button("🔄 Reset Status", use_container_width=True):
            film['status'] = 'pending'
            st.rerun()

    # Quick grade section
    st.subheader("🚀 Quick Grade")
    quick_score = quick_grade_interface(film)
    if quick_score:
        film['manual_score'] = quick_score
        film['status'] = 'graded'
        st.success(f"Quick graded {film['title']} as {quick_score}⭐")
        st.rerun()

    # Detailed scoring (your existing system)
    st.markdown("---")
    st.subheader(f"Detailed Scoring: **{new_title}**")
    score = st.session_state.scoring_system.get_scorecard_interface(new_title)

    if score:
        score["weighted_score"] = st.session_state.scoring_system.calculate_weighted_score(score["scores"])
        st.session_state.all_scores.append(score)
        film['status'] = 'graded'
        st.success(f"Detailed score saved! Weighted score: {score['weighted_score']}/5")

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import re
import hashlib
import time
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs

# ------------------------------------------------
# 1.  GRACE CONSTANTS  (new)
# ------------------------------------------------
GRACE_FLOOR        = 2.8
GRACE_CULTURE      = 0.45
GRACE_ORAL         = 0.35
GRACE_COMMUNITY    = 0.30
GRACE_INGENUITY    = 0.40
GRACE_RELEVANCE    = 0.25

# ------------------------------------------------
# 2.  Helper for oral/community phrases
# ------------------------------------------------

def anyphrase(text, phrases):
    """Return True if any phrase is in the text."""
    text = text.lower()
    return any(p.lower() in text for p in phrases)
# --- Utility function to get video ID safely ---
def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    return None

# --- Initialize OpenAI client ---
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 CSV Movie Reviews", "🎥 YouTube Film Analysis"])

# --------------------------
# TAB 2: YouTube Film Review
# --------------------------
with tab2:
    st.header("🎥 YouTube Film Analysis")
    st.caption("Analyze YouTube films using Dan Harmon's Story Circle + Joseph Campbell's Hero’s Journey")

    youtube_url = st.text_input("Paste a YouTube video URL to analyze:")

    if youtube_url:
        video_id = get_video_id(youtube_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL")
        else:
            try:
                yt = YouTube(youtube_url)
                st.video(youtube_url)
                st.markdown(f"**🎞️ Title:** {yt.title}")
                st.markdown(f"**📅 Published:** {yt.publish_date}")
                st.markdown(f"**🕒 Length:** {yt.length // 60} minutes")

                # --- Attempt to retrieve transcript ---
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = " ".join([seg["text"] for seg in transcript_list])
                    st.success("✅ Transcript retrieved successfully!")
                except (TranscriptsDisabled, NoTranscriptFound):
                    st.warning("⚠️ No transcript available. Using title + description.")
                    transcript_text = yt.title + " " + (yt.description or "")

                # --- AI Prompt ---
                prompt = f"""
You are a professional film festival juror. Review and score the film below using objective and metadata-informed criteria.

Film Metadata:
{film_metadata}

Public Reception:
{audience_reception}

Transcript Excerpt:
{transcript_text[:2500]}

Scoring Criteria (1–5 scale):
• Storytelling (35%) — narrative structure, character depth, emotional arc.
• Technical/Directing (25%) — cinematography, editing, pacing, sound.
• Artistic Vision (15%) — originality, aesthetic coherence, creative risk.
• Cultural Fidelity (15%) — authenticity, representation, context.
• Social Impact (10%) — message, relevance, influence.

Please output:
1️⃣ A concise synopsis.
2️⃣ Strengths and weaknesses.
3️⃣ Numeric scores per category.
4️⃣ Weighted final score (out of 5.00).
5️⃣ 2–3 Jury Notes referencing scenes or timestamps.
"""


                # --- Call OpenAI ---
                with st.spinner("🤖 AI reviewing in progress..."):
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                    )

                st.subheader("🧾 AI Review Summary")
                st.markdown(response.choices[0].message.content)

            except Exception as e:
                st.error(f"❌ Error processing YouTube video: {e}")
    else:
        st.info("Please enter a valid YouTube link to begin.")

# --------------------------
# Main App
# --------------------------
def main():
    st.set_page_config(page_title="FlickFinder", page_icon="🎬", layout="wide")

    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.error(f"OpenAI client initialization failed: {e}")
        client = None

    # Initialize scoring system
    if "scoring_system" not in st.session_state:
        st.session_state.scoring_system = ScoringSystem()
    if "all_scores" not in st.session_state:
        st.session_state.all_scores = []

    # Sidebar navigation
    with st.sidebar:
        st.header("🎬 FlickFinder")
        st.markdown("---")
        page_option = st.radio(
            "Navigate to:",
            ["🏠 Home", "🔗 FilmFreeway", "🎯 Score Films", "📊 Export", "📚 Saved Projects"]
        )

    # Page routing
    if page_option == "🏠 Home":
        home_interface()
    elif page_option == "🔗 FilmFreeway":
        if client:
            filmfreeway_interface(client)
        else:
            st.error("OpenAI client not initialized. Check your API key.")
    elif page_option == "🎯 Score Films":
        scoring_interface()
    elif page_option == "📊 Export":
        export_interface()
    elif page_option == "📚 Saved Projects":
        display_saved_projects()

# --------------------------
# Home page
# --------------------------
def home_interface():
    st.title("Welcome to FlickFinder 🎬")
    st.markdown("### Professional Film Evaluation Platform")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔗 FilmFreeway Integration**")
        st.markdown("Import and analyze projects directly from FilmFreeway")
    with col2:
        st.markdown("**🎯 Smart Scoring**")
        st.markdown("Weighted scoring with bias checks and qualitative feedback")
    with col3:
        st.markdown("**📊 Export Tools**")
        st.markdown("Generate PDF reports and CSV exports for festival management")
    st.markdown("---")
    st.markdown("Get started by importing films from FilmFreeway or scoring existing projects.")

# --------------------------
# Scoring interface
# --------------------------
def scoring_interface():
    st.header("🎯 Film Scoring")
    films_to_score = st.session_state.get("filmfreeway_projects", [])
    if not films_to_score:
        st.info("📥 No films available for scoring. Import some films from FilmFreeway section first.")
        st.markdown("---")
        st.subheader("Or add a film manually for testing:")
        manual_film = st.text_input("Film title for manual scoring:")
        if manual_film and st.button("Add for Scoring"):
            if "filmfreeway_projects" not in st.session_state:
                st.session_state.filmfreeway_projects = []
            st.session_state.filmfreeway_projects.append({
                "title": manual_film,
                "platform": "Manual Entry",
                "url": "N/A"
            })
            st.rerun()
        return

    film_titles = [project.get("title", f"Project {i+1}") for i, project in enumerate(films_to_score)]
    selected_film = st.selectbox("Select film to score:", film_titles)

    if selected_film:
        score_result = st.session_state.scoring_system.get_scorecard_interface(selected_film)
        if score_result:
            score_result["weighted_score"] = st.session_state.scoring_system.calculate_weighted_score(score_result["scores"])
            st.session_state.all_scores.append(score_result)
            st.success(f"✅ Score saved! Weighted score: {score_result['weighted_score']}/5")

            with st.expander("📊 View Score Summary"):
                col1, col2, col3, col4, col5 = st.columns(5)
                scores = score_result["scores"]
                with col1: st.metric("Storytelling", f"{scores['storytelling']}/5")
                with col2: st.metric("Technical", f"{scores['technical_directing']}/5")
                with col3: st.metric("Artistic", f"{scores['artistic_vision']}/5")
                with col4: st.metric("Cultural", f"{scores['cultural_fidelity']}/5")
                with col5: st.metric("Final Score", f"{score_result['weighted_score']}/5")

# --------------------------
# Run app
# --------------------------
if __name__ == "__main__":
    main()

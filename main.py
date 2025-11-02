import streamlit as st
from filmfreeway_analyzer import filmfreeway_interface, display_saved_projects
from scoring_system import ScoringSystem
from export_system import export_interface
from openai import OpenAI
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs
import pandas as pd

# --------------------------
# Utility Functions
# --------------------------
def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    return None

def fetch_transcript(video_id, yt):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([seg["text"] for seg in transcript_list])
        st.success("✅ Transcript retrieved successfully!")
    except (TranscriptsDisabled, NoTranscriptFound):
        st.warning("⚠️ No transcript available. Using title + description.")
        transcript_text = yt.title + " " + (yt.description or "")
    return transcript_text

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
    st.caption("Upload CSV or FilmFreeway projects to analyze and score films")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
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
    st.caption("Analyze YouTube films using Dan Harmon's Story Circle + Hero’s Journey")

    youtube_url = st.text_input("Paste a YouTube video URL to analyze:")
    youtube_title = st.text_input("🎬 Enter or Edit Film Title for Scoring:")

    # Save title before running AI review
    if st.button("💾 Save Title for Scoring"):
        if youtube_url and youtube_title:
            store_film_for_scoring(youtube_title, youtube_url, "YouTube")
            st.success(f"✅ Saved '{youtube_title}' for scoring.")
        else:
            st.warning("Please enter both a valid YouTube URL and film title.")

    if youtube_url:
        video_id = get_video_id(youtube_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL")
        else:
            try:
                yt = YouTube(youtube_url)
                st.video(youtube_url)

                if not youtube_title:
                    youtube_title = yt.title
                    st.info(f"Auto-filled film title: {youtube_title}")

                st.markdown(f"**📅 Published:** {yt.publish_date}")
                st.markdown(f"**🕒 Length:** {yt.length // 60} minutes")

                transcript_text = fetch_transcript(video_id, yt)

                # Only analyze after the title is saved
                if st.button("🎬 Run AI Film Review"):
                    prompt = f"""
You are a professional film festival reviewer.
Analyze and score this film based on:
- Dan Harmon's 8-Step Story Circle (minimum 70–80% adherence)
- Joseph Campbell's Hero's Journey

Weighted Categories (Total 100 points):
- Storytelling 25%
- Technical 20%
- Directing 20%
- Cultural/Social Impact 20%
- Artistic Vision 15%

Film title: {youtube_title}
Transcript: {transcript_text if transcript_text else '[No transcript available]'}
Provide:
1. Short synopsis
2. Strengths and weaknesses
3. Weighted numeric scores for each category
4. Final total score /100
5. Jury summary paragraph.
"""
                    with st.spinner("🤖 AI reviewing in progress..."):
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                        )

                    st.subheader("🧾 AI Review Summary")
                    st.markdown(response.choices[0].message.content)

            except Exception as e:
                st.error(f"❌ Error processing YouTube video: {e}")

# --------------------------
# Main App
# --------------------------
def main():
    st.set_page_config(page_title="FlickFinder", page_icon="🎬", layout="wide")

    if "scoring_system" not in st.session_state:
        st.session_state.scoring_system = ScoringSystem()
    if "all_scores" not in st.session_state:
        st.session_state.all_scores = []

    with st.sidebar:
        st.header("🎬 FlickFinder")
        st.markdown("---")
        page_option = st.radio(
            "Navigate to:",
            ["🏠 Home", "🔗 FilmFreeway", "🎯 Score Films", "📊 Export", "📚 Saved Projects"]
        )

    if page_option == "🏠 Home":
        home_interface()
    elif page_option == "🔗 FilmFreeway":
        filmfreeway_interface(client)
    elif page_option == "🎯 Score Films":
        scoring_interface()
    elif page_option == "📊 Export":
        export_interface()
    elif page_option == "📚 Saved Projects":
        display_saved_projects()

# --------------------------
# Home Page
# --------------------------
def home_interface():
    st.title("Welcome to FlickFinder 🎬")
    st.markdown("### Professional Film Evaluation Platform")
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("**🔗 FilmFreeway Integration**")
    with col2: st.markdown("**🎯 Smart Scoring**")
    with col3: st.markdown("**📊 Export Tools**")
    st.markdown("---")
    st.markdown("Get started by importing films or analyzing YouTube videos.")

# --------------------------
# Scoring Page
# --------------------------
def scoring_interface():
    st.header("🎯 Film Scoring")
    films_to_score = st.session_state.get("films_to_score", [])

    if not films_to_score:
        st.info("📥 No films available for scoring. Analyze a YouTube video or import from FilmFreeway first.")
        return

    film_titles = [f["title"] for f in films_to_score]
    selected_film = st.selectbox("Select film to score:", film_titles)

    if selected_film:
        film_obj = next(f for f in films_to_score if f["title"] == selected_film)
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
                with col5: st.metric("Final", f"{score_result['weighted_score']}/5")

# --------------------------
# Run App
# --------------------------
if __name__ == "__main__":
    main()

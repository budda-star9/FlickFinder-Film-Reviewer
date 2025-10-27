import streamlit as st
import pandas as pd
import json
import datetime
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials

# === PAGE CONFIG ===
st.set_page_config(page_title="🎬 FlickFinder: Film Evaluator", layout="wide")
st.title("🎞️ FlickFinder — AI Film Review & Story Analysis")

# === CLIENT SETUP ===
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === GOOGLE SHEETS CONNECTION ===
def get_google_sheet():
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    client = gspread.authorize(creds)
    sheet = client.open_by_url(st.secrets["GOOGLE_SHEETS_URL"]).sheet1
    return sheet

# === LOAD MOVIES ===
@st.cache_data
def load_movies():
    return pd.read_csv("movies.csv")

df = load_movies()

# === SIDEBAR ===
st.sidebar.header("🎬 Film Filters")
genres = st.sidebar.multiselect("Select Genres", df["genres"].unique())
language = st.sidebar.selectbox("Select Language", ["All"] + df["original_language"].unique().tolist())

filtered = df
if genres:
    filtered = filtered[filtered["genres"].isin(genres)]
if language != "All":
    filtered = filtered[filtered["original_language"] == language]

selected_movie = st.sidebar.selectbox("Select a Film", filtered["title"].tolist())
movie = df[df["title"] == selected_movie].iloc[0]

# === DISPLAY MOVIE INFO ===
st.subheader(f"{movie['title']} ({movie['release_date']})")
st.markdown(f"**Genres:** {movie['genres']} | **Rating:** {movie['vote_average']}")
if "filmfreeway_url" in movie:
    st.markdown(f"[🎥 View on FilmFreeway]({movie['filmfreeway_url']})", unsafe_allow_html=True)
st.markdown(f"**Summary:** {movie.get('story_summary', 'No summary available.')}")

reviewer_name = st.text_input("Reviewer Name")
extra_notes = st.text_area("Additional Reviewer Notes (optional)")

# === AI EVALUATION ===
if st.button("🧠 Run AI Evaluation"):
    with st.spinner("Evaluating film using Dan Harmon's Story Circle & Joseph Campbell’s Hero’s Journey..."):

        prompt = f"""
You are a professional film reviewer and narrative structure analyst.

Evaluate this film according to:
1️⃣ Dan Harmon's 8-Step Story Circle (each step 0–10)
2️⃣ Joseph Campbell’s 12-Stage Hero’s Journey (each stage 0–10)
3️⃣ Weighted Review Metrics:
   - Storytelling 25%
   - Technical 20%
   - Directing 20%
   - Cultural/Social Impact 20%
   - Artistic Vision 15%

The film must fulfill at least 70–80% of both narrative frameworks to meet standard.

Film Summary:
{movie.get('story_summary', '')}

Return ONLY JSON:
{{
  "StoryCircle": {{
    "You": 0-10, "Need": 0-10, "Go": 0-10, "Search": 0-10,
    "Find": 0-10, "Take": 0-10, "Return": 0-10, "Change": 0-10
  }},
  "HerosJourney": {{
    "OrdinaryWorld": 0-10, "CallToAdventure": 0-10, "Refusal": 0-10, "Mentor": 0-10,
    "CrossThreshold": 0-10, "TestsAlliesEnemies": 0-10, "ApproachCave": 0-10,
    "Ordeal": 0-10, "Reward": 0-10, "RoadBack": 0-10, "Resurrection": 0-10, "ReturnElixir": 0-10
  }},
  "Categories": {{
    "Storytelling": {{"score": 0-10, "comment": "short qualitative comment"}},
    "Technical": {{"score": 0-10, "comment": "short qualitative comment"}},
    "Directing": {{"score": 0-10, "comment": "short qualitative comment"}},
    "CulturalSocialImpact": {{"score": 0-10, "comment": "short qualitative comment"}},
    "ArtisticVision": {{"score": 0-10, "comment": "short qualitative comment"}}
  }},
  "TotalScore": 0-100,
  "MeetsStandard": "Yes" or "No",
  "Feedback": "3–4 sentence feedback combining story structure and craft."
}}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise film festival evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
        )

        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            st.error("⚠️ Could not parse AI response. Try again.")
            st.stop()

        st.success("✅ Evaluation complete!")

        # === DISPLAY STORY CIRCLE ===
        st.markdown("## 🧩 Dan Harmon's Story Circle")
        story_df = pd.DataFrame(result["StoryCircle"].items(), columns=["Step", "Score (0–10)"])
        st.dataframe(story_df, hide_index=True)

        # === DISPLAY HERO’S JOURNEY ===
        st.markdown("## 🧙‍♂️ Joseph Campbell’s Hero’s Journey")
        hero_df = pd.DataFrame(result["HerosJourney"].items(), columns=["Stage", "Score (0–10)"])
        st.dataframe(hero_df, hide_index=True)

        # === DISPLAY CATEGORIES ===
        st.markdown("## 🎥 Evaluation Categories (Weighted)")
        cat_data = []
        for k, v in result["Categories"].items():
            cat_data.append([k.replace("CulturalSocialImpact", "Cultural/Social Impact"), v["score"], v["comment"]])
        cat_df = pd.DataFrame(cat_data, columns=["Category", "Score (0–10)", "Comment"])
        st.dataframe(cat_df, hide_index=True)

        total_score = result["TotalScore"]
        status = "✅ Meets Standard" if total_score >= 70 else "⚠️ Below Standard"

        col1, col2 = st.columns(2)
        col1.metric("🎯 Weighted Total Score", f"{total_score:.1f} / 100")
        col2.metric("📊 Evaluation", status)
        st.markdown("### 💬 Feedback")
        st.write(result["Feedback"])

        # === SAVE DATA ===
        st.markdown("---")
        st.markdown("### 💾 Save This Review")

        save_data = {
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Reviewer": reviewer_name,
            "Film Title": movie["title"],
            "Total Score": total_score,
            "Meets Standard": result["MeetsStandard"],
            "Feedback": result["Feedback"],
            "Notes": extra_notes,
            **{f"StoryCircle_{k}": v for k, v in result["StoryCircle"].items()},
            **{f"HerosJourney_{k}": v for k, v in result["HerosJourney"].items()},
            **{f"{k}_Score": v["score"] for k, v in result["Categories"].items()},
            **{f"{k}_Comment": v["comment"] for k, v in result["Categories"].items()},
        }

        if st.button("📤 Save to Google Sheets"):
            try:
                sheet = get_google_sheet()
                sheet.append_row(list(save_data.values()))
                st.success("✅ Review saved to Google Sheets!")
            except Exception as e:
                st.error(f"⚠️ Could not save to Google Sheets: {e}")

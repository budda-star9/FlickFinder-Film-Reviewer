import streamlit as st
import pandas as pd
from openai import OpenAI

# --- SETUP ---
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", "your_api_key_here"))

# --- LOAD MOVIES ---
@st.cache_data
def load_movies():
    try:
        df = pd.read_csv("movies.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ movies.csv not found! Please add the file to your project folder.")
        return pd.DataFrame()

# --- STORY EVALUATION FUNCTION ---
def evaluate_story(story_text):
    prompt = f"""
    Evaluate this story using Dan Harmon's 8-Step Story Circle:
    1. You — A character is in a zone of comfort.
    2. Need — But they want something.
    3. Go — They enter an unfamiliar situation.
    4. Search — Adapt to it.
    5. Find — Get what they wanted.
    6. Take — Pay a heavy price for it.
    7. Return — Then return to their familiar situation.
    8. Change — Having changed.

    Identify which steps are clearly present (Yes/No). Calculate the completion
    percentage (steps_present ÷ 8). If completion ≥ 0.7, mark as "Complete."
    If < 0.7, list missing elements and suggestions to improve.

    Return results as JSON with keys:
    steps_present, completion_score, result, feedback.

    Story:
    {story_text}
    """

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "system", "content": "You are a story structure evaluator."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# --- STREAMLIT UI ---
st.title("🎬 FlickFinder: Story Evaluator & Movie Browser")
st.write("Upload your stories and check if they follow Dan Harmon's 8-Step Story Circle (70–80% minimum).")

tab1, tab2 = st.tabs(["📚 Movie Browser", "✍️ Story Submission"])

# --- TAB 1: MOVIE BROWSER ---
with tab1:
    st.header("Browse Movies")
    df = load_movies()
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("Upload a `movies.csv` file with columns like title, genres, language, etc.")

# --- TAB 2: STORY SUBMISSION ---
with tab2:
    st.header("Submit Your Story for Evaluation")
    story_text = st.text_area("Paste your story text here:", height=300)
    if st.button("Evaluate Story"):
        if story_text.strip():
            with st.spinner("Analyzing story structure..."):
                result = evaluate_story(story_text)
            st.success("✅ Evaluation Complete!")
            st.text_area("AI Feedback", result, height=400)
        else:
            st.warning("Please enter a story first.")



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load and prepare data
movies = pd.read_csv("movies.csv", encoding="utf-8-sig")
movies = movies.rename(columns={"id": "movieId"})

# Clean up data
movies["genres"] = movies["genres"].fillna("").apply(lambda x: " ".join(x.replace("|", " ").replace(",", " ").split()))
movies["release_year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
movies["vote_average"] = movies["vote_average"].fillna(0)

# Language code to full name mapping
language_map = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "fr": "French",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi"
}
movies["language_full"] = movies["original_language"].map(language_map)


# Sidebar filters
st.sidebar.header("🎛️ Filters")

# Genres
all_genres = sorted(set(g for s in movies["genres"].dropna() for g in s.split()))
selected_genres = st.sidebar.multiselect("🎭 Select Genre(s)", all_genres)

# Languages
languages = sorted(movies["language_full"].dropna().unique())
selected_language = st.sidebar.selectbox("🌐 Language", ["Any"] + languages)

# Rating filter
min_rating = st.sidebar.slider("⭐ Minimum Rating", 0.0, 10.0, 5.0, step=0.1)

# Release year filter
min_year = int(movies["release_year"].dropna().min())
max_year = int(movies["release_year"].dropna().max())
selected_year_range = st.sidebar.slider("📅 Release Year Range", min_year, max_year, (min_year, max_year))

# App title
st.title("🎬 Real-Time Movie Recommendation System")

# Apply filters
filtered_movies = movies.copy()

if selected_language != "Any":
    filtered_movies = filtered_movies[filtered_movies["language_full"] == selected_language]

if selected_genres:
    filtered_movies = filtered_movies[filtered_movies["genres"].apply(
        lambda g: all(genre in g.split() for genre in selected_genres))]

filtered_movies = filtered_movies[
    (filtered_movies["vote_average"] >= min_rating) &
    (filtered_movies["release_year"].notna()) &
    (filtered_movies["release_year"] >= selected_year_range[0]) &
    (filtered_movies["release_year"] <= selected_year_range[1])
]

filtered_movies = filtered_movies.reset_index(drop=True)

# Show filtered list
st.markdown(f"🎯 **{len(filtered_movies)} movies found** matching your filters.")

if filtered_movies.empty:
    st.warning("No movies found with the selected filters.")
else:
    st.markdown("### 📋 Filtered Movies")
    st.dataframe(filtered_movies[["title", "release_year", "vote_average"]])

    #st.dataframe(filtered_movies[["title", "release_year", "vote_average"]])

    selected_movie = st.selectbox("🎞️ Choose a Movie for Recommendations", filtered_movies["title"].tolist())

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(filtered_movies["genres"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(filtered_movies.index, index=filtered_movies["title"]).drop_duplicates()

    def get_recommendations(title, cosine_sim=cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return filtered_movies.iloc[movie_indices]

    if selected_movie:
        recommendations = get_recommendations(selected_movie)
        st.markdown("### 🧠 Recommended Movies")
        for _, row in recommendations.iterrows():
            st.markdown(f"**{row['title']}** ({int(row['release_year'])}) – ⭐ {row['vote_average']}")

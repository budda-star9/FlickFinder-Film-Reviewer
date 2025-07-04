# FlickFinder
A smart movie filtering tool with genre, language, and rating filters — built with Streamlit and scikit-learn.

This is a Streamlit-based Movie Recommendation System that lets users filter movies by genres, language, rating, and release year — and then get smart recommendations based on content similarity (TF-IDF and cosine similarity).

---

## 🚀 Features

- 🎭 **Genre Selection** — Choose one or more genres to narrow results.
- 🌐 **Language Filter** — Filter by original movie language (e.g., English, Hindi, Telugu).
- ⭐ **Minimum Rating** — Set a threshold for vote average.
- 📅 **Release Year Range** — Choose the time window for movie releases.
- 🧠 **Smart Recommendations** — Content-based recommendations using TF-IDF on genres.
- ✅ Real-time filtering with interactive sidebar

---

## 📁 Project Structure

movie-recommender/
│
├── movies.csv # Movie metadata (required)
├── main.py # Streamlit app code
├── requirements.txt # Python dependencies
└── README.md # This file


> 💡 You must provide your own `movies.csv` file with columns like `title`, `genres`, `original_language`, `vote_average`, and `release_date`.

---

## ⚙️ Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender

2. Create a virtual environment (optional but recommended)

python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\activate          # On Windows

3. Install dependencies

pip install -r requirements.txt

▶️ Run the App

streamlit run main.py

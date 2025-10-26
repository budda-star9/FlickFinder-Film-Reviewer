#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

csv_path = Path("movies.csv")

# New movie data
new_movies = pd.DataFrame([
    {
        "title": "The Wanderer",
        "genres": "Drama",
        "original_language": "English",
        "vote_average": 8.1,
        "release_date": "2024-06-12",
        "filmfreeway_url": "https://filmfreeway.com/TheWanderer",
        "story_summary": "A lonely traveler seeks redemption in a foreign land, facing the ghosts of his past."
    },
    {
        "title": "Skyward Bound",
        "genres": "Adventure",
        "original_language": "English",
        "vote_average": 7.5,
        "release_date": "2023-09-10",
        "filmfreeway_url": "https://filmfreeway.com/SkywardBound",
        "story_summary": "A pilot risks everything to discover freedom above the clouds."
    }
])

# Update or create CSV
if csv_path.exists():
    existing = pd.read_csv(csv_path)
    updated = pd.concat([existing, new_movies]).drop_duplicates(subset=["title"], keep="last")
else:
    updated = new_movies

updated.to_csv(csv_path, index=False)
print(f"✅ movies.csv has been updated at {csv_path.resolve()}")

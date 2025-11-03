import streamlit as st
import random

class ScoringSystem:
    def __init__(self):
        """Initialize scoring system."""
        self.categories = {
            "storytelling": 0.25,       # 25%
            "technical_directing": 0.20, # 20%
            "artistic_vision": 0.15,    # 15%
            "cultural_fidelity": 0.20,  # 20%
            "social_impact": 0.10       # 10%
        }

    def generate_random_score(self):
        """Generate a realistic random score between 2.5 and 5.0 for demonstration."""
        return round(random.uniform(2.5, 5.0), 2)

    def get_scorecard_interface(self, film_title):
        """Create a Streamlit interface for scoring a single film."""
        st.subheader(f"Scoring: {film_title}")
        scores = {}
        for cat in self.categories:
            scores[cat] = st.slider(
                f"{cat.replace('_', ' ').title()} (0-5)",
                min_value=0.0,
                max_value=5.0,
                value=self.generate_random_score(),
                step=0.1
            )

        return {"title": film_title, "scores": scores}

    def calculate_weighted_score(self, scores):
        """Calculate weighted total score out of 5."""
        weighted_sum = sum(scores[cat] * weight for cat, weight in self.categories.items())
        return round(weighted_sum, 2)

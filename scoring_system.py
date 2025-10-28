"""
Scoring system for FlickFinder with adjustable weights and bias checks
"""

import streamlit as st
import pandas as pd

class ScoringSystem:
    def __init__(self):
        self.default_weights = {
            'storytelling': 0.35,
            'technical_directing': 0.25,
            'artistic_vision': 0.15,
            'cultural_fidelity': 0.15,
            'social_impact': 0.10
        }
        
        self.score_descriptors = {
            1: "Poor - Fundamental issues",
            2: "Below Average - Significant room for improvement", 
            3: "Average - Meets basic expectations",
            4: "Good - Exceeds expectations in some areas",
            5: "Excellent - Outstanding achievement"
        }
    
    def get_scorecard_interface(self, film_title):
        """Render the scoring interface"""
        st.subheader(f"🎯 Scoring: {film_title}")
        
        scores = {}
        notes = ""
        
        # Scoring criteria
        with st.form(f"score_form_{film_title}"):
            st.markdown("### Scoring Criteria")
            
            # Storytelling (35%)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Storytelling (35%)** - Narrative structure, character development, emotional impact")
            with col2:
                scores['storytelling'] = st.slider("Storytelling Score", 1.0, 5.0, 3.0, 0.5, 
                                                 key=f"story_{film_title}", label_visibility="collapsed")
            
            # Technical/Directing (25%)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Technical/Directing (25%)** - Cinematography, editing, sound, pacing")
            with col2:
                scores['technical_directing'] = st.slider("Technical Score", 1.0, 5.0, 3.0, 0.5,
                                                        key=f"tech_{film_title}", label_visibility="collapsed")
            
            # Artistic Vision (15%)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Artistic Vision (15%)** - Originality, creative approach, visual style")
            with col2:
                scores['artistic_vision'] = st.slider("Artistic Score", 1.0, 5.0, 3.0, 0.5,
                                                    key=f"art_{film_title}", label_visibility="collapsed")
            
            # Cultural Fidelity (15%)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Cultural Fidelity (15%)** - Authentic representation, cultural context")
            with col2:
                scores['cultural_fidelity'] = st.slider("Cultural Score", 1.0, 5.0, 3.0, 0.5,
                                                       key=f"culture_{film_title}", label_visibility="collapsed")
            
            # Social Impact (10%)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Social Impact (10%)** - Message, relevance, potential impact")
            with col2:
                scores['social_impact'] = st.slider("Impact Score", 1.0, 5.0, 3.0, 0.5,
                                                  key=f"impact_{film_title}", label_visibility="collapsed")
            
            # Qualitative notes
            st.markdown("### Qualitative Assessment")
            st.markdown("**Please provide 2-3 sentences with specific examples:**")
            notes = st.text_area("Notes (include timestamped references if possible)", 
                               placeholder="e.g., '02:14-03:20: The cinematography in this scene effectively establishes mood...'",
                               key=f"notes_{film_title}")
            
            # Bias check
            st.markdown("### Bias & Ethics Check")
            bias_reflection = st.text_area("Reflect on representation and potential stereotypes:",
                                        placeholder="Consider: Are there any unconscious biases affecting your scoring? How is diversity represented?")
            
            # Hero's Journey toggle
            heros_journey = st.checkbox("Apply Hero's Journey framework", value=True,
                                      help="Toggle OFF for non-Western narrative structures")
            
            # Conflict of interest
            conflict = st.checkbox("I have a conflict of interest with this film", 
                                 help="If checked, your comments will be anonymized")
            
            submitted = st.form_submit_button("💾 Save Score")
            
            if submitted:
                if len(notes.strip().split()) < 10:  # Roughly 2-3 sentences
                    st.error("Please provide at least 2-3 sentences of qualitative notes")
                    return None
                
                return {
                    'scores': scores,
                    'notes': notes,
                    'bias_reflection': bias_reflection,
                    'heros_journey': heros_journey,
                    'conflict_of_interest': conflict,
                    'film_title': film_title
                }
        
        return None
    
    def calculate_weighted_score(self, scores, weights=None):
        """Calculate weighted final score"""
        if weights is None:
            weights = self.default_weights
        
        weighted_total = 0
        for criterion, score in scores.items():
            weighted_total += score * weights.get(criterion, 0)
        
        return round(weighted_total, 2)

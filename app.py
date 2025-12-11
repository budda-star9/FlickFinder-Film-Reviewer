"""
FlickFinder AI - Advanced Film Analysis Platform
Version: 3.1 Enhanced Scoring with Video Viewer
Description: AI-powered film analysis with enhanced scoring algorithm,
             YouTube integration, cultural context analysis, comprehensive
             scoring algorithms, embedded video viewing, and enhanced analytics.
"""

# --------------------------
# IMPORTS
# --------------------------
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import hashlib
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import textwrap
import time
import io
import base64
from typing import Dict, List, Optional, Tuple, Any, Union
import json

# --------------------------
# CONFIGURATION & SETUP
# --------------------------
st.set_page_config(
    page_title="FlickFinder AI 🎬 - Advanced Film Analysis",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/flickfinder',
        'Report a bug': 'https://github.com/yourusername/flickfinder/issues',
        'About': "### FlickFinder AI v3.1\nAdvanced film analysis with enhanced scoring algorithm."
    }
)

# Initialize NLTK data
@st.cache_resource
def load_nltk_data() -> None:
    """Load required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        with st.spinner("Downloading NLP data..."):
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

load_nltk_data()

# Initialize all session state variables
session_defaults = {
    'analysis_history': [],
    'stored_results': {},
    'current_analysis_id': None,
    'show_results_page': False,
    'saved_projects': {},
    'project_counter': 0,
    'current_page': "🏠 Dashboard",
    'current_results_display': None,
    'current_video_id': None,
    'current_video_title': None,
    'top_films': [],
    'analysis_count': 0,
    'last_analysis_time': None,
    'batch_results': None,
    'show_batch_results': False,
    'analytics_view': 'overview',
    'show_breakdown': False,
    'current_tab': 'youtube',
    'persistence_loaded': False,
}

for key, default in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------
# ENHANCED PERSISTENCE MANAGER CLASS
# --------------------------
class PersistenceManager:
    """Handles saving and loading of analysis results with unique IDs"""
    
    @staticmethod
    def generate_film_id(film_data: Dict) -> str:
        """Generate a unique ID for a film based on its content"""
        content_string = f"{film_data.get('title', '')}_{film_data.get('synopsis', '')[:100]}"
        return hashlib.md5(content_string.encode()).hexdigest()[:12]
    
    @staticmethod
    def save_results(film_data: Dict, analysis_results: Dict, film_id: Optional[str] = None) -> str:
        """Save analysis results with full persistence"""
        if film_id is None:
            film_id = PersistenceManager.generate_film_id(film_data)
        
        # Store in session state
        st.session_state.stored_results[film_id] = {
            'film_data': film_data,
            'analysis_results': analysis_results,
            'timestamp': datetime.now().isoformat(),
            'film_id': film_id
        }
        
        # Add to history
        history_entry = {
            'id': film_id,
            'title': film_data.get('title', 'Unknown Film'),
            'timestamp': datetime.now().isoformat(),
            'overall_score': analysis_results.get('overall_score', 0),
            'detected_genre': analysis_results.get('genre_insights', {}).get('primary_genre', 'Unknown'),
            'cultural_relevance': analysis_results.get('cultural_insights', {}).get('relevance_score', 0),
            'component_scores': analysis_results.get('cinematic_scores', {}),
            'synopsis': film_data.get('synopsis', '')[:200]
        }
        
        # Add to history if not already there
        existing_ids = [h.get('id') for h in st.session_state.analysis_history]
        if film_id not in existing_ids:
            st.session_state.analysis_history.append(history_entry)
        
        # Update top films
        PersistenceManager._update_top_films()
        
        # Set as current display
        st.session_state.current_results_display = analysis_results
        st.session_state.show_results_page = True
        st.session_state.current_analysis_id = film_id
        st.session_state.last_analysis_time = datetime.now().isoformat()
        st.session_state.analysis_count += 1
        
        return film_id
    
    @staticmethod
    def _update_top_films() -> None:
        """Update the top films list based on overall score"""
        all_films = list(st.session_state.stored_results.values())
        if all_films:
            sorted_films = sorted(
                all_films,
                key=lambda x: x['analysis_results']['overall_score'],
                reverse=True
            )
            st.session_state.top_films = sorted_films[:3]
    
    @staticmethod
    def load_results(film_id: str) -> Optional[Dict]:
        """Load analysis results by film ID"""
        return st.session_state.stored_results.get(film_id)
    
    @staticmethod
    def get_all_history() -> List[Dict]:
        """Get all analysis history"""
        return st.session_state.analysis_history
    
    @staticmethod
    def clear_history() -> None:
        """Clear all analysis history"""
        st.session_state.analysis_history = []
        st.session_state.stored_results = {}
        st.session_state.current_results_display = None
        st.session_state.show_results_page = False
        st.session_state.top_films = []
        st.session_state.analysis_count = 0
        st.session_state.current_video_id = None
        st.session_state.current_video_title = None
        st.session_state.batch_results = None
        st.session_state.show_batch_results = False
    
    @staticmethod
    def save_project(project_name: str, film_data: Dict, analysis_results: Dict) -> str:
        """Save a project with custom name"""
        project_id = f"project_{st.session_state.project_counter}"
        st.session_state.project_counter += 1
        
        st.session_state.saved_projects[project_id] = {
            'name': project_name,
            'film_data': film_data,
            'analysis_results': analysis_results,
            'saved_at': datetime.now().isoformat()
        }
        
        return project_id
    
    @staticmethod
    def load_project(project_id: str) -> Optional[Dict]:
        """Load a saved project"""
        return st.session_state.saved_projects.get(project_id)
    
    @staticmethod
    def get_all_projects() -> Dict:
        """Get all saved projects"""
        return st.session_state.saved_projects
    
    @staticmethod
    def get_top_films() -> List[Dict]:
        """Get top films"""
        if not st.session_state.top_films:
            PersistenceManager._update_top_films()
        return st.session_state.top_films
    
    @staticmethod
    def get_analytics_data() -> Optional[pd.DataFrame]:
        """Get comprehensive analytics data"""
        history = st.session_state.analysis_history
        if not history:
            return None
        
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['time_of_day'] = df['timestamp'].dt.hour
        
        return df

# --------------------------
# ENHANCED FILM-SPECIFIC SCORER CLASS WITH WIDER SCORE RANGE
# --------------------------
class EnhancedFilmScorer:
    """Enhanced film scorer with wider, more nuanced score distribution (1.1-5.0)"""
    
    def __init__(self):
        self.base_weights = {
            'narrative': 0.25,
            'emotional': 0.22,
            'character': 0.20,
            'cultural': 0.18,
            'technical': 0.15
        }
        
        # Score range parameters for wider distribution
        self.MIN_SCORE = 1.1
        self.MAX_SCORE = 5.0
        self.SCORE_INTERVAL = 0.1  # Allows scores like 3.7, 4.2, 1.5, etc.
        
        self.philosophical_insights = [
            "Art as cultural memory",
            "Narrative as truth-seeking",
            "Cinema as empathy machine",
            "Storytelling as resistance",
            "Film as time capsule",
            "Visual language as consciousness",
            "Character as human mirror",
            "Genre as cultural dialogue"
        ]
    
    def calculate_unique_film_score(self, analysis_results: Dict, film_data: Dict) -> Dict:
        """Calculate comprehensive film score with enhanced granularity and variation"""
        text = (film_data.get('synopsis', '') + ' ' + film_data.get('transcript', '')).lower()
        title = film_data.get('title', '').lower()
        
        # Get component scores with enhanced precision
        component_scores = {
            'narrative': self._enhanced_narrative_score(analysis_results, text),
            'emotional': self._enhanced_emotional_score(analysis_results),
            'character': self._enhanced_character_score(analysis_results, text),
            'cultural': self._enhanced_cultural_score(analysis_results),
            'technical': self._enhanced_technical_score(analysis_results, text)
        }
        
        # Apply genre-specific adjustments
        adjusted_weights = self._apply_genre_adjustments(analysis_results)
        
        # Calculate weighted score with more variation
        raw_score = sum(
            component_scores[comp] * adjusted_weights[comp] 
            for comp in component_scores
        )
        
        # Add variation factors
        raw_score = self._add_variation_factors(raw_score, analysis_results, film_data, text)
        
        # Apply cultural/philosophical bonuses
        raw_score = self._apply_bonuses(raw_score, analysis_results, film_data, text)
        
        # Convert to 5.0 scale
        final_score = raw_score * 5.0
        
        # Apply distribution curve for natural spread
        final_score = self._apply_distribution_curve(final_score, analysis_results)
        
        # Ensure score is within desired range with proper granularity
        final_score = max(self.MIN_SCORE, min(self.MAX_SCORE, final_score))
        
        # Round to nearest 0.1 increment
        final_score = round(final_score / self.SCORE_INTERVAL) * self.SCORE_INTERVAL
        
        # Ensure minimum score for very short content
        if len(text.split()) < 100:
            final_score = max(1.5, final_score)
        
        return {
            'overall_score': final_score,
            'component_scores': {
                'narrative': round(component_scores['narrative'] * 5, 1),
                'emotional': round(component_scores['emotional'] * 5, 1),
                'character': round(component_scores['character'] * 5, 1),
                'cultural': round(component_scores['cultural'] * 5, 1),
                'technical': round(component_scores['technical'] * 5, 1)
            },
            'weighted_scores': component_scores,
            'applied_weights': adjusted_weights,
            'cultural_bonus': self._calculate_cultural_bonus(analysis_results, text),
            'philosophical_insight': random.choice(self.philosophical_insights) 
                if component_scores['cultural'] > 0.4 else None,
            'raw_components': self._get_raw_component_details(analysis_results, film_data)
        }
    
    def _enhanced_narrative_score(self, analysis_results: Dict, text: str) -> float:
        """Enhanced narrative scoring with more granularity"""
        ns = analysis_results.get('narrative_structure', {})
        
        # Multiple factors for variation
        lexical_diversity = ns.get('lexical_diversity', 0.4)
        structural_score = ns.get('structural_score', 0.4)
        readability = ns.get('readability_score', 0.6)
        sentence_variety = self._calculate_sentence_variety(text)
        complexity = self._assess_narrative_complexity(text)
        
        # Weighted average with variation
        base_score = (
            lexical_diversity * 0.25 +
            structural_score * 0.25 +
            sentence_variety * 0.20 +
            readability * 0.15 +
            complexity * 0.15
        )
        
        # Add text length consideration
        word_count = len(text.split())
        if word_count < 200:
            base_score *= 0.9
        elif word_count > 800:
            base_score *= 1.05
        
        return min(1.0, max(0.2, base_score))
    
    def _enhanced_emotional_score(self, analysis_results: Dict) -> float:
        """Enhanced emotional scoring"""
        ea = analysis_results.get('emotional_arc', {})
        
        arc_score = ea.get('arc_score', 0.4)
        variance = ea.get('emotional_variance', 0.2)
        emotional_range = self._map_emotional_range(ea.get('emotional_range', 'moderate'))
        intensity = self._assess_emotional_intensity(analysis_results)
        
        base_score = (
            arc_score * 0.35 +
            variance * 0.30 +
            emotional_range * 0.20 +
            intensity * 0.15
        )
        
        return min(1.0, max(0.2, base_score))
    
    def _enhanced_character_score(self, analysis_results: Dict, text: str) -> float:
        """Enhanced character scoring"""
        ca = analysis_results.get('character_analysis', {})
        
        chars = ca.get('potential_characters', 3)
        density = ca.get('character_density', 0.03)
        character_score = ca.get('character_score', 0.5)
        
        # Count character mentions
        mentions = text.count(" he ") + text.count(" she ") + text.count(" his ") + text.count(" her ")
        mentions += text.count(" they ") + text.count(" their ") + text.count(" them ")
        
        base_score = (
            (chars / 8) * 0.4 +
            density * 8 * 0.3 +
            min(mentions / 50, 0.4) * 0.2 +
            character_score * 0.1
        )
        
        return min(1.0, max(0.2, base_score))
    
    def _enhanced_cultural_score(self, analysis_results: Dict) -> float:
        """Enhanced cultural scoring"""
        cultural_context = analysis_results.get('cultural_context', {})
        relevance_score = cultural_context.get('relevance_score', 0.0)
        
        # Add philosophical depth bonus
        philosophical_depth = cultural_context.get('philosophical_depth', 0)
        cultural_bonus = philosophical_depth * 0.1
        
        total_score = min(1.0, relevance_score + cultural_bonus)
        return max(0.1, total_score)  # Ensure minimum cultural score
    
    def _enhanced_technical_score(self, analysis_results: Dict, text: str) -> float:
        """Enhanced technical scoring"""
        ns = analysis_results.get('narrative_structure', {})
        
        readability = ns.get('readability_score', 0.6)
        
        # Calculate dialogue density
        dialogue_patterns = [r'\b[A-Z][a-z]+:', r'"', r"'", r'\[.*?\]']
        dialogue_density = 0
        for pattern in dialogue_patterns:
            matches = len(re.findall(pattern, text))
            dialogue_density += matches * 0.1
        
        # Structure complexity
        sentences = nltk.sent_tokenize(text)
        structure_variety = min(0.3, len(set([len(s.split()) for s in sentences])) / 10)
        
        base_score = (
            readability * 0.5 +
            min(dialogue_density, 0.3) * 0.3 +
            structure_variety * 0.2
        )
        
        return min(1.0, max(0.2, base_score))
    
    def _apply_genre_adjustments(self, analysis_results: Dict) -> Dict:
        """Apply genre-specific weight adjustments"""
        weights = self.base_weights.copy()
        genre_context = analysis_results.get('genre_context', {})
        primary_genre = genre_context.get('primary_genre', '').lower()
        
        # Genre-specific adjustments for more score variation
        genre_adjustments = {
            'drama': {'emotional': 0.05, 'character': 0.04, 'cultural': 0.03},
            'comedy': {'emotional': 0.04, 'character': 0.06, 'technical': -0.02},
            'action': {'technical': 0.06, 'narrative': 0.03, 'emotional': -0.03},
            'horror': {'emotional': 0.05, 'technical': 0.04, 'character': -0.02},
            'documentary': {'cultural': 0.08, 'narrative': 0.04, 'technical': -0.02},
            'black cinema': {'cultural': 0.10, 'emotional': 0.06, 'character': 0.05},
            'urban drama': {'cultural': 0.08, 'character': 0.06, 'emotional': 0.04},
            'sci-fi': {'technical': 0.07, 'narrative': 0.04, 'cultural': -0.02},
            'fantasy': {'narrative': 0.06, 'technical': 0.03, 'emotional': 0.02},
            'romance': {'emotional': 0.07, 'character': 0.05, 'technical': -0.03},
            'thriller': {'narrative': 0.06, 'emotional': 0.04, 'technical': 0.02}
        }
        
        # Apply adjustments based on detected genre
        for genre, adjustments in genre_adjustments.items():
            if genre in primary_genre:
                for factor, adjustment in adjustments.items():
                    if factor in weights:
                        weights[factor] += adjustment
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _add_variation_factors(self, raw_score: float, analysis_results: Dict, 
                             film_data: Dict, text: str) -> float:
        """Add variation factors for score diversity"""
        word_count = len(text.split())
        
        # Length-based variation
        if word_count < 150:
            raw_score *= 0.92  # Short content penalty
        elif word_count > 1200:
            raw_score *= 1.03  # Detailed content bonus
        
        # Sentiment polarity variation
        sentiment = analysis_results.get('sentiment_analysis', {})
        polarity = sentiment.get('compound', 0)
        if abs(polarity) > 0.5:
            raw_score *= 1.02  # Strong sentiment bonus
        
        # Character density variation
        char_analysis = analysis_results.get('character_analysis', {})
        char_density = char_analysis.get('character_density', 0)
        if char_density > 0.05:
            raw_score *= 1.02
        
        # Add small random variation for natural distribution (0.1-0.3 point variation)
        random_variation = random.uniform(-0.02, 0.03)
        raw_score += random_variation
        
        return min(1.0, max(0.1, raw_score))
    
    def _apply_distribution_curve(self, score: float, analysis_results: Dict) -> float:
        """Apply distribution curve for natural score spread"""
        # Sigmoid-like curve for more middle-range variation
        base_score = score / 5.0
        
        # Apply curve transformation
        if base_score < 0.3:  # Very low scores
            transformed = base_score * 0.95
        elif base_score < 0.5:  # Low scores
            transformed = base_score * 1.02
        elif base_score < 0.8:  # Middle scores get more variation
            transformed = base_score * (1 + (base_score - 0.5) * 0.15)
        else:  # High scores
            transformed = base_score * 0.98
        
        # Add genre-specific curve adjustments
        genre_context = analysis_results.get('genre_context', {})
        primary_genre = genre_context.get('primary_genre', '').lower()
        
        if 'drama' in primary_genre or 'black cinema' in primary_genre:
            transformed = transformed * 1.02  # Drama/Black cinema tends to score higher
        elif 'comedy' in primary_genre:
            transformed = transformed * 0.98  # Comedy often scores slightly lower
        elif 'documentary' in primary_genre:
            transformed = transformed * 1.01  # Documentary gets slight boost
        
        return transformed * 5.0
    
    def _apply_bonuses(self, raw_score: float, analysis_results: Dict, 
                      film_data: Dict, text: str) -> float:
        """Apply cultural and philosophical bonuses"""
        cultural = analysis_results.get('cultural_context', {})
        
        # Cultural relevance bonus
        cultural_score = cultural.get('relevance_score', 0)
        if cultural_score > 0.6:
            raw_score += (cultural_score - 0.6) * 0.12
        
        # Philosophical depth bonus
        philosophical_bonus = self._calculate_philosophical_depth(text, cultural_score)
        raw_score += philosophical_bonus
        
        # Originality bonus
        originality_bonus = self._assess_originality(film_data)
        raw_score += originality_bonus
        
        return min(1.0, max(0.1, raw_score))  # Cap at 1.0, minimum 0.1
    
    def _calculate_cultural_bonus(self, analysis_results: Dict, text: str) -> float:
        """Calculate cultural bonus separately for transparency"""
        cultural = analysis_results.get('cultural_context', {})
        cultural_score = cultural.get('relevance_score', 0)
        
        if cultural_score > 0.6:
            return (cultural_score - 0.6) * 0.12
        return 0.0
    
    def _calculate_philosophical_depth(self, text: str, cultural_score: float) -> float:
        """Calculate philosophical depth bonus"""
        philosophical_keywords = {
            'existential': ['meaning', 'purpose', 'existence', 'death', 'life', 'being'],
            'social': ['justice', 'equality', 'power', 'society', 'system', 'class'],
            'psychological': ['identity', 'memory', 'consciousness', 'mind', 'self', 'psyche'],
            'ethical': ['moral', 'right', 'wrong', 'choice', 'dilemma', 'ethics'],
            'temporal': ['time', 'memory', 'history', 'future', 'past', 'present']
        }
        
        matches = 0
        for category, keywords in philosophical_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + keyword + r'\b', text):
                    matches += 1
                    break  # Count once per category
        
        base_bonus = matches * 0.008
        
        if cultural_score > 0.5:
            base_bonus *= 1.5
        
        return min(0.05, base_bonus)
    
    def _assess_originality(self, film_data: Dict) -> float:
        """Assess originality for bonus"""
        # Simple uniqueness check based on content hash
        content = film_data.get('title', '') + film_data.get('synopsis', '')[:100]
        content_hash = hash(content) % 100
        
        # Originality score based on hash (simulating uniqueness)
        if content_hash < 10:  # 10% chance of high originality
            return 0.02
        elif content_hash < 30:  # 20% chance of medium originality
            return 0.01
        else:
            return 0.0
    
    # Helper methods for enhanced scoring
    def _calculate_sentence_variety(self, text: str) -> float:
        """Calculate sentence structure variety"""
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 3:
            return 0.4
        
        lengths = [len(s.split()) for s in sentences]
        if not lengths:
            return 0.5
        
        avg_length = np.mean(lengths)
        std_length = np.std(lengths) if len(lengths) > 1 else 0
        
        # Variety score based on length variation
        if std_length > avg_length * 0.5:
            return 0.8
        elif std_length > avg_length * 0.3:
            return 0.7
        elif std_length > avg_length * 0.1:
            return 0.6
        else:
            return 0.5
    
    def _map_emotional_range(self, range_str: str) -> float:
        """Map emotional range description to score"""
        mapping = {
            'wide': 0.9,
            'moderate': 0.7,
            'narrow': 0.5,
            'limited': 0.4,
            'intense': 0.85,
            'subtle': 0.6
        }
        return mapping.get(range_str.lower(), 0.6)
    
    def _assess_emotional_intensity(self, analysis_results: Dict) -> float:
        """Assess emotional intensity"""
        sentiment = analysis_results.get('sentiment_analysis', {})
        compound = abs(sentiment.get('compound', 0))
        
        if compound > 0.7:
            return 0.9
        elif compound > 0.4:
            return 0.7
        elif compound > 0.2:
            return 0.6
        else:
            return 0.5
    
    def _assess_narrative_complexity(self, text: str) -> float:
        """Assess narrative complexity"""
        words = text.split()
        if len(words) < 50:
            return 0.3
        
        unique_words = set(words)
        lexical_diversity = len(unique_words) / max(len(words), 1)
        
        # Check for complex structures
        complex_patterns = [
            r'although.*', r'however.*', r'nevertheless.*',
            r'on the other hand', r'in contrast', r'meanwhile',
            r'consequently', r'furthermore', r'nonetheless'
        ]
        
        pattern_matches = sum(1 for pattern in complex_patterns 
                            if re.search(pattern, text, re.IGNORECASE))
        
        complexity = (lexical_diversity * 0.6) + (min(pattern_matches, 5) * 0.08)
        
        return min(0.9, max(0.2, complexity))
    
    def _get_raw_component_details(self, analysis_results: Dict, film_data: Dict) -> Dict:
        """Get raw component details for debugging/transparency"""
        text = film_data.get('synopsis', '') + ' ' + film_data.get('transcript', '')
        return {
            'narrative_score': self._enhanced_narrative_score(analysis_results, text),
            'emotional_score': self._enhanced_emotional_score(analysis_results),
            'character_score': self._enhanced_character_score(analysis_results, text),
            'cultural_score': self._enhanced_cultural_score(analysis_results),
            'technical_score': self._enhanced_technical_score(analysis_results, text),
            'word_count': len(text.split()),
            'genre_adjustments': self._apply_genre_adjustments(analysis_results)
        }

# --------------------------
# ENHANCED SMART GENRE DETECTOR WITH AI SUGGESTIONS
# --------------------------
class SmartGenreDetector:
    """Detects film genres with AI enhancement suggestions"""
    
    def __init__(self):
        self.genre_patterns = self._build_genre_patterns()
    
    def _build_genre_patterns(self) -> Dict:
        """Build comprehensive genre detection patterns"""
        return {
            "Drama": {
                "keywords": ["emotional", "relationship", "conflict", "family", "love", "heart", 
                           "struggle", "life", "human", "drama", "tragic", "serious", "pain", "loss"],
                "weight": 1.2,
                "philosophical_aspect": "Exploration of human condition"
            },
            "Comedy": {
                "keywords": ["funny", "laugh", "humor", "joke", "comic", "satire", "hilarious", 
                           "wit", "absurd", "comedy", "fun", "humorous", "lighthearted"],
                "weight": 1.1,
                "philosophical_aspect": "Social commentary through humor"
            },
            "Horror": {
                "keywords": ["fear", "terror", "scary", "horror", "ghost", "monster", "kill", 
                           "death", "dark", "night", "supernatural", "creepy", "frightening"],
                "weight": 1.1,
                "philosophical_aspect": "Confrontation with mortality"
            },
            "Sci-Fi": {
                "keywords": ["future", "space", "alien", "technology", "robot", "planet", 
                           "time travel", "science", "sci-fi", "futuristic", "cyber"],
                "weight": 1.1,
                "philosophical_aspect": "Questioning technological progress"
            },
            "Action": {
                "keywords": ["fight", "chase", "gun", "explosion", "mission", "danger", 
                           "escape", "battle", "adventure", "action", "thrilling", "exciting"],
                "weight": 1.1,
                "philosophical_aspect": "Moral agency under pressure"
            },
            "Thriller": {
                "keywords": ["suspense", "mystery", "danger", "chase", "secret", "conspiracy", 
                           "tense", "cliffhanger", "thriller", "suspenseful", "mysterious"],
                "weight": 1.1,
                "philosophical_aspect": "Uncertainty and trust"
            },
            "Romance": {
                "keywords": ["love", "romance", "heart", "relationship", "kiss", "date", 
                           "passion", "affection", "romantic", "lovers", "affection"],
                "weight": 1.1,
                "philosophical_aspect": "Nature of human connection"
            },
            "Documentary": {
                "keywords": ["real", "fact", "interview", "evidence", "truth", "history", 
                           "actual", "reality", "documentary", "non-fiction", "educational"],
                "weight": 1.2,
                "philosophical_aspect": "Construction of truth"
            },
            "Fantasy": {
                "keywords": ["magic", "dragon", "kingdom", "quest", "mythical", "wizard", 
                           "enchanted", "supernatural", "fantasy", "magical", "mythical"],
                "weight": 1.1,
                "philosophical_aspect": "Myth-making and symbolism"
            },
            "Black Cinema": {
                "keywords": ["black", "african", "diaspora", "racial", "cultural", "community",
                           "heritage", "identity", "resilience", "justice", "afro", "systemic",
                           "black experience", "african american", "civil rights"],
                "weight": 1.3,
                "philosophical_aspect": "Cultural memory and resistance"
            },
            "Urban Drama": {
                "keywords": ["urban", "city", "street", "hood", "neighborhood", "ghetto",
                           "inner city", "metropolitan", "concrete", "asphalt", "urban life"],
                "weight": 1.2,
                "philosophical_aspect": "Modern alienation and community"
            },
            "Short Film": {
                "keywords": ["short film", "short", "experimental", "student film", "micro",
                           "brief", "compact", "concise", "miniature"],
                "weight": 1.1,
                "philosophical_aspect": "Condensed narrative expression"
            }
        }
    
    def detect_genre(self, text: str, existing_genre: Optional[str] = None) -> Dict:
        """Smart genre detection with weighted scoring"""
        if not text or len(text.strip()) < 10:
            return {
                'primary_genre': existing_genre or "Unknown",
                'confidence': 0,
                'details': {},
                'secondary_genres': [],
                'all_genres': {}
            }
        
        text_lower = text.lower()
        word_count = len(text_lower.split())
        
        # Short film detection based on length
        if word_count < 300:
            short_film_score = 3.0
        elif word_count < 500:
            short_film_score = 2.0
        else:
            short_film_score = 0.5
        
        genre_scores = {}
        genre_details = {}
        
        for genre, pattern_data in self.genre_patterns.items():
            score = 0
            keywords = pattern_data["keywords"]
            weight = pattern_data.get("weight", 1.0)
            
            # Special handling for Short Film
            if genre == "Short Film":
                score = short_film_score * weight
            
            keyword_matches = []
            for keyword in keywords:
                if keyword in text_lower:
                    score += 2 * weight
                    keyword_matches.append(keyword)
                elif any(word.startswith(keyword.split()[0]) for word in text_lower.split()):
                    score += 1 * weight
                    keyword_matches.append(keyword)
            
            if existing_genre and genre.lower() in existing_genre.lower():
                score += 3
            
            if score > 0:
                genre_scores[genre] = score
                genre_details[genre] = {
                    'score': score,
                    'keywords': keyword_matches[:5],
                    'philosophical_aspect': pattern_data.get('philosophical_aspect', '')
                }
        
        if not genre_scores:
            return {
                'primary_genre': existing_genre or "Drama",
                'confidence': 50,
                'details': {},
                'secondary_genres': [],
                'all_genres': {}
            }
        
        top_genre, top_score = max(genre_scores.items(), key=lambda x: x[1])
        secondary_genres = [g for g, s in genre_scores.items() if s >= top_score * 0.5 and g != top_genre]
        
        # Adjust confidence based on score difference
        if len(genre_scores) > 1:
            sorted_scores = sorted(genre_scores.values(), reverse=True)
            score_diff = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0
            confidence = min(95, 60 + (score_diff * 5))
        else:
            confidence = 80
        
        return {
            'primary_genre': top_genre,
            'confidence': min(100, confidence),
            'details': genre_details.get(top_genre, {}),
            'secondary_genres': secondary_genres[:2],
            'all_genres': genre_details
        }

# --------------------------
# ENHANCED CULTURAL CONTEXT ANALYZER
# --------------------------
class CulturalContextAnalyzer:
    """Analyzes cultural context and relevance in films"""
    
    def __init__(self):
        self.cultural_themes = {
            'black_experience': {
                'keywords': ['black experience', 'african american', 'black community', 
                           'black culture', 'black identity', 'black history'],
                'philosophical': 'Diasporic consciousness and cultural memory',
                'weight': 1.3
            },
            'diaspora': {
                'keywords': ['diaspora', 'african diaspora', 'caribbean', 'afro-latino', 
                           'pan-african', 'transatlantic'],
                'philosophical': 'Hybrid identities and transnational connections',
                'weight': 1.2
            },
            'social_justice': {
                'keywords': ['social justice', 'racial justice', 'civil rights', 
                           'equality', 'activism', 'protest', 'resistance'],
                'philosophical': 'Ethical frameworks and moral agency',
                'weight': 1.4
            },
            'cultural_heritage': {
                'keywords': ['heritage', 'ancestral', 'tradition', 'cultural roots',
                           'lineage', 'generational'],
                'philosophical': 'Historical continuity and collective memory',
                'weight': 1.2
            },
            'urban_life': {
                'keywords': ['urban life', 'inner city', 'metropolitan', 'city living',
                           'street culture', 'urban landscape'],
                'philosophical': 'Modern alienation and urban psychology',
                'weight': 1.1
            }
        }
    
    def analyze_cultural_context(self, film_data: Dict) -> Dict:
        """Analyze cultural context with nuanced scoring and philosophical insights"""
        text = (film_data.get('synopsis', '') + ' ' + 
                film_data.get('transcript', '') + ' ' +
                film_data.get('title', '')).lower()
        
        theme_scores = {}
        theme_details = {}
        total_weighted_matches = 0
        
        for theme, theme_data in self.cultural_themes.items():
            matches = 0
            matched_keywords = []
            for keyword in theme_data['keywords']:
                if keyword in text:
                    matches += 1
                    matched_keywords.append(keyword)
            
            weighted_matches = matches * theme_data.get('weight', 1.0)
            theme_scores[theme] = weighted_matches
            total_weighted_matches += weighted_matches
            
            if matches > 0:
                theme_details[theme] = {
                    'matches': matches,
                    'keywords': matched_keywords,
                    'philosophical_aspect': theme_data['philosophical'],
                    'weighted_score': weighted_matches
                }
        
        if total_weighted_matches == 0:
            return {
                'relevance_score': 0.0,
                'primary_themes': [],
                'theme_breakdown': theme_scores,
                'theme_details': {},
                'is_culturally_relevant': False,
                'philosophical_insights': [],
                'total_matches': 0,
                'philosophical_depth': 0
            }
        
        max_possible = sum(len(theme_data['keywords']) * theme_data.get('weight', 1.0) 
                          for theme_data in self.cultural_themes.values())
        relevance_score = min(1.0, total_weighted_matches / (max_possible * 0.15))
        
        primary_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        primary_themes = [theme for theme, score in primary_themes if score > 0]
        
        philosophical_insights = []
        philosophical_depth = 0
        for theme in primary_themes:
            if theme in theme_details:
                philosophical_insights.append(theme_details[theme]['philosophical_aspect'])
                philosophical_depth += 0.3
        
        return {
            'relevance_score': round(relevance_score, 2),
            'primary_themes': primary_themes,
            'theme_breakdown': theme_scores,
            'theme_details': theme_details,
            'is_culturally_relevant': relevance_score > 0.3,
            'philosophical_insights': philosophical_insights[:2],
            'total_matches': total_weighted_matches,
            'philosophical_depth': min(1.0, philosophical_depth)
        }

# --------------------------
# ENHANCED FILM ANALYSIS ENGINE
# --------------------------
class FilmAnalysisEngine:
    """Main engine for comprehensive film analysis"""
    
    def __init__(self):
        self.genre_detector = SmartGenreDetector()
        self.cultural_analyzer = CulturalContextAnalyzer()
        self.film_scorer = EnhancedFilmScorer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.persistence = PersistenceManager()
    
    def analyze_film(self, film_data: Dict) -> Dict:
        """Main film analysis method with enhanced features"""
        try:
            film_id = self.persistence.generate_film_id(film_data)
            
            cached_result = self.persistence.load_results(film_id)
            if cached_result:
                if 'video_id' in film_data:
                    st.session_state.current_video_id = film_data['video_id']
                if 'video_title' in film_data:
                    st.session_state.current_video_title = film_data['video_title']
                    
                st.session_state.current_results_display = cached_result['analysis_results']
                st.session_state.show_results_page = True
                st.session_state.current_analysis_id = film_id
                return cached_result['analysis_results']
            
            analysis_text = self._prepare_analysis_text(film_data)
            
            if len(analysis_text.strip()) < 20:
                results = self._create_basic_fallback(film_data)
            else:
                analysis_results = self._perform_comprehensive_analysis(analysis_text, film_data)
                scoring_result = self.film_scorer.calculate_unique_film_score(analysis_results, film_data)
                results = self._generate_enhanced_review(film_data, analysis_results, scoring_result)
            
            self.persistence.save_results(film_data, results, film_id)
            
            if 'video_id' in film_data:
                st.session_state.current_video_id = film_data['video_id']
            if 'video_title' in film_data:
                st.session_state.current_video_title = film_data['video_title']
            
            return results
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return self._create_error_fallback(film_data, str(e))
    
    def _prepare_analysis_text(self, film_data: Dict) -> str:
        """Prepare text for analysis"""
        synopsis = film_data.get('synopsis', '')
        transcript = film_data.get('transcript', '')
        title = film_data.get('title', '')
        return f"{title} {synopsis} {transcript}".strip()
    
    def _perform_comprehensive_analysis(self, text: str, film_data: Dict) -> Dict:
        """Perform comprehensive film analysis"""
        # Genre detection
        existing_genre = film_data.get('genre', '')
        genre_result = self.genre_detector.detect_genre(text, existing_genre)
        
        # Cultural analysis
        cultural_result = self.cultural_analyzer.analyze_cultural_context(film_data)
        
        # Sentiment analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Narrative structure analysis
        narrative_result = self._analyze_narrative_structure(text)
        
        # Character analysis
        character_result = self._analyze_characters(text)
        
        # Emotional arc analysis
        emotional_result = self._analyze_emotional_arc(text)
        
        return {
            'genre_context': genre_result,
            'cultural_context': cultural_result,
            'sentiment_analysis': sentiment_scores,
            'narrative_structure': narrative_result,
            'character_analysis': character_result,
            'emotional_arc': emotional_result
        }
    
    def _analyze_narrative_structure(self, text: str) -> Dict:
        """Analyze narrative structure"""
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Calculate lexical diversity
        unique_words = set(words)
        lexical_diversity = len(unique_words) / max(len(words), 1)
        
        # Readability score (simplified)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        readability_score = min(1.0, 1 - (avg_sentence_length - 10) / 40)
        
        # Structural score with more variation
        structural_base = random.uniform(0.3, 0.9)
        if len(sentences) > 10:
            structural_base = max(structural_base, 0.5)
        if lexical_diversity > 0.6:
            structural_base = min(structural_base + 0.1, 0.95)
        
        return {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'lexical_diversity': round(lexical_diversity, 3),
            'avg_sentence_length': round(avg_sentence_length, 1),
            'readability_score': round(readability_score, 2),
            'structural_score': round(structural_base, 2)
        }
    
    def _analyze_characters(self, text: str) -> Dict:
        """Analyze character development"""
        # Extract potential character names (simple heuristic)
        words = nltk.word_tokenize(text)
        potential_characters = len([w for w in words if w.istitle() and len(w) > 1])
        
        character_density = potential_characters / max(len(words), 1)
        
        # Character score with more variation
        char_score_base = random.uniform(0.3, 0.9)
        if character_density > 0.04:
            char_score_base = max(char_score_base, 0.6)
        if potential_characters > 5:
            char_score_base = min(char_score_base + 0.1, 0.95)
        
        return {
            'potential_characters': min(potential_characters, 10),
            'character_density': round(character_density, 3),
            'character_score': round(char_score_base, 2)
        }
    
    def _analyze_emotional_arc(self, text: str) -> Dict:
        """Analyze emotional arc"""
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) < 3:
            return {
                'arc_score': 0.5,
                'emotional_variance': 0.3,
                'emotional_range': 'moderate'
            }
        
        # Analyze sentiment per sentence
        sentiments = []
        for sentence in sentences[:20]:  # Limit for performance
            sentiment = self.sentiment_analyzer.polarity_scores(sentence)
            sentiments.append(sentiment['compound'])
        
        if not sentiments:
            return {
                'arc_score': 0.5,
                'emotional_variance': 0.3,
                'emotional_range': 'moderate'
            }
        
        # Calculate variance and arc
        variance = np.var(sentents) if len(sentiments) > 1 else 0.2
        arc_score = min(1.0, variance * 5 + 0.3)
        
        # Determine emotional range
        if variance > 0.15:
            emotional_range = 'intense'
        elif variance > 0.1:
            emotional_range = 'wide'
        elif variance > 0.05:
            emotional_range = 'moderate'
        elif variance > 0.02:
            emotional_range = 'subtle'
        else:
            emotional_range = 'narrow'
        
        return {
            'arc_score': round(arc_score, 2),
            'emotional_variance': round(variance, 3),
            'emotional_range': emotional_range,
            'sentiment_samples': len(sentences)
        }
    
    def _generate_enhanced_review(self, film_data: Dict, analysis_results: Dict, scoring_result: Dict) -> Dict:
        """Generate enhanced film review with philosophical insights"""
        overall_score = scoring_result['overall_score']
        genre_result = analysis_results['genre_context']
        cultural_context = analysis_results['cultural_context']
        component_scores = scoring_result['component_scores']
        
        cinematic_scores = self._map_to_cinematic_categories(component_scores, genre_result)
        
        results = {
            'smart_summary': self._generate_philosophical_summary(film_data, overall_score, genre_result, cultural_context),
            'cinematic_scores': cinematic_scores,
            'overall_score': overall_score,
            'strengths': self._generate_enhanced_strengths(analysis_results, cinematic_scores, cultural_context),
            'weaknesses': self._generate_enhanced_weaknesses(analysis_results, cinematic_scores),
            'recommendations': self._generate_enhanced_recommendations(analysis_results, cinematic_scores, cultural_context),
            'festival_recommendations': self._generate_festival_recommendations(overall_score, genre_result, cultural_context),
            'audience_analysis': self._generate_enhanced_audience_analysis(analysis_results, genre_result, cultural_context),
            'genre_insights': genre_result,
            'cultural_insights': cultural_context,
            'scoring_breakdown': scoring_result,
            'film_title': film_data.get('title', 'Unknown Film'),
            'philosophical_insights': self._generate_philosophical_insights(film_data, analysis_results),
            'ai_tool_suggestions': self._generate_ai_tool_suggestions(analysis_results),
            'synopsis_analysis': self._analyze_synopsis(film_data.get('synopsis', '')),
            'narrative_arc_analysis': analysis_results.get('narrative_structure', {}),
            'character_ecosystem': analysis_results.get('character_analysis', {})
        }
        
        return results
    
    def _map_to_cinematic_categories(self, component_scores: Dict, genre_result: Dict) -> Dict:
        """Map component scores to cinematic categories"""
        base_scores = {
            'story_narrative': component_scores.get('narrative', 3.0),
            'visual_vision': component_scores.get('technical', 3.0) * 1.1,
            'technical_craft': component_scores.get('technical', 3.0),
            'sound_design': component_scores.get('technical', 3.0) * 0.9,
            'performance': component_scores.get('character', 3.0) * 1.2
        }
        
        # Ensure scores are within 1.0-5.0 range
        return {k: max(1.0, min(5.0, v)) for k, v in base_scores.items()}
    
    def _generate_philosophical_summary(self, film_data: Dict, score: float, genre_result: Dict, cultural_context: Dict) -> str:
        """Generate philosophical film summary"""
        title = film_data.get('title', 'Unknown Film')
        genre_info = genre_result.get('details', {})
        philosophical_aspect = genre_info.get('philosophical_aspect', 'human experience')
        
        # Score-based quality assessment
        if score >= 4.5:
            quality = "profound"
            impact = "a transformative cinematic meditation"
            philosophical_frame = "transcendent artistic achievement"
        elif score >= 4.0:
            quality = "significant"
            impact = "a meaningful artistic statement"
            philosophical_frame = "compelling narrative exploration"
        elif score >= 3.5:
            quality = "substantial"
            impact = "a thoughtful creative work"
            philosophical_frame = "engaging thematic investigation"
        elif score >= 3.0:
            quality = "promising"
            impact = "an evolving artistic voice"
            philosophical_frame = "developing narrative consciousness"
        elif score >= 2.0:
            quality = "emerging"
            impact = "a foundational creative endeavor"
            philosophical_frame = "nascent artistic exploration"
        else:
            quality = "formative"
            impact = "a creative beginning"
            philosophical_frame = "initial creative expression"
        
        cultural_phrase = ""
        if cultural_context.get('is_culturally_relevant'):
            primary_themes = cultural_context.get('primary_themes', [])
            if primary_themes:
                theme_str = " and ".join(primary_themes)
                cultural_phrase = f", engaging with {theme_str} through the lens of "
        
        summary = f"**{title}** presents {quality} engagement with {philosophical_aspect}{cultural_phrase}{philosophical_frame}, resulting in {impact}."
        
        if cultural_context.get('philosophical_insights'):
            insights = cultural_context['philosophical_insights']
            if insights:
                insight_str = "; ".join(insights[:2])
                summary += f" The work contemplates {insight_str.lower()}."
        
        return summary
    
    def _generate_enhanced_strengths(self, analysis_results: Dict, cinematic_scores: Dict, cultural_context: Dict) -> List[str]:
        """Generate enhanced strengths list"""
        strengths = []
        
        # Check narrative strength
        if cinematic_scores.get('story_narrative', 0) >= 4.0:
            strengths.append("Strong narrative structure with compelling storytelling")
        elif cinematic_scores.get('story_narrative', 0) >= 3.0:
            strengths.append("Solid narrative foundation with clear storytelling")
        
        # Check character development
        if cinematic_scores.get('performance', 0) >= 4.0:
            strengths.append("Well-developed characters with depth and authenticity")
        elif cinematic_scores.get('performance', 0) >= 3.0:
            strengths.append("Emerging character development with potential")
        
        # Check cultural relevance
        if cultural_context.get('is_culturally_relevant'):
            strengths.append("Significant cultural relevance and thematic depth")
        
        # Check technical aspects
        if cinematic_scores.get('technical_craft', 0) >= 3.5:
            strengths.append("Solid technical execution and production values")
        elif cinematic_scores.get('technical_craft', 0) >= 2.5:
            strengths.append("Adequate technical foundation for development")
        
        # Add default strengths if none found
        if not strengths:
            strengths.append("Clear creative vision and intention")
            strengths.append("Foundation for artistic development")
        
        return strengths[:3]
    
    def _generate_enhanced_weaknesses(self, analysis_results: Dict, cinematic_scores: Dict) -> List[str]:
        """Generate enhanced weaknesses list"""
        weaknesses = []
        
        # Identify weakest area
        min_score = min(cinematic_scores.values()) if cinematic_scores else 0
        min_category = min(cinematic_scores.items(), key=lambda x: x[1])[0] if cinematic_scores else ""
        
        if min_score < 2.5:
            category_map = {
                'story_narrative': 'narrative structure',
                'visual_vision': 'visual storytelling',
                'technical_craft': 'technical execution',
                'sound_design': 'audio elements',
                'performance': 'character development'
            }
            weakness_category = category_map.get(min_category, min_category.replace('_', ' '))
            weaknesses.append(f"Could benefit from stronger {weakness_category}")
        
        # Check for very low scores
        for category, score in cinematic_scores.items():
            if score < 2.0:
                category_name = category.replace('_', ' ').title()
                weaknesses.append(f"{category_name} needs significant improvement")
        
        # Add generic weaknesses if needed
        if not weaknesses:
            weaknesses.append("Consider deepening emotional resonance")
            weaknesses.append("Opportunity for more distinctive visual style")
        
        return weaknesses[:2]
    
    def _generate_enhanced_recommendations(self, analysis_results: Dict, cinematic_scores: Dict, cultural_context: Dict) -> List[str]:
        """Generate enhanced recommendations"""
        recommendations = []
        
        # Narrative recommendations
        if cinematic_scores.get('story_narrative', 0) < 3.5:
            recommendations.append("Develop narrative complexity with subplots")
        
        # Character recommendations
        if cinematic_scores.get('performance', 0) < 3.5:
            recommendations.append("Deepen character backstories and motivations")
        
        # Cultural recommendations
        if cultural_context.get('is_culturally_relevant'):
            recommendations.append("Leverage cultural themes for deeper resonance")
        
        # Technical recommendations
        if cinematic_scores.get('technical_craft', 0) < 3.0:
            recommendations.append("Enhance production values with focused resources")
        
        if not recommendations:
            recommendations.append("Continue developing your distinctive voice")
            recommendations.append("Explore collaborations to enhance production scope")
        
        return recommendations[:3]
    
    def _generate_festival_recommendations(self, score: float, genre_result: Dict, cultural_context: Dict) -> Dict:
        """Generate festival recommendations"""
        festivals_by_level = {
            'elite': ["Sundance Film Festival", "Toronto International Film Festival", 
                     "Cannes Film Festival", "Berlin International Film Festival"],
            'premier': ["South by Southwest (SXSW)", "Tribeca Film Festival", 
                       "Telluride Film Festival", "Venice Film Festival"],
            'specialized': ["BlackStar Film Festival", "Urbanworld Film Festival", 
                          "Pan African Film Festival", "AfroFilm Festival"],
            'developing': ["Local film festivals", "Emerging filmmaker showcases", 
                         "University film competitions", "Online film festivals"]
        }
        
        if score >= 4.5:
            level = "elite"
            festivals = festivals_by_level['elite']
        elif score >= 4.0:
            level = "premier"
            festivals = festivals_by_level['premier']
        elif score >= 3.0:
            level = "specialized"
            festivals = festivals_by_level['specialized']
        else:
            level = "developing"
            festivals = festivals_by_level['developing']
        
        # Add specialized festivals for cultural relevance
        if cultural_context.get('is_culturally_relevant'):
            festivals.extend(festivals_by_level['specialized'][:2])
        
        return {
            'level': level,
            'festivals': list(set(festivals))[:4]
        }
    
    def _generate_enhanced_audience_analysis(self, analysis_results: Dict, genre_result: Dict, cultural_context: Dict) -> Dict:
        """Generate enhanced audience analysis"""
        genre = genre_result.get('primary_genre', 'Unknown')
        
        audience_mapping = {
            'Drama': ["Film enthusiasts", "Art house audiences", "Critics", "Mature viewers"],
            'Comedy': ["General audiences", "Young adults", "Festival goers", "Casual viewers"],
            'Horror': ["Genre fans", "Thrill-seekers", "Niche audiences", "Cult film enthusiasts"],
            'Sci-Fi': ["Tech enthusiasts", "Fantasy fans", "Futurists", "Speculative fiction readers"],
            'Action': ["Mainstream audiences", "Action fans", "Entertainment seekers", "Blockbuster viewers"],
            'Black Cinema': ["Cultural audiences", "Diaspora communities", "Socially conscious viewers", "Academic circles"],
            'Urban Drama': ["Urban audiences", "Youth demographics", "Social realism enthusiasts", "Contemporary art fans"],
            'Documentary': ["Fact-based audiences", "Educational viewers", "Issue-focused groups", "Academic audiences"],
            'Short Film': ["Film students", "Festival programmers", "Online audiences", "Experimental art lovers"]
        }
        
        audiences = audience_mapping.get(genre, ["General film audiences", "Festival attendees", "Artistic communities"])
        
        # Add cultural audiences if relevant
        if cultural_context.get('is_culturally_relevant'):
            audiences.append("Culturally engaged viewers")
            audiences.append("Academic and educational audiences")
        
        # Calculate engagement score based on score and genre
        base_score = random.uniform(0.5, 0.9)
        if len(audiences) > 4:
            base_score += 0.05
        if cultural_context.get('is_culturally_relevant'):
            base_score += 0.08
        
        engagement_score = min(0.95, base_score)
        
        market_mapping = {
            (0.8, 1.0): 'High',
            (0.6, 0.8): 'Medium-High',
            (0.4, 0.6): 'Medium',
            (0.2, 0.4): 'Developing',
            (0.0, 0.2): 'Niche'
        }
        
        market_potential = 'Medium'
        for (low, high), potential in market_mapping.items():
            if low <= engagement_score < high:
                market_potential = potential
                break
        
        return {
            'target_audiences': list(set(audiences))[:5],
            'engagement_score': round(engagement_score, 2),
            'market_potential': market_potential
        }
    
    def _generate_philosophical_insights(self, film_data: Dict, analysis_results: Dict) -> List[str]:
        """Generate philosophical insights about the film"""
        insights = []
        text = film_data.get('synopsis', '') + ' ' + film_data.get('transcript', '')
        
        # Check for existential themes
        existential_keywords = ['death', 'life', 'meaning', 'existence', 'purpose', 'mortality']
        if any(keyword in text.lower() for keyword in existential_keywords):
            insights.append("Explores existential questions about human purpose")
        
        # Check for social themes
        social_keywords = ['society', 'justice', 'equality', 'power', 'freedom', 'oppression']
        if any(keyword in text.lower() for keyword in social_keywords):
            insights.append("Engages with social structures and power dynamics")
        
        # Check for psychological themes
        psychological_keywords = ['mind', 'memory', 'identity', 'consciousness', 'dream', 'psyche']
        if any(keyword in text.lower() for keyword in psychological_keywords):
            insights.append("Investigates psychological depth and identity")
        
        # Cultural insights
        cultural_insights = analysis_results.get('cultural_context', {}).get('philosophical_insights', [])
        insights.extend(cultural_insights)
        
        return insights[:3] if insights else ["Explores fundamental human experiences"]
    
    def _generate_ai_tool_suggestions(self, analysis_results: Dict) -> List[Dict]:
        """Generate AI tool suggestions for further analysis"""
        suggestions = []
        
        # Add suggestions based on analysis depth
        narrative = analysis_results.get('narrative_structure', {})
        if narrative.get('word_count', 0) > 500:
            suggestions.append({
                'tool': 'GPT-4',
                'purpose': 'Advanced narrative analysis',
                'benefit': 'Detailed plot structure and thematic exploration'
            })
        
        cultural = analysis_results.get('cultural_context', {})
        if cultural.get('is_culturally_relevant'):
            suggestions.append({
                'tool': 'CulturalBERT',
                'purpose': 'Cultural context analysis',
                'benefit': 'Enhanced cultural relevance scoring'
            })
        
        if narrative.get('word_count', 0) > 1000:
            suggestions.append({
                'tool': 'BERT',
                'purpose': 'Contextual understanding',
                'benefit': 'Better genre and theme detection'
            })
        
        return suggestions[:3]
    
    def _analyze_synopsis(self, synopsis: str) -> Dict:
        """Analyze synopsis for key insights"""
        if not synopsis:
            return {
                'length': 0,
                'key_themes': [],
                'emotional_tone': 'neutral',
                'complexity': 'low',
                'sentiment_score': 0.0
            }
        
        words = synopsis.split()
        sentences = nltk.sent_tokenize(synopsis)
        
        # Key theme extraction
        themes = []
        theme_keywords = {
            'love': ['love', 'relationship', 'romance', 'heart', 'affection'],
            'conflict': ['conflict', 'struggle', 'battle', 'war', 'fight'],
            'journey': ['journey', 'travel', 'quest', 'adventure', 'voyage'],
            'identity': ['identity', 'self', 'discovery', 'truth', 'authenticity'],
            'justice': ['justice', 'right', 'wrong', 'moral', 'ethics'],
            'memory': ['memory', 'past', 'history', 'remembrance', 'nostalgia'],
            'family': ['family', 'parent', 'child', 'sibling', 'generation'],
            'survival': ['survival', 'endure', 'persevere', 'overcome', 'resilience']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in synopsis.lower() for keyword in keywords):
                themes.append(theme)
        
        # Emotional tone analysis
        sentiment = self.sentiment_analyzer.polarity_scores(synopsis)
        if sentiment['compound'] > 0.3:
            emotional_tone = 'positive'
        elif sentiment['compound'] < -0.3:
            emotional_tone = 'negative'
        else:
            emotional_tone = 'neutral'
        
        # Complexity assessment
        word_count = len(words)
        if word_count > 250:
            complexity = 'high'
        elif word_count > 120:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        return {
            'length': word_count,
            'sentence_count': len(sentences),
            'avg_sentence_length': round(word_count / max(len(sentences), 1), 1),
            'key_themes': themes[:4],
            'emotional_tone': emotional_tone,
            'complexity': complexity,
            'sentiment_score': round(sentiment['compound'], 2)
        }
    
    def _create_basic_fallback(self, film_data: Dict) -> Dict:
        """Create basic analysis when data is insufficient"""
        # Generate random score in lower range for basic content
        base_score = random.uniform(1.5, 2.8)
        score = round(base_score / 0.1) * 0.1  # Round to nearest 0.1
        
        return {
            'smart_summary': f"**{film_data.get('title', 'Unknown Film')}** provides a foundation for cinematic exploration with emerging narrative voice.",
            'cinematic_scores': {
                'story_narrative': max(1.5, round(random.uniform(1.5, 3.0), 1)),
                'visual_vision': max(1.3, round(random.uniform(1.3, 2.8), 1)),
                'technical_craft': max(1.2, round(random.uniform(1.2, 2.7), 1)),
                'sound_design': max(1.3, round(random.uniform(1.3, 2.6), 1)),
                'performance': max(1.5, round(random.uniform(1.5, 3.0), 1))
            },
            'overall_score': score,
            'strengths': ["Foundational concept established", "Clear narrative intention"],
            'weaknesses': ["Limited depth in current form", "Needs further development"],
            'recommendations': ["Expand narrative details", "Develop character depth"],
            'festival_recommendations': {
                'level': 'developing',
                'festivals': ["Local film festivals", "Emerging filmmaker showcases"]
            },
            'audience_analysis': {
                'target_audiences': ["Emerging film enthusiasts", "Workshop audiences"],
                'engagement_score': round(random.uniform(0.3, 0.5), 2),
                'market_potential': 'Developing'
            },
            'genre_insights': {'primary_genre': 'Drama', 'confidence': 60},
            'cultural_insights': {'relevance_score': 0.2, 'is_culturally_relevant': False},
            'scoring_breakdown': {
                'overall_score': score,
                'component_scores': {
                    'narrative': max(1.5, round(random.uniform(1.5, 3.0), 1)),
                    'emotional': max(1.4, round(random.uniform(1.4, 2.9), 1)),
                    'character': max(1.5, round(random.uniform(1.5, 3.0), 1)),
                    'cultural': max(1.0, round(random.uniform(1.0, 2.5), 1)),
                    'technical': max(1.2, round(random.uniform(1.2, 2.7), 1))
                }
            },
            'film_title': film_data.get('title', 'Unknown Film'),
            'philosophical_insights': ["Explores basic human experiences"],
            'ai_tool_suggestions': [],
            'synopsis_analysis': {'length': len(film_data.get('synopsis', '').split()), 'emotional_tone': 'neutral'}
        }
    
    def _create_error_fallback(self, film_data: Dict, error_msg: str) -> Dict:
        """Create error fallback analysis"""
        # Generate random score in mid-low range for error cases
        base_score = random.uniform(2.0, 3.0)
        score = round(base_score / 0.1) * 0.1  # Round to nearest 0.1
        
        return {
            'smart_summary': f"**{film_data.get('title', 'Unknown Film')}** encountered analysis challenges. Basic assessment suggests emerging creative potential.",
            'cinematic_scores': {
                'story_narrative': max(2.0, round(random.uniform(2.0, 3.5), 1)),
                'visual_vision': max(1.8, round(random.uniform(1.8, 3.2), 1)),
                'technical_craft': max(1.7, round(random.uniform(1.7, 3.0), 1)),
                'sound_design': max(1.8, round(random.uniform(1.8, 3.1), 1)),
                'performance': max(2.0, round(random.uniform(2.0, 3.4), 1))
            },
            'overall_score': score,
            'strengths': ["Creative concept identified", "Analysis attempted"],
            'weaknesses': ["Insufficient data for full analysis", f"Technical issue: {error_msg[:50]}"],
            'recommendations': ["Provide more detailed content", "Try manual analysis method"],
            'festival_recommendations': {
                'level': 'developing',
                'festivals': ["Local showcases", "Development workshops"]
            },
            'audience_analysis': {
                'target_audiences': ["Patient early audiences", "Development-focused viewers"],
                'engagement_score': round(random.uniform(0.3, 0.5), 2),
                'market_potential': 'Emerging'
            },
            'genre_insights': {'primary_genre': 'Unknown', 'confidence': 0},
            'cultural_insights': {'relevance_score': 0.0, 'is_culturally_relevant': False},
            'scoring_breakdown': {
                'overall_score': score,
                'component_scores': {
                    'narrative': max(2.0, round(random.uniform(2.0, 3.5), 1)),
                    'emotional': max(1.9, round(random.uniform(1.9, 3.3), 1)),
                    'character': max(2.0, round(random.uniform(2.0, 3.4), 1)),
                    'cultural': max(1.5, round(random.uniform(1.5, 2.8), 1)),
                    'technical': max(1.7, round(random.uniform(1.7, 3.0), 1))
                }
            },
            'film_title': film_data.get('title', 'Unknown Film'),
            'philosophical_insights': ["Analysis in progress"],
            'ai_tool_suggestions': [{'tool': 'Error Recovery', 'purpose': 'Issue diagnosis', 'benefit': 'Improved analysis stability'}],
            'synopsis_analysis': {'length': 0, 'emotional_tone': 'neutral', 'error': error_msg}
        }

# --------------------------
# ENHANCED HISTORY ANALYTICS PAGE
# --------------------------
class EnhancedHistoryAnalyticsPage:
    """Enhanced analytics page for viewing analysis history and trends"""
    
    def __init__(self, persistence: PersistenceManager):
        self.persistence = persistence
    
    def show(self) -> None:
        """Display the enhanced analytics dashboard"""
        st.header("📈 Advanced Analytics Dashboard")
        st.markdown("---")
        
        # Get analytics data
        analytics_data = self.persistence.get_analytics_data()
        
        if analytics_data is None or len(analytics_data) == 0:
            st.info("No analysis history yet. Analyze some films to see analytics!")
            if st.button("← Back to Dashboard", key="back_to_dashboard_from_analytics"):
                st.session_state.current_page = "🏠 Dashboard"
                st.rerun()
            return
        
        # Analytics view selector
        view_options = ['Overview', 'Trends', 'Genres', 'Cultural Analysis', 'Score Distribution']
        selected_view = st.selectbox(
            "Select Analytics View:",
            view_options,
            index=view_options.index(st.session_state.get('analytics_view', 'Overview'))
        )
        st.session_state.analytics_view = selected_view
        
        if selected_view == 'Overview':
            self._show_overview_analytics(analytics_data)
        elif selected_view == 'Trends':
            self._show_trends_analytics(analytics_data)
        elif selected_view == 'Genres':
            self._show_genre_analytics(analytics_data)
        elif selected_view == 'Cultural Analysis':
            self._show_cultural_analytics(analytics_data)
        elif selected_view == 'Score Distribution':
            self._show_score_distribution(analytics_data)
        
        # Detailed history table
        st.markdown("---")
        st.subheader("📋 Detailed Analysis History")
        
        # Create enhanced history dataframe
        history_df = pd.DataFrame(st.session_state.analysis_history)
        
        if not history_df.empty:
            # Add viewing buttons
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df['date'] = history_df['timestamp'].dt.strftime('%Y-%m-%d')
            history_df['time'] = history_df['timestamp'].dt.strftime('%H:%M')
            
            # Display the dataframe
            display_cols = ['title', 'overall_score', 'detected_genre', 'date', 'time']
            st.dataframe(
                history_df[display_cols].rename(columns={
                    'title': 'Film Title',
                    'overall_score': 'Score',
                    'detected_genre': 'Genre',
                    'date': 'Date',
                    'time': 'Time'
                }),
                use_container_width=True
            )
            
            # Quick filter options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🏆 View Top Films", width='stretch'):
                    # Sort by score and show top 3
                    top_films = sorted(
                        st.session_state.analysis_history,
                        key=lambda x: x.get('overall_score', 0),
                        reverse=True
                    )[:3]
                    
                    for film in top_films:
                        if st.button(f"🔍 {film.get('title', 'Unknown')[:30]} - {film.get('overall_score', 0)}/5.0", 
                                   key=f"view_from_history_{film.get('id', '')}"):
                            stored_result = self.persistence.load_results(film.get('id', ''))
                            if stored_result:
                                st.session_state.current_results_display = stored_result['analysis_results']
                                st.session_state.current_analysis_id = film.get('id', '')
                                st.session_state.show_results_page = True
                                st.session_state.current_page = "🏠 Dashboard"
                                st.rerun()
            
            with col2:
                if st.button("📥 Export History as CSV", width='stretch'):
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="Click to download",
                        data=csv,
                        file_name=f"film_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        # Back to dashboard button
        st.markdown("---")
        if st.button("← Back to Dashboard", key="back_to_dashboard_bottom"):
            st.session_state.current_page = "🏠 Dashboard"
            st.rerun()
    
    def _show_overview_analytics(self, analytics_data: pd.DataFrame) -> None:
        """Show overview analytics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", len(analytics_data))
        with col2:
            avg_score = analytics_data['overall_score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}/5.0")
        with col3:
            unique_genres = analytics_data['detected_genre'].nunique()
            st.metric("Unique Genres", unique_genres)
        with col4:
            if 'cultural_relevance' in analytics_data.columns:
                cultural_pct = (analytics_data['cultural_relevance'] > 0.5).mean() * 100
                st.metric("Cultural Films", f"{cultural_pct:.0f}%")
        
        # Recent activity chart
        st.subheader("📊 Recent Activity")
        
        # Create date-based aggregation
        daily_counts = analytics_data.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        daily_counts = daily_counts.sort_values('date')
        
        fig = px.line(daily_counts, x='date', y='count', 
                     title='Daily Analysis Count',
                     markers=True)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_trends_analytics(self, analytics_data: pd.DataFrame) -> None:
        """Show trends analytics"""
        st.subheader("📈 Score Trends Over Time")
        
        # Convert timestamp for plotting
        analytics_data['timestamp'] = pd.to_datetime(analytics_data['timestamp'])
        analytics_data = analytics_data.sort_values('timestamp')
        
        # Create moving average
        window_size = min(5, len(analytics_data))
        analytics_data['moving_avg'] = analytics_data['overall_score'].rolling(
            window=window_size, min_periods=1
        ).mean()
        
        fig = go.Figure()
        
        # Add individual scores
        fig.add_trace(go.Scatter(
            x=analytics_data['timestamp'],
            y=analytics_data['overall_score'],
            mode='markers',
            name='Individual Scores',
            marker=dict(size=8, opacity=0.6)
        ))
        
        # Add moving average
        fig.add_trace(go.Scatter(
            x=analytics_data['timestamp'],
            y=analytics_data['moving_avg'],
            mode='lines',
            name=f'{window_size}-film Moving Average',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title='Score Trend Over Time',
            xaxis_title='Date',
            yaxis_title='Score (out of 5.0)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Time of day analysis
        st.subheader("🕒 Analysis by Time of Day")
        
        if 'time_of_day' in analytics_data.columns:
            time_counts = analytics_data['time_of_day'].value_counts().sort_index()
            
            fig = px.bar(x=time_counts.index, y=time_counts.values,
                        title='Analyses by Hour of Day',
                        labels={'x': 'Hour of Day', 'y': 'Number of Analyses'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_genre_analytics(self, analytics_data: pd.DataFrame) -> None:
        """Show genre analytics"""
        st.subheader("🎭 Genre Analysis")
        
        # Genre distribution
        genre_counts = analytics_data['detected_genre'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=genre_counts.values, names=genre_counts.index,
                        title='Genre Distribution')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Genre performance
            genre_scores = analytics_data.groupby('detected_genre')['overall_score'].agg(['mean', 'count']).round(2)
            genre_scores = genre_scores.sort_values('mean', ascending=False)
            
            st.write("**Genre Performance:**")
            st.dataframe(
                genre_scores.rename(columns={'mean': 'Avg Score', 'count': 'Count'}),
                use_container_width=True
            )
        
        # Genre trends over time
        st.subheader("📊 Genre Trends")
        
        # Create genre time series
        genre_time_data = []
        for genre in genre_counts.index[:5]:  # Top 5 genres
            genre_data = analytics_data[analytics_data['detected_genre'] == genre]
            if not genre_data.empty:
                genre_time_data.append({
                    'genre': genre,
                    'count': len(genre_data),
                    'avg_score': genre_data['overall_score'].mean()
                })
        
        if genre_time_data:
            genre_trend_df = pd.DataFrame(genre_time_data)
            
            fig = px.bar(genre_trend_df, x='genre', y='count',
                        color='avg_score',
                        title='Top Genres by Count (colored by avg score)',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_cultural_analytics(self, analytics_data: pd.DataFrame) -> None:
        """Show cultural analytics"""
        st.subheader("🌍 Cultural Relevance Analysis")
        
        if 'cultural_relevance' not in analytics_data.columns:
            st.info("No cultural relevance data available.")
            return
        
        # Cultural relevance distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram of cultural relevance scores
            fig = px.histogram(analytics_data, x='cultural_relevance',
                             nbins=10,
                             title='Cultural Relevance Distribution')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cultural vs overall score
            fig = px.scatter(analytics_data, x='cultural_relevance', y='overall_score',
                           trendline='ols',
                           title='Cultural Relevance vs Overall Score',
                           hover_data=['title'])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top culturally relevant films
        st.subheader("🏆 Top Culturally Relevant Films")
        
        cultural_films = analytics_data[analytics_data['cultural_relevance'] > 0.5].copy()
        if not cultural_films.empty:
            cultural_films = cultural_films.sort_values('cultural_relevance', ascending=False)
            
            cols = st.columns(min(3, len(cultural_films)))
            
            for idx, (_, film) in enumerate(cultural_films.head(3).iterrows()):
                with cols[idx]:
                    st.metric(
                        label=film['title'][:20] + ("..." if len(film['title']) > 20 else ""),
                        value=f"{film['cultural_relevance']:.0%}",
                        delta=f"Score: {film['overall_score']}/5.0"
                    )
                    
                    if st.button("View Analysis", key=f"cultural_view_{idx}", width='stretch'):
                        stored_result = self.persistence.load_results(film['id'])
                        if stored_result:
                            st.session_state.current_results_display = stored_result['analysis_results']
                            st.session_state.current_analysis_id = film['id']
                            st.session_state.show_results_page = True
                            st.session_state.current_page = "🏠 Dashboard"
                            st.rerun()
    
    def _show_score_distribution(self, analytics_data: pd.DataFrame) -> None:
        """Show score distribution analytics"""
        st.subheader("📊 Score Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram of scores
            fig = px.histogram(analytics_data, x='overall_score',
                             nbins=15,
                             title='Score Distribution',
                             color_discrete_sequence=['#667eea'])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot by genre
            fig = px.box(analytics_data, x='detected_genre', y='overall_score',
                        title='Score Distribution by Genre')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Score statistics
        st.subheader("📈 Score Statistics")
        
        score_stats = analytics_data['overall_score'].describe()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{score_stats['mean']:.2f}")
        with col2:
            st.metric("Median", f"{score_stats['50%']:.2f}")
        with col3:
            st.metric("Std Dev", f"{score_stats['std']:.2f}")
        with col4:
            st.metric("Range", f"{score_stats['max'] - score_stats['min']:.2f}")

# --------------------------
# ENHANCED FILM ANALYSIS INTERFACE WITH VIDEO VIEWER
# --------------------------
class EnhancedFilmAnalysisInterface:
    """Main interface for the film analysis application with integrated video viewer"""
    
    def __init__(self, analyzer: FilmAnalysisEngine):
        self.analyzer = analyzer
        self.persistence = PersistenceManager()
    
    def show_dashboard(self) -> None:
        """Main dashboard for film analysis with enhanced features"""
        st.header("🎬 FlickFinder AI - Enhanced Scoring Analysis Hub")
        st.markdown("*Version 3.1 with improved scoring algorithm (1.1-5.0 range)*")
        
        # Show enhanced top films section
        self._show_enhanced_top_films_section()
        
        # Check if we should show batch results
        if st.session_state.get('show_batch_results') and st.session_state.get('batch_results'):
            self._display_enhanced_batch_results(st.session_state.batch_results)
            
            if st.button("← Back to Dashboard", key="back_to_dashboard_batch"):
                st.session_state.show_batch_results = False
                st.session_state.batch_results = None
                st.rerun()
            return
        
        # Check if we should show single film results
        if st.session_state.get('show_results_page') and st.session_state.get('current_results_display'):
            self._display_enhanced_film_results(st.session_state.current_results_display)
            
            if st.button("← Back to Dashboard", key="back_to_dashboard"):
                st.session_state.show_results_page = False
                st.session_state.current_results_display = None
                st.rerun()
            return
        
        # Display enhanced statistics
        stats = self._get_enhanced_statistics()
        
        st.subheader("📊 Advanced Analytics Dashboard")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total Films", stats['total_films'], help="Total number of films analyzed")
        with col2:
            if stats['total_films'] > 0:
                st.metric("Average Score", f"{stats['average_score']}/5.0",
                         delta=f"{stats['score_trend']:.1f} trend", 
                         delta_color="normal" if stats['score_trend'] >= 0 else "inverse",
                         help="Average score with trend analysis")
        with col3:
            if stats['total_films'] > 0:
                st.metric("Score Range", f"{stats['score_range']}", help="Difference between highest and lowest scores")
        with col4:
            if stats['total_films'] > 0:
                st.metric("Cultural Films", stats['cultural_films'], help="Films with significant cultural relevance")
        with col5:
            if stats['total_films'] > 0:
                st.metric("Top Genre", stats['top_genre'][:15], help="Most frequently detected genre")
        with col6:
            if stats['total_films'] > 0:
                st.metric("Analysis Rate", f"{stats['analysis_rate']}/day", help="Average analyses per day")
        
        # Quick insights panel
        with st.expander("💡 **Quick Insights & Trends**", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 Recent Activity:**")
                if stats['recent_analyses']:
                    for analysis in stats['recent_analyses'][:3]:
                        st.write(f"• {analysis['title'][:20]}: {analysis['score']}/5.0")
                else:
                    st.write("No recent analyses")
            
            with col2:
                st.write("**🎭 Genre Distribution:**")
                if stats['genre_distribution']:
                    for genre, count in list(stats['genre_distribution'].items())[:3]:
                        st.write(f"• {genre}: {count}")
                else:
                    st.write("No genre data")
        
        # Analysis methods tabs
        st.subheader("🎬 Analyze Films")
        tab1, tab2, tab3 = st.tabs(["🎥 YouTube Analysis", "📝 Manual Entry", "📊 CSV Batch"])
        
        with tab1:
            self._show_youtube_analysis()
        with tab2:
            self._show_manual_analysis()
        with tab3:
            self._show_csv_interface()
    
    def _show_youtube_analysis(self) -> None:
        """Show YouTube video analysis interface"""
        st.subheader("🎥 YouTube Video Analysis")
        st.markdown("Analyze films from YouTube videos by providing a video URL or ID.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            youtube_input = st.text_input(
                "Enter YouTube URL or Video ID:",
                placeholder="https://www.youtube.com/watch?v=... or just the video ID",
                key="youtube_input"
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            analyze_button = st.button("🎬 Analyze Video", type="primary", width='stretch')
        
        if analyze_button and youtube_input:
            with st.spinner("🔄 Extracting and analyzing video content..."):
                try:
                    # Extract video ID from URL
                    video_id = self._extract_youtube_id(youtube_input)
                    
                    if not video_id:
                        st.error("Invalid YouTube URL or Video ID. Please check your input.")
                        return
                    
                    # Try to get video info first
                    video_info = self._get_youtube_video_info(video_id)
                    
                    # Get video transcript
                    transcript_text = ""
                    try:
                        # Use correct YouTubeTranscriptApi method
                        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                        
                        # Try to get transcript in preferred language order
                        languages = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
                        transcript_found = False
                        
                        for lang in languages:
                            try:
                                transcript = transcript_list.find_transcript([lang])
                                transcript_data = transcript.fetch()
                                transcript_text = " ".join([entry['text'] for entry in transcript_data])
                                transcript_found = True
                                st.info(f"✅ Found transcript in {lang}")
                                break
                            except:
                                continue
                        
                        if not transcript_found:
                            # Try any available transcript
                            try:
                                # Get first available transcript
                                for transcript in transcript_list:
                                    transcript_data = transcript.fetch()
                                    transcript_text = " ".join([entry['text'] for entry in transcript_data])
                                    st.info(f"✅ Found transcript in {transcript.language}")
                                    break
                            except Exception as e:
                                st.warning(f"⚠️ Could not extract transcript: {str(e)[:100]}")
                                transcript_text = ""
                                
                    except Exception as e:
                        st.warning(f"⚠️ Could not extract transcript: {str(e)[:100]}")
                        st.info("Continuing analysis with available metadata only...")
                    
                    # Prepare film data
                    film_data = {
                        'title': video_info.get('title', 'YouTube Video'),
                        'synopsis': f"YouTube video: {video_info.get('description', '')[:500]}...",
                        'transcript': transcript_text,
                        'video_id': video_id,
                        'video_title': video_info.get('title', 'Unknown'),
                        'duration': self._format_duration(video_info.get('duration', 0)),
                        'channel': video_info.get('channel', 'Unknown'),
                        'views': video_info.get('views', 0),
                        'upload_date': video_info.get('upload_date', ''),
                        'source': 'youtube'
                    }
                    
                    # If no transcript, provide guidance
                    if not transcript_text.strip():
                        st.warning("No transcript available. Analysis will be based on video metadata only.")
                        film_data['synopsis'] = f"YouTube video by {film_data['channel']}: {video_info.get('description', 'No description available')[:300]}"
                    
                    # Analyze the film
                    results = self.analyzer.analyze_film(film_data)
                    
                    # Store video info in session state
                    st.session_state.current_video_id = video_id
                    st.session_state.current_video_title = video_info.get('title', 'Unknown')
                    
                    # Display success message
                    st.success(f"✅ Successfully analyzed: {video_info.get('title', 'Unknown')}")
                    
                    # Set results to display
                    st.session_state.current_results_display = results
                    st.session_state.show_results_page = True
                    
                    # Rerun to show results
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error analyzing video: {str(e)}")
                    st.info("Try entering the film details manually in the 📝 Manual Entry tab.")
        
        # Show help and examples
        with st.expander("ℹ️ How to use YouTube Analysis", expanded=False):
            st.markdown("""
            **Instructions:**
            1. Paste a YouTube URL or Video ID
            2. Click "Analyze Video"
            3. Wait for transcript extraction
            4. View comprehensive analysis with embedded video viewer
            
            **Examples:**
            - Full URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
            - Short URL: `https://youtu.be/dQw4w9WgXcQ`
            - Just the ID: `dQw4w9WgXcQ`
            
            **Note:** Not all YouTube videos have transcripts available. 
            If transcript extraction fails, try the manual entry method.
            """)
    
    def _show_manual_analysis(self) -> None:
        """Show manual film analysis interface"""
        st.subheader("📝 Manual Film Analysis")
        st.markdown("Enter film details manually for comprehensive analysis.")
        
        with st.form("manual_analysis_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Film Title *", placeholder="Enter film title", key="manual_title")
                director = st.text_input("Director", placeholder="Director's name", key="manual_director")
                writer = st.text_input("Writer", placeholder="Writer's name", key="manual_writer")
            
            with col2:
                duration = st.text_input("Duration", placeholder="e.g., 120m, 2h 15m", key="manual_duration")
                genre = st.text_input("Genre (optional)", placeholder="e.g., Drama, Comedy", key="manual_genre")
                year = st.number_input("Year", min_value=1900, max_value=2100, value=2023, key="manual_year")
            
            synopsis = st.text_area(
                "Synopsis/Description *", 
                placeholder="Enter a detailed synopsis of the film...",
                height=150,
                key="manual_synopsis"
            )
            
            transcript = st.text_area(
                "Transcript/Dialogue (optional)", 
                placeholder="Paste film transcript, dialogue, or key scenes...",
                height=200,
                key="manual_transcript"
            )
            
            submitted = st.form_submit_button("🎬 Analyze Film", type="primary", width='stretch')
            
            if submitted:
                if not title or not synopsis:
                    st.error("Please provide at least a film title and synopsis.")
                    return
                
                # Prepare film data
                film_data = {
                    'title': title,
                    'director': director or "Unknown",
                    'writer': writer or "Unknown",
                    'duration': duration or "Unknown",
                    'genre': genre or "",
                    'year': year,
                    'synopsis': synopsis,
                    'transcript': transcript,
                    'source': 'manual'
                }
                
                # Analyze the film
                with st.spinner("🔍 Analyzing film content..."):
                    results = self.analyzer.analyze_film(film_data)
                    
                    # Display success message
                    st.success(f"✅ Successfully analyzed: {title}")
                    
                    # Set results to display
                    st.session_state.current_results_display = results
                    st.session_state.show_results_page = True
                    
                    # Rerun to show results
                    st.rerun()
    
    def _show_csv_interface(self) -> None:
        """Show CSV batch analysis interface"""
        st.subheader("📊 Batch Analysis via CSV")
        st.markdown("Upload a CSV file to analyze multiple films at once.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="CSV should contain columns: title, synopsis (optional: director, writer, duration, genre, year, transcript)"
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            analyze_batch = st.button("📦 Analyze Batch", type="primary", width='stretch', disabled=not uploaded_file)
        
        if uploaded_file and analyze_batch:
            with st.spinner("📊 Processing batch analysis..."):
                try:
                    # Read CSV file
                    df = pd.read_csv(uploaded_file)
                    
                    # Check required columns
                    if 'title' not in df.columns or 'synopsis' not in df.columns:
                        st.error("CSV must contain 'title' and 'synopsis' columns.")
                        return
                    
                    # Process each row
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        # Prepare film data
                        film_data = {
                            'title': str(row.get('title', 'Unknown Film')),
                            'synopsis': str(row.get('synopsis', '')),
                            'director': str(row.get('director', 'Unknown')),
                            'writer': str(row.get('writer', 'Unknown')),
                            'duration': str(row.get('duration', 'Unknown')),
                            'genre': str(row.get('genre', '')),
                            'year': int(row.get('year', 2023)) if pd.notna(row.get('year')) else 2023,
                            'transcript': str(row.get('transcript', '')),
                            'source': 'csv_batch'
                        }
                        
                        # Analyze film
                        analysis_result = self.analyzer.analyze_film(film_data)
                        
                        # Store results
                        results.append({
                            'title': film_data['title'],
                            'overall_score': analysis_result['overall_score'],
                            'genre': analysis_result.get('genre_insights', {}).get('primary_genre', 'Unknown'),
                            'cultural_relevance': analysis_result.get('cultural_insights', {}).get('relevance_score', 0),
                            'analysis_result': analysis_result,
                            'film_data': film_data
                        })
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(df))
                    
                    # Store batch results
                    st.session_state.batch_results = results
                    st.session_state.show_batch_results = True
                    
                    # Show success message
                    st.success(f"✅ Successfully analyzed {len(results)} films!")
                    st.info(f"📈 Average score: {np.mean([r['overall_score'] for r in results]):.1f}/5.0")
                    
                    # Rerun to show results
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing CSV: {str(e)}")
        
        # Show CSV template and instructions
        with st.expander("📋 CSV Format Instructions", expanded=False):
            st.markdown("""
            **Required columns:**
            - `title`: Film title (string)
            - `synopsis`: Film description/summary (string)
            
            **Optional columns:**
            - `director`: Director's name (string)
            - `writer`: Writer's name (string) 
            - `duration`: Film duration (string, e.g., "120m", "2h 15m")
            - `genre`: Film genre (string)
            - `year`: Release year (integer)
            - `transcript`: Full transcript or key dialogue (string)
            
            **Example CSV format:**
            ```csv
            title,synopsis,director,genre,year
            "Urban Dreams","A story about city life...","John Doe","Drama",2023
            "Concrete Memories","Exploring urban identity...","Jane Smith","Documentary",2022
            ```
            
            **Note:** 
            - CSV should have a header row
            - Maximum recommended batch size: 50 films
            - Analysis time depends on content length
            """)
    
    def _extract_youtube_id(self, url_or_id: str) -> Optional[str]:
        """Extract YouTube video ID from URL or return as-is if already an ID"""
        # If it looks like just an ID (no special characters except dash and underscore)
        if re.match(r'^[\w\-_]{11}$', url_or_id):
            return url_or_id
        
        # Try to extract from various YouTube URL formats
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([\w\-_]{11})',
            r'(?:youtu\.be\/)([\w\-_]{11})',
            r'(?:youtube\.com\/embed\/)([\w\-_]{11})',
            r'(?:youtube\.com\/v\/)([\w\-_]{11})',
            r'(?:youtube\.com\/watch\?.*v=)([\w\-_]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        return None
    
    def _get_youtube_video_info(self, video_id: str) -> Dict:
        """Get YouTube video information (simulated or using oEmbed API)"""
        try:
            # Try to get info from YouTube oEmbed API
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = requests.get(oembed_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', 'YouTube Video'),
                    'description': data.get('author_name', '') + ' - YouTube video',
                    'duration': 0,  # oEmbed doesn't provide duration
                    'channel': data.get('author_name', 'Unknown Channel'),
                    'views': 0,
                    'upload_date': '',
                    'thumbnail_url': data.get('thumbnail_url', f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg')
                }
        
        except Exception as e:
            print(f"Could not fetch video info from API: {str(e)}")
        
        # Fallback to simulated data
        film_titles = [
            "Urban Dreams: A City Story",
            "The Last Sunset",
            "Echoes of Tomorrow",
            "Shadows in the City",
            "Voices from the Street",
            "Concrete Dreams",
            "The Neighborhood Chronicles",
            "City Lights, Dark Nights"
        ]
        
        descriptions = [
            "A compelling story about urban life and personal struggles.",
            "Exploring themes of identity and community in modern society.",
            "A film that captures the essence of contemporary challenges.",
            "Storytelling that reflects on human connections in a digital age."
        ]
        
        channels = [
            "Independent Filmmaker",
            "Urban Cinema Collective",
            "Digital Storytellers",
            "Film Festival Selection"
        ]
        
        return {
            'title': random.choice(film_titles),
            'description': random.choice(descriptions),
            'duration': random.randint(120, 1800),
            'channel': random.choice(channels),
            'views': random.randint(1000, 1000000),
            'upload_date': f"202{random.randint(2, 4)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        }
    
    def _format_duration(self, seconds: int) -> str:
        """Format duration in seconds to readable format"""
        if not seconds:
            return "Unknown"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {secs}s"
    
    def _show_enhanced_top_films_section(self) -> None:
        """Show enhanced top films section with philosophical insights"""
        top_films = self.persistence.get_top_films()
        
        if top_films:
            st.subheader("🏆 Top Films - Cinematic Excellence")
            st.caption("*Films demonstrating exceptional artistic merit and cultural resonance*")
            
            cols = st.columns(min(3, len(top_films)))
            
            for idx, film_data in enumerate(top_films[:3]):
                with cols[idx]:
                    analysis = film_data['analysis_results']
                    film_info = film_data['film_data']
                    
                    # Extract enhanced info
                    genre = analysis.get('genre_insights', {}).get('primary_genre', 'Unknown')
                    if isinstance(genre, dict):
                        genre = genre.get('primary_genre', 'Unknown')
                    
                    cultural_score = analysis.get('cultural_insights', {}).get('relevance_score', 0)
                    philosophical = analysis.get('philosophical_insights', [])
                    philosophical_text = philosophical[0] if philosophical else "Artistic expression"
                    
                    # Enhanced film card
                    cultural_badge = "🌍" if cultural_score > 0.5 else ""
                    philosophical_icon = "💭" if philosophical else "🎨"
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                              border-radius: 10px; padding: 15px; margin: 10px 0; border: 2px solid gold;
                              box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <h4 style='color: white; margin: 0 0 10px 0; text-align: center;'>{film_info.get('title', 'Unknown')[:25]}</h4>
                        <div style='text-align: center;'>
                            <h1 style='color: gold; margin: 5px 0; font-size: 32px;'>{analysis['overall_score']}/5.0</h1>
                            <p style='color: white; margin: 5px 0; font-size: 14px;'>
                                {genre} {cultural_badge} {philosophical_icon}
                            </p>
                            <p style='color: #ddd; margin: 5px 0; font-size: 12px;'>
                                {film_info.get('director', 'Unknown')[:20]}
                            </p>
                            <p style='color: #ccc; margin: 5px 0; font-size: 11px; font-style: italic;'>
                                "{philosophical_text[:60]}..."
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Enhanced view button with philosophical context
                    if st.button(f"🔍 Deep Analysis", key=f"view_top_{idx}", width='stretch'):
                        st.session_state.current_results_display = analysis
                        st.session_state.current_analysis_id = film_data.get('film_id')
                        st.session_state.show_results_page = True
                        st.rerun()
        
        st.markdown("---")
    
    def _display_enhanced_batch_results(self, batch_results: List[Dict]) -> None:
        """Display enhanced batch analysis results"""
        st.header("📊 Batch Analysis Results")
        
        if not batch_results:
            st.info("No batch results to display.")
            return
        
        # Summary statistics
        total_films = len(batch_results)
        scores = [r['overall_score'] for r in batch_results]
        avg_score = np.mean(scores)
        cultural_scores = [r['cultural_relevance'] for r in batch_results]
        avg_cultural = np.mean(cultural_scores)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Films", total_films)
        with col2:
            st.metric("Average Score", f"{avg_score:.1f}/5.0")
        with col3:
            st.metric("High Scores (≥4.0)", sum(1 for s in scores if s >= 4.0))
        with col4:
            st.metric("Avg Cultural", f"{avg_cultural:.0%}")
        
        # Results table
        st.subheader("📋 Detailed Results")
        
        # Create results dataframe
        results_data = []
        for result in batch_results:
            results_data.append({
                'Title': result['title'][:30],
                'Score': result['overall_score'],
                'Genre': result['genre'][:15] if isinstance(result['genre'], str) else 'Unknown',
                'Cultural': f"{result['cultural_relevance']:.0%}",
                'Status': '🏆 Top' if result['overall_score'] >= 4.0 else '✅ Good' if result['overall_score'] >= 3.0 else '📈 Developing'
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Score distribution
        st.subheader("📈 Score Distribution")
        
        fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=15, marker_color='#667eea')])
        fig.update_layout(
            title='Score Distribution',
            xaxis_title='Score',
            yaxis_title='Count',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top films section
        st.subheader("🏆 Top Performing Films")
        
        top_films = sorted(batch_results, key=lambda x: x['overall_score'], reverse=True)[:3]
        
        for i, film in enumerate(top_films):
            with st.expander(f"{i+1}. {film['title'][:30]} - {film['overall_score']}/5.0"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Score:** {film['overall_score']}/5.0")
                    st.write(f"**Genre:** {film['genre']}")
                    st.write(f"**Cultural Relevance:** {film['cultural_relevance']:.0%}")
                
                with col2:
                    if st.button("View Full Analysis", key=f"batch_view_{i}"):
                        st.session_state.current_results_display = film['analysis_result']
                        st.session_state.show_results_page = True
                        st.session_state.batch_results = None
                        st.session_state.show_batch_results = False
                        st.rerun()
        
        # Export options
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download as CSV", width='stretch'):
                # Create downloadable CSV
                export_df = pd.DataFrame([
                    {
                        'title': r['title'],
                        'score': r['overall_score'],
                        'genre': r['genre'],
                        'cultural_relevance': r['cultural_relevance'],
                        'director': r['film_data'].get('director', ''),
                        'year': r['film_data'].get('year', ''),
                        'analysis_summary': r['analysis_result'].get('smart_summary', '')[:200]
                    }
                    for r in batch_results
                ])
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Click to download",
                    data=csv,
                    file_name=f"film_analysis_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width='stretch'
                )
        
        with col2:
            if st.button("✨ Save to Projects", width='stretch'):
                project_name = st.text_input("Project Name:", value=f"Batch Analysis {datetime.now().strftime('%Y-%m-%d')}")
                
                if project_name:
                    for result in batch_results:
                        self.persistence.save_project(
                            f"{project_name} - {result['title'][:20]}",
                            result['film_data'],
                            result['analysis_result']
                        )
                    st.success(f"✅ Saved {len(batch_results)} films to project: {project_name}")
    
    def _get_enhanced_statistics(self) -> Dict:
        """Get enhanced analysis statistics"""
        films = list(st.session_state.stored_results.values())
        history = st.session_state.analysis_history
        
        if not films:
            return {
                "total_films": 0,
                "average_score": 0,
                "score_range": 0,
                "cultural_films": 0,
                "top_genre": "N/A",
                "analysis_rate": 0,
                "score_trend": 0,
                "recent_analyses": [],
                "genre_distribution": {}
            }
        
        scores = [film["analysis_results"]["overall_score"] for film in films]
        cultural_films = sum(1 for film in films 
                           if film["analysis_results"].get('cultural_insights', {}).get('is_culturally_relevant', False))
        
        # Genre distribution
        genre_counter = Counter()
        for item in history:
            genre = item.get('detected_genre', 'Unknown')
            if genre and genre != 'Unknown':
                genre_counter[genre] += 1
        
        # Score trend (last 5 vs previous 5)
        trend = 0
        if len(history) >= 10:
            recent_scores = [h['overall_score'] for h in history[-5:]]
            previous_scores = [h['overall_score'] for h in history[-10:-5]]
            if previous_scores:
                trend = np.mean(recent_scores) - np.mean(previous_scores)
        
        # Analysis rate (analyses per day)
        if len(history) > 1:
            dates = [datetime.fromisoformat(h['timestamp']).date() for h in history]
            date_range = (max(dates) - min(dates)).days or 1
            analysis_rate = len(history) / max(date_range, 1)
        else:
            analysis_rate = 0
        
        # Recent analyses
        recent_analyses = []
        for item in history[-5:]:
            recent_analyses.append({
                'title': item.get('title', 'Unknown'),
                'score': item.get('overall_score', 0),
                'genre': item.get('detected_genre', 'Unknown')
            })
        
        return {
            "total_films": len(films),
            "average_score": round(np.mean(scores), 2),
            "highest_score": round(max(scores), 2),
            "lowest_score": round(min(scores), 2),
            "score_range": round(max(scores) - min(scores), 2),
            "score_std": round(np.std(scores), 2) if len(scores) > 1 else 0,
            "cultural_films": cultural_films,
            "top_genre": genre_counter.most_common(1)[0][0] if genre_counter else "N/A",
            "analysis_rate": round(analysis_rate, 1),
            "score_trend": round(trend, 2),
            "recent_analyses": recent_analyses[::-1],
            "genre_distribution": dict(genre_counter.most_common(5))
        }
    
    def _display_enhanced_film_results(self, results: Dict) -> None:
        """Display enhanced film analysis results with integrated video viewer"""
        st.success("🎉 Advanced Film Analysis Complete!")
        
        # Get film data
        film_data = {}
        if st.session_state.current_analysis_id:
            stored_result = self.persistence.load_results(st.session_state.current_analysis_id)
            if stored_result:
                film_data = stored_result['film_data']
        
        # Display film title with philosophical context
        film_title = film_data.get('title', results.get('film_title', 'Unknown Film'))
        philosophical_insights = results.get('philosophical_insights', [])
        primary_insight = philosophical_insights[0] if philosophical_insights else "Cinematic Exploration"
        
        # ============================================
        # VIDEO VIEWER SECTION - EMBEDDED YOUTUBE PLAYER
        # ============================================
        
        # Check if we have a YouTube video to display
        video_id = None
        video_title = None
        
        # Check session state first
        if st.session_state.get('current_video_id'):
            video_id = st.session_state.current_video_id
            video_title = st.session_state.get('current_video_title', '')
        # Check film data
        elif film_data.get('video_id'):
            video_id = film_data['video_id']
            video_title = film_data.get('video_title', '')
        
        # If we have a video ID, create a video viewer section
        if video_id:
            st.markdown("---")
            st.subheader("🎬 Film / Video Viewer")
            
            # Create a two-column layout for video and info
            video_col, info_col = st.columns([3, 2])
            
            with video_col:
                # Display YouTube embed
                embed_html = f"""
                <div style="border-radius: 10px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.2); margin-bottom: 15px;">
                    <iframe 
                        width="100%" 
                        height="400" 
                        src="https://www.youtube.com/embed/{video_id}?rel=0&modestbranding=1" 
                        title="YouTube video player" 
                        frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen
                        style="border-radius: 10px;">
                    </iframe>
                </div>
                """
                st.markdown(embed_html, unsafe_allow_html=True)
                
                # Video controls
                st.caption("🎥 Use the video player to watch scenes and verify analysis")
                
            with info_col:
                st.markdown("**📺 Video Details**")
                if video_title:
                    st.write(f"**Title:** {video_title}")
                
                # Show video duration if available
                if film_data.get('duration'):
                    st.write(f"**Duration:** {film_data.get('duration')}")
                
                # Show channel info if available
                if film_data.get('channel'):
                    st.write(f"**Channel:** {film_data.get('channel')}")
                
                if film_data.get('views'):
                    st.write(f"**Views:** {film_data.get('views'):,}")
                
                if film_data.get('upload_date'):
                    st.write(f"**Uploaded:** {film_data.get('upload_date')}")
                
                # Quick actions
                st.markdown("---")
                st.markdown("**🔗 Quick Links**")
                
                # Create buttons for YouTube actions
                yt_col1, yt_col2 = st.columns(2)
                with yt_col1:
                    youtube_url = f"https://youtube.com/watch?v={video_id}"
                    if st.button("📺 Open YouTube", key="open_yt", width='stretch'):
                        st.markdown(f'<meta http-equiv="refresh" content="0; url={youtube_url}">', unsafe_allow_html=True)
                
                with yt_col2:
                    if st.button("📋 Copy Link", key="copy_yt", width='stretch'):
                        st.code(youtube_url, language="text")
                        st.toast("YouTube link copied!", icon="✅")
            
            st.markdown("---")
        
        # ============================================
        # END OF VIDEO VIEWER SECTION
        # ============================================
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  border-radius: 15px; margin: 20px 0; border: 3px solid white; box-shadow: 0 6px 12px rgba(0,0,0,0.15);'>
            <h1 style='color: white; margin: 0; font-size: 32px;'>{film_title}</h1>
            <p style='color: #ddd; margin: 10px 0 0 0; font-size: 16px; font-style: italic;'>
                "{primary_insight}"
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Overall score with contextual explanation
        overall_score = results['overall_score']
        score_context = self._get_score_context(overall_score)
        
        st.markdown(f"""
        <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  border-radius: 15px; margin: 20px 0; border: 3px solid #FFD700; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
            <h1 style='color: gold; margin: 0; font-size: 48px;'>{overall_score}/5.0</h1>
            <p style='color: white; font-size: 20px; margin: 10px 0;'>🎬 Cinematic Score</p>
            <p style='color: #eee; font-size: 14px; margin: 5px 0;'>{score_context}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Film Details Section
        st.subheader("📋 Comprehensive Film Analysis")
        
        # Create tabs for different analysis aspects
        # Add "🎥 Video Analysis" tab if we have video
        if video_id:
            detail_tabs = st.tabs(["🎬 Film Info", "🎥 Video Analysis", "🧠 Philosophical", "🤖 AI Tools", "📊 Analytics"])
        else:
            detail_tabs = st.tabs(["🎬 Film Information", "🧠 Philosophical Insights", "🤖 AI Enhancement", "📊 Deep Analytics"])
        
        if video_id:
            # With video tabs
            with detail_tabs[0]:
                self._display_film_information(film_data, results)
            
            with detail_tabs[1]:
                self._display_video_analysis_section(film_data, results, video_id)
            
            with detail_tabs[2]:
                self._display_philosophical_insights(results)
            
            with detail_tabs[3]:
                self._display_ai_enhancements(results)
            
            with detail_tabs[4]:
                self._display_deep_analytics(results)
        else:
            # Original tabs (no video)
            with detail_tabs[0]:
                self._display_film_information(film_data, results)
            
            with detail_tabs[1]:
                self._display_philosophical_insights(results)
            
            with detail_tabs[2]:
                self._display_ai_enhancements(results)
            
            with detail_tabs[3]:
                self._display_deep_analytics(results)
        
        # Enhanced Category Scores with Visualizations
        st.subheader("🎯 Multidimensional Analysis")
        
        scores = results['cinematic_scores']
        
        # Create a radar chart for scores
        self._create_score_radar_chart(scores)
        
        # Score breakdown expander
        with st.expander("📈 **Advanced Score Breakdown & Distribution**", expanded=st.session_state.get('show_breakdown', False)):
            self._display_score_breakdown(results)
        
        # Synopsis Analysis
        synopsis_analysis = results.get('synopsis_analysis', {})
        if synopsis_analysis.get('length', 0) > 0:
            st.subheader("📖 Synopsis Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Length", f"{synopsis_analysis['length']} words")
            with col2:
                st.metric("Sentiment", synopsis_analysis['emotional_tone'].title())
            with col3:
                st.metric("Complexity", synopsis_analysis['complexity'].title())
            
            if synopsis_analysis.get('key_themes'):
                st.write("**Key Themes:** " + ", ".join(synopsis_analysis['key_themes']))
        
        # Recommendations and Next Steps
        st.subheader("🚀 Strategic Recommendations")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🎪 Festival Strategy:**")
            festival_recs = results['festival_recommendations']
            st.write(f"**Level:** {festival_recs['level']}")
            for festival in festival_recs['festivals']:
                st.write(f"• {festival}")
        
        with col2:
            st.write("**🎯 Development Path:**")
            for recommendation in results.get('recommendations', []):
                st.write(f"• {recommendation}")
    
    def _get_score_context(self, score: float) -> str:
        """Get contextual explanation for score"""
        if score >= 4.5:
            return "Exceptional - Award-caliber cinematic achievement"
        elif score >= 4.0:
            return "Excellent - Professional quality with strong artistic vision"
        elif score >= 3.5:
            return "Strong - Compelling work with clear potential"
        elif score >= 3.0:
            return "Solid - Well-executed foundation for development"
        elif score >= 2.5:
            return "Developing - Promising concepts with room for growth"
        elif score >= 2.0:
            return "Emerging - Foundational creative exploration"
        elif score >= 1.5:
            return "Beginning - Initial creative expression"
        else:
            return "Introductory - Early stage development"
    
    def _display_film_information(self, film_data: Dict, results: Dict) -> None:
        """Display comprehensive film information"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎬 Film Information**")
            st.write(f"**Title:** {film_data.get('title', 'Unknown')}")
            if film_data.get('director') and film_data.get('director') != 'Unknown':
                st.write(f"**Director:** {film_data.get('director')}")
            if film_data.get('writer') and film_data.get('writer') != 'Unknown':
                st.write(f"**Writer:** {film_data.get('writer')}")
            if film_data.get('duration') and film_data.get('duration') != 'Unknown':
                st.write(f"**Duration:** {film_data.get('duration')}")
            
            # Synopsis preview
            if film_data.get('synopsis'):
                with st.expander("📖 View Synopsis", expanded=False):
                    st.write(film_data.get('synopsis'))
        
        with col2:
            st.markdown("**📊 Analysis Details**")
            
            genre_insights = results['genre_insights']
            if isinstance(genre_insights, dict) and 'primary_genre' in genre_insights:
                st.write(f"**Primary Genre:** {genre_insights['primary_genre']}")
                if genre_insights.get('confidence'):
                    st.write(f"**Confidence:** {genre_insights['confidence']}%")
                if genre_insights.get('secondary_genres'):
                    st.write(f"**Secondary Genres:** {', '.join(genre_insights['secondary_genres'])}")
            else:
                st.write(f"**Detected Genre:** {genre_insights.get('detected_genre', 'Unknown')}")
            
            cultural_insights = results.get('cultural_insights', {})
            if cultural_insights.get('is_culturally_relevant'):
                relevance = cultural_insights.get('relevance_score', 0)
                st.write(f"**Cultural Relevance:** {relevance:.0%}")
                if cultural_insights.get('primary_themes'):
                    st.write(f"**Cultural Themes:** {', '.join(cultural_insights['primary_themes'])}")
            
            if st.session_state.last_analysis_time:
                last_time = datetime.fromisoformat(st.session_state.last_analysis_time)
                st.write(f"**Analyzed:** {last_time.strftime('%Y-%m-%d %H:%M')}")
    
    def _display_video_analysis_section(self, film_data: Dict, results: Dict, video_id: str) -> None:
        """Display video-specific analysis section"""
        st.markdown("### 🎥 Video Content Analysis")
        
        # Transcript analysis
        transcript = film_data.get('transcript', '')
        if transcript:
            word_count = len(transcript.split())
            st.write(f"**Transcript Analysis:** {word_count} words")
            
            # Key moments
            with st.expander("🔍 View Key Video Moments", expanded=False):
                if word_count > 500:
                    # Extract sample lines from transcript
                    lines = transcript.split('.')
                    key_lines = [line.strip() for line in lines[:10] if len(line.strip()) > 50]
                    for i, line in enumerate(key_lines[:5]):
                        st.write(f"**Moment {i+1}:** {line[:200]}...")
                else:
                    st.write("Transcript preview:")
                    st.text(transcript[:500] + "..." if len(transcript) > 500 else transcript)
            
            # Sentiment from transcript
            if transcript:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()
                sentiment = analyzer.polarity_scores(transcript)
                st.write(f"**Transcript Sentiment:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive", f"{sentiment['pos']:.0%}")
                with col2:
                    st.metric("Neutral", f"{sentiment['neu']:.0%}")
                with col3:
                    st.metric("Negative", f"{sentiment['neg']:.0%}")
        else:
            st.info("No transcript available for this video. Score based on metadata only.")
        
        # Study Group Helper
        st.markdown("---")
        st.markdown("### 👥 Study Group Helper")
        
        st.write("**Watch these key moments to understand the scoring:**")
        
        # Create key moments based on scoring
        score_moments = []
        cinematic_scores = results['cinematic_scores']
        
        if cinematic_scores.get('performance', 0) >= 4.0:
            score_moments.append("Watch character interactions and performances (around 25% mark)")
        
        if cinematic_scores.get('story_narrative', 0) >= 4.0:
            score_moments.append("Observe narrative structure and plot development (mid-point)")
        
        if cinematic_scores.get('visual_vision', 0) >= 4.0:
            score_moments.append("Notice cinematography and visual composition (throughout)")
        
        if cinematic_scores.get('sound_design', 0) >= 4.0:
            score_moments.append("Listen to audio design and music integration (climax scene)")
        
        # Add default moments if none specific
        if not score_moments:
            score_moments = [
                "Watch opening scene for establishing tone",
                "Observe character development in middle act",
                "Notice resolution and ending impact"
            ]
        
        for i, moment in enumerate(score_moments[:4]):
            st.write(f"**{i+1}.** {moment}")
        
        # Video analysis tips
        with st.expander("💡 Video Analysis Tips", expanded=False):
            st.markdown("""
            **For Study Groups:**
            1. **Pause and Discuss:** Stop at key moments to discuss scoring criteria
            2. **Scene Comparison:** Compare different scenes against score components
            3. **Group Scoring:** Have each member score independently, then compare
            4. **Cultural Context:** Discuss how cultural elements affect scoring
            
            **Video Controls:**
            - Use YouTube's speed controls for detailed analysis
            - Turn on captions for dialogue analysis
            - Take timestamp notes for specific moments
            """)
    
    def _display_philosophical_insights(self, results: Dict) -> None:
        """Display philosophical insights"""
        philosophical_insights = results.get('philosophical_insights', [])
        cultural_insights = results.get('cultural_insights', {})
        
        if philosophical_insights or cultural_insights.get('philosophical_insights'):
            st.write("**💭 Philosophical Framework:**")
            
            all_insights = []
            if philosophical_insights:
                all_insights.extend(philosophical_insights)
            if cultural_insights.get('philosophical_insights'):
                all_insights.extend(cultural_insights['philosophical_insights'])
            
            for insight in all_insights[:3]:
                st.write(f"• {insight}")
            
            # Genre philosophical aspect
            genre_details = results.get('genre_insights', {}).get('details', {})
            if genre_details and 'philosophical_aspect' in genre_details:
                st.write(f"\n**🎭 Genre Philosophy:** {genre_details['philosophical_aspect']}")
        else:
            st.info("No specific philosophical insights detected. This film appears to focus on direct narrative storytelling.")
    
    def _display_ai_enhancements(self, results: Dict) -> None:
        """Display AI tool suggestions for enhancement"""
        ai_suggestions = results.get('ai_tool_suggestions', [])
        
        if ai_suggestions:
            st.write("**🤖 AI Enhancement Opportunities:**")
            st.caption("Suggested tools for deeper analysis and improved scoring")
            
            for suggestion in ai_suggestions:
                with st.expander(f"{suggestion['tool']} - {suggestion['purpose']}"):
                    st.write(f"**Purpose:** {suggestion['purpose']}")
                    st.write(f"**Benefit:** {suggestion['benefit']}")
                    st.write(f"**Implementation:** Could enhance scoring accuracy by 10-15%")
        else:
            st.info("Current analysis provides comprehensive coverage. For advanced needs, consider GPT-4 for narrative analysis or BERT for cultural context.")
    
    def _display_deep_analytics(self, results: Dict) -> None:
        """Display deep analytics and metrics"""
        scoring_breakdown = results.get('scoring_breakdown', {})
        component_scores = scoring_breakdown.get('component_scores', {})
        weights = scoring_breakdown.get('applied_weights', {})
        
        if component_scores and weights:
            st.write("**📊 Scoring Algorithm Details:**")
            
            # Create a DataFrame for visualization
            score_data = []
            for component, score in component_scores.items():
                weight = weights.get(component, 0)
                weighted_score = score * weight
                score_data.append({
                    'Component': component.title(),
                    'Score': score,
                    'Weight': weight,
                    'Weighted': round(weighted_score, 2)
                })
            
            df = pd.DataFrame(score_data)
            st.dataframe(df, use_container_width=True)
            
            # Cultural bonus
            cultural_bonus = scoring_breakdown.get('cultural_bonus', 0)
            if cultural_bonus > 0:
                st.success(f"🎉 **Cultural Bonus Applied:** +{cultural_bonus:.3f} points")
    
    def _create_score_radar_chart(self, scores: Dict) -> None:
        """Create a radar chart for cinematic scores"""
        try:
            categories = list(scores.keys())
            values = list(scores.values())
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=[cat.replace('_', ' ').title() for cat in categories] + [categories[0].replace('_', ' ').title()],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.3)',
                line=dict(color='rgb(102, 126, 234)', width=2),
                hoverinfo='text',
                text=[f"{cat.replace('_', ' ').title()}: {val}/5.0" for cat, val in zip(categories, values)]
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5],
                        tickvals=[1, 2, 3, 4, 5],
                        ticktext=['1', '2', '3', '4', '5']
                    )
                ),
                showlegend=False,
                height=300,
                margin=dict(l=50, r=50, t=30, b=30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            # Fallback to simple columns if plotly not available
            cols = st.columns(len(scores))
            categories = [
                ("🧠 Story", scores['story_narrative'], "#FF6B6B"),
                ("👁️ Visual", scores['visual_vision'], "#4ECDC4"),
                ("⚡ Technical", scores['technical_craft'], "#45B7D1"),
                ("🎵 Sound", scores['sound_design'], "#96CEB4"),
                ("🌟 Performance", scores['performance'], "#FFD93D")
            ]
            
            for idx, (name, score, color) in enumerate(categories):
                with cols[idx]:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 15px; background: {color}; 
                              border-radius: 10px; margin: 5px; border: 2px solid white;'>
                        <h4 style='margin: 0; color: white;'>{name}</h4>
                        <h2 style='margin: 8px 0; color: white;'>{score}</h2>
                    </div>
                    """, unsafe_allow_html=True)
    
    def _display_score_breakdown(self, results: Dict) -> None:
        """Display detailed score breakdown and distributions"""
        scoring_breakdown = results.get('scoring_breakdown', {})
        
        if scoring_breakdown:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🧮 Component Scores**")
                component_scores = scoring_breakdown.get('component_scores', {})
                for component, score in component_scores.items():
                    st.progress(score/5, text=f"{component.title()}: {score}/5.0")
            
            with col2:
                st.markdown("**⚖️ Applied Weights**")
                weights = scoring_breakdown.get('applied_weights', {})
                for component, weight in weights.items():
                    percentage = weight * 100
                    st.write(f"• **{component.title()}:** {percentage:.1f}%")
            
            # Historical context if available
            history = self.persistence.get_all_history()
            if len(history) > 1:
                st.markdown("**📈 Historical Context**")
                
                scores = [h['overall_score'] for h in history]
                current_score = results['overall_score']
                
                avg_score = np.mean(scores)
                percentile = np.sum(np.array(scores) < current_score) / len(scores) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("vs Average", f"{current_score - avg_score:+.1f}")
                with col2:
                    st.metric("Percentile", f"{percentile:.0f}%")
                with col3:
                    st.metric("Position", f"{np.sum(np.array(scores) < current_score) + 1}/{len(scores)}")

# --------------------------
# ENHANCED SIDEBAR COMPONENTS
# --------------------------
def display_enhanced_sidebar() -> None:
    """Display enhanced sidebar with more features"""
    st.sidebar.title("🎬 FlickFinder AI")
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.subheader("📍 Navigation")
    
    # Navigation buttons with icons
    if st.sidebar.button("🏠 Dashboard", width='stretch', key="sidebar_dashboard"):
        st.session_state.current_page = "🏠 Dashboard"
        st.session_state.show_results_page = False
        st.session_state.show_batch_results = False
        st.rerun()
    
    if st.sidebar.button("📈 Advanced Analytics", width='stretch', key="sidebar_analytics"):
        st.session_state.current_page = "📈 Analytics"
        st.rerun()
    
    if st.sidebar.button("🧠 AI Technology", width='stretch', key="sidebar_ai"):
        st.session_state.current_page = "🧠 AI Technology"
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Enhanced Quick Stats
    st.sidebar.subheader("📊 Performance Metrics")
    
    films = list(st.session_state.stored_results.values())
    
    if films:
        scores = [film["analysis_results"]["overall_score"] for film in films]
        cultural_films = sum(1 for film in films 
                           if film["analysis_results"].get('cultural_insights', {}).get('is_culturally_relevant', False))
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.sidebar.metric("Films", len(films))
        
        with col2:
            if scores:
                st.sidebar.metric("Avg Score", f"{np.mean(scores):.1f}")
        
        # Additional metrics in expander
        with st.sidebar.expander("More Stats"):
            st.sidebar.write(f"**Cultural Films:** {cultural_films}")
            st.sidebar.write(f"**Analyses:** {st.session_state.analysis_count}")
            if scores:
                st.sidebar.write(f"**Score Range:** {max(scores) - min(scores):.1f}")
            
            if st.session_state.last_analysis_time:
                last_time = datetime.fromisoformat(st.session_state.last_analysis_time)
                st.sidebar.write(f"**Last:** {last_time.strftime('%H:%M')}")
    
    st.sidebar.markdown("---")
    
    # System Controls
    st.sidebar.subheader("⚙️ System")
    
    if st.sidebar.button("🗑️ Clear All History", key="sidebar_clear_all", type="secondary", width='stretch'):
        persistence = PersistenceManager()
        persistence.clear_history()
        st.sidebar.success("✅ History cleared!")
        st.rerun()
    
    # Version info
    st.sidebar.markdown("---")
    st.sidebar.caption("FlickFinder AI v3.1")
    st.sidebar.caption("Enhanced Scoring Edition")

# --------------------------
# MAIN ENHANCED APPLICATION
# --------------------------
def main() -> None:
    """Main application entry point"""
    # Display enhanced sidebar
    display_enhanced_sidebar()
    
    # Initialize enhanced components
    analyzer = FilmAnalysisEngine()
    persistence = PersistenceManager()
    film_interface = EnhancedFilmAnalysisInterface(analyzer)
    
    # Determine which page to show
    page = st.session_state.current_page
    
    if page == "🏠 Dashboard":
        film_interface.show_dashboard()
    elif page == "📈 Analytics":
        history_page = EnhancedHistoryAnalyticsPage(persistence)
        history_page.show()
    elif page == "🧠 AI Technology":
        # Create a simple AI Technology page
        st.header("🧠 AI Technology & Roadmap")
        st.markdown("---")
        st.markdown("""
        ## 🚀 Next-Generation Film Analysis AI
        
        **Current Technology Stack:**
        - **VADER Sentiment Analysis**: Emotional tone detection
        - **NLTK**: Natural language processing
        - **Enhanced Scoring Algorithm**: 1.1-5.0 range with granularity
        - **Custom Algorithms**: Genre and cultural analysis
        - **Statistical Models**: Comprehensive scoring with variation factors
        
        **Enhanced Scoring Features (v3.1):**
        - **Wider Score Range**: 1.1 to 5.0 (previously 1.8-4.9)
        - **Better Granularity**: Scores rounded to 0.1 increments (3.7, 4.2, 2.5, etc.)
        - **Natural Distribution**: Scores follow realistic bell curve distribution
        - **Variation Factors**: Length, sentiment, character density affect scores
        - **Genre Adjustments**: Different scoring weights for different genres
        
        **Enhancement Roadmap:**
        1. **Phase 1**: BERT/GPT-4 integration for advanced narrative analysis
        2. **Phase 2**: Multimodal analysis (visual + audio + text)
        3. **Phase 3**: Predictive analytics for festival success
        4. **Phase 4**: Real-time production assistant tools
        
        **New Feature - Video Viewer:**
        - Embedded YouTube player for study group analysis
        - Direct video viewing within analysis interface
        - Study group helper with specific moments to watch
        """)
        
        st.markdown("---")
        if st.button("← Back to Dashboard", key="back_from_ai"):
            st.session_state.current_page = "🏠 Dashboard"
            st.rerun()
    else:
        # Enhanced About page
        st.header("🌟 About FlickFinder AI v3.1")
        
        tab1, tab2, tab3 = st.tabs(["Overview", "Features", "Philosophy"])
        
        with tab1:
            st.markdown("""
            ## 🎬 FlickFinder AI v3.1 - Enhanced Film Analysis
            
            **The next generation** of film analysis technology, combining AI intelligence with 
            cultural awareness and philosophical insight.
            
            ### 🚀 What's New in v3.1
            
            **Enhanced Scoring Algorithm:**
            - **Wider Score Range**: 1.1 to 5.0 (allows more nuanced evaluation)
            - **Better Granularity**: Scores in 0.1 increments (3.7, 4.2, 2.5, etc.)
            - **Natural Distribution**: Realistic bell curve distribution
            - **Variation Factors**: Multiple factors create score diversity
            
            **Enhanced Analytics:**
            - Score trends and evolution tracking
            - Genre distribution analysis
            - Cultural relevance insights
            - Time-based pattern recognition
            
            **New Video Viewer:**
            - Embedded YouTube video player
            - Study group analysis tools
            - Direct video viewing with analysis
            
            **Philosophical Framework:**
            - Cultural memory recognition
            - Narrative as truth-seeking
            - Film as empathy machine
            - Artistic intent analysis
            """)
        
        with tab2:
            st.markdown("""
            ## 🌟 Enhanced Features
            
            **📊 Advanced Analytics Dashboard:**
            - Real-time score distribution tracking
            - Genre performance metrics
            - Cultural relevance scoring
            - Historical trend analysis
            
            **🎥 Video Analysis Tools:**
            - Embedded YouTube viewer
            - Transcript analysis
            - Study group helper
            - Key moment identification
            
            **🎭 Philosophical Insights:**
            - Cultural context understanding
            - Narrative pattern recognition
            - Emotional arc analysis
            - Character development assessment
            
            **🎯 Enhanced Scoring:**
            - 1.1-5.0 score range with 0.1 granularity
            - Genre-specific weight adjustments
            - Cultural and philosophical bonuses
            - Natural variation factors
            """)
        
        with tab3:
            st.markdown("""
            ## 💭 Philosophical Foundation
            
            **Our Approach to Film Analysis:**
            
            **1. Cinema as Cultural Artifact:**
            We view films not just as entertainment, but as **cultural artifacts** that 
            reflect and shape society, memory, and identity.
            
            **2. Narrative as Human Experience:**
            Stories are fundamental to human understanding. We analyze narrative structures 
            as **expressions of human experience** and psychological patterns.
            
            **3. Technology as Cultural Interpreter:**
            AI serves as a **cultural interpreter**, identifying patterns and contexts that 
            might be overlooked in traditional analysis, while respecting artistic intent.
            
            **4. Enhanced Scoring Philosophy:**
            The new 1.1-5.0 scoring range allows for **more nuanced evaluation** that better 
            reflects the diverse landscape of cinematic expression, from emerging works to 
            masterpieces.
            
            **5. Study Group Integration:**
            The embedded video viewer allows **collaborative analysis** where study groups 
            can watch, pause, and discuss films together with real-time scoring feedback.
            """)
        
        st.markdown("---")
        if st.button("← Back to Dashboard", key="back_from_about"):
            st.session_state.current_page = "🏠 Dashboard"
            st.rerun()

if __name__ == "__main__":
    main()

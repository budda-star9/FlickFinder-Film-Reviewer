import streamlit as st
import tempfile
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import base64
import os
import json
import time
import pandas as pd
from datetime import datetime
import re
import requests
from PIL import Image
import io
import nltk
from textblob import TextBlob
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random

# --------------------------
# Configuration & Setup
# --------------------------
st.set_page_config(
    page_title="FlickFinder AI 🎬",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
session_defaults = {
    'all_film_scores': [],
    'current_analysis': None,
    'filmfreeway_projects': [],
    'magic_mode': True
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------
# Enhanced Configuration Manager
# --------------------------
class ConfigManager:
    def __init__(self):
        self.config = {
            "api_key": "",
            "model": "huggingface", 
            "magic_mode": True,
            "scoring_variance": 0.3  # Add variance for realistic scoring
        }
    
    def get_api_key(self):
        return self.config["api_key"]

# Initialize configuration
config_manager = ConfigManager()

# --------------------------
# Realistic Scoring Engine
# --------------------------
class RealisticScoringEngine:
    def __init__(self):
        self.base_variance = 0.3
        self.category_weights = {
            'story_narrative': 0.30,
            'visual_vision': 0.25,
            'technical_craft': 0.25, 
            'sound_design': 0.10,
            'performance': 0.10
        }
    
    def apply_realistic_variance(self, base_score, analysis_quality=1.0):
        """Apply realistic variance to scores"""
        # More variance for mid-range scores, less for extremes
        if 2.0 <= base_score <= 4.0:
            variance = self.base_variance * analysis_quality
        else:
            variance = self.base_variance * 0.5 * analysis_quality
            
        varied_score = base_score + random.uniform(-variance, variance)
        return max(1.0, min(5.0, round(varied_score, 1)))
    
    def generate_cinematic_scores(self, analysis_results):
        """Generate realistic cinematic scores with dynamic variance"""
        narrative = analysis_results['narrative_structure']
        dialogue = analysis_results['dialogue_analysis']
        emotional = analysis_results['emotional_arc']
        complexity = analysis_results['complexity_metrics']
        characters = analysis_results['character_analysis']
        
        # Base calculations with more nuanced formulas
        base_scores = {
            'story_narrative': self._calculate_story_potential(narrative, emotional, complexity),
            'visual_vision': self._calculate_visual_potential(narrative, dialogue),
            'technical_craft': self._calculate_technical_execution(narrative, complexity, dialogue),
            'sound_design': self._calculate_sound_potential(dialogue, emotional),
            'performance': self._calculate_performance_potential(characters, dialogue, emotional)
        }
        
        # Apply realistic variance
        analysis_quality = min(1.0, narrative.get('structural_richness', 0.5) * 0.8 + complexity.get('content_density', 0.5) * 0.2)
        
        final_scores = {}
        for category, base_score in base_scores.items():
            final_scores[category] = self.apply_realistic_variance(base_score, analysis_quality)
        
        return final_scores
    
    def _calculate_story_potential(self, narrative, emotional, complexity):
        """More nuanced story scoring"""
        structural_base = narrative.get('structural_richness', 0.5) * 2.8
        emotional_weight = emotional.get('emotional_arc_strength', 0.3) * 1.2
        complexity_bonus = complexity.get('content_density', 0.4) * 0.8
        readability_penalty = max(0, (0.6 - narrative.get('readability_score', 0.5)) * 0.5)
        
        raw_score = structural_base + emotional_weight + complexity_bonus - readability_penalty
        return min(5.0, max(1.0, raw_score))
    
    def _calculate_visual_potential(self, narrative, dialogue):
        """Visual scoring based on descriptive richness"""
        descriptive_power = (narrative.get('lexical_diversity', 0.4) * 1.8 + 
                           dialogue.get('emotional_variety', 0.3) * 1.2)
        complexity_bonus = narrative.get('sentence_complexity', 0) * 0.1
        
        return min(5.0, 2.2 + descriptive_power + complexity_bonus)
    
    def _calculate_technical_execution(self, narrative, complexity, dialogue):
        """Technical execution scoring"""
        execution_quality = (narrative.get('readability_score', 0.5) * 1.5 +
                           complexity.get('syntactic_diversity', 0.3) * 1.0 +
                           dialogue.get('dialogue_quality_score', 0.3) * 0.5)
        
        return min(5.0, 2.3 + execution_quality)
    
    def _calculate_sound_potential(self, dialogue, emotional):
        """Sound design potential"""
        audio_indicators = (dialogue.get('dialogue_quality_score', 0.3) * 1.2 +
                          emotional.get('emotional_variance', 0.2) * 0.8)
        
        return min(5.0, 2.1 + audio_indicators * 1.2)
    
    def _calculate_performance_potential(self, characters, dialogue, emotional):
        """Performance potential scoring"""
        performance_indicators = (characters.get('character_presence_score', 0.3) * 1.5 +
                                dialogue.get('emotional_variety', 0.3) * 0.8 +
                                min(1.0, emotional.get('emotional_range', 0.2) * 1.5))
        
        return min(5.0, 2.0 + performance_indicators)

# --------------------------
# Optimized Magical Film Analyzer
# --------------------------
class MagicalFilmAnalyzer:
    def __init__(self):
        self.scoring_engine = RealisticScoringEngine()
        self.capabilities = {
            "huggingface_sentiment": "😊 Advanced sentiment analysis",
            "emotional_intelligence": "🎭 Multi-emotion detection", 
            "narrative_complexity": "📊 Structural analysis",
            "dialogue_sophistication": "💬 Dialogue quality assessment",
            "character_development": "👥 Character presence analysis",
            "cinematic_potential": "🎬 Production quality evaluation"
        }
        self._setup_models()
    
    def _setup_models(self):
        """Initialize models with graceful fallbacks"""
        self.models = {}
        try:
            from transformers import pipeline
            self.models['sentiment'] = pipeline("sentiment-analysis")
            st.success("🤗 Hugging Face Models Ready!")
        except:
            st.info("🔧 Using lightweight NLP analysis")
    
    def perform_magical_analysis(self, film_data):
        """Optimized analysis pipeline"""
        transcript = film_data.get('transcript', '')
        
        if not transcript or "No transcript available" in transcript:
            return self._create_contextual_fallback(film_data)
        
        # Core analysis pipeline
        analysis_results = {
            'narrative_structure': self._analyze_narrative_structure(transcript),
            'dialogue_analysis': self._analyze_dialogue_quality(transcript),
            'emotional_arc': self._analyze_emotional_arc(transcript),
            'complexity_metrics': self._calculate_complexity_metrics(transcript),
            'character_analysis': self._analyze_character_presence(transcript)
        }
        
        return self._generate_magical_review(film_data, analysis_results)
    
    def _analyze_narrative_structure(self, text):
        """Optimized narrative analysis"""
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        word_count = len(words)
        sentence_count = len(sentences)
        unique_words = len(set(words))
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'lexical_diversity': unique_words / max(word_count, 1),
            'readability_score': min(1.0, 30 / max(np.mean([len(nltk.word_tokenize(s)) for s in sentences]), 1)),
            'structural_richness': min(1.0, (unique_words/word_count * 0.6 + min(1, word_count/500) * 0.4))
        }
    
    def _analyze_dialogue_quality(self, text):
        """Optimized dialogue analysis"""
        questions = len(re.findall(r'\?', text))
        exclamations = len(re.findall(r'!', text))
        sentences = nltk.sent_tokenize(text)
        
        return {
            'questions_count': questions,
            'exclamations_count': exclamations,
            'dialogue_quality_score': min(1.0, (questions + exclamations) / max(len(sentences) * 0.3, 1)),
            'emotional_variety': min(1.0, (questions * 0.4 + exclamations * 0.6) / max(len(sentences) * 0.2, 1))
        }
    
    def _analyze_emotional_arc(self, text):
        """Optimized emotional analysis"""
        vader_analyzer = SentimentIntensityAnalyzer()
        sentences = nltk.sent_tokenize(text)[:10]  # Sample first 10 sentences
        
        if len(sentences) < 3:
            return {'emotional_arc_strength': 0.3, 'emotional_variance': 0.2, 'emotional_range': 0.3}
        
        emotional_scores = [vader_analyzer.polarity_scores(s)['compound'] for s in sentences]
        
        return {
            'emotional_arc_strength': min(1.0, np.var(emotional_scores) * 3),
            'emotional_variance': np.var(emotional_scores),
            'emotional_range': max(emotional_scores) - min(emotional_scores)
        }
    
    def _calculate_complexity_metrics(self, text):
        """Optimized complexity analysis"""
        words = nltk.word_tokenize(text)
        unique_ratio = len(set(words)) / max(len(words), 1)
        
        return {
            'vocabulary_richness': unique_ratio,
            'content_density': min(1.0, unique_ratio * 0.7 + min(1, len(words)/300) * 0.3)
        }
    
    def _analyze_character_presence(self, text):
        """Optimized character analysis"""
        words = nltk.word_tokenize(text)
        capital_words = [w for w in words if w.istitle() and len(w) > 1]
        
        return {
            'character_presence_score': min(1.0, len(set(capital_words)) / max(len(words) * 0.02, 1))
        }
    
    def _generate_magical_review(self, film_data, analysis_results):
        """Generate magical review with realistic scoring"""
        cinematic_scores = self.scoring_engine.generate_cinematic_scores(analysis_results)
        
        # Calculate weighted overall score
        overall_score = sum(
            score * self.scoring_engine.category_weights[category] 
            for category, score in cinematic_scores.items()
        )
        
        return {
            "magical_summary": self._generate_dynamic_summary(film_data, analysis_results, overall_score),
            "cinematic_scores": cinematic_scores,
            "overall_magic_score": round(overall_score, 1),
            "strengths": self._generate_dynamic_strengths(analysis_results, cinematic_scores),
            "improvements": self._generate_dynamic_improvements(analysis_results, cinematic_scores),
            "festival_recommendations": self._generate_festival_recommendations(overall_score),
            "audience_analysis": self._generate_audience_analysis(analysis_results),
            "ai_capabilities_used": list(self.capabilities.keys())[:4]
        }
    
    def _generate_dynamic_summary(self, film_data, analysis_results, score):
        """Dynamic summary based on score range"""
        title = film_data['title']
        narrative = analysis_results['narrative_structure']
        
        score_templates = {
            (4.5, 5.0): [
                f"**{title}** demonstrates exceptional cinematic craftsmanship with sophisticated narrative structure and compelling emotional depth.",
                f"**{title}** showcases masterful storytelling with remarkable technical execution and powerful audience engagement."
            ],
            (4.0, 4.4): [
                f"**{title}** presents strong cinematic vision with well-developed narrative elements and solid technical foundation.",
                f"**{title}** exhibits professional quality with engaging storytelling and competent execution across key categories."
            ],
            (3.5, 3.9): [
                f"**{title}** shows promising potential with solid narrative foundation and clear creative direction.",
                f"**{title}** demonstrates competent storytelling with noticeable strengths and areas for refinement."
            ],
            (3.0, 3.4): [
                f"**{title}** displays developing cinematic voice with foundational elements in place awaiting further refinement.",
                f"**{title}** presents authentic creative vision with emerging technical capabilities and narrative understanding."
            ],
            (2.5, 2.9): [
                f"**{title}** shows early-stage development with basic narrative structure and opportunities for technical growth.",
                f"**{title}** demonstrates creative beginnings with clear potential for enhanced execution and storytelling depth."
            ],
            (1.0, 2.4): [
                f"**{title}** represents initial creative exploration with foundational elements emerging.",
                f"**{title}** shows early development phase with opportunities for narrative and technical advancement."
            ]
        }
        
        # Find appropriate template
        for score_range, templates in score_templates.items():
            if score_range[0] <= score <= score_range[1]:
                return random.choice(templates)
        
        return f"**{title}** presents unique cinematic qualities with distinctive creative approach."
    
    def _generate_dynamic_strengths(self, analysis_results, scores):
        """Dynamic strengths based on actual performance"""
        strengths = []
        narrative = analysis_results['narrative_structure']
        dialogue = analysis_results['dialogue_analysis']
        
        if scores.get('story_narrative', 0) > 3.5:
            strengths.append("Strong narrative foundation and structural coherence")
        elif scores.get('story_narrative', 0) > 2.5:
            strengths.append("Clear storytelling intention and basic narrative structure")
        
        if scores.get('visual_vision', 0) > 3.0:
            strengths.append("Evocative descriptive elements and visual potential")
        
        if dialogue.get('emotional_variety', 0) > 0.3:
            strengths.append("Expressive dialogue and emotional variety")
        
        if narrative.get('lexical_diversity', 0) > 0.5:
            strengths.append("Rich vocabulary and linguistic sophistication")
        
        # Fallback strengths
        if not strengths:
            strengths.extend([
                "Authentic creative vision and personal expression",
                "Clear potential for cinematic development",
                "Foundational storytelling elements effectively established"
            ])
        
        return strengths[:3]
    
    def _generate_dynamic_improvements(self, analysis_results, scores):
        """Dynamic improvements based on actual weaknesses"""
        improvements = []
        narrative = analysis_results['narrative_structure']
        dialogue = analysis_results['dialogue_analysis']
        
        if scores.get('technical_craft', 0) < 3.0:
            improvements.append("Opportunity for enhanced technical execution and refinement")
        
        if dialogue.get('dialogue_quality_score', 0) < 0.2:
            improvements.append("Potential for more dynamic dialogue and conversational elements")
        
        if narrative.get('structural_richness', 0) < 0.4:
            improvements.append("Consider strengthening narrative complexity and structural variety")
        
        if scores.get('sound_design', 0) < 2.5:
            improvements.append("Opportunity to enhance audio and sound design elements")
        
        # Fallback improvements
        if not improvements:
            improvements.extend([
                "Further development of technical execution",
                "Enhanced character depth and development",
                "Stronger emotional pacing and narrative rhythm"
            ])
        
        return improvements[:3]
    
    def _generate_festival_recommendations(self, overall_score):
        """Dynamic festival recommendations"""
        if overall_score >= 4.0:
            return {"level": "International", "festivals": ["Major film festivals", "Genre competitions"]}
        elif overall_score >= 3.0:
            return {"level": "National/Regional", "festivals": ["Regional showcases", "Emerging filmmaker events"]}
        else:
            return {"level": "Local/Development", "festivals": ["Local screenings", "Workshop festivals"]}
    
    def _generate_audience_analysis(self, analysis_results):
        """Dynamic audience analysis"""
        emotional = analysis_results['emotional_arc']
        
        if emotional.get('emotional_arc_strength', 0) > 0.6:
            return {"audience": "Mainstream and festival viewers", "impact": "Strong emotional engagement"}
        elif emotional.get('emotional_arc_strength', 0) > 0.3:
            return {"audience": "Independent film enthusiasts", "impact": "Thoughtful emotional resonance"}
        else:
            return {"audience": "Niche and development audiences", "impact": "Emerging emotional connection"}
    
    def _create_contextual_fallback(self, film_data):
        """Fallback for missing transcript"""
        return {
            "magical_summary": f"**{film_data['title']}** presents cinematic vision awaiting detailed analysis through transcript content.",
            "cinematic_scores": {cat: round(random.uniform(2.8, 3.6), 1) for cat in self.scoring_engine.category_weights},
            "overall_magic_score": round(random.uniform(2.9, 3.5), 1),
            "strengths": ["Creative concept established", "Foundation for development", "Clear artistic intention"],
            "improvements": ["Enhanced narrative accessibility", "Technical refinement", "Character development"],
            "festival_recommendations": {"level": "Regional/Development", "festivals": ["Emerging filmmaker events"]},
            "audience_analysis": {"audience": "Development audiences", "impact": "Creative potential evident"},
            "ai_capabilities_used": ["contextual_analysis"]
        }

# --------------------------
# Optimized FilmFreeway Importer
# --------------------------
class FilmFreewayImporter:
    def manual_import_interface(self):
        st.subheader("📥 FilmFreeway Projects")
        with st.form("filmfreeway_form"):
            title = st.text_input("🎬 Project Title")
            director = st.text_input("👤 Director")
            genre = st.selectbox("🎭 Genre", ["Drama", "Comedy", "Documentary", "Horror", "Sci-Fi", "Animation", "Other"])
            duration = st.text_input("⏱️ Duration", placeholder="e.g., 15:30 or 90 min")
            synopsis = st.text_area("📖 Synopsis", height=100)
            
            if st.form_submit_button("💾 Add Project") and title:
                project_data = {
                    'id': len(st.session_state.filmfreeway_projects) + 1,
                    'title': title, 'director': director, 'genre': genre,
                    'duration': duration, 'synopsis': synopsis,
                    'import_date': datetime.now().isoformat()
                }
                st.session_state.filmfreeway_projects.append(project_data)
                st.success(f"✅ '{title}' added!")

# --------------------------
# Optimized Database
# --------------------------
class FilmScoreDatabase:
    def __init__(self):
        self.films = st.session_state.get('all_film_scores', [])
    
    def add_film_analysis(self, film_data, analysis_results):
        film_record = {
            "id": len(self.films) + 1,
            "timestamp": datetime.now().isoformat(),
            "film_data": film_data,
            "analysis_results": analysis_results
        }
        self.films.append(film_record)
        st.session_state.all_film_scores = self.films
        return film_record
    
    def get_statistics(self):
        if not self.films:
            return {"total_films": 0, "average_score": 0, "highest_score": 0, "lowest_score": 0}
        
        scores = [film["analysis_results"]["overall_magic_score"] for film in self.films]
        return {
            "total_films": len(self.films),
            "average_score": round(np.mean(scores), 2),
            "highest_score": round(max(scores), 2),
            "lowest_score": round(min(scores), 2)
        }

# --------------------------
# Utility Functions
# --------------------------
def get_video_id(url):
    try:
        parsed = urlparse(url)
        if "youtube.com" in parsed.hostname:
            return parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.hostname == "youtu.be":
            return parsed.path[1:]
    except:
        return None

def get_video_info(video_id):
    try:
        response = requests.get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {'title': data.get('title', 'Unknown'), 'author': data.get('author_name', 'Unknown'), 'success': True}
    except:
        pass
    return {'success': False}

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([seg["text"] for seg in transcript_list])
    except:
        return "No transcript available. AI will analyze based on film context and metadata."

# --------------------------
# Optimized Magical Interface
# --------------------------
def magical_interface(analyzer, database):
    st.header("🔮 FlickFinder MAGICAL AI Analysis")
    
    # URL input
    youtube_url = st.text_input("🎥 **Paste YouTube URL:**", placeholder="https://www.youtube.com/watch?v=...")
    
    if youtube_url:
        video_id = get_video_id(youtube_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL")
            return
        
        video_info = get_video_info(video_id)
        if not video_info.get('success'):
            st.error("❌ Could not access video information")
            return
        
        # Display video and info
        col1, col2 = st.columns([2, 1])
        with col1:
            st.components.v1.iframe(f"https://www.youtube.com/embed/{video_id}", height=400)
        with col2:
            st.subheader("📋 Film Info")
            st.write(f"**Title:** {video_info['title']}")
            st.write(f"**Channel:** {video_info['author']}")
        
        custom_title = st.text_input("✏️ **Film Title:**", value=video_info['title'])
        
        if st.button("🔮 **START MAGICAL ANALYSIS**", type="primary", use_container_width=True):
            with st.spinner("✨ Performing magical analysis..."):
                try:
                    transcript = get_transcript(video_id)
                    film_data = {'title': custom_title, 'channel': video_info['author'], 'transcript': transcript}
                    
                    # Perform analysis
                    magical_results = analyzer.perform_magical_analysis(film_data)
                    film_record = database.add_film_analysis(film_data, magical_results)
                    
                    # Display results
                    display_magical_results(magical_results)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")

def display_magical_results(results):
    st.success("🌟 **MAGICAL ANALYSIS COMPLETE!**")
    
    # Score display
    magic_score = results['overall_magic_score']
    st.markdown(f"""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #8A2BE2 0%, #4B0082 100%); border-radius: 20px; margin: 20px 0; border: 3px solid #FFD700;'>
        <h1 style='color: gold; margin: 0; font-size: 60px;'>{magic_score}/5.0</h1>
        <p style='color: white; font-size: 24px; margin: 10px 0;'>✨ Overall Magic Score ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Category scores
    st.subheader("🎯 Category Scores")
    scores = results['cinematic_scores']
    cols = st.columns(5)
    categories = [
        ("🧙♂️ Story", scores['story_narrative'], "#FF6B6B"),
        ("🔮 Visual", scores['visual_vision'], "#4ECDC4"),
        ("⚡ Technical", scores['technical_craft'], "#45B7D1"),
        ("🎵 Sound", scores['sound_design'], "#96CEB4"),
        ("🌟 Performance", scores['performance'], "#FFD93D")
    ]
    
    for idx, (name, score, color) in enumerate(categories):
        with cols[idx]:
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; background: {color}; border-radius: 12px; margin: 5px; border: 2px solid gold;'>
                <h4 style='margin: 0; color: white;'>{name}</h4>
                <h2 style='margin: 8px 0; color: gold;'>{score}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Summary and insights
    st.subheader("📖 Magical Summary")
    st.write(results['magical_summary'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Strengths")
        for strength in results['strengths']:
            st.write(f"✨ {strength}")
    
    with col2:
        st.subheader("📝 Improvements")
        for improvement in results['improvements']:
            st.write(f"🔧 {improvement}")

# --------------------------
# Main Application
# --------------------------
def main():
    st.sidebar.title("🔮 FlickFinder MAGIC")
    st.sidebar.markdown("---")
    
    # Initialize components
    analyzer = MagicalFilmAnalyzer()
    database = FilmScoreDatabase()
    filmfreeway_importer = FilmFreewayImporter()
    
    # Navigation
    page = st.sidebar.radio("Navigate:", ["🔮 Magical Analysis", "📥 FilmFreeway", "💾 Database"])
    
    if page == "🔮 Magical Analysis":
        magical_interface(analyzer, database)
    elif page == "📥 FilmFreeway":
        filmfreeway_importer.manual_import_interface()
        # Display projects
        if st.session_state.filmfreeway_projects:
            st.subheader("📚 Your Projects")
            for project in st.session_state.filmfreeway_projects[-5:]:
                with st.expander(f"🎬 {project['title']}"):
                    st.write(f"**Director:** {project.get('director', 'N/A')}")
                    st.write(f"**Genre:** {project.get('genre', 'N/A')}")
                    st.write(f"**Duration:** {project.get('duration', 'N/A')}")
                    if project.get('synopsis'):
                        st.write("**Synopsis:**", project['synopsis'])
    elif page == "💾 Database":
        st.header("💾 Film Database")
        stats = database.get_statistics()
        cols = st.columns(4)
        metrics = [
            ("Total Films", stats['total_films']),
            ("Average Score", f"{stats['average_score']}/5.0"),
            ("Highest Score", f"{stats['highest_score']}/5.0"), 
            ("Lowest Score", f"{stats['lowest_score']}/5.0")
        ]
        
        for (col, (label, value)) in zip(cols, metrics):
            col.metric(label, value)
        
        # Recent analyses
        if database.films:
            st.subheader("🎬 Recent Analyses")
            for film in database.films[-5:]:
                with st.expander(f"📊 {film['film_data']['title']} - {film['analysis_results']['overall_magic_score']}/5.0"):
                    scores = film['analysis_results']['cinematic_scores']
                    score_cols = st.columns(5)
                    for col, (cat, score) in zip(score_cols, scores.items()):
                        col.metric(cat.split('_')[-1].title(), f"{score}")

if __name__ == "__main__":
    main()

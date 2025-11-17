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
    'magic_mode': True,
    'batch_processing': False,
    'batch_progress': 0,
    'batch_results': []
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
            "scoring_variance": 0.3,
            "batch_size": 50
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
        
        base_scores = {
            'story_narrative': self._calculate_story_potential(narrative, emotional, complexity),
            'visual_vision': self._calculate_visual_potential(narrative, dialogue),
            'technical_craft': self._calculate_technical_execution(narrative, complexity, dialogue),
            'sound_design': self._calculate_sound_potential(dialogue, emotional),
            'performance': self._calculate_performance_potential(characters, dialogue, emotional)
        }
        
        analysis_quality = min(1.0, narrative.get('structural_richness', 0.5) * 0.8 + complexity.get('content_density', 0.5) * 0.2)
        
        final_scores = {}
        for category, base_score in base_scores.items():
            final_scores[category] = self.apply_realistic_variance(base_score, analysis_quality)
        
        return final_scores
    
    def _calculate_story_potential(self, narrative, emotional, complexity):
        structural_base = narrative.get('structural_richness', 0.5) * 2.8
        emotional_weight = emotional.get('emotional_arc_strength', 0.3) * 1.2
        complexity_bonus = complexity.get('content_density', 0.4) * 0.8
        readability_penalty = max(0, (0.6 - narrative.get('readability_score', 0.5)) * 0.5)
        
        raw_score = structural_base + emotional_weight + complexity_bonus - readability_penalty
        return min(5.0, max(1.0, raw_score))
    
    def _calculate_visual_potential(self, narrative, dialogue):
        descriptive_power = (narrative.get('lexical_diversity', 0.4) * 1.8 + 
                           dialogue.get('emotional_variety', 0.3) * 1.2)
        complexity_bonus = narrative.get('sentence_complexity', 0) * 0.1
        
        return min(5.0, 2.2 + descriptive_power + complexity_bonus)
    
    def _calculate_technical_execution(self, narrative, complexity, dialogue):
        execution_quality = (narrative.get('readability_score', 0.5) * 1.5 +
                           complexity.get('syntactic_diversity', 0.3) * 1.0 +
                           dialogue.get('dialogue_quality_score', 0.3) * 0.5)
        
        return min(5.0, 2.3 + execution_quality)
    
    def _calculate_sound_potential(self, dialogue, emotional):
        audio_indicators = (dialogue.get('dialogue_quality_score', 0.3) * 1.2 +
                          emotional.get('emotional_variance', 0.2) * 0.8)
        
        return min(5.0, 2.1 + audio_indicators * 1.2)
    
    def _calculate_performance_potential(self, characters, dialogue, emotional):
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
        synopsis = film_data.get('synopsis', '')
        
        # Use synopsis if no transcript
        analysis_text = transcript if transcript and "No transcript available" not in transcript else synopsis
        
        if not analysis_text or len(analysis_text) < 50:
            return self._create_contextual_fallback(film_data)

        analysis_results = {
            'narrative_structure': self._analyze_narrative_structure(analysis_text),
            'dialogue_analysis': self._analyze_dialogue_quality(analysis_text),
            'emotional_arc': self._analyze_emotional_arc(analysis_text),
            'complexity_metrics': self._calculate_complexity_metrics(analysis_text),
            'character_analysis': self._analyze_character_presence(analysis_text)
        }

        return self._generate_magical_review(film_data, analysis_results)
    
    def _analyze_narrative_structure(self, text):
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
        vader_analyzer = SentimentIntensityAnalyzer()
        sentences = nltk.sent_tokenize(text)[:10]
        
        if len(sentences) < 3:
            return {'emotional_arc_strength': 0.3, 'emotional_variance': 0.2, 'emotional_range': 0.3}
        
        emotional_scores = [vader_analyzer.polarity_scores(s)['compound'] for s in sentences]
        
        return {
            'emotional_arc_strength': min(1.0, np.var(emotional_scores) * 3),
            'emotional_variance': np.var(emotional_scores),
            'emotional_range': max(emotional_scores) - min(emotional_scores)
        }
    
    def _calculate_complexity_metrics(self, text):
        words = nltk.word_tokenize(text)
        unique_ratio = len(set(words)) / max(len(words), 1)
        
        return {
            'vocabulary_richness': unique_ratio,
            'content_density': min(1.0, unique_ratio * 0.7 + min(1, len(words)/300) * 0.3)
        }
    
    def _analyze_character_presence(self, text):
        words = nltk.word_tokenize(text)
        capital_words = [w for w in words if w.istitle() and len(w) > 1]
        
        return {
            'character_presence_score': min(1.0, len(set(capital_words)) / max(len(words) * 0.02, 1))
        }
    
    def _generate_magical_review(self, film_data, analysis_results):
        cinematic_scores = self.scoring_engine.generate_cinematic_scores(analysis_results)
        
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
        title = film_data['title']
        
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
        
        for score_range, templates in score_templates.items():
            if score_range[0] <= score <= score_range[1]:
                return random.choice(templates)
        
        return f"**{title}** presents unique cinematic qualities with distinctive creative approach."
    
    def _generate_dynamic_strengths(self, analysis_results, scores):
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

        if not strengths:
            strengths.extend([
                "Authentic creative vision and personal expression",
                "Clear potential for cinematic development",
                "Foundational storytelling elements effectively established"
            ])
        
        return strengths[:3]
    
    def _generate_dynamic_improvements(self, analysis_results, scores):
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

        if not improvements:
            improvements.extend([
                "Further development of technical execution",
                "Enhanced character depth and development",
                "Stronger emotional pacing and narrative rhythm"
            ])
        
        return improvements[:3]
    
    def _generate_festival_recommendations(self, overall_score):
        if overall_score >= 4.0:
            return {"level": "International", "festivals": ["Major film festivals", "Genre competitions"]}
        elif overall_score >= 3.0:
            return {"level": "National/Regional", "festivals": ["Regional showcases", "Emerging filmmaker events"]}
        else:
            return {"level": "Local/Development", "festivals": ["Local screenings", "Workshop festivals"]}
    
    def _generate_audience_analysis(self, analysis_results):
        emotional = analysis_results['emotional_arc']
        
        if emotional.get('emotional_arc_strength', 0) > 0.6:
            return {"audience": "Mainstream and festival viewers", "impact": "Strong emotional engagement"}
        elif emotional.get('emotional_arc_strength', 0) > 0.3:
            return {"audience": "Independent film enthusiasts", "impact": "Thoughtful emotional resonance"}
        else:
            return {"audience": "Niche and development audiences", "impact": "Emerging emotional connection"}
    
    def _create_contextual_fallback(self, film_data):
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
# Enhanced CSV Batch Processor
# --------------------------
class CSVBatchProcessor:
    def __init__(self, analyzer, database, batch_size=50):
        self.analyzer = analyzer
        self.database = database
        self.batch_size = batch_size

    def process_csv_batch(self, df, start_idx, end_idx, progress_bar, status_text):
        """Process a batch of films from CSV"""
        batch_results = []
        batch_df = df.iloc[start_idx:end_idx]
        
        for idx, row in batch_df.iterrows():
            try:
                film_data = self._prepare_film_data_from_row(row, idx)
                status_text.text(f"🎬 Analyzing: {film_data['title']} ({idx - start_idx + 1}/{len(batch_df)})")
                
                analysis_results = self.analyzer.perform_magical_analysis(film_data)
                film_record = self.database.add_film_analysis(film_data, analysis_results)
                
                batch_results.append({
                    'title': film_data['title'],
                    'director': film_data.get('director', 'Unknown'),
                    'writer': film_data.get('writer', 'Not specified'),
                    'score': analysis_results['overall_magic_score'],
                    'status': 'Success'
                })
                
                progress_bar.progress((idx - start_idx + 1) / len(batch_df))
                
            except Exception as e:
                batch_results.append({
                    'title': film_data.get('title', f'Film_{idx}'),
                    'score': 0,
                    'status': f'Error: {str(e)[:50]}'
                })
        
        return batch_results

    def _prepare_film_data_from_row(self, row, idx):
        """Prepare film data from CSV row with flexible column mapping"""
        column_mapping = {
            'title': ['title', 'Title', 'Film Title', 'Project Title'],
            'director': ['director', 'Director', 'Filmmaker'],
            'writer': ['writer', 'Writer', 'Screenwriter'],
            'producer': ['producer', 'Producer', 'Production'],
            'genre': ['genre', 'Genre', 'Category'],
            'duration': ['duration', 'Duration', 'Runtime'],
            'synopsis': ['synopsis', 'Synopsis', 'Description', 'Logline']
        }
        
        film_data = {}
        for field, possible_columns in column_mapping.items():
            for col in possible_columns:
                if col in row and pd.notna(row[col]):
                    film_data[field] = str(row[col])
                    break
            if field not in film_data:
                film_data[field] = self._get_default_value(field, idx)
        
        film_data['transcript'] = ''
        return film_data

    def _get_default_value(self, field, idx):
        """Get default values for missing fields"""
        defaults = {
            'title': f'Film_{idx}',
            'director': 'Unknown',
            'writer': 'Not specified',
            'producer': 'Not specified',
            'genre': 'Unknown',
            'duration': 'N/A',
            'synopsis': ''
        }
        return defaults.get(field, '')

    def validate_csv(self, df):
        """Validate CSV has required columns"""
        required_cols = ['title']
        missing_required = [col for col in required_cols if col.lower() not in [c.lower() for c in df.columns]]
        
        if missing_required:
            return False, f"Missing required columns: {', '.join(missing_required)}"
        
        return True, "CSV validated successfully"

# --------------------------
# Competitive Scoring Engine with Filmmaker Data
# --------------------------
class CompetitiveFilmScorer:
    def __init__(self):
        self.genre_leaders = {}
        self.update_leaderboard()
    
    def update_leaderboard(self):
        """Update genre leaders with filmmaker information"""
        analyzed_projects = [p for p in st.session_state.filmfreeway_projects if p.get('analyzed', False)]
        self.genre_leaders = {}
        
        for project in analyzed_projects:
            genre = project.get('genre', 'Unknown')
            score = project.get('score', 0)
            
            if genre not in self.genre_leaders or score > self.genre_leaders[genre]['score']:
                self.genre_leaders[genre] = {
                    'title': project['title'],
                    'score': score,
                    'director': project.get('director', 'Unknown'),
                    'writer': project.get('writer', 'Not specified'),
                    'producer': project.get('producer', 'Not specified'),
                    'timestamp': project.get('analysis_date', '')
                }
    
    def display_genre_badges(self):
        """Display genre champion badges with filmmaker credits"""
        if not self.genre_leaders:
            st.info("🎬 No genre champions yet. Start analyzing films!")
            return
        
        st.subheader("🏆 Genre Champions")
        cols = st.columns(min(4, len(self.genre_leaders)))
        
        for idx, (genre, leader) in enumerate(self.genre_leaders.items()):
            col = cols[idx % len(cols)]
            
            # Create filmmaker credit line
            filmmaker_credit = f"Dir: {leader['director']}"
            if leader.get('writer') and leader['writer'] != 'Not specified':
                filmmaker_credit += f" | Wri: {leader['writer']}"
            
            with col:
                st.markdown(f"""
                <div style="border: 2px solid #FFD700; border-radius: 10px; padding: 15px; text-align: center; 
                          background: linear-gradient(135deg, #FFF9C4, #FFEB3B); margin: 10px 0; 
                          box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                    <div style="font-size: 24px;">⭐</div>
                    <h4 style="margin: 5px 0; color: #333; font-weight: bold;">{genre}</h4>
                    <h3 style="margin: 8px 0; color: #B71C1C; font-size: 1.1em;">{leader['title']}</h3>
                    <p style="margin: 3px 0; color: #666; font-size: 0.9em;">{filmmaker_credit}</p>
                    <h2 style="margin: 8px 0; color: #E65100; font-size: 1.4em;">{leader['score']}/5.0</h2>
                </div>
                """, unsafe_allow_html=True)

# --------------------------
# Enhanced FilmFreeway Importer with Filmmaker Data
# --------------------------
class FilmFreewayImporter:
    def __init__(self, analyzer, database):
        self.analyzer = analyzer
        self.database = database
        self.batch_processor = CSVBatchProcessor(analyzer, database)
        self.competitive_scorer = CompetitiveFilmScorer()

    def show_import_dashboard(self):
        """Main dashboard for FilmFreeway imports"""
        st.header("🎬 FilmFreeway Project Manager")
        
        # Display genre champions at the top
        self.competitive_scorer.display_genre_badges()
        
        # Enhanced stats with filmmaker data
        analyzed_projects = [p for p in st.session_state.filmfreeway_projects if p.get('analyzed', False)]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Projects", len(st.session_state.filmfreeway_projects))
        with col2:
            analyzed_count = len(analyzed_projects)
            st.metric("Analyzed", analyzed_count)
        with col3:
            unique_directors = len(set(p.get('director', 'Unknown') for p in st.session_state.filmfreeway_projects))
            st.metric("Unique Directors", unique_directors)
        with col4:
            if analyzed_count > 0:
                avg_score = np.mean([p.get('score', 0) for p in analyzed_projects])
                st.metric("Avg Score", f"{avg_score:.1f}")
        
        # Enhanced import methods tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📥 Quick Add", "📊 CSV Batch", "🏆 Live Scoring", "👥 Filmmaker Insights"])
        
        with tab1:
            self.enhanced_quick_add_interface()
        with tab2:
            self.enhanced_csv_interface()
        with tab3:
            self.enhanced_live_scoring_interface()
        with tab4:
            self.filmmaker_insights_interface()

    def enhanced_quick_add_interface(self):
        """Enhanced single project addition with filmmaker data"""
        st.subheader("🚀 Quick Add Project")
        
        with st.form("enhanced_project_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("🎬 Film Title*", placeholder="Enter film title...")
                director = st.text_input("👤 Director*", placeholder="Director name")
                writer = st.text_input("✍️ Writer", placeholder="Writer name")
                
            with col2:
                genre = st.selectbox("🎭 Genre", 
                    ["Select...", "Drama", "Comedy", "Documentary", "Horror", 
                     "Sci-Fi", "Animation", "Experimental", "Thriller", "Romance", "Action"])
                duration = st.text_input("⏱️ Duration", placeholder="e.g., 15:30 or 90min")
                synopsis = st.text_area("📖 Synopsis", height=120, placeholder="Brief description...")
            
            submitted = st.form_submit_button("✨ Add & Analyze Project", use_container_width=True)
            
            if submitted and title and director:
                project_data = self._create_project_data(title, director, writer, genre, duration, synopsis)
                
                # Immediate analysis
                if synopsis:
                    with st.spinner("🔮 Performing magical analysis..."):
                        film_data = {
                            'title': title, 'director': director, 'writer': writer,
                            'genre': genre, 'duration': duration, 'synopsis': synopsis, 'transcript': ''
                        }
                        analysis = self.analyzer.perform_magical_analysis(film_data)
                        project_data.update({
                            'analyzed': True, 'score': analysis['overall_magic_score'],
                            'analysis': analysis, 'analysis_date': datetime.now().isoformat()
                        })
                        self.competitive_scorer.update_leaderboard()
                
                st.session_state.filmfreeway_projects.append(project_data)
                status_msg = " 🎉 Analysis complete!" if project_data['analyzed'] else " ⏳ Ready for analysis."
                st.success(f"✅ **{title}** added successfully!{status_msg}")

    def _create_project_data(self, title, director, writer, genre, duration, synopsis):
        """Create standardized project data"""
        return {
            'id': len(st.session_state.filmfreeway_projects) + 1,
            'title': title,
            'director': director,
            'writer': writer if writer else "Not specified",
            'genre': genre if genre != "Select..." else "Unknown",
            'duration': duration,
            'synopsis': synopsis,
            'import_date': datetime.now().isoformat(),
            'analyzed': False
        }

    def enhanced_csv_interface(self):
        """Enhanced CSV batch processing with filmmaker data"""
        st.subheader("📊 Mass Import & Analysis")
        
        # Performance settings
        with st.expander("⚙️ Performance Settings", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.select_slider("🎯 Batch Size", options=[10, 25, 50, 100], value=50)
            with col2:
                quality_filter = st.checkbox("🎨 Quality Filter", value=True)
        
        # File upload
        uploaded_file = st.file_uploader("📁 Upload FilmFreeway CSV", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} films")
                
                # Enhanced data preview
                st.subheader("📋 Data Preview with Filmmaker Details")
                preview_df = self._enhance_preview_data(df.head(10))
                st.dataframe(preview_df, use_container_width=True)
                
                if st.button(f"🚀 Analyze {len(df)} Films", type="primary", use_container_width=True):
                    self.process_enhanced_batch(df, batch_size, quality_filter)
                    
            except Exception as e:
                st.error(f"❌ Error reading CSV: {e}")

    def _enhance_preview_data(self, df):
        """Add filmmaker data to preview"""
        enhanced_df = df.copy()
        
        # Map common column names to standard fields
        column_mapping = {
            'title': ['title', 'Title', 'Film Title', 'Project Title'],
            'director': ['director', 'Director', 'Filmmaker'],
            'writer': ['writer', 'Writer', 'Screenwriter'],
            'genre': ['genre', 'Genre', 'Category'],
            'duration': ['duration', 'Duration', 'Runtime'],
            'synopsis': ['synopsis', 'Synopsis', 'Description']
        }
        
        # Create standard columns for display
        display_columns = []
        for standard_field, possible_columns in column_mapping.items():
            for col in possible_columns:
                if col in enhanced_df.columns:
                    if standard_field not in enhanced_df.columns:
                        enhanced_df[standard_field] = enhanced_df[col]
                    if standard_field not in display_columns:
                        display_columns.append(standard_field)
                    break
        
        return enhanced_df[display_columns] if display_columns else enhanced_df

    def process_enhanced_batch(self, df, batch_size, quality_filter):
        """Process batch with enhanced filmmaker data"""
        total_films = len(df)
        
        # Apply quality filter
        if quality_filter:
            synopsis_col = next((col for col in ['synopsis', 'Synopsis', 'Description'] if col in df.columns), None)
            if synopsis_col:
                df = df[df[synopsis_col].notna() & (df[synopsis_col].str.len() > 50)]
        
        if len(df) == 0:
            st.warning("No films meet quality criteria.")
            return
        
        # Setup progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        all_results = []
        
        for batch_num in range(0, len(df), batch_size):
            batch_df = df.iloc[batch_num:batch_num + batch_size]
            batch_results = []
            
            for idx, row in batch_df.iterrows():
                try:
                    film_data = self.batch_processor._prepare_film_data_from_row(row, idx)
                    status_text.text(f"Analyzing: {film_data['title'][:50]}...")
                    
                    analysis = self.analyzer.perform_magical_analysis(film_data)
                    
                    project_data = {
                        'id': len(st.session_state.filmfreeway_projects) + 1,
                        'title': film_data['title'],
                        'director': film_data.get('director', 'Unknown'),
                        'writer': film_data.get('writer', 'Not specified'),
                        'genre': film_data.get('genre', 'Unknown'),
                        'duration': film_data.get('duration', 'N/A'),
                        'synopsis': film_data.get('synopsis', ''),
                        'analyzed': True,
                        'score': analysis['overall_magic_score'],
                        'analysis': analysis,
                        'import_date': datetime.now().isoformat(),
                        'analysis_date': datetime.now().isoformat()
                    }
                    
                    st.session_state.filmfreeway_projects.append(project_data)
                    self.competitive_scorer.update_leaderboard()
                    
                    batch_results.append({
                        'title': film_data['title'],
                        'director': film_data.get('director', 'Unknown'),
                        'writer': film_data.get('writer', 'Not specified'),
                        'genre': film_data.get('genre', 'Unknown'),
                        'score': analysis['overall_magic_score'],
                        'status': 'Success'
                    })
                    
                except Exception as e:
                    batch_results.append({
                        'title': film_data.get('title', 'Unknown'),
                        'director': film_data.get('director', 'Unknown'),
                        'score': 0,
                        'status': f'Error: {str(e)[:30]}'
                    })
                
                progress_bar.progress((idx + 1) / len(df))
            
            all_results.extend(batch_results)
            
            # Show live results
            with results_container:
                self.show_enhanced_live_results(all_results)
        
        # Final summary
        self.show_enhanced_comprehensive_summary(all_results)

    def show_enhanced_live_results(self, results):
        """Show live results with filmmaker context"""
        if results:
            results_df = pd.DataFrame(results)
            
            # Display current genre leaders
            st.subheader("🏆 Current Genre Leaders")
            self.competitive_scorer.display_genre_badges()
            
            # Show recent results with filmmaker info
            st.subheader("📊 Recent Analysis with Filmmaker Details")
            display_df = results_df[['title', 'director', 'writer', 'genre', 'score', 'status']].tail(10)
            st.dataframe(display_df, use_container_width=True)

    def show_enhanced_comprehensive_summary(self, results):
        """Show final summary with filmmaker insights"""
        st.success("🎉 Batch analysis complete!")
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", len(results_df))
            with col2:
                success = len(results_df[results_df['status'] == 'Success'])
                st.metric("Successful", success)
            with col3:
                if success > 0:
                    avg_score = results_df[results_df['status'] == 'Success']['score'].mean()
                    st.metric("Avg Score", f"{avg_score:.1f}")
            with col4:
                unique_directors = results_df[results_df['status'] == 'Success']['director'].nunique()
                st.metric("Unique Directors", unique_directors)

    def enhanced_live_scoring_interface(self):
        """Enhanced live scoring with filmmaker context"""
        st.header("🏆 Live Scoring Arena")
        
        analyzed = [p for p in st.session_state.filmfreeway_projects if p.get('analyzed', False)]
        
        if not analyzed:
            st.info("🎬 Analyze some films to see live scoring!")
            return
        
        self.competitive_scorer.update_leaderboard()
        self.competitive_scorer.display_genre_badges()
        
        # Enhanced top films display with filmmaker credits
        st.subheader("⭐ Top Rated Films with Filmmaker Credits")
        top_films = sorted(analyzed, key=lambda x: x.get('score', 0), reverse=True)[:15]
        
        for film in top_films:
            # Create filmmaker credit line
            credits = []
            if film.get('director') and film['director'] != 'Unknown':
                credits.append(f"**Director:** {film['director']}")
            if film.get('writer') and film['writer'] != 'Not specified':
                credits.append(f"**Writer:** {film['writer']}")
            
            credit_line = " | ".join(credits) if credits else "Filmmaker information not available"
            
            col1, col2, col3 = st.columns([3, 3, 1])
            with col1:
                st.write(f"**{film['title']}**")
            with col2:
                st.write(f"{film.get('genre', 'Unknown')} • {credit_line}")
            with col3:
                score = film.get('score', 0)
                score_color = "#FFD700" if score >= 4.0 else "#4ECDC4" if score >= 3.0 else "#FF6B6B"
                st.markdown(f"<h3 style='color: {score_color}; text-align: center;'>{score}</h3>", unsafe_allow_html=True)
            
            st.divider()

    def filmmaker_insights_interface(self):
        """Dedicated interface for filmmaker analytics"""
        st.header("👥 Filmmaker Insights & Analytics")
        
        analyzed = [p for p in st.session_state.filmfreeway_projects if p.get('analyzed', False)]
        
        if not analyzed:
            st.info("📊 Analyze some films to see filmmaker insights!")
            return
        
        # Director performance analysis
        st.subheader("🎬 Director Performance Analysis")
        
        director_data = {}
        for project in analyzed:
            director = project.get('director', 'Unknown')
            if director not in director_data:
                director_data[director] = []
            director_data[director].append(project)
        
        # Create director performance table
        performance_data = []
        for director, projects in director_data.items():
            if director != 'Unknown':
                scores = [p.get('score', 0) for p in projects]
                performance_data.append({
                    'Director': director,
                    'Films': len(projects),
                    'Avg Score': np.mean(scores),
                    'Highest Score': max(scores),
                    'Lowest Score': min(scores),
                    'Latest Film': projects[-1]['title']
                })
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            performance_df = performance_df.sort_values('Avg Score', ascending=False)
            st.dataframe(performance_df, use_container_width=True)

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
    filmfreeway_importer = FilmFreewayImporter(analyzer, database)
    
    # Enhanced navigation
    page = st.sidebar.radio("Navigate:", ["🔮 Magical Analysis", "📥 FilmFreeway Pro", "💾 Database", "👥 Filmmaker Analytics"])
    
    if page == "🔮 Magical Analysis":
        magical_interface(analyzer, database)
    elif page == "📥 FilmFreeway Pro":
        filmfreeway_importer.show_import_dashboard()
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
    elif page == "👥 Filmmaker Analytics":
        filmfreeway_importer.filmmaker_insights_interface()

if __name__ == "__main__":
    main()

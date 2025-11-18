import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

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
if 'all_film_scores' not in st.session_state:
    st.session_state.all_film_scores = []
if 'filmfreeway_projects' not in st.session_state:
    st.session_state.filmfreeway_projects = []

# --------------------------
# Smart Genre Detector
# --------------------------
class SmartGenreDetector:
    def __init__(self):
        self.genre_patterns = self._build_genre_patterns()
    
    def _build_genre_patterns(self):
        """Build comprehensive genre detection patterns"""
        return {
            "Drama": {
                "keywords": ["emotional", "relationship", "conflict", "family", "love", "heart", "struggle", "life", "human", "drama"],
            },
            "Comedy": {
                "keywords": ["funny", "laugh", "humor", "joke", "comic", "satire", "hilarious", "wit", "absurd", "comedy"],
            },
            "Horror": {
                "keywords": ["fear", "terror", "scary", "horror", "ghost", "monster", "kill", "death", "dark", "night"],
            },
            "Sci-Fi": {
                "keywords": ["future", "space", "alien", "technology", "robot", "planet", "time travel", "science", "sci-fi", "future"],
            },
            "Action": {
                "keywords": ["fight", "chase", "gun", "explosion", "mission", "danger", "escape", "battle", "adventure", "action"],
            },
            "Thriller": {
                "keywords": ["suspense", "mystery", "danger", "chase", "secret", "conspiracy", "tense", "cliffhanger", "thriller"],
            },
            "Romance": {
                "keywords": ["love", "romance", "heart", "relationship", "kiss", "date", "passion", "affection", "romantic"],
            },
            "Documentary": {
                "keywords": ["real", "fact", "interview", "evidence", "truth", "history", "actual", "reality", "documentary"],
            },
            "Fantasy": {
                "keywords": ["magic", "dragon", "kingdom", "quest", "mythical", "wizard", "enchanted", "supernatural", "fantasy"],
            }
        }
    
    def detect_genre(self, text, existing_genre=None):
        """Smart genre detection from text content"""
        if not text or len(text.strip()) < 10:
            return existing_genre or "Unknown"
        
        text_lower = text.lower()
        
        # Calculate genre scores
        genre_scores = {}
        
        for genre, pattern_data in self.genre_patterns.items():
            score = 0
            
            # Keyword matching
            for keyword in pattern_data["keywords"]:
                if keyword in text_lower:
                    score += 2  # Base score for keyword match
            
            genre_scores[genre] = score
        
        # Get top genre
        if genre_scores:
            top_genre = max(genre_scores.items(), key=lambda x: x[1])
            if top_genre[1] > 0:  # Any match is enough for genre
                return top_genre[0]
        
        return existing_genre or "Drama"  # Default fallback
    
    def get_confidence(self, text, detected_genre):
        """Get confidence score for genre detection"""
        if detected_genre == "Unknown" or not text:
            return 0.0
        
        pattern_data = self.genre_patterns.get(detected_genre, {})
        
        if not pattern_data:
            return 0.0
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in pattern_data["keywords"] if keyword in text_lower)
        max_possible = len(pattern_data["keywords"])
        
        confidence = min(1.0, keyword_matches / max_possible * 2)
        return round(confidence, 2)

# --------------------------
# Film Analysis Engine
# --------------------------
class FilmAnalysisEngine:
    def __init__(self):
        self.genre_detector = SmartGenreDetector()
    
    def analyze_film(self, film_data):
        """Analyze film data with genre detection"""
        transcript = film_data.get('transcript', '')
        synopsis = film_data.get('synopsis', '')
        existing_genre = film_data.get('genre', 'Unknown')
        
        # Smart genre detection
        analysis_text = self._select_analysis_text(transcript, synopsis)
        detected_genre = self.genre_detector.detect_genre(analysis_text, existing_genre)
        confidence = self.genre_detector.get_confidence(analysis_text, detected_genre)
        
        # Update film data
        film_data['detected_genre'] = detected_genre
        film_data['genre_confidence'] = confidence
        film_data['original_genre'] = existing_genre
        
        if not analysis_text or len(analysis_text) < 50:
            return self._create_fallback_analysis(film_data, detected_genre)

        # Enhanced analysis with genre context
        analysis_results = {
            'narrative_structure': self._analyze_narrative_structure(analysis_text),
            'emotional_arc': self._analyze_emotional_arc(analysis_text),
            'character_analysis': self._analyze_character_presence(analysis_text),
            'genre_context': {
                'detected_genre': detected_genre,
                'confidence': confidence,
                'genre_alignment': self._analyze_genre_alignment(analysis_text, detected_genre)
            }
        }

        return self._generate_film_review(film_data, analysis_results)
    
    def _select_analysis_text(self, transcript, synopsis):
        """Smart text selection for analysis"""
        if transcript and "No transcript available" not in transcript:
            return transcript
        elif synopsis:
            return synopsis
        else:
            return ""
    
    def _analyze_narrative_structure(self, text):
        """Enhanced narrative analysis"""
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        word_count = len(words)
        unique_words = len(set(words))
        
        return {
            'word_count': word_count,
            'sentence_count': len(sentences),
            'lexical_diversity': unique_words / max(word_count, 1),
            'readability_score': min(1.0, 30 / max(np.mean([len(nltk.word_tokenize(s)) for s in sentences]), 1)),
            'structural_richness': min(1.0, (unique_words/word_count * 0.6 + min(1, word_count/500) * 0.4))
        }
    
    def _analyze_emotional_arc(self, text):
        """Enhanced emotional analysis"""
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
    
    def _analyze_character_presence(self, text):
        """Enhanced character analysis"""
        words = nltk.word_tokenize(text)
        capital_words = [w for w in words if w.istitle() and len(w) > 1]
        
        character_score = min(1.0, len(set(capital_words)) / max(len(words) * 0.02, 1))
        
        return {
            'character_presence_score': character_score,
            'unique_characters': len(set(capital_words))
        }
    
    def _analyze_genre_alignment(self, text, detected_genre):
        """Analyze how well content aligns with detected genre"""
        text_lower = text.lower()
        patterns = self.genre_detector.genre_patterns.get(detected_genre, {})
        
        if not patterns:
            return 0.5
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in patterns["keywords"] if keyword in text_lower)
        max_keywords = len(patterns["keywords"])
        
        keyword_score = keyword_matches / max_keywords if max_keywords > 0 else 0
        return min(1.0, keyword_score * 1.5)
    
    def _generate_film_review(self, film_data, analysis_results):
        """Generate film review with automatic genre insights"""
        cinematic_scores = self._generate_cinematic_scores(analysis_results, film_data)
        
        overall_score = np.mean(list(cinematic_scores.values()))
        detected_genre = film_data.get('detected_genre', 'Unknown')
        
        return {
            "smart_summary": self._generate_smart_summary(film_data, overall_score, detected_genre),
            "cinematic_scores": cinematic_scores,
            "overall_score": round(overall_score, 1),
            "strengths": self._generate_strengths(analysis_results, cinematic_scores, detected_genre),
            "recommendations": self._generate_recommendations(analysis_results, cinematic_scores, detected_genre),
            "festival_recommendations": self._generate_festival_recommendations(overall_score, detected_genre),
            "audience_analysis": self._generate_audience_analysis(analysis_results, detected_genre),
            "genre_insights": analysis_results['genre_context']
        }
    
    def _generate_cinematic_scores(self, analysis_results, film_data):
        """Generate realistic cinematic scores"""
        narrative = analysis_results['narrative_structure']
        emotional = analysis_results['emotional_arc']
        characters = analysis_results['character_analysis']
        genre = film_data.get('detected_genre', 'Unknown')
        
        base_scores = {
            'story_narrative': self._calculate_story_potential(narrative, emotional, genre),
            'visual_vision': self._calculate_visual_potential(narrative, genre),
            'technical_craft': self._calculate_technical_execution(narrative, genre),
            'sound_design': self._calculate_sound_potential(emotional, genre),
            'performance': self._calculate_performance_potential(characters, emotional, genre)
        }
        
        # Apply realistic variance
        final_scores = {}
        for category, base_score in base_scores.items():
            varied_score = base_score + random.uniform(-0.3, 0.3)
            final_scores[category] = max(1.0, min(5.0, round(varied_score, 1)))
        
        return final_scores
    
    def _calculate_story_potential(self, narrative, emotional, genre):
        structural_base = narrative.get('structural_richness', 0.5) * 2.8
        emotional_weight = emotional.get('emotional_arc_strength', 0.3) * 1.2
        raw_score = structural_base + emotional_weight
        return min(5.0, max(1.0, raw_score))
    
    def _calculate_visual_potential(self, narrative, genre):
        descriptive_power = narrative.get('lexical_diversity', 0.4) * 1.8
        return min(5.0, 2.2 + descriptive_power)
    
    def _calculate_technical_execution(self, narrative, genre):
        execution_quality = narrative.get('readability_score', 0.5) * 1.5
        return min(5.0, 2.3 + execution_quality)
    
    def _calculate_sound_potential(self, emotional, genre):
        audio_indicators = emotional.get('emotional_variance', 0.2) * 0.8
        return min(5.0, 2.1 + audio_indicators * 1.2)
    
    def _calculate_performance_potential(self, characters, emotional, genre):
        performance_indicators = (characters.get('character_presence_score', 0.3) * 1.5 +
                                min(1.0, emotional.get('emotional_range', 0.2) * 1.5))
        return min(5.0, 2.0 + performance_indicators)
    
    def _generate_smart_summary(self, film_data, score, detected_genre):
        """Generate smart summary with genre detection insights"""
        title = film_data['title']
        original_genre = film_data.get('original_genre', 'Unknown')
        genre_confidence = film_data.get('genre_confidence', 0.0)
        
        # Genre detection insights
        genre_insight = ""
        if detected_genre != original_genre and original_genre != "Unknown":
            if genre_confidence > 0.7:
                genre_insight = f" AI analysis suggests this is better classified as {detected_genre}."
            else:
                genre_insight = f" AI detects elements of {detected_genre} alongside the specified {original_genre} genre."
        elif detected_genre == "Unknown":
            genre_insight = " Genre analysis was inconclusive - consider providing more descriptive content."
        else:
            genre_insight = f" Confirmed as {detected_genre} genre with {genre_confidence*100:.0f}% confidence."
        
        if score >= 4.5:
            return f"**{title}** demonstrates exceptional cinematic craftsmanship with sophisticated narrative structure.{genre_insight}"
        elif score >= 4.0:
            return f"**{title}** presents strong cinematic vision with well-developed narrative elements.{genre_insight}"
        elif score >= 3.5:
            return f"**{title}** shows promising potential with solid creative foundation.{genre_insight}"
        elif score >= 3.0:
            return f"**{title}** displays developing cinematic voice with foundational elements.{genre_insight}"
        else:
            return f"**{title}** shows creative beginnings with clear potential for development.{genre_insight}"
    
    def _generate_strengths(self, analysis_results, scores, detected_genre):
        """Generate smart, genre-aware strengths"""
        strengths = []
        
        if scores.get('story_narrative', 0) > 3.5:
            strengths.append("Strong narrative foundation and structural coherence")
        elif scores.get('story_narrative', 0) > 2.5:
            strengths.append("Clear storytelling intention and basic narrative structure")
        
        if scores.get('visual_vision', 0) > 3.0:
            strengths.append("Evocative descriptive elements and visual potential")
        
        if scores.get('performance', 0) > 3.5:
            strengths.append("Compelling character presence and performance potential")
        
        if not strengths:
            strengths.extend([
                "Authentic creative vision and personal expression",
                "Clear potential for cinematic development",
                "Foundational storytelling elements effectively established"
            ])
        
        return strengths[:3]
    
    def _generate_recommendations(self, analysis_results, scores, detected_genre):
        """Generate smart, genre-aware improvements"""
        recommendations = []
        
        if scores.get('technical_craft', 0) < 3.0:
            recommendations.append("Opportunity for enhanced technical execution and refinement")
        
        if scores.get('sound_design', 0) < 3.0:
            recommendations.append("Consider enhancing audio elements and sound design")
        
        if scores.get('story_narrative', 0) < 3.0:
            recommendations.append("Potential for strengthening narrative complexity and depth")
        
        if not recommendations:
            recommendations.extend([
                "Further development of technical execution",
                "Enhanced character depth and development", 
                "Stronger emotional pacing and narrative rhythm"
            ])
        
        return recommendations[:3]
    
    def _generate_festival_recommendations(self, overall_score, detected_genre):
        """Generate smart festival recommendations considering genre"""
        if overall_score >= 4.5:
            return {"level": "International Premier", "festivals": ["Sundance", "Cannes", "Toronto IFF", "Berlin International"]}
        elif overall_score >= 4.0:
            return {"level": "International Showcase", "festivals": ["SXSW", "Tribeca", "Venice", "Locarno"]}
        elif overall_score >= 3.5:
            return {"level": "National/Regional", "festivals": ["Regional showcases", "Emerging filmmaker programs"]}
        else:
            return {"level": "Development Focus", "festivals": ["Local screenings", "Workshop festivals", "Pitch events"]}
    
    def _generate_audience_analysis(self, analysis_results, detected_genre):
        """Generate smart audience analysis with genre context"""
        emotional = analysis_results['emotional_arc']
        
        if emotional.get('emotional_arc_strength', 0) > 0.6:
            return {"audience": "Film enthusiasts and festival viewers", "impact": "Strong emotional engagement"}
        elif emotional.get('emotional_arc_strength', 0) > 0.3:
            return {"audience": "General audiences and indie fans", "impact": "Thoughtful emotional resonance"}
        else:
            return {"audience": "Niche and development audiences", "impact": "Emerging creative voice"}
    
    def _create_fallback_analysis(self, film_data, detected_genre):
        """Smart fallback for limited content"""
        genre_insight = f" Genre analysis suggests {detected_genre} based on available content." if detected_genre != "Unknown" else ""
        
        return {
            "smart_summary": f"**{film_data['title']}** presents cinematic vision that would benefit from more content for deeper AI analysis.{genre_insight}",
            "cinematic_scores": {cat: round(random.uniform(2.8, 3.6), 1) for cat in ['story_narrative', 'visual_vision', 'technical_craft', 'sound_design', 'performance']},
            "overall_score": round(random.uniform(2.9, 3.5), 1),
            "strengths": ["Creative concept established", "Foundation for artistic development", "Clear narrative intention"],
            "recommendations": ["Enhanced content depth for analysis", "Technical refinement opportunities", "Character development focus"],
            "festival_recommendations": {"level": "Development Focus", "festivals": ["Emerging filmmaker workshops"]},
            "audience_analysis": {"audience": "Development and festival workshop audiences", "impact": "Creative potential awaiting full realization"},
            "genre_insights": {
                "detected_genre": detected_genre,
                "confidence": film_data.get('genre_confidence', 0.0),
                "original_genre": film_data.get('original_genre', 'Unknown'),
                "genre_alignment": 0.5
            }
        }

# --------------------------
# Film Database
# --------------------------
class FilmDatabase:
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
        
        scores = [film["analysis_results"]["overall_score"] for film in self.films]
        return {
            "total_films": len(self.films),
            "average_score": round(np.mean(scores), 2),
            "highest_score": round(max(scores), 2),
            "lowest_score": round(min(scores), 2)
        }

# --------------------------
# Smart CSV Batch Processor for Films
# --------------------------
class FilmCSVProcessor:
    def __init__(self, analyzer, database):
        self.analyzer = analyzer
        self.database = database

    def process_film_csv(self, df, progress_bar, status_text):
        """Process film CSV batch"""
        results = []
        
        for idx, row in df.iterrows():
            try:
                film_data = self._prepare_film_data_from_row(row, idx)
                status_text.text(f"🎬 Analyzing: {film_data['title']} ({idx + 1}/{len(df)})")
                
                analysis_results = self.analyzer.analyze_film(film_data)
                film_record = self.database.add_film_analysis(film_data, analysis_results)
                
                # Get genre insights
                genre_insights = analysis_results.get('genre_insights', {})
                detected_genre = genre_insights.get('detected_genre', 'Unknown')
                confidence = genre_insights.get('confidence', 0.0)
                
                results.append({
                    'title': film_data['title'],
                    'director': film_data.get('director', 'Unknown'),
                    'original_genre': film_data.get('genre', 'Unknown'),
                    'detected_genre': detected_genre,
                    'confidence': f"{confidence:.0%}",
                    'score': analysis_results['overall_score'],
                    'status': 'Success'
                })
                
            except Exception as e:
                results.append({
                    'title': film_data.get('title', f'Film_{idx}'),
                    'director': 'Error',
                    'original_genre': 'Error',
                    'detected_genre': 'Error',
                    'confidence': '0%',
                    'score': 0,
                    'status': f'Error: {str(e)[:50]}'
                })
            
            progress_bar.progress((idx + 1) / len(df))
        
        return results

    def _prepare_film_data_from_row(self, row, idx):
        """Prepare film data from CSV row with proper film column mapping"""
        # Film-specific column mapping
        column_mapping = {
            'title': ['title', 'Title', 'Film Title', 'Project Title', 'Name'],
            'director': ['director', 'Director', 'Filmmaker'],
            'writer': ['writer', 'Writer', 'Screenwriter'],
            'producer': ['producer', 'Producer', 'Production'],
            'genre': ['genre', 'Genre', 'Category', 'Type'],
            'duration': ['duration', 'Duration', 'Runtime', 'Length'],
            'synopsis': ['synopsis', 'Synopsis', 'Description', 'Logline', 'Summary']
        }
        
        film_data = {}
        
        for field, possible_columns in column_mapping.items():
            value_found = False
            for col in possible_columns:
                if col in row and pd.notna(row[col]):
                    film_data[field] = str(row[col])
                    value_found = True
                    break
            
            if not value_found:
                # Set default value
                film_data[field] = self._get_default_value(field, idx)
        
        # Add transcript field (use synopsis as fallback)
        film_data['transcript'] = film_data.get('synopsis', '')
        
        return film_data

    def _get_default_value(self, field, idx):
        """Get default values for missing film fields"""
        defaults = {
            'title': f'Film_{idx + 1}',
            'director': 'Unknown Director',
            'writer': 'Not specified',
            'producer': 'Not specified',
            'genre': 'Unknown',
            'duration': 'N/A',
            'synopsis': 'No description available'
        }
        return defaults.get(field, '')

# --------------------------
# Film Analysis Interface
# --------------------------
class FilmAnalysisInterface:
    def __init__(self, analyzer, database):
        self.analyzer = analyzer
        self.database = database
        self.csv_processor = FilmCSVProcessor(analyzer, database)

    def show_dashboard(self):
        """Main dashboard for film analysis"""
        st.header("🎬 FlickFinder AI - Film Analysis Hub")
        
        # Display statistics
        stats = self.database.get_statistics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Films", stats['total_films'])
        with col2:
            st.metric("Analyzed", stats['total_films'])
        with col3:
            if stats['total_films'] > 0:
                st.metric("Average Score", f"{stats['average_score']}/5.0")
        with col4:
            if stats['total_films'] > 0:
                st.metric("Highest Score", f"{stats['highest_score']}/5.0")
        
        # Analysis methods tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎥 YouTube Analysis", "📝 Manual Entry", "📊 CSV Batch", "🏆 Top Films"])
        
        with tab1:
            self._show_youtube_analysis()
        with tab2:
            self._show_manual_analysis()
        with tab3:
            self._show_csv_interface()
        with tab4:
            self._show_top_films()

    def _show_youtube_analysis(self):
        """YouTube-based film analysis"""
        st.subheader("🎥 Analyze from YouTube")
        
        youtube_url = st.text_input("**Paste YouTube URL:**", placeholder="https://www.youtube.com/watch?v=...")
        
        if youtube_url:
            video_id = self._get_video_id(youtube_url)
            if not video_id:
                st.error("❌ Invalid YouTube URL")
                return
            
            video_info = self._get_video_info(video_id)
            if not video_info.get('success'):
                st.error("❌ Could not access video information")
                return
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.components.v1.iframe(f"https://www.youtube.com/embed/{video_id}", height=300)
            with col2:
                st.write(f"**Title:** {video_info['title']}")
                st.write(f"**Channel:** {video_info['author']}")
            
            custom_title = st.text_input("✏️ **Film Title:**", value=video_info['title'])
            
            if st.button("🧠 **START FILM ANALYSIS**", type="primary", use_container_width=True):
                with st.spinner("🧠 Performing AI analysis..."):
                    try:
                        transcript = self._get_transcript(video_id)
                        film_data = {
                            'title': custom_title, 
                            'channel': video_info['author'], 
                            'transcript': transcript,
                            'synopsis': transcript[:500] + "..." if len(transcript) > 500 else transcript
                        }
                        
                        results = self.analyzer.analyze_film(film_data)
                        self.database.add_film_analysis(film_data, results)
                        self._display_film_results(results)
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")

    def _show_manual_analysis(self):
        """Manual film entry analysis"""
        st.subheader("📝 Manual Film Analysis")
        
        with st.form("manual_film_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("🎬 Film Title*", placeholder="Enter film title...")
                director = st.text_input("👤 Director", placeholder="Director name")
                genre = st.selectbox("🎭 Genre", 
                    ["Select...", "Drama", "Comedy", "Documentary", "Horror", 
                     "Sci-Fi", "Animation", "Thriller", "Romance", "Action", "Fantasy"])
            
            with col2:
                writer = st.text_input("✍️ Writer", placeholder="Writer name")
                duration = st.text_input("⏱️ Duration", placeholder="e.g., 90min")
                synopsis = st.text_area("📖 Synopsis", height=120, placeholder="Film description or logline...")
            
            submitted = st.form_submit_button("🎯 Analyze Film", use_container_width=True)
            
            if submitted and title:
                film_data = {
                    'title': title,
                    'director': director,
                    'writer': writer,
                    'genre': genre if genre != "Select..." else "Unknown",
                    'duration': duration,
                    'synopsis': synopsis,
                    'transcript': synopsis  # Use synopsis as transcript for manual entry
                }
                
                with st.spinner("🔮 Performing AI analysis..."):
                    try:
                        results = self.analyzer.analyze_film(film_data)
                        self.database.add_film_analysis(film_data, results)
                        self._display_film_results(results)
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")

    def _show_csv_interface(self):
        """CSV batch processing interface"""
        st.subheader("📊 Batch CSV Analysis")
        
        uploaded_file = st.file_uploader("📁 Upload Film CSV", type=['csv'], 
                                       help="Upload a CSV file with film data. Required columns: title, director, genre, synopsis")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} films")
                
                # Show data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button(f"🚀 Analyze {len(df)} Films", type="primary", use_container_width=True):
                    self._process_film_batch(df)
                    
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")

    def _process_film_batch(self, df):
        """Process film batch with progress tracking"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        with results_container:
            st.subheader("🔄 Processing Films...")
            
            results = self.csv_processor.process_film_csv(df, progress_bar, status_text)
            
            # Display results
            st.success("🎉 Batch analysis complete!")
            results_df = pd.DataFrame(results)
            
            # Show summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processed", len(results_df))
            with col2:
                success_count = len(results_df[results_df['status'] == 'Success'])
                st.metric("Successful", success_count)
            with col3:
                if success_count > 0:
                    avg_score = results_df[results_df['status'] == 'Success']['score'].mean()
                    st.metric("Average Score", f"{avg_score:.1f}/5.0")
            
            # Show results table
            st.subheader("📊 Analysis Results")
            display_df = results_df[['title', 'director', 'original_genre', 'detected_genre', 'confidence', 'score', 'status']]
            st.dataframe(display_df, use_container_width=True)

    def _show_top_films(self):
        """Display top films"""
        films = st.session_state.get('all_film_scores', [])
        
        if not films:
            st.info("🎬 No films analyzed yet. Start by analyzing some films!")
            return
        
        # Sort by overall score
        sorted_films = sorted(films, 
                            key=lambda x: x['analysis_results']['overall_score'], 
                            reverse=True)
        
        st.subheader("🏆 Top Rated Films")
        
        for film in sorted_films[:10]:  # Show top 10
            analysis = film['analysis_results']
            data = film['film_data']
            
            with st.expander(f"🎬 {data['title']} - {analysis['overall_score']}/5.0"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Director:** {data.get('director', 'Unknown')}")
                    st.write(f"**Genre:** {analysis['genre_insights']['detected_genre']}")
                    st.write(f"**Duration:** {data.get('duration', 'N/A')}")
                
                with col2:
                    scores = analysis['cinematic_scores']
                    st.write(f"**Story:** {scores['story_narrative']}/5.0")
                    st.write(f"**Visual:** {scores['visual_vision']}/5.0")
                    st.write(f"**Technical:** {scores['technical_craft']}/5.0")
                
                st.write("**Strengths:**")
                for strength in analysis['strengths']:
                    st.write(f"✨ {strength}")

    def _display_film_results(self, results):
        """Display film analysis results"""
        st.success("🎉 Film Analysis Complete!")
        
        # Overall score
        overall_score = results['overall_score']
        st.markdown(f"""
        <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  border-radius: 15px; margin: 20px 0; border: 3px solid #FFD700;'>
            <h1 style='color: gold; margin: 0; font-size: 48px;'>{overall_score}/5.0</h1>
            <p style='color: white; font-size: 20px; margin: 10px 0;'>🎬 Overall Magic Score</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Genre insights
        genre_insights = results['genre_insights']
        if genre_insights['detected_genre'] != "Unknown":
            confidence_percent = genre_insights['confidence'] * 100
            st.info(f"🎭 **Genre Analysis**: Detected **{genre_insights['detected_genre']}** ({confidence_percent:.0f}% confidence)")
        
        # Category scores
        st.subheader("🎯 Category Analysis")
        scores = results['cinematic_scores']
        cols = st.columns(5)
        
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
                <div style='text-align: center; padding: 15px; background: {color}; border-radius: 10px; margin: 5px; border: 2px solid white;'>
                    <h4 style='margin: 0; color: white;'>{name}</h4>
                    <h2 style='margin: 8px 0; color: white;'>{score}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Summary and insights
        st.subheader("📖 Analysis Summary")
        st.write(results['smart_summary'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Strengths")
            for strength in results['strengths']:
                st.write(f"✨ {strength}")
        
        with col2:
            st.subheader("💡 Recommendations")
            for recommendation in results['recommendations']:
                st.write(f"🔧 {recommendation}")
        
        # Festival recommendations
        festival_recs = results['festival_recommendations']
        st.subheader("🎪 Festival Recommendations")
        st.write(f"**Level:** {festival_recs['level']}")
        st.write("**Suggested Festivals:**")
        for festival in festival_recs['festivals']:
            st.write(f"• {festival}")
        
        # Audience analysis
        audience = results['audience_analysis']
        st.subheader("🎯 Target Audience")
        st.write(f"**Primary Audience**: {audience['audience']}")
        st.write(f"**Expected Impact**: {audience['impact']}")

    def _get_video_id(self, url):
        """Extract video ID from YouTube URL"""
        try:
            parsed = urlparse(url)
            if "youtube.com" in parsed.hostname:
                return parse_qs(parsed.query).get("v", [None])[0]
            elif parsed.hostname == "youtu.be":
                return parsed.path[1:]
        except:
            return None

    def _get_video_info(self, video_id):
        """Get video information from YouTube"""
        try:
            response = requests.get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {'title': data.get('title', 'Unknown'), 'author': data.get('author_name', 'Unknown'), 'success': True}
        except:
            pass
        return {'success': False}

    def _get_transcript(self, video_id):
        """Get transcript from YouTube video"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([seg["text"] for seg in transcript_list])
        except:
            return "No transcript available. AI will analyze based on film context and metadata."

# --------------------------
# Main Application
# --------------------------
def main():
    st.sidebar.title("🎬 FlickFinder AI")
    st.sidebar.markdown("---")
    
    # Initialize components
    analyzer = FilmAnalysisEngine()
    database = FilmDatabase()
    film_interface = FilmAnalysisInterface(analyzer, database)
    
    # Navigation
    page = st.sidebar.radio("Navigate:", 
        ["🏠 Dashboard", "📈 Analytics", "ℹ️ About"])
    
    if page == "🏠 Dashboard":
        film_interface.show_dashboard()
    elif page == "📈 Analytics":
        film_interface._show_top_films()
    else:
        st.header("ℹ️ About FlickFinder AI")
        st.write("""
        **FlickFinder AI** is an intelligent film analysis tool that uses AI to evaluate cinematic content.
        
        **Features:**
        - 🎬 Smart genre detection from transcripts and descriptions
        - 📊 Multi-category scoring (Story, Visual, Technical, Sound, Performance)
        - 💡 AI-powered recommendations for improvement
        - 🎪 Festival recommendations based on quality
        - 📁 Batch CSV processing for multiple films
        - 🎯 Target audience analysis
        
        **Supported Input Methods:**
        1. **YouTube URLs** - Analyze films directly from YouTube
        2. **Manual Entry** - Enter film details manually
        3. **CSV Upload** - Batch analyze multiple films
        
        **CSV Format:** Your CSV should include columns like:
        - `title`, `director`, `writer`, `genre`, `duration`, `synopsis`
        """)

if __name__ == "__main__":
    main()

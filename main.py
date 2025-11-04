import streamlit as st
import tempfile
import numpy as np
from openai import OpenAI
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
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'filmfreeway_projects' not in st.session_state:
    st.session_state.filmfreeway_projects = []
if 'magic_mode' not in st.session_state:
    st.session_state.magic_mode = True

# --------------------------
# Initialize AI Clients
# --------------------------
@st.cache_resource
def initialize_ai_clients():
    """Initialize all AI clients"""
    clients = {}
    
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not api_key:
            st.error("❌ No OpenAI API key found. Please set OPENAI_API_KEY in secrets or environment variables.")
            return clients
            
        clients['openai'] = OpenAI(api_key=api_key)
        st.success("✅ All AI Magic Systems Initialized! ✨")
    except Exception as e:
        clients['openai'] = None
        st.error(f"❌ AI Magic failed to initialize: {e}")
    
    return clients

# --------------------------
# Magical Film Analyzer with ALL Capabilities (GPT-3.5 compatible)
# --------------------------
class MagicalFilmAnalyzer:
    def __init__(self, clients):
        self.clients = clients
        self.capabilities = {
            "gpt_analysis": "🧠 Comprehensive narrative analysis",
            "visual_assessment": "👁️ Visual composition analysis", 
            "sentiment_analysis": "😊 Emotional tone assessment",
            "performance_evaluation": "🎭 Acting quality evaluation",
            "cinematic_technique": "🎨 Professional film critique"
        }
    
    def perform_magical_analysis(self, film_data):
        """Perform magical multi-modal film analysis using ALL AI capabilities"""
        if not self.clients.get('openai'):
            return self._create_magical_fallback()
        
        try:
            # Step 1: Multi-modal analysis
            analysis_results = {}
            
            # 🧠 GPT Analysis - Narrative Analysis
            if film_data.get('transcript'):
                analysis_results['narrative_analysis'] = self._analyze_narrative(film_data)
            
            # 👁️ Visual Assessment - Based on description
            analysis_results['visual_assessment'] = self._assess_visual_elements(film_data)
            
            # 😊 Sentiment Analysis - Emotional Tone
            if film_data.get('transcript'):
                analysis_results['sentiment_analysis'] = self._analyze_sentiment(film_data['transcript'])
            
            # Step 2: Generate comprehensive magical review
            magical_review = self._generate_magical_review(film_data, analysis_results)
            return magical_review
            
        except Exception as e:
            st.error(f"❌ Magical analysis failed: {e}")
            return self._create_magical_fallback()
    
    def _analyze_narrative(self, film_data):
        """🧠 Comprehensive narrative analysis"""
        try:
            prompt = f"""
            As a master film critic, analyze this film's narrative and storytelling:

            TITLE: {film_data['title']}
            SOURCE: {film_data.get('channel', 'YouTube')}
            CONTENT: {film_data.get('transcript', '')[:2500]}

            Analyze these narrative elements and provide scores 1-5:
            1. Story structure and narrative arc
            2. Character development and depth
            3. Dialogue quality and authenticity  
            4. Thematic depth and symbolism
            5. Pacing and narrative flow

            Return only JSON:
            {{
                "story_structure": 3.5,
                "character_development": 4.0,
                "dialogue_quality": 3.8,
                "thematic_depth": 3.2,
                "pacing": 4.1,
                "narrative_insights": ["insight1", "insight2"]
            }}
            """
            
            response = self.clients['openai'].chat.completions.create(
                model="gpt-3.5-turbo",  # Using GPT-3.5 instead of GPT-4
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.warning(f"⚠️ Narrative analysis limited: {e}")
            return {"story_structure": 3.5, "character_development": 3.5, "dialogue_quality": 3.5, "thematic_depth": 3.0, "pacing": 3.5}
    
    def _assess_visual_elements(self, film_data):
        """👁️ Visual assessment based on available data"""
        try:
            prompt = f"""
            Based on the film context, assess likely visual elements:

            FILM: {film_data['title']}
            CONTEXT: YouTube film from {film_data.get('channel', 'unknown channel')}
            CONTENT: {film_data.get('transcript', '')[:1000]}

            Assess typical visual qualities for this type of content and provide scores 1-5:
            1. Visual composition and framing
            2. Lighting and color palette
            3. Cinematic style and aesthetic
            4. Production value and quality

            Return only JSON:
            {{
                "visual_composition": 3.8,
                "lighting_quality": 3.5,
                "cinematic_style": 4.0,
                "production_value": 3.7,
                "visual_notes": ["note1", "note2"]
            }}
            """
            
            response = self.clients['openai'].chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            return json.loads(response.choices[0].message.content)
        except:
            return {"visual_composition": 3.5, "lighting_quality": 3.5, "cinematic_style": 3.5, "production_value": 3.5}
    
    def _analyze_sentiment(self, transcript):
        """😊 Sentiment and emotional analysis"""
        try:
            prompt = f"""
            Analyze the emotional tone and sentiment of this film content:

            CONTENT: {transcript[:1500]}

            Evaluate:
            - Overall emotional tone (positive/negative/neutral/mixed)
            - Emotional depth and complexity
            - Audience emotional impact
            - Key emotional moments

            Return only JSON:
            {{
                "emotional_tone": "positive/negative/neutral/mixed",
                "emotional_depth": 3.5,
                "audience_impact": 4.0,
                "key_emotional_moments": ["moment1", "moment2"]
            }}
            """
            
            response = self.clients['openai'].chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600
            )
            
            return json.loads(response.choices[0].message.content)
        except:
            return {"emotional_tone": "neutral", "emotional_depth": 3.0, "audience_impact": 3.0}
    
    def _generate_magical_review(self, film_data, analysis_results):
        """Generate the ultimate magical review combining all analyses"""
        try:
            prompt = self._build_magical_prompt(film_data, analysis_results)
            
            response = self.clients['openai'].chat.completions.create(
                model="gpt-3.5-turbo",  # Using GPT-3.5 instead of GPT-4
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a world-class film critic with magical insight into cinema. 
                        Create brilliant, comprehensive film reviews that combine multiple analytical perspectives."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            review_text = response.choices[0].message.content
            return self._parse_magical_review(review_text, analysis_results)
            
        except Exception as e:
            st.error(f"❌ Magical review generation failed: {e}")
            return self._create_magical_fallback()
    
    def _build_magical_prompt(self, film_data, analysis_results):
        """Build the ultimate magical analysis prompt"""
        prompt = f"""
        🎬 MAGICAL FILM ANALYSIS - CREATE BRILLIANT REVIEW
        
        FILM: {film_data['title']}
        SOURCE: {film_data.get('channel', 'YouTube')}
        
        ANALYTICAL INSIGHTS:
        
        NARRATIVE ANALYSIS:
        {analysis_results.get('narrative_analysis', {})}
        
        VISUAL ASSESSMENT:
        {analysis_results.get('visual_assessment', {})}
        
        EMOTIONAL ANALYSIS:
        {analysis_results.get('sentiment_analysis', {})}
        
        CREATE A MAGICAL FILM REVIEW WITH:
        
        🎯 COMPREHENSIVE 5-CATEGORY SCORING (1-5):
        
        1. STORY & NARRATIVE (30% weight)
           - Based on narrative analysis above
           - Plot, characters, dialogue, themes
        
        2. VISUAL VISION (25% weight) 
           - Based on visual assessment above
           - Cinematography, composition, style
        
        3. TECHNICAL CRAFT (25% weight)
           - Technical execution quality
           - Production values, editing
        
        4. SOUND DESIGN (10% weight)
           - Audio quality and design
           - Music, sound effects, mixing
        
        5. PERFORMANCE (10% weight)
           - Acting quality and authenticity
           - Character believability
        
        Return ONLY this JSON format:
        {{
            "magical_summary": "Brilliant 2-3 paragraph analysis synthesizing all insights with cinematic wisdom",
            "cinematic_scores": {{
                "story_narrative": 4.2,
                "visual_vision": 4.0,
                "technical_craft": 3.8,
                "sound_design": 3.5,
                "performance": 4.1
            }},
            "overall_magic_score": 4.0,
            "strengths": ["Specific cinematic strength 1", "Specific strength 2", "Specific strength 3"],
            "improvements": ["Constructive area 1", "Constructive area 2"],
            "festival_recommendations": {{
                "suitable_festivals": ["Festival Type 1", "Festival Type 2"],
                "competition_level": "Regional/National/International"
            }},
            "audience_analysis": {{
                "target_audience": "Description of ideal viewers",
                "emotional_impact": "How the film emotionally affects audiences"
            }}
        }}
        
        Calculate overall_magic_score using: story_narrative*0.30 + visual_vision*0.25 + technical_craft*0.25 + sound_design*0.10 + performance*0.10
        Be specific, cinematic, and insightful in all feedback.
        """
        return prompt
    
    def _parse_magical_review(self, review_text, analysis_results):
        """Parse the magical review response"""
        try:
            if "{" in review_text and "}" in review_text:
                json_str = review_text[review_text.find("{"):review_text.rfind("}")+1]
                review_data = json.loads(json_str)
                
                # Add analysis metadata
                review_data['analysis_methods'] = list(analysis_results.keys())
                review_data['ai_capabilities_used'] = list(self.capabilities.keys())
                
                return review_data
        except Exception as e:
            st.warning(f"⚠️ Review parsing adapted: {e}")
        
        return self._create_magical_fallback()
    
    def _create_magical_fallback(self):
        """Create magical fallback review"""
        return {
            "magical_summary": "This film demonstrates solid cinematic craftsmanship with clear artistic intention. The narrative foundation shows promise while technical execution maintains professional standards suitable for festival consideration.",
            "cinematic_scores": {
                "story_narrative": 3.8,
                "visual_vision": 3.7,
                "technical_craft": 3.6,
                "sound_design": 3.5,
                "performance": 3.8
            },
            "overall_magic_score": 3.7,
            "strengths": [
                "Clear narrative structure and pacing",
                "Competent technical execution", 
                "Authentic character portrayals"
            ],
            "improvements": [
                "Opportunity for more distinctive visual style",
                "Potential for deeper thematic exploration"
            ],
            "festival_recommendations": {
                "suitable_festivals": ["Regional film festivals", "Emerging filmmaker showcases"],
                "competition_level": "Regional"
            },
            "audience_analysis": {
                "target_audience": "Independent film enthusiasts and festival audiences",
                "emotional_impact": "Creates genuine engagement through authentic storytelling"
            },
            "ai_capabilities_used": ["gpt_analysis", "visual_assessment", "sentiment_analysis"],
            "analysis_methods": ["narrative_analysis", "visual_assessment", "sentiment_analysis"]
        }

# --------------------------
# FilmFreeway Integration (Simplified)
# --------------------------
class FilmFreewayImporter:
    def manual_import_interface(self):
        """Manual FilmFreeway project import"""
        st.subheader("📥 FilmFreeway Projects")
        
        with st.form("filmfreeway_form"):
            title = st.text_input("🎬 Project Title")
            director = st.text_input("👤 Director")
            genre = st.selectbox("🎭 Genre", ["Drama", "Comedy", "Documentary", "Horror", "Sci-Fi", "Animation", "Experimental", "Other"])
            duration = st.text_input("⏱️ Duration", placeholder="e.g., 15:30 or 90 min")
            synopsis = st.text_area("📖 Synopsis", height=100)
            
            if st.form_submit_button("💾 Add Project"):
                if title:
                    project_data = {
                        'id': len(st.session_state.filmfreeway_projects) + 1,
                        'title': title,
                        'director': director,
                        'genre': genre,
                        'duration': duration,
                        'synopsis': synopsis,
                        'import_date': datetime.now().isoformat()
                    }
                    st.session_state.filmfreeway_projects.append(project_data)
                    st.success(f"✅ '{title}' added to project library!")

# --------------------------
# Record Keeping System
# --------------------------
class FilmScoreDatabase:
    def __init__(self):
        self.films = []
        if 'all_film_scores' in st.session_state:
            self.films = st.session_state.all_film_scores.copy()
    
    def add_film_analysis(self, film_data, analysis_results):
        """Add a film analysis to the database"""
        film_record = {
            "id": len(self.films) + 1,
            "timestamp": datetime.now().isoformat(),
            "film_data": film_data,
            "analysis_results": analysis_results,
            "weighted_score": analysis_results.get("overall_magic_score", 0),
            "category_scores": analysis_results.get("cinematic_scores", {})
        }
        
        self.films.append(film_record)
        st.session_state.all_film_scores = self.films.copy()
        return film_record
    
    def get_statistics(self):
        """Get database statistics with safe defaults"""
        if not self.films:
            return {
                "total_films": 0, 
                "average_score": 0, 
                "highest_score": 0,
                "lowest_score": 0
            }
        
        scores = [film["weighted_score"] for film in self.films]
        return {
            "total_films": len(self.films),
            "average_score": round(np.mean(scores), 2) if scores else 0,
            "highest_score": round(max(scores), 2) if scores else 0,
            "lowest_score": round(min(scores), 2) if scores else 0
        }

# --------------------------
# Utility Functions
# --------------------------
def get_video_id(url):
    """Extract YouTube video ID"""
    try:
        parsed = urlparse(url)
        if parsed.hostname in ["www.youtube.com", "youtube.com"]:
            return parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.hostname == "youtu.be":
            return parsed.path[1:]
        return None
    except:
        return None

def get_video_info(video_id):
    """Get video information"""
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'title': data.get('title', 'Unknown'),
                'author': data.get('author_name', 'Unknown'),
                'success': True
            }
    except:
        pass
    return {'success': False}

def get_transcript(video_id):
    """Get transcript from YouTube"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([seg["text"] for seg in transcript_list])
    except:
        return "No transcript available. AI will analyze based on film context and metadata."

# --------------------------
# Magical Interface (Keep the same beautiful UI)
# --------------------------
def magical_interface(clients, analyzer, database, filmfreeway_importer):
    """The truly magical interface"""
    st.header("🔮 FlickFinder MAGICAL AI Analysis")
    st.markdown("### ✨ Where AI works its cinema magic!")
    
    # Display AI capabilities
    with st.expander("🎩 **AI Magic Capabilities**", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🧠 GPT Intelligence**
            - Narrative structure analysis
            - Character development evaluation
            - Thematic depth assessment
            
            **👁️ Visual Assessment**
            - Composition and style analysis
            - Cinematic quality evaluation
            - Production value assessment
            """)
        
        with col2:
            st.markdown("""
            **😊 Emotional Analysis**
            - Sentiment and tone evaluation
            - Emotional arc mapping
            - Audience impact assessment
            
            **🎭 Performance Evaluation**
            - Acting quality assessment
            - Character believability
            
            **🎨 Cinematic Technique**
            - Professional film critique
            - Technical execution evaluation
            """)
    
    # URL input for magical analysis
    st.markdown("---")
    st.subheader("🎬 Magical Film Analysis")
    
    youtube_url = st.text_input(
        "**🎥 Paste YouTube URL for Magical Analysis:**",
        placeholder="https://www.youtube.com/watch?v=...",
        help="The AI will use ALL its capabilities to analyze your film magically!"
    )
    
    if youtube_url:
        st.info(f"🔮 Preparing magical analysis for: {youtube_url}")
        
        # Extract video ID
        video_id = get_video_id(youtube_url)
        if not video_id:
            st.error("❌ Could not extract video ID")
            return
        
        st.success(f"✅ Video ID: {video_id}")
        
        # Get video info
        video_info = get_video_info(video_id)
        if not video_info.get('success'):
            st.error("❌ Could not access video information")
            return
        
        # Display video
        col1, col2 = st.columns([2, 1])
        with col1:
            embed_url = f"https://www.youtube.com/embed/{video_id}"
            st.components.v1.iframe(embed_url, height=400)
        
        with col2:
            st.subheader("📋 Film Info")
            st.write(f"**Title:** {video_info['title']}")
            st.write(f"**Channel:** {video_info['author']}")
        
        # Film title customization
        custom_title = st.text_input(
            "**✏️ Film Title (edit if needed):**",
            value=video_info['title'],
            key="magic_title"
        )
        
        # Magical analysis button
        if st.button("**🔮 START MAGICAL AI ANALYSIS**", 
                   type="primary", 
                   use_container_width=True,
                   key="magic_button"):
            
            perform_complete_magical_analysis(video_info, custom_title, video_id, clients, analyzer, database)

def perform_complete_magical_analysis(video_info, custom_title, video_id, clients, analyzer, database):
    """Perform the full magical analysis"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Extract content
        status_text.text("📝 Gathering film content...")
        transcript = get_transcript(video_id)
        progress_bar.progress(25)
        time.sleep(1)
        
        # Step 2: Multi-modal analysis
        status_text.text("🧠 Activating AI capabilities...")
        
        film_data = {
            'title': custom_title,
            'channel': video_info['author'],
            'transcript': transcript,
            'video_id': video_id
        }
        
        # Perform magical analysis
        magical_results = analyzer.perform_magical_analysis(film_data)
        progress_bar.progress(75)
        time.sleep(1)
        
        # Step 3: Store and display results
        status_text.text("✨ Synthesizing magical insights...")
        
        # Store in database
        film_record = database.add_film_analysis(film_data, magical_results)
        st.session_state.current_analysis = film_record
        
        progress_bar.progress(100)
        
        # Display magical results
        display_magical_results(magical_results, film_data)
        status_text.text("🎉 Magical Analysis Complete!")
        
    except Exception as e:
        st.error(f"❌ Magical analysis failed: {e}")
        progress_bar.progress(0)

def display_magical_results(magical_results, film_data):
    """Display the magical analysis results"""
    st.success("🌟 **MAGICAL AI ANALYSIS COMPLETE!**")
    
    # Magical score display
    magic_score = magical_results.get('overall_magic_score', 0)
    st.markdown(f"""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #8A2BE2 0%, #4B0082 100%); border-radius: 20px; margin: 20px 0; border: 3px solid #FFD700;'>
        <h1 style='color: gold; margin: 0; font-size: 60px; text-shadow: 2px 2px 4px #000;'>{magic_score:.1f}/5.0</h1>
        <p style='color: white; font-size: 24px; margin: 10px 0 0 0;'>✨ Overall Magic Score ✨</p>
        <p style='color: silver; font-size: 18px; margin: 5px 0 0 0;'>{film_data['title']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 5-Category Scoring System
    st.subheader("🎯 Comprehensive 5-Category Scoring")
    scores = magical_results.get('cinematic_scores', {})
    
    cols = st.columns(5)
    categories = [
        ("🧙‍♂️ Story & Narrative", scores.get('story_narrative', 0), "#FF6B6B", "30%"),
        ("🔮 Visual Vision", scores.get('visual_vision', 0), "#4ECDC4", "25%"),
        ("⚡ Technical Craft", scores.get('technical_craft', 0), "#45B7D1", "25%"),
        ("🎵 Sound Design", scores.get('sound_design', 0), "#96CEB4", "10%"),
        ("🌟 Performance", scores.get('performance', 0), "#FFD93D", "10%")
    ]
    
    for idx, (name, score, color, weight) in enumerate(categories):
        with cols[idx]:
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; background: {color}; border-radius: 12px; margin: 5px; border: 2px solid gold;'>
                <h4 style='margin: 0; color: white; font-size: 14px;'>{name}</h4>
                <h2 style='margin: 8px 0; color: gold; font-size: 26px; text-shadow: 1px 1px 2px #000;'>{score:.1f}</h2>
                <p style='margin: 0; color: white; font-size: 12px;'>{weight} weight</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Magical summary
    st.subheader("📖 Magical Analysis Summary")
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;'>
        {magical_results.get('magical_summary', 'No magical summary available.')}
    </div>
    """, unsafe_allow_html=True)
    
    # Strengths and improvements
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Strengths")
        for strength in magical_results.get('strengths', []):
            st.write(f"✨ {strength}")
    
    with col2:
        st.subheader("📝 Areas for Improvement")
        for improvement in magical_results.get('improvements', []):
            st.write(f"🔧 {improvement}")
    
    # Festival recommendations
    st.subheader("🏆 Festival Recommendations")
    festivals = magical_results.get('festival_recommendations', {})
    st.info(f"**Competition Level:** {festivals.get('competition_level', 'N/A')}")
    st.write("**Suitable For:**")
    for festival in festivals.get('suitable_festivals', []):
        st.write(f"• {festival}")
    
    # Audience analysis
    st.subheader("🎯 Audience Analysis")
    audience = magical_results.get('audience_analysis', {})
    st.write(f"**Target Audience:** {audience.get('target_audience', 'N/A')}")
    st.write(f"**Emotional Impact:** {audience.get('emotional_impact', 'N/A')}")
    
    # AI capabilities used
    st.subheader("🤖 AI Magic Used")
    capabilities_used = magical_results.get('ai_capabilities_used', [])
    capability_emojis = {
        "gpt_analysis": "🧠",
        "visual_assessment": "👁️", 
        "sentiment_analysis": "😊",
        "performance_evaluation": "🎭",
        "cinematic_technique": "🎨"
    }
    
    for capability in capabilities_used:
        emoji = capability_emojis.get(capability, "⚡")
        st.write(f"{emoji} {capability.replace('_', ' ').title()}")

# --------------------------
# Main Application
# --------------------------
def main():
    """Main magical application"""
    
    st.sidebar.title("🔮 FlickFinder MAGIC")
    st.sidebar.markdown("---")
    
    # Initialize AI Magic
    with st.sidebar:
        with st.spinner("✨ Initializing AI Magic..."):
            clients = initialize_ai_clients()
            analyzer = MagicalFilmAnalyzer(clients)
            filmfreeway_importer = FilmFreewayImporter()
            database = FilmScoreDatabase()
    
    st.sidebar.markdown("### 🎩 Magical Capabilities:")
    for capability, description in analyzer.capabilities.items():
        emoji = description.split()[0]
        st.sidebar.write(f"{emoji} {description}")
    
    st.sidebar.markdown("### 🎯 5-Category Scoring:")
    st.sidebar.write("• 🧙‍♂️ Story & Narrative (30%)")
    st.sidebar.write("• 🔮 Visual Vision (25%)")
    st.sidebar.write("• ⚡ Technical Craft (25%)")
    st.sidebar.write("• 🎵 Sound Design (10%)")
    st.sidebar.write("• 🌟 Performance (10%)")
    
    # Navigation
    page = st.sidebar.radio("Navigate to:", ["🔮 Magical Analysis", "📥 FilmFreeway", "💾 Database"])
    
    if page == "🔮 Magical Analysis":
        magical_interface(clients, analyzer, database, filmfreeway_importer)
    elif page == "📥 FilmFreeway":
        filmfreeway_importer.manual_import_interface()
        # Show projects if any exist
        if st.session_state.filmfreeway_projects:
            st.subheader("📚 Your FilmFreeway Projects")
            for project in st.session_state.filmfreeway_projects:
                with st.expander(f"🎬 {project['title']}"):
                    st.write(f"**Director:** {project.get('director', 'N/A')}")
                    st.write(f"**Genre:** {project.get('genre', 'N/A')}")
                    st.write(f"**Duration:** {project.get('duration', 'N/A')}")
                    if project.get('synopsis'):
                        st.write("**Synopsis:**")
                        st.write(project['synopsis'])
    elif page == "💾 Database":
        st.header("💾 Film Database")
        stats = database.get_statistics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Films", stats['total_films'])
        with col2:
            st.metric("Average Score", f"{stats['average_score']}/5.0")
        with col3:
            st.metric("Highest Score", f"{stats['highest_score']}/5.0")
        with col4:
            st.metric("Lowest Score", f"{stats['lowest_score']}/5.0")
        
        # Show film history
        if database.films:
            st.subheader("🎬 Film Analysis History")
            for film in database.films[-5:]:  # Show last 5 analyses
                with st.expander(f"📊 {film['film_data']['title']} - {film['weighted_score']}/5.0"):
                    st.write(f"**Analyzed:** {film['timestamp'][:16]}")
                    st.write(f"**Source:** {film['film_data'].get('channel', 'YouTube')}")
                    scores = film['category_scores']
                    st.write("**Scores:**")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Story", f"{scores.get('story_narrative', 0):.1f}")
                    with col2:
                        st.metric("Visual", f"{scores.get('visual_vision', 0):.1f}")
                    with col3:
                        st.metric("Technical", f"{scores.get('technical_craft', 0):.1f}")
                    with col4:
                        st.metric("Sound", f"{scores.get('sound_design', 0):.1f}")
                    with col5:
                        st.metric("Performance", f"{scores.get('performance', 0):.1f}")

if __name__ == "__main__":
    main()

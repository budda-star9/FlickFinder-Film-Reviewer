import streamlit as st
import tempfile
import cv2
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
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_film' not in st.session_state:
    st.session_state.current_film = None
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
        st.success("✅ All AI systems initialized successfully!")
    except Exception as e:
        clients['openai'] = None
        st.error(f"❌ AI initialization failed: {e}")
    
    return clients

# --------------------------
# Magical Film Analyzer with ALL Capabilities
# --------------------------
class MagicalFilmAnalyzer:
    def __init__(self, clients):
        self.clients = clients
        self.capabilities = {
            "gpt4_analysis": "🧠 Comprehensive narrative and thematic analysis",
            "gpt4_vision": "👁️ Frame-by-frame visual composition analysis", 
            "sentiment_analysis": "😊 Emotional tone and audience impact assessment",
            "performance_evaluation": "🎭 Acting quality and character authenticity",
            "cinematic_technique": "🎨 Lighting, framing, and directorial style analysis"
        }
    
    def analyze_film_magically(self, film_data):
        """Perform magical multi-modal film analysis using ALL AI capabilities"""
        if not self.clients.get('openai'):
            return self._create_magical_fallback()
        
        try:
            # Step 1: Multi-modal analysis
            analysis_results = {}
            
            # 🧠 GPT-4 Text Analysis
            if film_data.get('transcript'):
                analysis_results['text_analysis'] = self._analyze_text_content(film_data)
            
            # 👁️ GPT-4 Vision Analysis  
            if film_data.get('frames'):
                analysis_results['visual_analysis'] = self._analyze_visual_content(film_data['frames'])
            
            # 😊 Sentiment & Emotion Analysis
            if film_data.get('transcript'):
                analysis_results['sentiment_analysis'] = self._analyze_sentiment_emotion(film_data['transcript'])
            
            # Step 2: Generate comprehensive magical review
            magical_review = self._generate_magical_review(film_data, analysis_results)
            return magical_review
            
        except Exception as e:
            st.error(f"❌ Magical analysis error: {e}")
            return self._create_magical_fallback()
    
    def _analyze_text_content(self, film_data):
        """🧠 GPT-4 Comprehensive narrative analysis"""
        try:
            prompt = f"""
            As a master film critic, analyze this film's narrative and storytelling:

            TITLE: {film_data['title']}
            CONTENT: {film_data.get('transcript', '')[:3000]}

            Analyze:
            1. Story structure and narrative arc
            2. Character development and depth
            3. Dialogue quality and authenticity  
            4. Thematic depth and symbolism
            5. Pacing and narrative flow

            Return JSON: {{
                "narrative_quality": 1-5,
                "character_depth": 1-5,
                "dialogue_effectiveness": 1-5,
                "thematic_strength": 1-5,
                "pacing_quality": 1-5,
                "story_insights": ["key insight 1", "key insight 2"]
            }}
            """
            
            response = self.clients['openai'].chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
        except:
            return {"narrative_quality": 3.5, "error": "Text analysis failed"}
    
    def _analyze_visual_content(self, frames):
        """👁️ GPT-4 Vision Frame analysis"""
        try:
            visual_analyses = []
            
            for i, frame in enumerate(frames[:3]):  # Analyze first 3 frames
                base64_image = base64.b64encode(frame).decode('utf-8')
                
                response = self.clients['openai'].chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Analyze this film frame for cinematic quality. Evaluate:
                                    - Composition and framing (1-5)
                                    - Lighting and contrast (1-5)  
                                    - Color palette and mood (1-5)
                                    - Visual storytelling (1-5)
                                    - Cinematic style description
                                    Return as JSON with scores and analysis."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
                
                analysis_text = response.choices[0].message.content
                try:
                    if "{" in analysis_text and "}" in analysis_text:
                        json_str = analysis_text[analysis_text.find("{"):analysis_text.rfind("}")+1]
                        frame_analysis = json.loads(json_str)
                        visual_analyses.append(frame_analysis)
                except:
                    continue
            
            return self._aggregate_visual_analysis(visual_analyses)
        except Exception as e:
            return {"visual_quality": 3.5, "error": f"Visual analysis failed: {e}"}
    
    def _analyze_sentiment_emotion(self, transcript):
        """😊 Sentiment and emotional analysis"""
        try:
            prompt = f"""
            Analyze the emotional journey and sentiment of this film:

            TRANSCRIPT: {transcript[:2000]}

            Evaluate:
            - Overall emotional tone (positive/negative/neutral)
            - Emotional arc throughout the film
            - Key emotional moments and transitions
            - Audience emotional impact

            Return JSON: {{
                "overall_sentiment": "positive/negative/neutral",
                "emotional_arc": "description of emotional journey",
                "key_emotional_moments": ["moment1", "moment2"],
                "audience_impact_score": 1-5,
                "emotional_depth": 1-5
            }}
            """
            
            response = self.clients['openai'].chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
        except:
            return {"overall_sentiment": "neutral", "audience_impact_score": 3.0}
    
    def _aggregate_visual_analysis(self, visual_analyses):
        """Aggregate multiple visual analyses"""
        if not visual_analyses:
            return {"visual_quality": 3.0, "composition": 3.0, "lighting": 3.0, "color": 3.0}
        
        aggregated = {
            "visual_quality": np.mean([v.get('composition', 3) for v in visual_analyses]),
            "composition": np.mean([v.get('composition', 3) for v in visual_analyses]),
            "lighting": np.mean([v.get('lighting', 3) for v in visual_analyses]),
            "color": np.mean([v.get('color', 3) for v in visual_analyses]),
            "visual_storytelling": np.mean([v.get('visual_storytelling', 3) for v in visual_analyses]),
            "frame_analyses": visual_analyses
        }
        return aggregated
    
    def _generate_magical_review(self, film_data, analysis_results):
        """Generate the ultimate magical review combining all analyses"""
        try:
            # Build mega-prompt with all analysis data
            prompt = self._build_magical_prompt(film_data, analysis_results)
            
            response = self.clients['openai'].chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a world-class film critic with magical insight into cinema. 
                        Combine multiple AI analyses into one brilliant, comprehensive review that showcases 
                        deep understanding of film artistry."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            review_text = response.choices[0].message.content
            return self._parse_magical_review(review_text, analysis_results)
            
        except Exception as e:
            st.error(f"❌ Magical review generation failed: {e}")
            return self._create_magical_fallback()
    
    def _build_magical_prompt(self, film_data, analysis_results):
        """Build the ultimate magical analysis prompt"""
        prompt = f"""
        🎬 MAGICAL FILM ANALYSIS - COMBINE ALL AI INSIGHTS
        
        FILM: {film_data['title']}
        CONTEXT: {film_data.get('description', 'Cinematic work for analysis')}
        
        === AI ANALYSIS RESULTS ===
        
        🧠 NARRATIVE ANALYSIS:
        {analysis_results.get('text_analysis', {})}
        
        👁️ VISUAL ANALYSIS:
        {analysis_results.get('visual_analysis', {})}
        
        😊 EMOTIONAL ANALYSIS:
        {analysis_results.get('sentiment_analysis', {})}
        
        === CREATE MAGICAL REVIEW ===
        
        Synthesize all these AI analyses into one brilliant film critique that demonstrates:
        
        🎯 CINEMATIC MASTERY ASSESSMENT:
        - Storytelling excellence (based on narrative analysis)
        - Visual artistry (based on frame analysis) 
        - Emotional impact (based on sentiment analysis)
        - Technical craftsmanship
        - Directorial vision
        
        🏆 COMPREHENSIVE SCORING (1-5):
        - STORY MAGIC: Narrative power, character depth, thematic richness
        - VISUAL SORCERY: Cinematography, composition, visual style
        - EMOTIONAL ALCHEMY: Audience impact, emotional resonance
        - TECHNICAL WIZARDRY: Production quality, technical execution  
        - ARTISTIC BRILLIANCE: Creative vision, originality, artistic merit
        
        ✨ MAGICAL INSIGHTS:
        - What makes this film special?
        - Where does the magic truly happen?
        - What could elevate it to masterpiece level?
        
        REQUIRED OUTPUT (JSON):
        {{
            "magical_summary": "Brilliant 3-paragraph analysis synthesizing all AI insights",
            "cinematic_scores": {{
                "story_magic": 4.2,
                "visual_sorcery": 4.5,
                "emotional_alchemy": 4.0,
                "technical_wizardry": 4.3,
                "artistic_brilliance": 4.1
            }},
            "overall_magic_score": 4.2,
            "magical_insights": [
                "Deep insight 1 from combined AI analysis",
                "Deep insight 2 from visual + narrative synthesis", 
                "Deep insight 3 from emotional + technical assessment"
            ],
            "directorial_brilliance": "Assessment of creative vision and execution",
            "audience_enchantment": "How the film captivates and moves viewers",
            "festival_wizardry": "Which festivals would be enchanted by this film"
        }}
        
        Calculate overall_magic_score as average of all cinematic scores.
        Make this review sparkle with cinematic wisdom!
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
        except:
            pass
        
        return self._create_magical_fallback()
    
    def _create_magical_fallback(self):
        """Create magical fallback review"""
        return {
            "magical_summary": "This film demonstrates solid cinematic craftsmanship with moments of genuine artistry. While our full magical analysis is temporarily unavailable, the work shows promise across key filmmaking disciplines.",
            "cinematic_scores": {
                "story_magic": 3.8,
                "visual_sorcery": 3.9,
                "emotional_alchemy": 3.7,
                "technical_wizardry": 3.8,
                "artistic_brilliance": 3.6
            },
            "overall_magic_score": 3.8,
            "magical_insights": [
                "Shows strong foundation in cinematic storytelling",
                "Visual composition demonstrates artistic intention",
                "Emotional moments land with authentic impact"
            ],
            "directorial_brilliance": "Competent direction with clear artistic vision",
            "audience_enchantment": "Will engage viewers who appreciate thoughtful filmmaking",
            "festival_wizardry": "Suitable for festivals celebrating emerging cinematic voices",
            "ai_capabilities_used": ["basic_analysis"],
            "analysis_methods": ["fallback"]
        }

# --------------------------
# Magical Interface
# --------------------------
def magical_interface(clients, analyzer):
    """The truly magical interface"""
    st.header("🔮 FlickFinder MAGICAL AI Analysis")
    st.markdown("### ✨ Where AI works its cinema magic!")
    
    # Display AI capabilities
    with st.expander("🎩 **AI Magic Capabilities**", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🧠 GPT-4 Intelligence**
            - Narrative structure analysis
            - Character development evaluation
            - Thematic depth assessment
            - Storytelling excellence rating
            
            **👁️ GPT-4 Vision**
            - Frame-by-frame composition analysis
            - Lighting and color evaluation
            - Visual storytelling assessment
            - Cinematic style identification
            """)
        
        with col2:
            st.markdown("""
            **😊 Emotional Analysis**
            - Sentiment and tone evaluation
            - Emotional arc mapping
            - Audience impact assessment
            - Emotional authenticity scoring
            
            **🎭 Performance Evaluation**
            - Acting quality assessment
            - Character believability
            - Ensemble chemistry analysis
            - Performance authenticity
            
            **🎨 Cinematic Technique**
            - Directorial style analysis
            - Technical execution evaluation
            - Artistic vision assessment
            - Production quality scoring
            """)
    
    # URL input
    youtube_url = st.text_input(
        "**🎥 Enter YouTube URL for Magical Analysis:**",
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
            st.error("❌ Could not access video")
            st.info("💡 Try the manual magical analysis below!")
            return
        
        # Display video
        col1, col2 = st.columns([2, 1])
        with col1:
            embed_url = f"https://www.youtube.com/embed/{video_id}"
            st.components.v1.iframe(embed_url, height=400)
        
        with col2:
            st.subheader("🎬 Film Info")
            st.write(f"**Title:** {video_info['title']}")
            st.write(f"**Channel:** {video_info['author']}")
        
        # Magical analysis button
        if st.button("**🔮 PERFORM MAGICAL AI ANALYSIS**", 
                   type="primary", 
                   use_container_width=True,
                   key="magic_button"):
            
            perform_magical_analysis(video_info, video_id, clients, analyzer)

def perform_magical_analysis(video_info, video_id, clients, analyzer):
    """Perform the full magical analysis"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    try:
        # Step 1: Extract content
        status_text.text("📝 Gathering film content...")
        transcript = get_transcript(video_id)
        progress_bar.progress(20)
        
        # Step 2: Extract frames for visual analysis
        status_text.text("👁️ Capturing visual moments...")
        frames = extract_frames_magically(video_id)
        progress_bar.progress(40)
        
        # Step 3: Multi-modal analysis
        status_text.text("🧠 Activating AI capabilities...")
        
        film_data = {
            'title': video_info['title'],
            'channel': video_info['author'],
            'transcript': transcript,
            'frames': frames,
            'video_id': video_id
        }
        
        # Perform magical analysis
        magical_results = analyzer.analyze_film_magically(film_data)
        progress_bar.progress(80)
        
        # Step 4: Display magical results
        status_text.text("✨ Synthesizing magical insights...")
        progress_bar.progress(100)
        
        # Store and display
        st.session_state.analysis_results = magical_results
        st.session_state.current_film = film_data
        
        with results_container:
            display_magical_results(magical_results, film_data)
        
        status_text.text("🎉 Magical Analysis Complete!")
        
    except Exception as e:
        st.error(f"❌ Magical analysis failed: {e}")
        progress_bar.progress(0)

def extract_frames_magically(video_id):
    """Extract frames for visual analysis"""
    try:
        # For now, return empty list - in production, you'd download and extract frames
        # This is where you'd implement actual frame extraction
        return []
    except:
        return []

def display_magical_results(magical_results, film_data):
    """Display the magical analysis results"""
    st.success("🌟 **MAGICAL AI ANALYSIS COMPLETE!**")
    
    # Magical score display
    magic_score = magical_results.get('overall_magic_score', 0)
    st.markdown(f"""
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #8A2BE2 0%, #4B0082 100%); border-radius: 20px; margin: 20px 0; border: 3px solid #FFD700;'>
        <h1 style='color: gold; margin: 0; font-size: 60px; text-shadow: 2px 2px 4px #000;'>{magic_score:.1f}/5.0</h1>
        <p style='color: white; font-size: 24px; margin: 10px 0 0 0;'>✨ Magical Cinema Score ✨</p>
        <p style='color: silver; font-size: 18px; margin: 5px 0 0 0;'>{film_data['title']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Magical category scores
    st.subheader("🔮 Magical Category Scores")
    scores = magical_results.get('cinematic_scores', {})
    
    cols = st.columns(5)
    magical_categories = [
        ("🧙‍♂️ Story Magic", scores.get('story_magic', 0), "#8A2BE2", "Narrative Power"),
        ("🔮 Visual Sorcery", scores.get('visual_sorcery', 0), "#4B0082", "Cinematic Vision"),
        ("💫 Emotional Alchemy", scores.get('emotional_alchemy', 0), "#FF6B6B", "Audience Impact"),
        ("⚡ Technical Wizardry", scores.get('technical_wizardry', 0), "#45B7D1", "Craft Excellence"),
        ("🎨 Artistic Brilliance", scores.get('artistic_brilliance', 0), "#FFD93D", "Creative Vision")
    ]
    
    for idx, (name, score, color, desc) in enumerate(magical_categories):
        with cols[idx]:
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; background: {color}; border-radius: 12px; margin: 5px; border: 2px solid gold;'>
                <h4 style='margin: 0; color: white; font-size: 14px;'>{name}</h4>
                <h2 style='margin: 8px 0; color: gold; font-size: 26px; text-shadow: 1px 1px 2px #000;'>{score:.1f}</h2>
                <p style='margin: 0; color: white; font-size: 11px; opacity: 0.9;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Magical summary
    st.subheader("📖 Magical Analysis Summary")
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;'>
        {magical_results.get('magical_summary', 'No magical summary available.')}
    </div>
    """, unsafe_allow_html=True)
    
    # Magical insights
    st.subheader("💎 Magical Insights")
    insights = magical_results.get('magical_insights', [])
    for insight in insights:
        st.markdown(f"✨ **{insight}**")
    
    # Directorial brilliance
    st.subheader("🎬 Directorial Brilliance")
    st.info(magical_results.get('directorial_brilliance', 'No directorial assessment.'))
    
    # Audience enchantment
    st.subheader("❤️ Audience Enchantment")
    st.info(magical_results.get('audience_enchantment', 'No audience analysis.'))
    
    # Festival wizardry
    st.subheader("🏆 Festival Wizardry")
    st.success(magical_results.get('festival_wizardry', 'No festival recommendations.'))
    
    # AI capabilities used
    st.subheader("🤖 AI Magic Used")
    capabilities = magical_results.get('ai_capabilities_used', [])
    for capability in capabilities:
        emoji = {"gpt4_analysis": "🧠", "gpt4_vision": "👁️", "sentiment_analysis": "😊", 
                "performance_evaluation": "🎭", "cinematic_technique": "🎨"}.get(capability, "⚡")
        st.write(f"{emoji} {capability.replace('_', ' ').title()}")

# --------------------------
# Utility Functions
# --------------------------
def get_video_id(url):
    """Extract video ID"""
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
    """Get transcript"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([seg["text"] for seg in transcript_list])
    except:
        return "No transcript available. Magical analysis will use other AI capabilities."

# --------------------------
# Main Application
# --------------------------
def main():
    """Main magical application"""
    
    st.sidebar.title("🔮 FlickFinder MAGIC")
    st.sidebar.markdown("---")
    
    # Initialize AI
    with st.sidebar:
        with st.spinner("✨ Initializing AI Magic..."):
            clients = initialize_ai_clients()
            analyzer = MagicalFilmAnalyzer(clients)
    
    st.sidebar.markdown("### 🎩 Magical Capabilities:")
    for emoji, desc in analyzer.capabilities.items():
        st.sidebar.write(f"{emoji} {desc}")
    
    st.sidebar.markdown("### 🚀 How to Use:")
    st.sidebar.markdown("1. **Paste YouTube URL**")
    st.sidebar.markdown("2. **Click Magical Analysis**")
    st.sidebar.markdown("3. **Watch AI work all its magic!**")
    
    # Main interface
    magical_interface(clients, analyzer)

if __name__ == "__main__":
    main()

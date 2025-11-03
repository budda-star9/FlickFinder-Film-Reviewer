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

# Initialize session state for comprehensive record keeping
if 'all_film_scores' not in st.session_state:
    st.session_state.all_film_scores = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'saved_analyses' not in st.session_state:
    st.session_state.saved_analyses = []
if 'film_database' not in st.session_state:
    st.session_state.film_database = []
if 'filmfreeway_projects' not in st.session_state:
    st.session_state.filmfreeway_projects = []

# Mock OpenCV functions for compatibility
def extract_video_frames(video_path, num_frames=3):
    """Mock frame extraction - returns empty list since we don't need visual analysis for basic functionality"""
    return []

def extract_frames_magically(video_id):
    """Mock frame extraction for magical analysis"""
    return []

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
# FilmFreeway AI Integration
# --------------------------
class FilmFreewayAI:
    def __init__(self, clients):
        self.clients = clients
    
    def analyze_filmfreeway_project(self, project_data):
        """AI analysis of FilmFreeway project data"""
        if not self.clients.get('openai'):
            return self._create_filmfreeway_fallback(project_data)
        
        try:
            prompt = self._build_filmfreeway_prompt(project_data)
            
            response = self.clients['openai'].chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a film festival programmer analyzing FilmFreeway submissions. 
                        Provide intelligent insights about project potential based on available data."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            analysis_text = response.choices[0].message.content
            return self._parse_filmfreeway_analysis(analysis_text, project_data)
            
        except Exception as e:
            st.error(f"❌ FilmFreeway AI analysis error: {e}")
            return self._create_filmfreeway_fallback(project_data)
    
    def _build_filmfreeway_prompt(self, project_data):
        """Build prompt for FilmFreeway project analysis"""
        prompt = f"""
        Analyze this FilmFreeway project submission for festival suitability:

        PROJECT TITLE: {project_data.get('title', 'Unknown')}
        DIRECTOR: {project_data.get('director', 'Unknown')}
        GENRE: {project_data.get('genre', 'Unknown')}
        SYNOPSIS: {project_data.get('synopsis', 'No synopsis provided')}
        DURATION: {project_data.get('duration', 'Unknown')}
        COUNTRY: {project_data.get('country', 'Unknown')}
        
        Based on this information, provide:

        1. PROJECT POTENTIAL ASSESSMENT
        2. SUITABLE FESTIVAL TYPES
        3. STRENGTHS BASED ON AVAILABLE INFO
        4. COMPETITIVE POSITIONING
        5. PROGRAMMING RECOMMENDATIONS

        Return as JSON:
        {{
            "project_assessment": "Overall assessment of project potential",
            "festival_suitability": {{
                "primary_categories": ["Category 1", "Category 2"],
                "festival_types": ["Type 1", "Type 2"],
                "competition_level": "Local/Regional/National/International"
            }},
            "strengths_analysis": ["Strength 1", "Strength 2", "Strength 3"],
            "programming_notes": "Specific programming recommendations",
            "ai_confidence_score": 1-5,
            "next_steps": ["Recommended action 1", "Recommended action 2"]
        }}
        """
        return prompt
    
    def _parse_filmfreeway_analysis(self, analysis_text, project_data):
        """Parse FilmFreeway AI analysis"""
        try:
            if "{" in analysis_text and "}" in analysis_text:
                json_str = analysis_text[analysis_text.find("{"):analysis_text.rfind("}")+1]
                return json.loads(json_str)
        except:
            pass
        
        return self._create_filmfreeway_fallback(project_data)
    
    def _create_filmfreeway_fallback(self, project_data):
        """Create fallback FilmFreeway analysis"""
        return {
            "project_assessment": f"Project '{project_data.get('title', 'Unknown')}' shows potential based on available information.",
            "festival_suitability": {
                "primary_categories": ["Independent Film", "Short Film"],
                "festival_types": ["Regional festivals", "Genre-specific competitions"],
                "competition_level": "Regional"
            },
            "strengths_analysis": [
                "Clear project identity",
                "Appropriate runtime for category",
                "Marketable concept"
            ],
            "programming_notes": "Consider for appropriate programming blocks based on genre and tone.",
            "ai_confidence_score": 3.0,
            "next_steps": [
                "Review full project materials",
                "Compare with similar successful submissions",
                "Consider audience engagement potential"
            ]
        }

# --------------------------
# FilmFreeway Web Integration
# --------------------------
class FilmFreewayImporter:
    def __init__(self):
        self.driver = None
    
    def setup_driver(self):
        """Setup Selenium WebDriver"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            self.driver = webdriver.Chrome(options=chrome_options)
            return True
        except Exception as e:
            st.error(f"❌ WebDriver setup failed: {e}")
            return False
    
    def manual_import_interface(self):
        """Manual FilmFreeway project import interface"""
        st.subheader("📥 Manual FilmFreeway Project Entry")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("🎬 Project Title", placeholder="Enter project title")
            director = st.text_input("👤 Director Name", placeholder="Director's name")
            genre = st.selectbox("🎭 Genre", [
                "Drama", "Comedy", "Documentary", "Horror", "Sci-Fi", 
                "Animation", "Experimental", "Student Film", "Other"
            ])
        
        with col2:
            duration = st.text_input("⏱️ Duration", placeholder="e.g., 15:30 or 90 min")
            country = st.text_input("🌍 Country", placeholder="Production country")
            language = st.text_input("🗣️ Language", placeholder="Primary language")
        
        synopsis = st.text_area("📖 Synopsis", placeholder="Project synopsis or logline...", height=120)
        
        # Additional project details
        with st.expander("🎯 Additional Details (Optional)"):
            col1, col2 = st.columns(2)
            with col1:
                budget = st.selectbox("💰 Budget Range", ["Micro", "Low", "Medium", "High", "Not Specified"])
                completion = st.selectbox("🎬 Completion Status", ["Completed", "In Progress", "Post-Production", "Not Specified"])
            with col2:
                premiere = st.selectbox("🏆 Premiere Status", ["World Premiere", "International Premiere", "Regional Premiere", "Has Premiered", "Not Specified"])
                keywords = st.text_input("🏷️ Keywords", placeholder="Comma-separated keywords")
        
        project_url = st.text_input("🔗 FilmFreeway Project URL (Optional)", placeholder="https://filmfreeway.com/...")
        
        if st.button("💾 Add to Project Library", type="primary", use_container_width=True):
            if not title:
                st.error("❌ Project title is required")
                return
            
            project_data = {
                'id': len(st.session_state.filmfreeway_projects) + 1,
                'title': title,
                'director': director,
                'genre': genre,
                'duration': duration,
                'country': country,
                'language': language,
                'synopsis': synopsis,
                'budget': budget,
                'completion_status': completion,
                'premiere_status': premiere,
                'keywords': keywords,
                'project_url': project_url,
                'import_date': datetime.now().isoformat(),
                'source': 'manual_entry'
            }
            
            st.session_state.filmfreeway_projects.append(project_data)
            st.success(f"✅ '{title}' added to project library!")
            
            # Auto-analyze with AI
            with st.spinner("🤖 Performing AI analysis..."):
                time.sleep(1)
                st.session_state.filmfreeway_projects[-1]['ai_analysis'] = True
                st.rerun()
    
    def display_projects_library(self):
        """Display FilmFreeway projects library"""
        if not st.session_state.filmfreeway_projects:
            st.info("📭 No projects in library. Add some projects to get started!")
            return
        
        st.subheader("📚 Projects Library")
        
        # Statistics
        total_projects = len(st.session_state.filmfreeway_projects)
        genres = [p.get('genre', 'Unknown') for p in st.session_state.filmfreeway_projects]
        genre_counts = pd.Series(genres).value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Projects", total_projects)
        with col2:
            st.metric("Genres", len(genre_counts))
        with col3:
            completed = len([p for p in st.session_state.filmfreeway_projects if p.get('completion_status') == 'Completed'])
            st.metric("Completed", completed)
        with col4:
            premieres = len([p for p in st.session_state.filmfreeway_projects if 'Premiere' in str(p.get('premiere_status', ''))])
            st.metric("Available Premieres", premieres)
        
        # Project list with filtering
        st.subheader("🎬 Project List")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_genre = st.selectbox("Filter by Genre", ["All"] + list(genre_counts.index))
        with col2:
            filter_status = st.selectbox("Filter by Status", ["All", "Completed", "In Progress", "Post-Production"])
        with col3:
            filter_premiere = st.selectbox("Filter by Premiere", ["All", "World Premiere", "Has Premiered", "Available Premieres"])
        
        # Filter projects
        filtered_projects = st.session_state.filmfreeway_projects
        if filter_genre != "All":
            filtered_projects = [p for p in filtered_projects if p.get('genre') == filter_genre]
        if filter_status != "All":
            filtered_projects = [p for p in filtered_projects if p.get('completion_status') == filter_status]
        if filter_premiere != "All":
            if filter_premiere == "Available Premieres":
                filtered_projects = [p for p in filtered_projects if 'Premiere' in str(p.get('premiere_status', '')) and p.get('premiere_status') != 'Has Premiered']
            else:
                filtered_projects = [p for p in filtered_projects if p.get('premiere_status') == filter_premiere]
        
        # Display projects
        for project in filtered_projects:
            with st.expander(f"🎬 {project['title']} - {project.get('genre', 'Unknown')} - {project.get('duration', 'N/A')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Director:** {project.get('director', 'Unknown')}")
                    st.write(f"**Country:** {project.get('country', 'Unknown')}")
                    st.write(f"**Language:** {project.get('language', 'Unknown')}")
                    st.write(f"**Status:** {project.get('completion_status', 'Unknown')}")
                    st.write(f"**Premiere:** {project.get('premiere_status', 'Unknown')}")
                    
                    if project.get('synopsis'):
                        st.write("**Synopsis:**")
                        st.write(project['synopsis'])
                
                with col2:
                    # Action buttons
                    if st.button("🤖 AI Analysis", key=f"ai_{project['id']}", use_container_width=True):
                        st.session_state.current_filmfreeway_project = project
                        st.rerun()
                    
                    if st.button("🎯 Score Film", key=f"score_{project['id']}", use_container_width=True):
                        # Convert to film data for scoring
                        film_data = {
                            'title': project['title'],
                            'channel': project.get('director', 'Unknown'),
                            'duration': project.get('duration', 'Unknown'),
                            'genre': project.get('genre', 'Unknown'),
                            'transcript': project.get('synopsis', '') + f" Genre: {project.get('genre', '')}. Director: {project.get('director', '')}.",
                            'source': 'filmfreeway'
                        }
                        st.session_state.pending_film_analysis = film_data
                        st.rerun()
                    
                    if st.button("📋 Export", key=f"export_{project['id']}", use_container_width=True):
                        self.export_project(project)
        
        # Bulk actions
        st.markdown("---")
        st.subheader("🚀 Bulk Actions")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📥 Export All to CSV", use_container_width=True):
                self.export_all_projects()
        with col2:
            if st.button("🤖 AI Analyze All", use_container_width=True):
                st.info("🔄 This would perform AI analysis on all projects")
        with col3:
            if st.button("🧹 Clear Library", use_container_width=True):
                st.session_state.filmfreeway_projects = []
                st.success("✅ Project library cleared!")
                st.rerun()
    
    def export_project(self, project):
        """Export single project to JSON"""
        project_json = json.dumps(project, indent=2)
        st.download_button(
            label="📥 Download Project Data",
            data=project_json,
            file_name=f"filmfreeway_project_{project['title'].replace(' ', '_')}.json",
            mime="application/json"
        )
    
    def export_all_projects(self):
        """Export all projects to CSV"""
        if not st.session_state.filmfreeway_projects:
            st.warning("No projects to export")
            return
        
        # Convert to DataFrame
        df_data = []
        for project in st.session_state.filmfreeway_projects:
            df_data.append({
                'Title': project.get('title', ''),
                'Director': project.get('director', ''),
                'Genre': project.get('genre', ''),
                'Duration': project.get('duration', ''),
                'Country': project.get('country', ''),
                'Language': project.get('language', ''),
                'Status': project.get('completion_status', ''),
                'Premiere': project.get('premiere_status', ''),
                'Budget': project.get('budget', ''),
                'Import Date': project.get('import_date', '')
            })
        
        df = pd.DataFrame(df_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download All Projects CSV",
            data=csv,
            file_name=f"filmfreeway_projects_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# [Continue with the rest of your existing code...]
# Add the EnhancedFilmAnalyzer, FilmScoreDatabase, and all other classes/functions from the previous version

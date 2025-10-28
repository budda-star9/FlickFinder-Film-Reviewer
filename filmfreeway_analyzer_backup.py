"""
FilmFreeway Analyzer Module for FlickFinder
Handles FilmFreeway project analysis with hybrid approach (URL validation + scraping + AI)
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re

class FilmFreewayAnalyzer:
    def __init__(self, openai_client):
        self.client = openai_client
        
    def process_filmfreeway_url(self, url):
        """Validate and parse FilmFreeway URLs"""
        if not url:
            return None
            
        # Basic URL validation
        if 'filmfreeway.com' not in url:
            st.error("❌ Please enter a valid FilmFreeway URL")
            return None
            
        # Clean and normalize URL
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # Extract project identifier
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        
        project_data = {
            'url': url,
            'platform': 'FilmFreeway',
            'project_id': path_parts[-1] if path_parts else None,
            'is_valid': True
        }
        
        return project_data
    
    def extract_basic_info(self, url):
        """Extract basic information from FilmFreeway page"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 FlickFinder/1.0'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts and styles to clean up text
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try multiple selectors for title
            title = None
            title_selectors = [
                'h1',
                '.title',
                '[class*="title"]',
                '[class*="Title"]',
                'head title'
            ]
            
            for selector in title_selectors:
                element = soup.select_one(selector)
                if element and element.get_text().strip():
                    title = element.get_text().strip()
                    break
            
            # Extract meta description
            description = soup.find('meta', attrs={'name': 'description'})
            description = description['content'] if description else None
            
            # Try to find any text content
            text_elements = soup.find_all(['p', 'div', 'span'], string=True)
            content_text = ' '.join([elem.get_text().strip() for elem in text_elements[:10] if elem.get_text().strip()])
            content_text = ' '.join(content_text.split()[:200])  # Limit length
            
            return {
                'page_exists': True,
                'title': title,
                'description': description,
                'content_preview': content_text,
                'status_code': response.status_code
            }
            
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ Could not fetch page details: {str(e)}")
            return {
                'page_exists': False,
                'error': str(e)
            }
        except Exception as e:
            st.warning(f"⚠️ Error parsing page: {str(e)}")
            return {
                'page_exists': False,
                'error': str(e)
            }
    
    def enhance_with_ai(self, project_data):
        """Use OpenAI to generate insights about the project"""
        try:
            prompt = f"""
            Analyze this FilmFreeway project and provide a comprehensive analysis in the following format:
            
            PROJECT OVERVIEW:
            [Brief summary based on available information]
            
            POTENTIAL GENRE/CATEGORIES:
            [List 3-5 possible genres or film categories]
            
            KEY THEMES & ELEMENTS:
            [Identify 3-5 main themes or notable elements]
            
            SIMILAR FILMS/REFERENCES:
            [Suggest 3-5 similar films or references]
            
            PRODUCTION INSIGHTS:
            [Any observations about production scale, style, or approach]
            
            Project Information:
            - URL: {project_data['url']}
            - Title: {project_data.get('title', 'Not specified')}
            - Description: {project_data.get('description', 'Not available')}
            - Content Preview: {project_data.get('content_preview', 'Limited information available')}
            
            Provide thoughtful, film-industry relevant insights. If information is limited, make educated guesses based on typical FilmFreeway projects.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a film industry expert who analyzes FilmFreeway projects. Provide insightful, professional analysis that would help filmmakers and reviewers understand the project better."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"❌ AI analysis failed: {str(e)}")
            return f"AI analysis unavailable. Error: {str(e)}"
    
    def analyze_project(self, url):
        """Main method to analyze a FilmFreeway project"""
        # Step 1: Validate URL
        project_data = self.process_filmfreeway_url(url)
        if not project_data:
            return None
        
        # Step 2: Extract basic info
        with st.spinner("🔄 Fetching project information..."):
            basic_info = self.extract_basic_info(url)
        
        # Combine data
        combined_data = {**project_data, **basic_info}
        
        # Step 3: AI enhancement
        if combined_data.get('page_exists'):
            with st.spinner("🤔 Analyzing with AI..."):
                ai_analysis = self.enhance_with_ai(combined_data)
                combined_data['ai_analysis'] = ai_analysis
        else:
            combined_data['ai_analysis'] = "Unable to analyze: Could not access project page."
        
        return combined_data

def display_filmfreeway_results(result):
    """Display the analysis results in an organized way"""
    st.success("✅ Analysis Complete!")
    
    # Basic info card
    with st.container():
        st.subheader("📋 Project Information")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if result.get('title'):
                st.markdown(f"**Title:** {result['title']}")
            st.markdown(f"**Platform:** {result['platform']}")
            st.markdown(f"**URL:** [View on FilmFreeway]({result['url']})")
            
            if result.get('description'):
                with st.expander("Project Description"):
                    st.write(result['description'])
        
        with col2:
            status_color = "🟢" if result.get('page_exists') else "🔴"
            st.markdown(f"**Status:** {status_color} {'Accessible' if result.get('page_exists') else 'Not Accessible'}")
    
    # AI Analysis section
    if result.get('ai_analysis'):
        st.subheader("🤖 AI Analysis")
        
        # Parse the AI response into sections
        analysis_text = result['ai_analysis']
        sections = {
            'PROJECT OVERVIEW:': '',
            'POTENTIAL GENRE/CATEGORIES:': '',
            'KEY THEMES & ELEMENTS:': '',
            'SIMILAR FILMS/REFERENCES:': '',
            'PRODUCTION INSIGHTS:': ''
        }
        
        current_section = None
        for line in analysis_text.split('\n'):
            line = line.strip()
            if any(section in line for section in sections.keys()):
                current_section = line
                sections[current_section] = ''
            elif current_section and line:
                sections[current_section] += line + '\n'
        
        # Display sections in a nice format
        for section_title, section_content in sections.items():
            if section_content.strip():
                with st.expander(f"📖 {section_title.replace(':', '')}"):
                    st.write(section_content.strip())
    
    # Action buttons
    st.markdown("---")
    st.subheader("💾 Save to FlickFinder")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Save Project", use_container_width=True, key="save_ff_project"):
            save_to_database(result)
    
    with col2:
        if st.button("📊 Compare Similar", use_container_width=True, key="compare_ff"):
            st.info("Feature coming soon: Compare with similar projects")
    
    with col3:
        if st.button("🔄 Analyze Another", use_container_width=True, key="analyze_another_ff"):
            st.rerun()

def save_to_database(project_data):
    """Save project data to FlickFinder database"""
    try:
        # Store in session state for now - integrate with your actual database later
        if 'filmfreeway_projects' not in st.session_state:
            st.session_state.filmfreeway_projects = []
        
        st.session_state.filmfreeway_projects.append(project_data)
        
        # Display what would be saved
        st.success(f"✅ Project '{project_data.get('title', 'Unknown')}' saved successfully!")
        
        # Show saved data preview
        with st.expander("📊 View Saved Data"):
            st.json({
                'title': project_data.get('title'),
                'platform': project_data.get('platform'),
                'url': project_data.get('url'),
                'ai_analysis_preview': project_data.get('ai_analysis', '')[:200] + '...' if project_data.get('ai_analysis') else 'No analysis'
            })
            
    except Exception as e:
        st.error(f"❌ Error saving project: {str(e)}")

def filmfreeway_interface(openai_client):
    """Streamlit interface for FilmFreeway analysis"""
    st.header("🎬 FilmFreeway Project Analyzer")
    st.markdown("Analyze FilmFreeway projects and get AI-powered insights for your FlickFinder database.")
    
    # URL input
    url = st.text_input(
        "Enter FilmFreeway Project URL:",
        placeholder="https://filmfreeway.com/YourProjectName",
        key="filmfreeway_url"
    )
    
    # Example URLs for testing
    with st.expander("💡 Example URLs (for testing)"):
        st.code("""
https://filmfreeway.com/MyShortFilm
https://filmfreeway.com/MyDocumentary  
https://filmfreeway.com/MyFeatureFilm
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        analyze_btn = st.button("🔍 Analyze Project", use_container_width=True, key="analyze_ff_main")
    
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True, key="clear_ff")
    
    if clear_btn:
        st.rerun()
    
    if analyze_btn and url:
        analyzer = FilmFreewayAnalyzer(openai_client)
        result = analyzer.analyze_project(url)
        
        if result:
            display_filmfreeway_results(result)
    elif analyze_btn and not url:
        st.warning("⚠️ Please enter a FilmFreeway URL first.")

def get_saved_projects():
    """Retrieve saved FilmFreeway projects from session state"""
    return st.session_state.get('filmfreeway_projects', [])

def display_saved_projects():
    """Display previously analyzed projects"""
    saved_projects = get_saved_projects()
    
    if saved_projects:
        st.subheader("📚 Saved FilmFreeway Projects")
        
        for i, project in enumerate(saved_projects):
            with st.expander(f"🎬 {project.get('title', f'Project {i+1}')}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**URL:** {project.get('url')}")
                    if project.get('description'):
                        st.write(f"**Description:** {project.get('description')}")
                
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_ff_{i}"):
                        st.session_state.filmfreeway_projects.pop(i)
                        st.rerun()
    else:
        st.info("No FilmFreeway projects saved yet. Analyze some projects first!")

import streamlit as st
from filmfreeway_analyzer import filmfreeway_interface, display_saved_projects
from scoring_system import ScoringSystem
from export_system import export_interface
from openai import OpenAI

def main():
    st.set_page_config(page_title="FlickFinder", page_icon="🎬", layout="wide")
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.error(f"OpenAI client initialization failed: {e}")
        client = None
    
    # Initialize systems
    if 'scoring_system' not in st.session_state:
        st.session_state.scoring_system = ScoringSystem()
    if 'all_scores' not in st.session_state:
        st.session_state.all_scores = []
    
    # Navigation
    with st.sidebar:
        st.header("🎬 FlickFinder")
        st.markdown("---")
        
        page_option = st.radio(
            "Navigate to:",
            ["🏠 Home", "🔗 FilmFreeway", "🎯 Score Films", "📊 Export", "📚 Saved Projects"]
        )
    
    # Page routing
    if page_option == "🏠 Home":
        home_interface()
    elif page_option == "🔗 FilmFreeway":
        if client:
            filmfreeway_interface(client)
        else:
            st.error("OpenAI client not initialized. Check your API key.")
    elif page_option == "🎯 Score Films":
        scoring_interface()
    elif page_option == "📊 Export":
        export_interface()
    elif page_option == "📚 Saved Projects":
        display_saved_projects()

def home_interface():
    """Home page interface"""
    st.title("Welcome to FlickFinder 🎬")
    st.markdown("### Professional Film Evaluation Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔗 FilmFreeway Integration**")
        st.markdown("Import and analyze projects directly from FilmFreeway")
    
    with col2:
        st.markdown("**🎯 Smart Scoring**")
        st.markdown("Weighted scoring with bias checks and qualitative feedback")
    
    with col3:
        st.markdown("**📊 Export Tools**")
        st.markdown("Generate PDF reports and CSV exports for festival management")
    
    st.markdown("---")
    st.markdown("Get started by importing films from FilmFreeway or scoring existing projects.")

def scoring_interface():
    """Scoring interface for films"""
    st.header("🎯 Film Scoring")
    
    # Get films to score (from saved projects or manual entry)
    films_to_score = st.session_state.get('filmfreeway_projects', [])
    
    if not films_to_score:
        st.info("📥 No films available for scoring. Import some films from the FilmFreeway section first.")
        
        # Allow manual film entry for testing
        st.markdown("---")
        st.subheader("Or add a film manually for testing:")
        manual_film = st.text_input("Film title for manual scoring:")
        if manual_film and st.button("Add for Scoring"):
            if 'filmfreeway_projects' not in st.session_state:
                st.session_state.filmfreeway_projects = []
            
            st.session_state.filmfreeway_projects.append({
                'title': manual_film,
                'platform': 'Manual Entry',
                'url': 'N/A'
            })
            st.rerun()
        return
    
    film_titles = [project.get('title', f'Project {i+1}') for i, project in enumerate(films_to_score)]
    
    selected_film = st.selectbox("Select film to score:", film_titles)
    
    if selected_film:
        score_result = st.session_state.scoring_system.get_scorecard_interface(selected_film)
        
        if score_result:
            # Calculate weighted score
            score_result['weighted_score'] = st.session_state.scoring_system.calculate_weighted_score(
                score_result['scores']
            )
            
            # Store score
            st.session_state.all_scores.append(score_result)
            st.success(f"✅ Score saved! Weighted score: {score_result['weighted_score']}/5")
            
            # Show score summary
            with st.expander("📊 View Score Summary"):
                col1, col2, col3, col4, col5 = st.columns(5)
                
                scores = score_result['scores']
                with col1:
                    st.metric("Storytelling", f"{scores['storytelling']}/5")
                with col2:
                    st.metric("Technical", f"{scores['technical_directing']}/5")
                with col3:
                    st.metric("Artistic", f"{scores['artistic_vision']}/5")
                with col4:
                    st.metric("Cultural", f"{scores['cultural_fidelity']}/5")
                with col5:
                    st.metric("Final Score", f"{score_result['weighted_score']}/5")

if __name__ == "__main__":
    main()

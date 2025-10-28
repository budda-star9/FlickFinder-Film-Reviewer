"""
Export system for PDF and CSV outputs
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime

class ExportSystem:
    def generate_score_pdf(self, film_data, scores_data):
        """Generate PDF score sheet (simplified for Streamlit Cloud)"""
        # For now, create a downloadable text report
        # In production, use ReportLab or WeasyPrint for actual PDF
        
        report = f"""
FLICKFINDER SCORE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

FILM: {film_data.get('title', 'Unknown')}
PLATFORM: {film_data.get('platform', 'Unknown')}
URL: {film_data.get('url', 'N/A')}

SCORES:
--------
Storytelling: {scores_data['scores']['storytelling']}/5 (35%)
Technical/Directing: {scores_data['scores']['technical_directing']}/5 (25%)
Artistic Vision: {scores_data['scores']['artistic_vision']}/5 (15%)
Cultural Fidelity: {scores_data['scores']['cultural_fidelity']}/5 (15%)
Social Impact: {scores_data['scores']['social_impact']}/5 (10%)

WEIGHTED SCORE: {scores_data.get('weighted_score', 'N/A')}/5

QUALITATIVE NOTES:
{scores_data['notes']}

BIAS REFLECTION:
{scores_data.get('bias_reflection', 'Not provided')}

HERO'S JOURNEY FRAMEWORK: {'Applied' if scores_data.get('heros_journey') else 'Not Applied'}
CONFLICT OF INTEREST: {'Yes' if scores_data.get('conflict_of_interest') else 'No'}
"""
        
        return report
    
    def export_to_csv(self, all_scores):
        """Export all scores to CSV format"""
        if not all_scores:
            st.warning("No scores to export")
            return None
        
        # Create DataFrame
        rows = []
        for score_data in all_scores:
            row = {
                'film_title': score_data.get('film_title'),
                'timestamp': datetime.now().isoformat(),
                'storytelling_score': score_data['scores']['storytelling'],
                'technical_score': score_data['scores']['technical_directing'],
                'artistic_score': score_data['scores']['artistic_vision'],
                'cultural_score': score_data['scores']['cultural_fidelity'],
                'impact_score': score_data['scores']['social_impact'],
                'weighted_score': score_data.get('weighted_score'),
                'notes_preview': score_data['notes'][:100] + '...' if len(score_data['notes']) > 100 else score_data['notes'],
                'heros_journey': score_data.get('heros_journey', False),
                'conflict_interest': score_data.get('conflict_of_interest', False)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df

def export_interface():
    """Streamlit interface for exports"""
    st.header("📊 Export Results")
    
    if 'all_scores' not in st.session_state or not st.session_state.all_scores:
        st.warning("No scores available for export. Please score some films first.")
        return
    
    exporter = ExportSystem()
    scores_data = st.session_state.all_scores
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Individual PDF Reports")
        selected_film = st.selectbox("Select film for PDF report:", 
                                   options=[s['film_title'] for s in scores_data])
        
        if selected_film:
            film_scores = next((s for s in scores_data if s['film_title'] == selected_film), None)
            if film_scores:
                pdf_content = exporter.generate_score_pdf(
                    {'title': selected_film}, 
                    film_scores
                )
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_content,
                    file_name=f"flickfinder_score_{selected_film.replace(' ', '_')}.txt",
                    mime="text/plain"
                )
    
    with col2:
        st.subheader("📊 Bulk CSV Export")
        csv_df = exporter.export_to_csv(scores_data)
        
        if csv_df is not None:
            csv_data = csv_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV (All Scores)",
                data=csv_data,
                file_name=f"flickfinder_scores_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Show preview
            with st.expander("📋 CSV Preview"):
                st.dataframe(csv_df)

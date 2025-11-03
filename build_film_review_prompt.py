# build_film_review_prompt.py
def build_film_review_prompt(film_metadata: str, transcript_text: str, audience_reception: str = "", visual_context: str = "") -> str:
    """
    Build a comprehensive AI prompt for film review and scoring.
    
    Args:
        film_metadata (str): Film info including title, channel, published date, length, tags.
        transcript_text (str): Transcript excerpt or fallback description.
        audience_reception (str): Public reception metrics (views, likes, comments).
        visual_context (str): Optional visual analysis cues from sampled frames.
    
    Returns:
        str: Fully formed AI prompt for scoring.
    """
    
    prompt = f"""
You are a professional film festival juror and critic. 
Analyze and score this short film using all available information. 
Your review must be as realistic and detailed as possible, like a seasoned jury member.

Film Metadata:
{film_metadata}

Public Reception:
{audience_reception}

Transcript Excerpt (or description for context):
{transcript_text[:2500]}

{f'Visual Analysis Notes:\n{visual_context}' if visual_context else ''}

Weighted Scoring Criteria (1–5 scale):
• Storytelling (35%) — narrative structure, character development, emotional impact.
• Technical/Directing (25%) — cinematography, editing, pacing, sound design.
• Artistic Vision (15%) — originality, aesthetic coherence, creative risk.
• Cultural Fidelity (15%) — authenticity, context, and representation.
• Social Impact (10%) — relevance, influence, message.

Please provide the following:
1️⃣ A concise synopsis of the film (2–3 sentences).
2️⃣ Strengths and weaknesses, referencing specific moments or timestamps if possible.
3️⃣ Numeric scores (1–5) for each weighted category.
4️⃣ Weighted final score (out of 5.00) calculated based on the percentages above.
5️⃣ Jury Notes: 2–3 sentences highlighting memorable aspects, potential improvements, or notable cinematic choices.
6️⃣ Optional: Commentary on visual style, framing, lighting, and color if Visual Analysis is enabled.

Make your evaluation professional, realistic, and constructive as if for an actual festival submission.
"""
    return prompt

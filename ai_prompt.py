# ai_prompt.py
# ---------------------------------
# 🎬 Centralized AI Review Prompt Template
# ---------------------------------

def build_film_review_prompt(film_metadata, transcript_text, audience_reception="N/A"):
    """
    Constructs a structured AI prompt for scoring films using metadata + transcripts.
    """

    return f"""
You are a professional film festival juror. Review and score the film below using objective and metadata-informed criteria.

Film Metadata:
{film_metadata}

Public Reception:
{audience_reception}

Transcript Excerpt:
{transcript_text[:2500]}

Scoring Criteria (1–5 scale):
• Storytelling (35%) — narrative structure, character depth, emotional arc.
• Technical/Directing (25%) — cinematography, editing, pacing, sound.
• Artistic Vision (15%) — originality, aesthetic coherence, creative risk.
• Cultural Fidelity (15%) — authenticity, representation, context.
• Social Impact (10%) — message, relevance, influence.

Please output:
1️⃣ A concise synopsis.
2️⃣ Strengths and weaknesses.
3️⃣ Numeric scores per category.
4️⃣ Weighted final score (out of 5.00).
5️⃣ 2–3 Jury Notes referencing scenes or timestamps.
"""

from __future__ import annotations

import json

import streamlit as st
from google import genai

from hbd_ideation_coach import parse_probs_from_csv, HBDProbabilities



def call_llm_for_hbd_probs(prompt: str) -> HBDProbabilities:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-flash"

    def run_once(p: str) -> HBDProbabilities:
        response = client.models.generate_content(
            model=model_name,
            contents=p,
            config={
                "temperature": 0.2,
                "max_output_tokens": 600,
            },
        )

        text = getattr(response, "text", None)
        if not text:
            text = str(response)

        st.write("RAW MODEL OUTPUT:", text)

        return parse_probs_from_csv(text)


    try:
        return run_once(prompt)
    except Exception as e:
        st.write("Retrying with JSON-fix instruction. Error:", str(e))

        fix_prompt = (
            prompt
            + "\n\nIMPORTANT: Your previous output was not valid JSON matching the schema. "
              "Output ONLY a single valid JSON object with ALL 5 keys. No extra text."
        )
        return run_once(fix_prompt)
def call_llm_for_coaching(
    user_question_theme: str,
    user_text: str,
    user_action: str,
    probs: "HBDProbabilities",
    nudge_dimension: str,
) -> dict:
    """
    Returns a dict with keys:
      - mirror: str
      - directions: List[str] (3 items)
      - question: str
    No JSON parsing needed; we ask for a simple JSON-like text format.
    """

    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-flash"

    # Convert probs to a simple text summary for Gemini
    prob_lines = "\n".join([f"- {k}: {v:.2f}" for k, v in probs.probs.items()])

    coaching_prompt = f"""
You are a warm mentor and playful explorer curiosity coach.

Goal:
Given the theme and the user's current idea direction, help them develop better ideation through curiosity and creativity.

The coaching must be tailored using the user's thinking profile:
HBDi probabilities (normalized):
{prob_lines}

Nudge target dimension (the one to strengthen next):
{nudge_dimension}

User theme/question:
{user_question_theme}

User idea text:
{user_text}

User action (how they engaged):
{user_action}

Output format (plain text, no markdown, no code fences):
MIRROR: <1 short paragraph describing what they did well, strength-based, no deficits>
DIRECTIONS:
1) <Try exploring... idea direction #1>
2) <Try exploring... idea direction #2>
3) <Try exploring... idea direction #3>
QUESTION: <1 curiosity question that invites the next turn>

Important:
- Do NOT mention JSON.
- Do NOT include any other sections besides MIRROR, DIRECTIONS (with 1-3), and QUESTION.
""".strip()

    response = client.models.generate_content(
        model=model_name,
        contents=coaching_prompt,
        config={
            "temperature": 0.7,
            "max_output_tokens": 450,
        },
    )

    text = getattr(response, "text", None)
    if not text:
        text = str(response)

    # Lightweight parsing of the plain-text format into a dict
    mirror = ""
    directions: list[str] = []
    question = ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    mode = None
    for ln in lines:
        if ln.startswith("MIRROR:"):
            mode = "mirror"
            mirror = ln.replace("MIRROR:", "", 1).strip()
        elif ln.startswith("DIRECTIONS:"):
            mode = "dirs"
        elif ln.startswith("QUESTION:"):
            mode = "question"
            question = ln.replace("QUESTION:", "", 1).strip()
        elif mode == "dirs":
            if ln[0].isdigit() and ")" in ln:
                directions.append(ln.split(")", 1)[1].strip())

    # Fallback if formatting is slightly different
    if not directions and "1)" in text and "2)" in text and "3)" in text:
        # crude extraction
        parts = text.split("DIRECTIONS:", 1)[-1]
        for i in ["1)", "2)", "3)"]:
            # not perfect, but enough for a first working prototype
            pass

    return {
        "mirror": mirror or text,
        "directions": directions[:3],
        "question": question or "",
    }

from __future__ import annotations

import json
import time
import re

import streamlit as st
from google import genai

from hbd_ideation_coach import (
    HBDProbabilities,
    parse_probs_from_csv,
    extract_json_object,
)


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
        st.write("Retrying with fallback. Error:", str(e))
        fix_prompt = (
            prompt
            + "\n\nIMPORTANT: Output ONLY the 5 comma-separated floats (no other text)."
        )
        return run_once(fix_prompt)


def call_llm_for_coaching(
    user_question_theme: str,
    user_text: str,
    user_action: str,
    probs: HBDProbabilities,
    nudge_dimension: str,
) -> dict:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-flash"

    prob_lines = "\n".join([f"- {k}: {v:.2f}" for k, v in probs.probs.items()])

    coaching_prompt = f"""
You are a warm mentor and playful explorer curiosity coach.

Theme/question:
{user_question_theme}

User idea text:
{user_text}

User action:
{user_action}

Nudge target dimension:
{nudge_dimension}

HBDi probabilities:
{prob_lines}

Return ONLY valid JSON with EXACTLY these keys:
- "mirror" (string)
- "directions" (array of exactly 3 strings)
- "question" (string)

Rules:
- No extra keys
- No markdown
- No commentary
- Output must be directly parseable by json.loads()
""".strip()

    attempts = 3
    last_err = None

    for i in range(attempts):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=coaching_prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 500,
                },
            )

            text = getattr(response, "text", None)
            if not text:

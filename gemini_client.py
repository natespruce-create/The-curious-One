from __future__ import annotations

import json
import re
import time

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

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"temperature": 0.2, "max_output_tokens": 600},
    )

    text = getattr(response, "text", None) or str(response)
    st.write("RAW MODEL OUTPUT:", text)
    return parse_probs_from_csv(text)


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
Return ONLY valid JSON (no markdown). Exactly these keys:
mirror, directions, question

mirror: string (<= 20 words)
directions: array of exactly 3 strings (<= 12 words each)
question: string (<= 15 words)

Theme/question: {user_question_theme}
User idea: {user_text}
User action: {user_action}
Nudge dimension: {nudge_dimension}
HBDi probabilities: {prob_lines}

Output must be parseable by json.loads().
""".strip()

    response = client.models.generate_content(
        model=model_name,
        contents=coaching_prompt,
        config={"temperature": 0.5, "max_output_tokens": 220},
    )

    text = getattr(response, "text", None) or str(response)
    st.write("COACH RAW MODEL OUTPUT:", text)

    data = json.loads(extract_json_object(text))
    return {
        "mirror": data.get("mirror", ""),
        "directions": data.get("directions", [])[:3],
        "question": data.get("question", ""),
    }

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

    def build_prompt() -> str:
        return f"""
Return ONLY valid JSON (no markdown). Exactly these keys:
mirror, directions, question

mirror: string (<= 12 words)
directions: array of exactly 3 strings (<= 8 words each)
question: string (<= 10 words)

Theme/question: {user_question_theme}
User idea: {user_text}
User action: {user_action}
Nudge dimension: {nudge_dimension}
HBDi probabilities:
{prob_lines}

Output must be directly parseable by json.loads().
""".strip()

    def validate(data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("Coach JSON is not an object.")
        if "mirror" not in data or "directions" not in data or "question" not in data:
            raise ValueError("Coach JSON missing required keys.")
        if not isinstance(data["directions"], list) or len(data["directions"]) != 3:
            raise ValueError("Coach JSON directions must be a list of exactly 3 items.")
        return data

    attempts = 3
    last_err = None
    prompt = build_prompt()

    for i in range(attempts):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"temperature": 0.3, "max_output_tokens": 180},
            )

            text = getattr(response, "text", None) or str(response)
            st.write("COACH RAW MODEL OUTPUT:", text)

            data = json.loads(extract_json_object(text))
            data = validate(data)

            return {
                "mirror": data["mirror"],
                "directions": data["directions"][:3],
                "question": data["question"],
            }

        except Exception as e:
            last_err = e
            st.write(f"Coach parse/retry {i+1}/{attempts}. Error:", str(e))

            # Force a re-output of correct JSON only (no truncation-friendly reformat)
            prompt = prompt + "\n\nREPLY AGAIN. ONLY the JSON object. No truncation. No extra text."

    raise last_err

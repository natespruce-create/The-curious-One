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

    def run_once(p: str) -> HBDProbabilities:
        response = client.models.generate_content(
            model=model_name,
            contents=p,
            config={"temperature": 0.2, "max_output_tokens": 200},
        )

        text = getattr(response, "text", None) or str(response)
        st.write("RAW MODEL OUTPUT:", text)
        return parse_probs_from_csv(text)

    try:
        return run_once(prompt)
    except Exception as e:
        st.write("Retrying HBDi probs. Error:", str(e))
        fix_prompt = prompt + "\n\nIMPORTANT: Output ONLY the 5 comma-separated floats (no other text)."
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
You are a warm curiosity coach.

Return ONLY valid JSON (no markdown) with EXACTLY these keys:
- "mirror": string (max 20 words)
- "directions": array of exactly 3 strings (each max 12 words)
- "question": string (max 15 words)

Theme/question: {user_question_theme}
User idea: {user_text}
User action: {user_action}
Nudge dimension: {nudge_dimension}
HBDi probabilities: {prob_lines}

Output must be directly parseable by json.loads().
""".strip()


    attempts = 3
    last_err = None

    for i in range(attempts):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=coaching_prompt,
                config={"temperature": 0.7, "max_output_tokens": 500},
            )

            text = getattr(response, "text", None) or str(response)
            st.write("COACH RAW MODEL OUTPUT:", text)

            data = json.loads(extract_json_object(text))
            return {
                "mirror": data.get("mirror", ""),
                "directions": data.get("directions", [])[:3],
                "question": data.get("question", ""),
            }

        except Exception as e:
            last_err = e
            st.write(f"Coach retry {i+1}/{attempts}. Error:", str(e))
            time.sleep(1.5 * (i + 1))

    raise last_err

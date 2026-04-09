from __future__ import annotations
from hbd_ideation_coach import extract_json_object

import json
import re
import time

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

       data = json.loads(extract_json_object(text))  # reuse your existing extractor
return {
    "mirror": data.get("mirror", ""),
    "directions": data.get("directions", [])[:3],
    "question": data.get("question", ""),
}


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
                    "max_output_tokens": 450,
                },
            )
            text = getattr(response, "text", None)
            if not text:
                text = str(response)
            st.write("COACH RAW MODEL OUTPUT:", text)

            # Parse expected format
            mirror = ""
            directions: list[str] = []
            question = ""

            m = re.search(r"MIRROR:\s*(.*?)(?:\nDIRECTIONS:|\nQUESTION:|$)", text, re.DOTALL)
            if m:
                mirror = m.group(1).strip()

            for idx in [1, 2, 3]:
                dm = re.search(rf"{idx}\)\s*(.*?)(?=\n\d\)|\nQUESTION:|$)", text, re.DOTALL)
                if dm:
                    directions.append(dm.group(1).strip())

            qm = re.search(r"QUESTION:\s*(.*)$", text, re.DOTALL)
            if qm:
                question = qm.group(1).strip()

            return {
                "mirror": mirror or text,
                "directions": directions[:3],
                "question": question,
            }

        except Exception as e:
            last_err = e
            st.write(f"Gemini server error (attempt {i+1}/{attempts}). Retrying…")
            time.sleep(1.5 * (i + 1))

    raise last_err

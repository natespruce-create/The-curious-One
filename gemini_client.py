from __future__ import annotations

import json

import streamlit as st
from google import genai

from hbd_ideation_coach import parse_probs_from_csv


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

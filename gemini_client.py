from __future__ import annotations

import json

import streamlit as st
from google import genai

from hbd_ideation_coach import HBDProbabilities, extract_json_object


def call_llm_for_hbd_probs(prompt: str) -> HBDProbabilities:
    api_key = st.secrets["GEMINI_API_KEY"]

    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-flash"

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "temperature": 0.2,
            "max_output_tokens": 300,
        },
    )

    text = getattr(response, "text", None)
    if not text:
        text = str(response)

    json_str = extract_json_object(text)
    data = json.loads(json_str)

    return HBDProbabilities.from_json_dict(data)

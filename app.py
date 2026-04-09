import streamlit as st
import plotly.graph_objects as go

from hbd_ideation_coach import CoachState, build_prompt_payload_for_llm, radar_chart
from gemini_client import call_llm_for_coaching

with st.spinner("Coaching your next curiosity moves…"):
    coaching_text = call_llm_for_coaching(
        user_question_theme=user_question_theme,
        user_text=user_text,
        user_action=user_action,
        probs=probs,
        nudge_dimension=rotation_target,
    )


        st.subheader("Coach mirror (strength-based)")
        st.write(coaching_text["mirror"])

        st.subheader("Try exploring (3 directions)")
        for d in coaching_text["directions"]:
            st.write(f"• {d}")

        st.subheader("Curiosity question")
        st.write(coaching_text["question"])


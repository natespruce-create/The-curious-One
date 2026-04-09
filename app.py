import streamlit as st
import plotly.graph_objects as go

from hbd_ideation_coach import CoachState, build_prompt_payload_for_llm, radar_chart
from gemini_client import call_llm_for_hbd_probs

st.set_page_config(page_title="Curiosity Coach", layout="wide")
st.title("Curiosity Coach (HBDi Spider Web)")

if "state" not in st.session_state:
    st.session_state.state = CoachState()

st.subheader("1) Theme / question")
user_question_theme = st.text_area(
    "Enter the theme/question to explore (e.g., 'localised unemployment')",
    height=80,
)

st.subheader("2) Your idea text (what you're exploring)")
user_text = st.text_area("Enter your idea / description", height=140)

st.subheader("3) What you just did")
user_action = st.selectbox(
    "Select your action",
    [
        "generated_idea_direction",
        "edited_idea_direction",
        "selected_idea_direction",
        "regenerated_after_rejecting",
    ],
)

if st.button("Generate + show spider web"):
    if not user_question_theme.strip() or not user_text.strip():
        st.warning("Please fill theme/question and your idea text.")
    else:
        prompt = build_prompt_payload_for_llm(
            user_question_theme=user_question_theme,
            user_text=user_text,
            user_action=user_action,
        )

        with st.spinner("Calling Gemini (flash) and analyzing thinking..."):
            probs = call_llm_for_hbd_probs(prompt)

        st.write("Dominant thinking dimension:", probs.dominant())

        # rotation target (least-used) can be displayed too
        rotation_target = st.session_state.state.least_used_with_history_tiebreak(probs)
        st.session_state.state.least_used_history.append(rotation_target)

        st.write("Coach rotation target (least used):", rotation_target)

        fig = radar_chart(probs, title="HBDi Spider Web (after your idea edit)")
        st.plotly_chart(fig, use_container_width=True)

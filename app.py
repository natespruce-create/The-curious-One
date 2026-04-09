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
user_text = st.text_area(
    "Enter your idea / description",
    height=140,
)

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

if st.button("Generate + Coach (curiosity)"):
    if not user_question_theme.strip() or not user_text.strip():
        st.warning("Please fill theme/question and your idea text.")
    else:
        # 1) Build prompt for HBDi probs (CSV/whatever you’re using now)
        probs_prompt = build_prompt_payload_for_llm(
            user_question_theme=user_question_theme,
            user_text=user_text,
            user_action=user_action,
        )

        with st.spinner("Reading your thinking style (HBDi)…"):
            probs = call_llm_for_hbd_probs(probs_prompt)

        # 2) Compute which dimension to nudge next (least-used, history-based)
        rotation_target = st.session_state.state.least_used_with_history_tiebreak(probs)
        st.session_state.state.least_used_history.append(rotation_target)

        st.write("Coach will nudge:", rotation_target)

        # 3) Now generate the curiosity coaching (no spider web yet)
        from gemini_client import call_llm_for_coaching  # we’ll add this next

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


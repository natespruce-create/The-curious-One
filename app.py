import streamlit as st

from hbd_ideation_coach import CoachState, build_prompt_payload_for_llm
from hbd_ideation_coach import radar_chart  # not used now, but harmless
from gemini_client import call_llm_for_hbd_probs, call_llm_for_coaching

st.set_page_config(page_title="Curiosity Coach", layout="wide")
st.title("Curiosity Coach (HBDi)")

if "state" not in st.session_state:
    st.session_state.state = CoachState()

st.subheader("1) Theme / question")
user_question_theme = st.text_area(
    "Enter the theme/question to explore (e.g., 'localised unemployment')",
    height=80,
)

st.subheader("2) Your idea text")
user_text = st.text_area(
    "Enter your current idea / what you’re exploring",
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

if st.button("Generate + Coach"):
    if not user_question_theme.strip() or not user_text.strip():
        st.warning("Please fill theme/question and your idea text.")
        st.stop()

    # 1) Get HBDi probabilities (used only to tailor coaching)
    probs_prompt = build_prompt_payload_for_llm(
        user_question_theme=user_question_theme,
        user_text=user_text,
        user_action=user_action,
    )

    with st.spinner("Reading your thinking style (HBDi)…"):
        probs = call_llm_for_hbd_probs(probs_prompt)

    rotation_target = st.session_state.state.least_used_with_history_tiebreak(probs)
    st.session_state.state.least_used_history.append(rotation_target)

    # 2) Generate curiosity coaching
    with st.spinner("Coaching your next curiosity moves…"):
        coaching_text = call_llm_for_coaching(
            user_question_theme=user_question_theme,
            user_text=user_text,
            user_action=user_action,
            probs=probs,
            nudge_dimension=rotation_target,
        )

    st.subheader("Coach mirror (strength-based)")
    st.write(coaching_text.get("mirror", ""))

    st.subheader("Try exploring (3 directions)")
    for d in coaching_text.get("directions", []):
        st.write(f"• {d}")

    st.subheader("Curiosity question")
    st.write(coaching_text.get("question", ""))

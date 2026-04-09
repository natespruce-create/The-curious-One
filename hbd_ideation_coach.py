from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import plotly.graph_objects as go

DIMENSIONS = [
    "idea_generation",
    "improved_curiosity",
    "making_connections",
    "seeing_new_things",
    "realizing_ability",
]


@dataclass
class HBDProbabilities:
    # normalized distribution over 5 dimensions, sum ~ 1
    probs: Dict[str, float]

    @staticmethod
    def from_json_dict(d: Dict) -> "HBDProbabilities":
        missing = [k for k in DIMENSIONS if k not in d]
        if missing:
            raise ValueError(f"Missing dimensions in LLM output: {missing}")

        probs: Dict[str, float] = {}
        for k in DIMENSIONS:
            v = d[k]
            if not isinstance(v, (int, float)):
                raise ValueError(f"Dimension {k} must be numeric, got {type(v)}")
            v = float(v)
            if v < 0:
                raise ValueError(f"Dimension {k} must be non-negative, got {v}")
            probs[k] = v

        s = sum(probs.values())
        if s <= 0:
            raise ValueError("Sum of probabilities must be > 0")

        # Normalize to sum exactly 1 (tolerates float drift)
        probs = {k: v / s for k, v in probs.items()}
        return HBDProbabilities(probs=probs)

    def dominant(self) -> str:
        return max(self.probs.items(), key=lambda kv: kv[1])[0]

    def least_used(self) -> str:
        return min(self.probs.items(), key=lambda kv: kv[1])[0]


@dataclass
class CoachState:
    # Used for history tie-break: choose least-used earlier in session
    least_used_history: List[str] = field(default_factory=list)

    def least_used_with_history_tiebreak(self, current_probs: HBDProbabilities) -> str:
        items = list(current_probs.probs.items())
        min_val = min(v for _, v in items)
        tol = 1e-9
        candidates = [k for k, v in items if abs(v - min_val) <= tol]

        if len(candidates) == 1:
            return candidates[0]

        # Tie-break by least used earlier:
        counts = {c: self.least_used_history.count(c) for c in candidates}
        min_count = min(counts.values())
        tied = [c for c in candidates if counts[c] == min_count]

        return sorted(tied)[0]


def build_prompt_payload_for_llm(user_question_theme: str, user_text: str, user_action: str) -> str:
    return f"""
You are classifying how a user is thinking across five HBDi training dimensions.

Theme/question being explored:
{user_question_theme}

User message (or idea edit) text:
{user_text}

User action (how they engaged):
{user_action}

Return a single valid JSON object with EXACTLY these keys (NO omissions, NO extra keys):
- idea_generation
- improved_curiosity
- making_connections
- seeing_new_things
- realizing_ability

Constraints:
- All 5 values must be non-negative floats.
- The 5 values must be normalized to sum to exactly 1.0 (within 0.01).
- Output must be valid JSON that Python can parse with json.loads().

Output rules (must follow):
- JSON ONLY
- No markdown fences
- No trailing commas
- No explanation text


""".strip()


def radar_chart(probs: HBDProbabilities, title: str = "HBDi Spider Web (5-dim distribution)") -> go.Figure:
    labels = DIMENSIONS
    values = [probs.probs[d] for d in DIMENSIONS]

    # close loop
    r = values + [values[0]]
    theta = labels + [labels[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=r,
                theta=theta,
                fill="toself",
                mode="lines+markers",
                name="probability",
            )
        ]
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def extract_json_object(text: str) -> str:
    """
    Robustly extract a JSON object from model output.
    Handles code fences and extra text around JSON.
    """
    # 1) If code-fenced JSON exists, extract inside the fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # 2) Otherwise, grab the first {...} block (greedy-ish but robust enough)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError("No JSON object found in model output.")


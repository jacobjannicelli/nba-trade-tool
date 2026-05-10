import os
import json

import anthropic
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import helpers

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Trade Impact Tool",
    page_icon="🏀",
    layout="wide",
)

# ── API key (Streamlit Cloud secrets first, then env var) ─────────────────────
api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))

# ── Load models & data once at startup ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading models and data…")
def load_everything():
    reg_pipe = joblib.load("final_trade_impact_regressor.pkl")
    clf_pipe = joblib.load("final_trade_impact_classifier.pkl")
    FEATURES = joblib.load("final_model_features.pkl")
    df       = pd.read_csv("final_modeling_dataset.csv")

    # Reindex df to the exact 250-column shape the models expect.
    # Columns missing from the dataset are filled with 0; the pipeline's
    # internal imputer handles any remaining variance.
    df_features = df.reindex(columns=FEATURES, fill_value=0)

    # Build scaled feature matrix once — used by similarity engine
    imp  = SimpleImputer(strategy="median")
    sc   = StandardScaler()
    X_sc = sc.fit_transform(imp.fit_transform(df_features))

    return reg_pipe, clf_pipe, FEATURES, df, X_sc, imp, sc


reg_pipe, clf_pipe, FEATURES, df_data, X_sc, imp, sc = load_everything()
helpers.init(reg_pipe, clf_pipe, FEATURES, df_data, X_sc, imp, sc)


# ── Claude helpers ────────────────────────────────────────────────────────────
def _client():
    return anthropic.Anthropic(api_key=api_key)


def parse_trade_query(query: str) -> dict:
    """Use Claude to convert a free-form query into {player_name, receiving_team, season}."""
    import re

    response = _client().messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system=(
            "You are an NBA trade analyst. Parse trade queries into structured data. "
            "Return ONLY valid JSON — no explanation, no markdown fences."
        ),
        messages=[{
            "role": "user",
            "content": (
                f'Parse this NBA trade/acquisition query into JSON.\n\n'
                f'Query: "{query}"\n\n'
                "Return a JSON object with:\n"
                '- player_name: full player name (required)\n'
                '- receiving_team: team acquiring the player, full name preferred '
                '(optional — omit if unclear)\n'
                '- season: year as integer (optional — omit if not mentioned; '
                'use the year the season ends, e.g. 2023 for the 2022-23 season)\n\n'
                "If you cannot identify the player with reasonable confidence, return:\n"
                '{"error": "Could not identify the player. Please provide a more specific query."}\n\n'
                "Return ONLY JSON."
            ),
        }],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    # If there's extra text around the JSON, extract the first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "error": (
                f"Could not parse Claude's response as JSON. "
                f"Raw response: {raw!r} — Error: {exc}"
            )
        }


def generate_explanation(prompt: str) -> str:
    """Generate an analyst-style explanation from the pre-built prompt."""
    response = _client().messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_comp_narration(comps: list, player: str, to_team: str) -> str:
    """Narrate the top 3 historical comps as a short prose paragraph."""
    top3 = comps[:3]
    comp_lines = "\n".join([
        f"- {c['player']} to {c['to_team']} ({c['season']}): "
        f"actual win% change = {c['actual_win_pct_change']:+.3f}, "
        f"similarity = {c['similarity_score']:.3f}"
        for c in top3
    ])
    response = _client().messages.create(
        model="claude-sonnet-4-5",
        max_tokens=350,
        messages=[{
            "role": "user",
            "content": (
                f"You are an NBA analyst. In 2–3 sentences of flowing prose (no bullets), "
                f"narrate the top 3 historical comparable trades to the acquisition of "
                f"{player} by {to_team}.\n\n"
                "For each comp, mention who the player was, which team acquired them, "
                "and what actually happened to that team's win percentage afterward. "
                "Focus on what makes them relevant comparisons.\n\n"
                f"Comps:\n{comp_lines}"
            ),
        }],
    )
    return response.content[0].text


# ── UI helpers ────────────────────────────────────────────────────────────────
def _score_color(score: float) -> str:
    if score >= 65:
        return "#2ecc71"   # green
    if score >= 45:
        return "#f39c12"   # orange
    return "#e74c3c"       # red


def _score_emoji(score: float) -> str:
    if score >= 65:
        return "🟢"
    if score >= 45:
        return "🟡"
    return "🔴"


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🏀 NBA Trade Acquisition Impact Tool")
st.markdown(
    "Type a trade in plain English. "
    "The tool predicts its historical impact using machine learning, "
    "then explains it with AI analysis."
)

if not api_key:
    st.warning(
        "⚠️ `ANTHROPIC_API_KEY` is not set. "
        "Add it as an environment variable or a Streamlit secret to enable AI features."
    )

query = st.text_input(
    "Describe a trade or signing",
    placeholder="e.g. 'the Celtics acquiring Jrue Holiday' or 'Lakers trading for Anthony Davis'",
)

analyze = st.button("Analyze Trade", type="primary")

if analyze and not query.strip():
    st.warning("Please enter a trade description before clicking Analyze.")

if analyze and query.strip():
    if not api_key:
        st.error("Cannot run analysis without an `ANTHROPIC_API_KEY`.")
        st.stop()

    with st.spinner("Analyzing trade…"):

        # Step 1 — parse natural language
        parsed = parse_trade_query(query)
        if "error" in parsed:
            st.error(f"**Query Error:** {parsed['error']}")
            st.stop()

        player_name    = parsed.get("player_name")
        receiving_team = parsed.get("receiving_team")
        season         = parsed.get("season")

        # Step 2 — model prediction + similarity comps
        prediction = helpers.predict_acquisition_impact(player_name, receiving_team, season)
        if "error" in prediction:
            st.error(
                f"**Player Not Found:** {prediction['error']}\n\n"
                "This player may not appear in the historical trades dataset (2000–2025). "
                "Try a different spelling, or remove the team/season filter and try again."
            )
            st.stop()

        comps = helpers.find_similar_acquisitions(player_name, receiving_team, season, n=5)

        # Step 3 — AI explanation
        explanation_prompt = helpers.build_explanation_prompt(prediction, comps)
        explanation = generate_explanation(explanation_prompt)

        # Step 4 — comp narration
        comp_narration = generate_comp_narration(
            comps, prediction["player"], prediction["to_team"]
        )

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()

    # Warn if we fell back to most-recent trade because the requested team wasn't found
    if prediction.get("used_fallback"):
        team_str = f" to **{receiving_team}**" if receiving_team else ""
        st.warning(
            f"The exact trade of **{prediction['player']}**{team_str} wasn't found in the "
            f"dataset. Showing results for their most recent trade "
            f"(**{prediction['from_team']} → {prediction['to_team']}, {prediction['season']}**) instead."
        )

    # Parsed trade identity
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Player",  prediction["player"])
    c2.metric("From",    prediction["from_team"])
    c3.metric("To",      prediction["to_team"])
    c4.metric("Season",  str(prediction["season"]))

    st.divider()

    # ── Key metrics row ───────────────────────────────────────────────────────
    score    = prediction["impact_score_0_to_100"]
    win_chg  = prediction["predicted_win_pct_change"]
    prob_imp = prediction["prob_improvement"]
    interp   = prediction["interpretation"]
    color    = _score_color(score)
    emoji    = _score_emoji(score)

    col_score, col_win, col_prob = st.columns(3)

    with col_score:
        st.markdown(f"#### {emoji} Impact Score")
        st.markdown(
            f"<p style='font-size:3rem;font-weight:700;color:{color};margin:0'>"
            f"{score}/100</p>",
            unsafe_allow_html=True,
        )
        st.progress(int(score) / 100)
        st.caption(f"Interpretation: **{interp}**")

    with col_win:
        wins_proj = round(win_chg * 82)
        st.metric(
            label="Predicted Win% Change",
            value=f"{win_chg:+.1%}",
            delta=f"{wins_proj:+d} wins over 82 games",
        )
        if prediction.get("team_win_pct_trade_season") is not None:
            st.caption(
                f"Team win% in trade season: "
                f"{prediction['team_win_pct_trade_season']:.3f}"
            )

    with col_prob:
        st.metric(
            label="Probability of Improvement",
            value=f"{prob_imp:.1%}",
            delta=f"{prob_imp - 0.50:+.1%} vs 50% baseline",
        )

    st.divider()

    # ── Player profile ────────────────────────────────────────────────────────
    profile_items = [
        ("PPG",            prediction.get("player_pts_per_game"), ".1f"),
        ("BPM",            prediction.get("player_bpm"),          "+.1f"),
        ("VORP",           prediction.get("player_vorp"),         ".1f"),
        ("Age at Trade",   prediction.get("player_age"),          "d"),
    ]
    visible = [(lbl, val, fmt) for lbl, val, fmt in profile_items if val is not None]

    if visible:
        st.markdown("**Player Profile at Time of Trade**")
        p_cols = st.columns(len(visible))
        for col, (lbl, val, fmt) in zip(p_cols, visible):
            col.metric(lbl, format(val, fmt))
        st.divider()

    # ── AI explanation ────────────────────────────────────────────────────────
    st.markdown("### AI Trade Analysis")
    st.markdown(explanation)

    st.divider()

    # ── Historical comps table ────────────────────────────────────────────────
    st.markdown("### Most Similar Historical Acquisitions")
    if isinstance(comps, list) and comps:
        comps_df = pd.DataFrame([
            {
                "Player":           c["player"],
                "Season":           c["season"],
                "From":             c["from_team"],
                "To":               c["to_team"],
                "Actual Win% Chg":  f"{c['actual_win_pct_change']:+.3f}",
                "Similarity":       f"{c['similarity_score']:.3f}",
            }
            for c in comps
        ])
        st.dataframe(comps_df, use_container_width=True, hide_index=True)
    else:
        st.info("No comparable acquisitions found.")

    st.divider()

    # ── Comp narration ────────────────────────────────────────────────────────
    st.markdown("### What Happened to These Teams")
    st.markdown(comp_narration)

    st.divider()

    # ── Caveats ───────────────────────────────────────────────────────────────
    with st.expander("⚠️ Model Caveats"):
        for caveat in prediction["model_caveats"]:
            st.markdown(f"- {caveat}")
        st.markdown(
            "- The model uses a time-aware train/test split (trained pre-2020, tested on 2020+) "
            "and is best used as decision support, not a definitive trade grade."
        )

    # ── About the model ───────────────────────────────────────────────────────
    with st.expander("📊 About the Model"):
        st.markdown("""
**Regressor — Gradient Boosting**
| | |
|---|---|
| Dataset | Trades only · 3,302 rows · 2000–2025 |
| Target | Receiving team win% change |
| MAE | 0.1005 (~±8 wins over 82 games) |
| RMSE | 0.1185 |
| R² | 0.180 |
| Baseline | MAE 0.1018 · R² < 0 (always predict mean) |

**Classifier — Random Forest**
| | |
|---|---|
| Dataset | All acquisitions · 5,745 rows · 2000–2025 |
| Target | Did the receiving team improve? (binary) |
| AUC | 0.614 |
| F1 | 0.641 |
| Accuracy | 62.7% |
| Baseline | 59.7% (majority class) |

**Features:** 250 — player stats, team trajectory, player fit, roster activity, SOS/MOV, age context

**Train/Test Split:** Time-aware — trained pre-2020, tested on 2020+

**Leakage check:** None detected
        """)

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Module-level globals — set via init() before calling any function
df = None
FEATURES = None   # full 250-feature list from the pkl — never filtered
reg_pipe = None
clf_pipe = None
_imp = None
_sc = None
_X_sc = None


def init(reg_pipe_, clf_pipe_, features_, df_, x_sc_, imp_, sc_):
    """Initialize module globals. Must be called once at app startup."""
    global df, FEATURES, reg_pipe, clf_pipe, _X_sc, _imp, _sc
    df       = df_
    FEATURES = list(features_)
    reg_pipe = reg_pipe_
    clf_pipe = clf_pipe_
    _X_sc    = x_sc_
    _imp     = imp_
    _sc      = sc_


def _normalize_name(name: str) -> str:
    """Collapse internal whitespace and strip edges."""
    return " ".join(name.strip().split())


def _player_mask(name: str) -> "pd.Series":
    """
    Return a boolean mask over df matching player_name.
    Tries in order:
      1. Exact case-insensitive match (after whitespace normalisation)
      2. Substring match on the full name
      3. Last-name-only substring match
    """
    cleaned = _normalize_name(name).lower()

    # 1 — exact
    mask = df["player_name"].str.strip().str.lower() == cleaned
    if mask.any():
        return mask

    # 2 — full name as substring (handles "Jr." suffixes, etc.)
    mask = df["player_name"].str.lower().str.contains(cleaned, na=False, regex=False)
    if mask.any():
        return mask

    # 3 — last name only
    last = cleaned.split()[-1]
    return df["player_name"].str.lower().str.contains(last, na=False, regex=False)


def _team_mask(col: "pd.Series", abbrev_col: "pd.Series", team: str) -> "pd.Series":
    """
    Return a boolean mask matching a team string against full-name and abbrev columns.
    Strips leading 'the ' so 'the Boston Celtics' matches 'Boston Celtics'.
    """
    t = re.sub(r"^the\s+", "", team.strip().lower())
    return (
        col.str.lower().str.contains(t, na=False, regex=False) |
        (abbrev_col.str.lower() == t)
    )


def _label(score):
    if score <= -0.08: return 'strong negative'
    if score <= -0.02: return 'mild negative'
    if score <   0.02: return 'neutral'
    if score <   0.08: return 'mild positive'
    return 'strong positive'


def _impact_score(change, lo=-0.20, hi=0.20):
    """Scale predicted win% change to a 0–100 impact score."""
    return round((max(lo, min(hi, change)) - lo) / (hi - lo) * 100, 1)


def predict_acquisition_impact(player_name, receiving_team=None, season=None):
    """
    Estimate the historical impact of a player acquisition on the receiving team.

    Args:
        player_name:    str  — player to look up (case-insensitive)
        receiving_team: str  — (optional) team name or abbreviation
        season:         int  — (optional) season year

    Returns:
        dict with all fields needed for AI explanation generation.
        Returns {'error': '...'} if the player is not found.
    """
    player_mask = _player_mask(player_name)
    mask = player_mask.copy()
    if receiving_team:
        # receiving_team is matched against secondary_team only — that is the acquiring team
        mask &= _team_mask(df['secondary_team'], df['secondary_abbrev'], receiving_team)
    if season:
        mask &= df['season'] == int(season)

    sub = df[mask].sort_values('season', ascending=False)

    # If the team/season filters produced no rows, fall back to player-only (most recent)
    # and surface a flag so the caller can warn the user.
    used_fallback = False
    if sub.empty and (receiving_team or season):
        sub = df[player_mask].sort_values('season', ascending=False)
        used_fallback = True

    if sub.empty:
        return {'error': f"No trade rows found for '{player_name}' with the given filters."}

    # Multiple rows (same player traded to same team more than once) → use most recent
    row   = sub.iloc[[0]]
    # Pre-fill all NaN with 0 so the pipeline's internal imputer receives clean data
    X_row = row.reindex(columns=FEATURES, fill_value=0).fillna(0)

    pred = float(reg_pipe.predict(X_row)[0])
    prob = float(clf_pipe.predict_proba(X_row)[0, 1])

    base_pred = row['recv_hist_baseline_win_pct_pred'].values[0]
    trans_wp  = row['receiving_team_trans_win_pct'].values[0]
    pre_wp    = row['receiving_team_pre_win_pct'].values[0] if 'receiving_team_pre_win_pct' in row.columns else None

    return {
        # Identity
        'player':                     row['player_name'].values[0],
        'season':                     int(row['season'].values[0]),
        'from_team':                  row['primary_team'].values[0],
        'to_team':                    row['secondary_team'].values[0],
        'transaction_text':           str(row['text'].values[0]) if 'text' in row.columns else '',

        # Team context at time of trade
        'team_win_pct_prior_season':  round(float(pre_wp), 4)    if pre_wp  is not None and pd.notna(pre_wp)  else None,
        'team_win_pct_trade_season':  round(float(trans_wp), 4)  if pd.notna(trans_wp)  else None,
        'team_baseline_prediction':   round(float(base_pred), 4) if pd.notna(base_pred) else None,
        'team_expected_change':       round(float(base_pred - trans_wp), 4)
                                      if pd.notna(base_pred) and pd.notna(trans_wp) else None,

        # Player context
        'player_bpm':         round(float(row['player_trans_bpm'].values[0]), 2)
                              if 'player_trans_bpm' in row.columns else None,
        'player_vorp':        round(float(row['player_trans_vorp'].values[0]), 2)
                              if 'player_trans_vorp' in row.columns else None,
        'player_pts_per_game': round(float(row['player_trans_pts_per_game'].values[0]), 1)
                               if 'player_trans_pts_per_game' in row.columns else None,
        'player_age':         int(row['player_trans_age'].values[0])
                              if 'player_trans_age' in row.columns else None,

        # Model outputs
        'predicted_win_pct_change': round(pred, 4),
        'prob_improvement':         round(prob, 4),
        'impact_score_0_to_100':    _impact_score(pred),
        'interpretation':           _label(pred),

        # Caveats for AI to use
        'model_caveats': [
            'This estimate is based on historical similar acquisitions, not a guaranteed causal outcome.',
            'Draft picks and cash considerations are not valued by the model.',
            'Player stats reflect season-level profiles and may not capture exact pre-trade timing.',
            'The model does not account for injuries, chemistry, or coaching changes.',
        ],

        # True when the requested team/season wasn't found and we fell back to most recent
        'used_fallback': used_fallback,
    }


def find_similar_acquisitions(player_name, receiving_team=None, season=None, n=5):
    """
    Return the n most historically similar trade acquisitions.

    Args:
        player_name:    str — player to look up
        receiving_team: str — (optional) team name or abbreviation
        season:         int — (optional) season year
        n:              int — number of comps to return (default 5)

    Returns:
        list of dicts, each describing a similar historical acquisition.
        Returns {'error': '...'} if the player is not found.
    """
    player_mask = _player_mask(player_name)
    mask = player_mask.copy()
    if receiving_team:
        mask &= _team_mask(df['secondary_team'], df['secondary_abbrev'], receiving_team)
    if season:
        mask &= df['season'] == int(season)

    sub = df[mask].sort_values('season', ascending=False)

    if sub.empty and (receiving_team or season):
        sub = df[player_mask].sort_values('season', ascending=False)

    if sub.empty:
        return {'error': f"No rows found for '{player_name}'."}

    idx_list  = df.index.tolist()
    query_pos = idx_list.index(sub.index[0])
    sims      = cosine_similarity(_X_sc[query_pos].reshape(1, -1), _X_sc)[0]
    sims[query_pos] = -999

    results = []
    for p in np.argsort(sims)[::-1][:n]:
        row = df.iloc[p]
        residual = row.get('residual_win_pct')
        results.append({
            'player':                row['player_name'],
            'season':                int(row['season']),
            'from_team':             row['primary_team'],
            'to_team':               row['secondary_team'],
            'actual_win_pct_change': round(float(row['receiving_team_post_win_pct_change']), 4),
            'actual_residual':       round(float(residual), 4) if residual is not None and pd.notna(residual) else None,
            'similarity_score':      round(float(sims[p]), 4),
            'transaction_text':      str(row.get('text', ''))[:150],
        })
    return results


def build_explanation_prompt(prediction: dict, comps: list) -> str:
    """
    Build a prompt for Claude to generate a natural-language trade explanation.
    Pass the output of predict_acquisition_impact() and find_similar_acquisitions() directly.
    """
    if 'error' in prediction:
        return f'Error: {prediction["error"]}'

    p = prediction
    comp_text = '\n'.join([
        f"  - {c['player']} to {c['to_team']} ({c['season']}): "
        f"actual win% change = {c['actual_win_pct_change']:+.3f}, "
        f"similarity = {c['similarity_score']:.3f}"
        for c in comps
    ]) if isinstance(comps, list) else '  None available'

    return f"""You are an NBA trade analyst assistant. Explain the following trade prediction
in clear, concise language for a basketball fan. Be honest about uncertainty.
Do not use bullet points. Write 3–4 sentences.

Trade: {p['player']} from {p['from_team']} to {p['to_team']} (season {p['season']})

Player profile at time of trade:
  Age: {p.get('player_age', 'N/A')}
  Points per game: {p.get('player_pts_per_game', 'N/A')}
  Box Plus/Minus (BPM): {p.get('player_bpm', 'N/A')}
  Value Over Replacement Player (VORP): {p.get('player_vorp', 'N/A')}

Receiving team context:
  Win% prior season: {p.get('team_win_pct_prior_season', 'N/A')}
  Win% in trade season: {p.get('team_win_pct_trade_season', 'N/A')}
  Expected win% next season (before this trade): {p.get('team_baseline_prediction', 'N/A')}
  Expected change before trade: {p.get('team_expected_change', 'N/A')}

Model prediction:
  Predicted win% change: {p['predicted_win_pct_change']:+.4f}
  Probability of improvement: {p['prob_improvement']:.1%}
  Impact score (0–100): {p['impact_score_0_to_100']}
  Interpretation: {p['interpretation']}

Most similar historical acquisitions and their actual outcomes:
{comp_text}

Important caveats to acknowledge:
{chr(10).join('  - ' + c for c in p['model_caveats'])}
"""

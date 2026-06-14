import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import poisson, skellam
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="World Cup AI Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#060814;--card:#0e1426;--card2:#111b33;--ink:#f8fafc;--muted:#94a3b8;--gold:#f8c145;--green:#2dd4bf;--red:#fb7185;--blue:#60a5fa;--line:rgba(255,255,255,.10);} 
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
.stApp {background: radial-gradient(circle at top left, #172554 0%, #060814 35%, #030712 100%); color: var(--ink);} 
.block-container {padding-top: 1.2rem; max-width: 1500px;}
.hero {padding: 1.2rem 1.4rem; border:1px solid var(--line); border-radius: 24px; background: linear-gradient(135deg, rgba(96,165,250,.20), rgba(45,212,191,.10), rgba(248,193,69,.08)); box-shadow: 0 24px 80px rgba(0,0,0,.35);} 
.hero h1 {font-size: 2.25rem; margin: 0; font-weight: 800; letter-spacing:-.04em;}
.hero p {color: var(--muted); margin:.35rem 0 0; font-size:1.02rem;}
.card {background: linear-gradient(180deg, rgba(255,255,255,.065), rgba(255,255,255,.025)); border: 1px solid var(--line); border-radius: 20px; padding: 1rem 1rem; box-shadow: 0 16px 40px rgba(0,0,0,.25);} 
.metric-label {font-size:.78rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em;}
.metric-value {font-size:1.8rem; font-weight:800; color:#fff; margin-top:.15rem;}
.match-title {font-size:1.35rem; font-weight:800; color:#fff;}
.badge {display:inline-block; padding:.28rem .55rem; border-radius:999px; font-size:.75rem; font-weight:800; border:1px solid var(--line); margin-right:.35rem;}
.badge-green {background:rgba(45,212,191,.14); color:#5eead4;}
.badge-blue {background:rgba(96,165,250,.16); color:#93c5fd;}
.badge-gold {background:rgba(248,193,69,.15); color:#fde68a;}
.badge-red {background:rgba(251,113,133,.15); color:#fda4af;}
.small-muted {color: var(--muted); font-size:.86rem;}
hr {border-color: var(--line)!important;}
[data-testid="stMetricValue"] {font-weight:800;}
.stDataFrame {border-radius: 16px; overflow:hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
BASE_RATING = 1500.0
HOME_ADV = 100.0
HOSTS = {"Mexico", "Canada", "United States"}
CONTINENTAL_FINALS = {
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "CONCACAF Championship", "Gold Cup", "Oceania Nations Cup", "FIFA Confederations Cup",
}
MAJOR_40 = {"UEFA Nations League", "CONCACAF Nations League"}
DATA_PATH = Path(__file__).parent / "data" / "results.csv"

# ============================================================
# MODEL FUNCTIONS
# ============================================================
def k_factor(tournament: str) -> float:
    if tournament == "FIFA World Cup":
        return 60.0
    if tournament in CONTINENTAL_FINALS:
        return 50.0
    if "qualification" in str(tournament).lower() or tournament in MAJOR_40:
        return 40.0
    if tournament == "Friendly":
        return 20.0
    return 30.0


def mov_multiplier(goal_diff: int) -> float:
    n = abs(int(goal_diff))
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    if n == 3:
        return 1.75
    return 1.75 + (n - 3) / 8.0


def expectancy(dr: float) -> float:
    return 1.0 / (10.0 ** (-dr / 400.0) + 1.0)


def rating_history_frame(results: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = results.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    ratings = {}
    rows = []
    for r in df.itertuples(index=False):
        home, away = r.home_team, r.away_team
        hs, aw = int(r.home_score), int(r.away_score)
        rh, ra = ratings.get(home, BASE_RATING), ratings.get(away, BASE_RATING)
        dr = rh - ra + (0.0 if bool(r.neutral) else HOME_ADV)
        rows.append({
            "date": r.date, "home_team": home, "away_team": away,
            "home_score": hs, "away_score": aw, "tournament": r.tournament,
            "neutral": bool(r.neutral), "elo_home_pre": rh, "elo_away_pre": ra, "dr": dr,
        })
        we = expectancy(dr)
        w = 1.0 if hs > aw else 0.5 if hs == aw else 0.0
        delta = k_factor(r.tournament) * mov_multiplier(hs - aw) * (w - we)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return pd.DataFrame(rows), ratings


def fit_goals_model(hist: pd.DataFrame, fit_from="1998-01-01") -> dict:
    fit_df = hist[(hist["date"] >= pd.Timestamp(fit_from)) & (hist["tournament"] != "Friendly")].copy()
    x = fit_df["dr"].to_numpy(dtype=float) / 400.0
    hg = fit_df["home_score"].to_numpy(dtype=float)
    ag = fit_df["away_score"].to_numpy(dtype=float)

    def nll(params):
        a, b = params
        lh = np.exp(a + b * x)
        la = np.exp(a - b * x)
        return -(np.sum(hg * np.log(lh) - lh) + np.sum(ag * np.log(la) - la))

    res = minimize(nll, x0=[0.2, 0.5], method="Nelder-Mead")
    a, b = res.x
    return {"a": float(a), "b": float(b), "n": int(len(x)), "success": bool(res.success)}


def expected_goals(dr: float, params: dict) -> tuple[float, float]:
    x = dr / 400.0
    return float(np.exp(params["a"] + params["b"] * x)), float(np.exp(params["a"] - params["b"] * x))


def outcome_probs(dr: float, params: dict) -> tuple[float, float, float]:
    lh, la = expected_goals(dr, params)
    p_draw = float(skellam.pmf(0, lh, la))
    p_away = float(skellam.cdf(-1, lh, la))
    p_home = 1.0 - p_draw - p_away
    return p_home, p_draw, p_away


def exact_score(lh: float, la: float, max_goals: int = 6) -> tuple[str, float]:
    best, bestp = "0-0", -1.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, lh) * poisson.pmf(a, la)
            if p > bestp:
                best, bestp = f"{h}-{a}", float(p)
    return best, bestp


def recent_form_from_past(results: pd.DataFrame, team: str, before, n=8):
    df = results[(results["date"] < pd.Timestamp(before)) & ((results["home_team"] == team) | (results["away_team"] == team))]
    df = df.dropna(subset=["home_score", "away_score"]).sort_values("date").tail(n)
    form, gf, ga = [], 0, 0
    for _, r in df.iterrows():
        is_home = r["home_team"] == team
        f = int(r["home_score"] if is_home else r["away_score"])
        a = int(r["away_score"] if is_home else r["home_score"])
        gf += f; ga += a
        form.append("W" if f > a else "D" if f == a else "L")
    unbeaten = 0
    for x in reversed(form):
        if x in ["W", "D"]: unbeaten += 1
        else: break
    return {"form":"".join(form), "gf":gf, "ga":ga, "avg_gf":gf/max(len(df),1), "avg_ga":ga/max(len(df),1), "unbeaten":unbeaten, "n":len(df)}


def make_training_features(hist: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = hist.copy().sort_values("date").reset_index(drop=True)
    rows = []
    stats = {}
    for r in df.itertuples(index=False):
        home, away = r.home_team, r.away_team
        hs, aw = int(r.home_score), int(r.away_score)
        sh = stats.get(home, {"gf": [], "ga": [], "pts": []})
        sa = stats.get(away, {"gf": [], "ga": [], "pts": []})
        lh, la = expected_goals(float(r.dr), params)
        def avg(x): return float(np.mean(x[-8:])) if len(x) else 0.0
        row = {
            "date": r.date, "home_team": home, "away_team": away,
            "dr": float(r.dr), "neutral": int(bool(r.neutral)),
            "elo_home_pre": float(r.elo_home_pre), "elo_away_pre": float(r.elo_away_pre),
            "xg_home": lh, "xg_away": la, "total_xg": lh + la,
            "home_form_gf": avg(sh["gf"]), "home_form_ga": avg(sh["ga"]), "home_form_pts": avg(sh["pts"]),
            "away_form_gf": avg(sa["gf"]), "away_form_ga": avg(sa["ga"]), "away_form_pts": avg(sa["pts"]),
            "ftr": "H" if hs > aw else "D" if hs == aw else "A",
            "over15": int(hs + aw > 1.5), "over25": int(hs + aw > 2.5),
            "btts": int(hs > 0 and aw > 0),
        }
        row["dc_1x"] = int(row["ftr"] in ["H", "D"])
        row["dc_12"] = int(row["ftr"] in ["H", "A"])
        row["dc_x2"] = int(row["ftr"] in ["D", "A"])
        rows.append(row)
        hp, ap = (3, 0) if hs > aw else (1, 1) if hs == aw else (0, 3)
        stats.setdefault(home, {"gf": [], "ga": [], "pts": []})
        stats.setdefault(away, {"gf": [], "ga": [], "pts": []})
        stats[home]["gf"].append(hs); stats[home]["ga"].append(aw); stats[home]["pts"].append(hp)
        stats[away]["gf"].append(aw); stats[away]["ga"].append(hs); stats[away]["pts"].append(ap)
    return pd.DataFrame(rows)


def make_future_features(future: pd.DataFrame, results: pd.DataFrame, ratings: dict, params: dict) -> pd.DataFrame:
    rows = []
    for _, r in future.sort_values("date").iterrows():
        home, away = r["home_team"], r["away_team"]
        rh, ra = ratings.get(home, BASE_RATING), ratings.get(away, BASE_RATING)
        host_adv = HOME_ADV if (not bool(r.get("neutral", True))) else 0.0
        dr = rh - ra + host_adv
        lh, la = expected_goals(dr, params)
        hf = recent_form_from_past(results, home, r["date"])
        af = recent_form_from_past(results, away, r["date"])
        ls, lsp = exact_score(lh, la)
        rows.append({
            "date": r["date"], "home_team": home, "away_team": away, "tournament": r.get("tournament", "FIFA World Cup"),
            "neutral": int(bool(r.get("neutral", True))), "elo_home_pre": rh, "elo_away_pre": ra, "dr": dr,
            "xg_home": lh, "xg_away": la, "total_xg": lh + la,
            "home_form_gf": hf["avg_gf"], "home_form_ga": hf["avg_ga"], "home_form_pts": np.nan,
            "away_form_gf": af["avg_gf"], "away_form_ga": af["avg_ga"], "away_form_pts": np.nan,
            "home_form": hf["form"], "away_form": af["form"],
            "home_goals_recent": f"{hf['gf']}-{hf['ga']}", "away_goals_recent": f"{af['gf']}-{af['ga']}",
            "likely_score": ls, "likely_score_prob": lsp,
        })
    return pd.DataFrame(rows)


def train_models(train_df: pd.DataFrame):
    features = ["dr", "neutral", "elo_home_pre", "elo_away_pre", "xg_home", "xg_away", "total_xg", "home_form_gf", "home_form_ga", "away_form_gf", "away_form_ga"]
    X = train_df[features]
    models = {}
    # Multiclass FTR
    ftr_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", HistGradientBoostingClassifier(max_iter=160, learning_rate=0.045, random_state=42))])
    ftr_pipe.fit(X, train_df["ftr"])
    models["ftr"] = ftr_pipe
    # Binary targets
    for target in ["over15", "over25", "btts", "dc_1x", "dc_12", "dc_x2"]:
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=260, min_samples_leaf=25, random_state=42, n_jobs=-1, class_weight="balanced_subsample"))])
        pipe.fit(X, train_df[target])
        models[target] = pipe
    return models, features


def predict_all(future_feat: pd.DataFrame, models: dict, features: list, params: dict) -> pd.DataFrame:
    df = future_feat.copy()
    X = df[features]
    # Poisson FTR
    pois = np.array([outcome_probs(x, params) for x in df["dr"]])
    # ML FTR classes may be A,D,H; align
    m = models["ftr"]
    proba = m.predict_proba(X)
    classes = list(m.named_steps["model"].classes_) if "model" in m.named_steps else list(m.classes_)
    ml_map = {c: proba[:, i] for i, c in enumerate(classes)}
    ml_h = ml_map.get("H", np.zeros(len(df))); ml_d = ml_map.get("D", np.zeros(len(df))); ml_a = ml_map.get("A", np.zeros(len(df)))
    df["ml_p_home"], df["ml_p_draw"], df["ml_p_away"] = ml_h, ml_d, ml_a
    df["p_home"] = 0.55 * pois[:,0] + 0.45 * ml_h
    df["p_draw"] = 0.55 * pois[:,1] + 0.45 * ml_d
    df["p_away"] = 0.55 * pois[:,2] + 0.45 * ml_a
    s = df[["p_home", "p_draw", "p_away"]].sum(axis=1)
    df["p_home"], df["p_draw"], df["p_away"] = df["p_home"]/s, df["p_draw"]/s, df["p_away"]/s
    # Trained target probs blended with derived where applicable
    def bin_prob(target):
        return models[target].predict_proba(X)[:, 1]
    df["ml_p_over15"] = bin_prob("over15"); df["ml_p_over25"] = bin_prob("over25"); df["ml_p_btts"] = bin_prob("btts")
    df["ml_p_1x"] = bin_prob("dc_1x"); df["ml_p_12"] = bin_prob("dc_12"); df["ml_p_x2"] = bin_prob("dc_x2")
    df["poisson_p_under15"] = poisson.cdf(1, df["total_xg"]); df["poisson_p_over15"] = 1 - df["poisson_p_under15"]
    df["poisson_p_under25"] = poisson.cdf(2, df["total_xg"]); df["poisson_p_over25"] = 1 - df["poisson_p_under25"]
    df["p_over15"] = 0.50 * df["poisson_p_over15"] + 0.50 * df["ml_p_over15"]
    df["p_under15"] = 1 - df["p_over15"]
    df["p_over25"] = 0.50 * df["poisson_p_over25"] + 0.50 * df["ml_p_over25"]
    df["p_under25"] = 1 - df["p_over25"]
    df["p_btts"] = df["ml_p_btts"]
    df["p_btts_no"] = 1 - df["p_btts"]
    df["p_1x"] = 0.50 * (df["p_home"] + df["p_draw"]) + 0.50 * df["ml_p_1x"]
    df["p_12"] = 0.50 * (df["p_home"] + df["p_away"]) + 0.50 * df["ml_p_12"]
    df["p_x2"] = 0.50 * (df["p_draw"] + df["p_away"]) + 0.50 * df["ml_p_x2"]
    # Picks
    df["presentation_prediction"] = df.apply(lambda r: max({r.home_team:r.p_home, "Draw":r.p_draw, r.away_team:r.p_away}, key={r.home_team:r.p_home, "Draw":r.p_draw, r.away_team:r.p_away}.get), axis=1)
    df["presentation_prediction_prob"] = df[["p_home", "p_draw", "p_away"]].max(axis=1)
    dc_cols = {"1X":"p_1x", "12":"p_12", "X2":"p_x2"}
    df["double_chance_pick"] = df[list(dc_cols.values())].idxmax(axis=1).map({v:k for k,v in dc_cols.items()})
    df["double_chance_prob"] = df[list(dc_cols.values())].max(axis=1)
    df["ou15_pick"] = np.where(df["p_over15"] >= df["p_under15"], "Over 1.5", "Under 1.5"); df["ou15_prob"] = df[["p_over15", "p_under15"]].max(axis=1)
    df["ou25_pick"] = np.where(df["p_over25"] >= df["p_under25"], "Over 2.5", "Under 2.5"); df["ou25_prob"] = df[["p_over25", "p_under25"]].max(axis=1)
    df["btts_pick"] = np.where(df["p_btts"] >= df["p_btts_no"], "BTTS Yes", "BTTS No"); df["btts_prob"] = df[["p_btts", "p_btts_no"]].max(axis=1)
    def tier(p):
        return "Elite" if p >= .85 else "Very Strong" if p >= .75 else "Strong" if p >= .70 else "Moderate" if p >= .60 else "Weak"
    def best_safe(r):
        picks = {"1X":r.p_1x,"12":r.p_12,"X2":r.p_x2,"Over 1.5":r.p_over15,"Under 1.5":r.p_under15,"Over 2.5":r.p_over25,"Under 2.5":r.p_under25,"BTTS Yes":r.p_btts,"BTTS No":r.p_btts_no}
        ranked = sorted(picks.items(), key=lambda kv: kv[1], reverse=True)
        return pd.Series({"best_safe_pick":ranked[0][0], "best_safe_pick_prob":ranked[0][1], "best_safe_pick_tier":tier(ranked[0][1]), "top3_safe_picks":"; ".join([f"{k} {v:.1%}" for k,v in ranked[:3]])})
    df = pd.concat([df, df.apply(best_safe, axis=1)], axis=1)
    return df


def h2h_text(results, home, away, before):
    h = results[(results["date"] < pd.Timestamp(before)) & (((results["home_team"]==home)&(results["away_team"]==away)) | ((results["home_team"]==away)&(results["away_team"]==home)))]
    h = h.dropna(subset=["home_score","away_score"]).sort_values("date")
    if h.empty: return f"No previous recorded meeting between {home} and {away}."
    hw=d=aw=0
    for _, r in h.iterrows():
        hg = int(r.home_score if r.home_team == home else r.away_score)
        ag = int(r.away_score if r.home_team == home else r.home_score)
        if hg>ag: hw += 1
        elif hg==ag: d += 1
        else: aw += 1
    last = h.iloc[-1]
    return f"H2H for {home}: {hw}W-{d}D-{aw}L in {len(h)} meetings; last: {last.date.date()} {last.home_team} {int(last.home_score)}-{int(last.away_score)} {last.away_team}."


def pretty_form_str(s):
    return " ".join({"W":"✅","D":"➖","L":"❌"}.get(x,x) for x in str(s))

# ============================================================
# DATA PIPELINE
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes=None):
    if uploaded_bytes is not None:
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
    else:
        df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["home_score", "away_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].fillna(True).astype(bool)
    else:
        df["neutral"] = True
    return df.sort_values("date").reset_index(drop=True)


@st.cache_resource(show_spinner=True)
def build_engine(uploaded_bytes=None):
    results = load_data(uploaded_bytes)
    hist, ratings = rating_history_frame(results)
    params = fit_goals_model(hist)
    train_df = make_training_features(hist, params)
    # Keep modern/informative era for ML where feature behavior is relevant.
    train_ml = train_df[train_df["date"] >= pd.Timestamp("1998-01-01")].copy()
    models, features = train_models(train_ml)
    future = results[(results["tournament"] == "FIFA World Cup") & (results["home_score"].isna() | results["away_score"].isna())].copy()
    future_feat = make_future_features(future, results, ratings, params)
    preds = predict_all(future_feat, models, features, params)
    return results, hist, ratings, params, train_ml, models, features, preds

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### ⚙️ Dashboard Controls")
upload = st.sidebar.file_uploader("Upload another results.csv", type=["csv"])
uploaded_bytes = upload.getvalue() if upload is not None else None
st.sidebar.caption("Default file: data/results.csv. Uploading a new file rebuilds the model.")

with st.spinner("Building World Cup AI engine: Elo + Poisson + trained ML safe markets..."):
    results, hist, ratings, params, train_ml, models, features, predictions = build_engine(uploaded_bytes)

fixture_dates = sorted(predictions["date"].dt.date.unique()) if len(predictions) else []
if fixture_dates:
    today = date.today()
    default_idx = fixture_dates.index(today) if today in fixture_dates else 0
    selected_date = st.sidebar.selectbox("Fixture date", fixture_dates, index=default_idx)
else:
    selected_date = date.today()
team_filter = st.sidebar.multiselect("Filter teams", sorted(set(predictions["home_team"]).union(predictions["away_team"])) if len(predictions) else [])
min_safe = st.sidebar.slider("Minimum best-safe probability", 0.50, 0.95, 0.60, 0.01)

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class='hero'>
  <h1>⚽ World Cup AI Predictor</h1>
  <p>Elo + Poisson xG + trained ML targets for FTR, Double Chance, Over/Under 1.5, Over/Under 2.5 and BTTS — with detailed match explanations.</p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='card'><div class='metric-label'>Historical Matches</div><div class='metric-value'>{len(results.dropna(subset=['home_score'])):,}</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='card'><div class='metric-label'>Teams Rated</div><div class='metric-value'>{len(ratings):,}</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='card'><div class='metric-label'>Future WC Fixtures</div><div class='metric-value'>{len(predictions):,}</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='card'><div class='metric-label'>Goals Fit Matches</div><div class='metric-value'>{params['n']:,}</div></div>", unsafe_allow_html=True)

# ============================================================
# FILTERED DATA
# ============================================================
day = predictions[predictions["date"].dt.date == selected_date].copy()
if team_filter:
    day = day[(day["home_team"].isin(team_filter)) | (day["away_team"].isin(team_filter))]
day = day[day["best_safe_pick_prob"] >= min_safe]

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📅 Matchday Predictions", "🏆 All Fixtures", "📊 Model Analytics", "🧠 Methodology"])

with tab1:
    st.subheader(f"Predictions for {selected_date}")
    if day.empty:
        st.warning("No fixtures match the selected filters.")
    for _, r in day.iterrows():
        home, away = r.home_team, r.away_team
        primary = r.presentation_prediction
        conf = r.presentation_prediction_prob
        badge_cls = "badge-green" if conf >= .65 else "badge-gold" if conf >= .45 else "badge-red"
        st.markdown(f"""
        <div class='card'>
          <div class='match-title'>{home} <span class='small-muted'>vs</span> {away}</div>
          <span class='badge {badge_cls}'>Prediction: {primary} ({conf:.1%})</span>
          <span class='badge badge-blue'>Likely Score: {r.likely_score}</span>
          <span class='badge badge-green'>Best Safe: {r.best_safe_pick} ({r.best_safe_pick_prob:.1%})</span>
          <span class='badge badge-gold'>Top 3: {r.top3_safe_picks}</span>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1.05, 1.05, 1.2])
        with c1:
            st.markdown("**Match Probabilities**")
            prob_df = pd.DataFrame({"Outcome":[home,"Draw",away], "Probability":[r.p_home,r.p_draw,r.p_away]})
            fig = px.bar(prob_df, x="Probability", y="Outcome", orientation="h", text=prob_df["Probability"].map(lambda x: f"{x:.1%}"), range_x=[0,1])
            fig.update_layout(height=230, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Safe Markets**")
            safe_df = pd.DataFrame({
                "Market":["1X","12","X2","Over 1.5","Under 1.5","Over 2.5","Under 2.5","BTTS Yes","BTTS No"],
                "Probability":[r.p_1x,r.p_12,r.p_x2,r.p_over15,r.p_under15,r.p_over25,r.p_under25,r.p_btts,r.p_btts_no]
            }).sort_values("Probability", ascending=False)
            st.dataframe(safe_df.style.format({"Probability":"{:.1%}"}), use_container_width=True, hide_index=True)
        with c3:
            hf = recent_form_from_past(results, home, r.date)
            af = recent_form_from_past(results, away, r.date)
            st.markdown("**Model Explanation**")
            st.write(f"Primary view: **{primary} ({conf:.1%})**. Best safer pick: **{r.best_safe_pick} ({r.best_safe_pick_prob:.1%})**. Top safer options: {r.top3_safe_picks}.")
            st.write(f"Elo: {home} **{r.elo_home_pre:.0f}** vs {away} **{r.elo_away_pre:.0f}**; adjusted Elo diff **{r.dr:.0f}**. Poisson xG: **{r.xg_home:.2f}-{r.xg_away:.2f}**.")
            st.write(f"Recent form: {home} {pretty_form_str(hf['form'])} ({hf['gf']}-{hf['ga']}); {away} {pretty_form_str(af['form'])} ({af['gf']}-{af['ga']}).")
            st.caption(h2h_text(results, home, away, r.date))
        st.divider()

with tab2:
    st.subheader("All Future World Cup Fixtures")
    show_cols = ["date","home_team","away_team","presentation_prediction","presentation_prediction_prob","likely_score","best_safe_pick","best_safe_pick_prob","top3_safe_picks","p_home","p_draw","p_away","p_1x","p_12","p_x2","p_over15","p_under15","p_over25","p_under25","p_btts","p_btts_no"]
    table = predictions[show_cols].copy()
    st.dataframe(table.style.format({c:"{:.1%}" for c in table.select_dtypes(include=[float]).columns if c not in []}), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download predictions CSV", table.to_csv(index=False).encode("utf-8"), "worldcup_ai_predictions.csv", "text/csv")

with tab3:
    st.subheader("Model Analytics")
    c1, c2 = st.columns(2)
    with c1:
        top_ratings = pd.DataFrame(sorted(ratings.items(), key=lambda x: -x[1])[:30], columns=["Team", "Elo"])
        fig = px.bar(top_ratings, x="Elo", y="Team", orientation="h", title="Top 30 Elo Ratings")
        fig.update_layout(height=650, yaxis={"categoryorder":"total ascending"}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        market_counts = predictions["best_safe_pick"].value_counts().reset_index()
        market_counts.columns = ["Best Safe Pick", "Count"]
        fig = px.pie(market_counts, names="Best Safe Pick", values="Count", hole=.45, title="Best Safe Pick Distribution")
        fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Goals model parameters**")
        st.json(params)

with tab4:
    st.subheader("Methodology")
    st.markdown("""
    **Pipeline**
    1. Replay international football history using an Elo engine with match-importance K factors, margin-of-victory adjustment and home advantage.
    2. Fit a Poisson goals model where Elo difference maps to expected goals.
    3. Train ML models for full-time result, Over 1.5, Over 2.5, BTTS and Double Chance targets.
    4. Blend statistical Poisson probabilities and trained ML probabilities.
    5. Produce professional-style justifications using Elo, xG, safe-market probabilities, recent form and head-to-head history.

    **Important note:** These are probabilistic estimates, not guarantees. Use them as decision support, not certainty.
    """)
    st.markdown("**Feature columns used by ML:**")
    st.code("\n".join(features))

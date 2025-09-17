# ui.py — compact layout, refined sidebar, clearer Motion & Baseline,
# steadier Timelines (smoothing + motion gating), PPG quality badge,
# Recent Actions compact (smaller font, tight spacing),
# "Intervention effect" shown only once (under Motion & Baseline).
import os, time
import httpx
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

AGENT_URL  = os.getenv("AGENT_URL", "http://127.0.0.1:8001")
POLL_SEC   = float(os.getenv("POLL_SEC", "1.0"))
WINDOW_SEC = int(os.getenv("WINDOW_SEC", "180"))

MOVE_GATE = float(os.getenv("MOVE_GATE", "0.03"))
EMA_ALPHA = float(os.getenv("EMA_ALPHA", "0.30"))

st.set_page_config(page_title="NeuroFocus Copilot", layout="wide")
st.title("🧠 NeuroFocus Copilot")

# ---------- Global CSS tweaks ----------
st.markdown(
    """
    <style>
      /* Sidebar buttons */
      [data-testid="stSidebar"] .block-container { padding-top: 0.6rem; }
      [data-testid="stSidebar"] .stButton > button {
          width: 100%; padding: 0.45rem 0.6rem; border-radius: 10px;
          white-space: nowrap; font-weight: 600;
      }
      [data-testid="stSidebar"] .stButton { margin-bottom: .35rem; }

      /* Badges & section subtitles */
      .badge{display:inline-block;padding:.25rem .5rem;border-radius:999px;font-size:.85rem;font-weight:600}
      .section-subtitle{font-size:.95rem;font-weight:700;margin:.25rem 0 .4rem 0;}

      /* Smaller metric numbers globally (Focus, HR, HRV, Movement, EEG baseline) */
      div[data-testid="stMetricValue"]{ font-size:1.6rem; }     /* default is ~2.5rem */
      div[data-testid="stMetricLabel"]{ font-size:0.85rem; }
      div[data-testid="stMetricDelta"]{ font-size:0.85rem; }

      /* Recent actions: compact spacing + smaller font */
      .recent-compact { line-height: 1.25; }
      .recent-compact .title { font-size:0.95rem; font-weight:700; margin:0.1rem 0; }
      .recent-compact .meta  { font-size:0.82rem; color:rgba(0,0,0,.65); margin:.05rem 0 .2rem 0; }
      .recent-compact hr { margin:4px 0; opacity:.15; }
      .recent-compact [data-testid="stMarkdownContainer"] p { margin:0.15rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def fetch_snapshot(seconds: int) -> dict:
    r = httpx.get(f"{AGENT_URL}/snapshot", params={"seconds": seconds}, timeout=5.0)
    r.raise_for_status()
    return r.json()

def start_calibration():
    httpx.post(f"{AGENT_URL}/calibrate/start", timeout=3.0)

def stop_calibration() -> dict:
    r = httpx.post(f"{AGENT_URL}/calibrate/stop", timeout=5.0)
    r.raise_for_status()
    return r.json()

def effect_around(df: pd.DataFrame, ts: float, baseline_ratio: float, seconds: int = 20):
    seg = df[(df["ts"] >= ts - seconds) & (df["ts"] <= ts + seconds)].copy()
    if seg.empty:
        return None
    b = float(baseline_ratio) or 1.0
    seg["focus"] = 50 + 100 * (b - seg["ratio_smooth"].fillna(seg["ratio"])) / (b + 1e-6)
    pre  = seg[seg["ts"] < ts]["focus"].median() if (seg["ts"] < ts).any() else None
    post = seg[seg["ts"] >= ts]["focus"].median() if (seg["ts"] >= ts).any() else None
    if pre is None or post is None:
        return None
    return float(post - pre)

def compute_ppg_quality(df: pd.DataFrame):
    if "hr_bpm" not in df.columns or len(df) == 0:
        return None
    hr = df["hr_bpm"].replace(0, np.nan)
    valid = hr.notna()
    if valid.sum() < 5:
        return {"score": 0.0, "label": "No signal", "continuity": 0.0, "stability": 0.0,
                "motion": float(df.get("move_idx", pd.Series([0])).fillna(0).mean()),
                "plausibility": 0.0, "color": "#ef4444"}
    continuity = float(valid.mean())
    hr_s = df.get("hr_bpm_smooth", hr).astype(float)
    diffs = np.abs(np.diff(hr_s.dropna().values))
    stability = float(1.0 - min(1.0, (np.median(diffs) if diffs.size else 0.0) / 15.0))
    mov_series = df.get("move_idx", pd.Series([0.0] * len(df))).fillna(0.0)
    motion_level = float(mov_series.rolling(10, min_periods=1).mean().iloc[-1])
    motion_score = float(1.0 - min(1.0, motion_level / max(MOVE_GATE, 1e-6)))
    plaus_series = hr_s[(hr_s > 0)]
    plausibility = float(((plaus_series >= 40) & (plaus_series <= 180)).mean()) if len(plaus_series) else 0.0
    score = 0.40 * continuity + 0.25 * stability + 0.20 * motion_score + 0.15 * plausibility
    if score >= 0.80: label, color = "Excellent", "#16a34a"
    elif score >= 0.60: label, color = "Good", "#22c55e"
    elif score >= 0.40: label, color = "Fair", "#f59e0b"
    else: label, color = "Poor", "#ef4444"
    return {"score": float(score), "label": label, "color": color,
            "continuity": continuity, "stability": stability,
            "motion": motion_level, "plausibility": plausibility}

def render_quality_badge(q):
    if not q:
        st.caption("PPG quality: no signal")
        return
    txt = f"PPG quality: {q['label']}  ·  score {q['score']:.2f}"
    st.markdown(f"<span class='badge' style='background:{q['color']}; color:white'>{txt}</span>",
                unsafe_allow_html=True)
    st.caption(
        f"continuity {q['continuity']:.2f} · stability {q['stability']:.2f} · "
        f"motion {q['motion']:.3f} · plausibility {q['plausibility']:.2f}"
    )

def build_recap_bullets(df: pd.DataFrame, baseline_ratio: float, ts: float, seconds: int = 60):
    start, end = ts - seconds, ts
    seg = df[(df["ts"] >= start) & (df["ts"] <= end)].copy()
    if seg.empty:
        return ["No recent data.", "", ""]
    b = float(baseline_ratio) or 1.0
    seg["focus"] = 50 + 100 * (b - seg["ratio_smooth"].fillna(seg["ratio"])) / (b + 1e-6)
    engaged_pct = 100.0 * (seg["ratio_smooth"].fillna(seg["ratio"]) < b).mean()
    motion_pct  = 100.0 * (seg.get("move_idx", pd.Series(0, index=seg.index)) >= MOVE_GATE).mean()
    hr_med = seg.get("hr_bpm_smooth", seg.get("hr_bpm", pd.Series(np.nan))).median()
    hr_txt = f"HR ~{hr_med:.0f} bpm" if pd.notna(hr_med) and hr_med > 0 else "HR unavailable"
    return [
        f"Focus was **{engaged_pct:.0f}% engaged** over the last {seconds}s.",
        f"Motion spikes (move_idx ≥ {MOVE_GATE:.02f}) for **{motion_pct:.0f}%** of the time.",
        f"{hr_txt}; try a **60s breathing** reset if focus dips again."
    ]

def fmt_reltime(ts: float) -> str:
    if not ts: return "—"
    dt = max(0, int(time.time() - ts))
    if dt < 60: return f"{dt}s ago"
    m, s = divmod(dt, 60)
    if m < 60: return f"{m}m {s}s ago"
    h, m = divmod(m, 60)
    return f"{h}h {m}m ago"

def latest_action(acts, typ):
    best = None
    for a in acts:
        if a.get("type") == typ:
            if (best is None) or float(a.get("ts", 0)) > float(best.get("ts", 0)):
                best = a
    return best

def latest_actions_by_type(actions, keep_types=("recap_last_minute","breathing_coach","focus_mode_on")):
    last = {}
    for a in actions:
        t = a.get("type")
        if keep_types and t not in keep_types:
            continue
        ts = float(a.get("ts", 0))
        if t not in last or ts > float(last[t].get("ts", -1)):
            last[t] = a
    return sorted(last.values(), key=lambda x: float(x.get("ts", 0)), reverse=True)

# ---------- Sidebar ----------
st.sidebar.header("Calibration")
c1, c2 = st.sidebar.columns(2, vertical_alignment="center")
if c1.button("Start 15s Calibration"):
    try:
        start_calibration()
        st.session_state["cal_end"] = time.time() + 15
    except Exception as e:
        st.sidebar.error(f"Start failed: {e}")
if c2.button("Stop & Set Baseline"):
    try:
        res = stop_calibration()
        b = res.get("baseline", {}).get("ratio")
        st.sidebar.success(f"Baseline set: {b:.3f}" if b else "Baseline set")
        st.session_state.pop("cal_end", None)
    except Exception as e:
        st.sidebar.error(f"Stop failed: {e}")

if "cal_end" in st.session_state:
    remain = max(0, int(st.session_state["cal_end"] - time.time()))
    st.sidebar.info(f"Collecting… auto-stop in {remain}s")
    if remain == 0:
        try: stop_calibration()
        except Exception: pass
        st.session_state.pop("cal_end", None)

st.sidebar.divider()
auto_refresh = st.sidebar.checkbox("Auto-refresh", True)
st.sidebar.caption(f"Agent: {AGENT_URL}")

# Demo triggers
st.sidebar.markdown("#### Demo (force actions)")
d1, d2, d3 = st.sidebar.columns([1, 1, 1.35], vertical_alignment="center")
if d1.button("Recap"):       httpx.post(f"{AGENT_URL}/demo/trigger", params={"action": "recap_last_minute"})
if d2.button("Breathe"):     httpx.post(f"{AGENT_URL}/demo/trigger", params={"action": "breathing_coach"})
if d3.button("Focus Mode"):  httpx.post(f"{AGENT_URL}/demo/trigger", params={"action": "focus_mode_on"})
st.sidebar.caption("Use these to demonstrate state changes without waiting for signals.")

# ---------- Data ----------
try:
    snap = fetch_snapshot(WINDOW_SEC)
except Exception as e:
    st.error(
        "Agent not reachable at "
        f"{AGENT_URL}. Start it with:\n\n"
        "```bash\nuv run uvicorn src.agent_orchestrator:app --port 8001 --reload\n```"
        f"\n\nError: {e}"
    )
    st.stop()

baseline = snap.get("baseline", {}) or {}
samps = snap.get("samples", []) or []
acts  = snap.get("actions", []) or []

# Build dataframe
if samps:
    df = pd.DataFrame(samps).sort_values("ts")
else:
    df = pd.DataFrame(columns=["ts","ratio","alpha","beta","conf","hr_bpm","hrv_rmssd_ms","move_idx","intent"])

# ---- Smoothing & motion-gating (UI-only) ----
if len(df):
    mask = df["move_idx"] >= MOVE_GATE if "move_idx" in df.columns else pd.Series(False, index=df.index)
    ratio_clean = df["ratio"].where(~mask, np.nan).interpolate(limit_direction="both")
    df["ratio_smooth"] = ratio_clean.rolling(5, min_periods=1).median().ewm(alpha=EMA_ALPHA, adjust=False).mean()
    if "hr_bpm" in df.columns:
        hr = df["hr_bpm"].replace(0, np.nan).interpolate(limit_direction="both")
        df["hr_bpm_smooth"] = hr.rolling(5, min_periods=1).median().ewm(alpha=EMA_ALPHA, adjust=False).mean()
    df["t"] = pd.to_datetime(df["ts"], unit="s")

# ---------- Main layout ----------
left, mid, right = st.columns([1, 1, 2])

# LEFT — Focus & Motion (+ Recap bullets + Intervention effect)
with left:
    st.markdown("##### Focus Level")
    focus_val, conf_val = 0, 0.0
    if len(df) and baseline.get("ratio"):
        r_latest = df.iloc[-1].get("ratio_smooth", np.nan)
        if pd.isna(r_latest): r_latest = df.iloc[-1].get("ratio", 0.0)
        r = float(r_latest); b = float(baseline["ratio"]) or 1.0
        focus_val = int(max(0, min(100, 50 + 100 * (b - r) / (b + 1e-6))))
        conf_val = float(df.iloc[-1].get("conf", 0.0))
    st.metric("Now", f"{focus_val}")
    st.progress(focus_val / 100)
    st.caption(f"Confidence: {conf_val:.2f}")
    if len(df) and baseline.get("ratio"):
        engaged_pct = 100.0 * (df["ratio_smooth"].fillna(df["ratio"]) < float(baseline["ratio"])).mean()
        st.caption(f"Engaged in last {WINDOW_SEC}s: **{engaged_pct:.0f}%**")

    with st.expander("Motion & Baseline", expanded=True):
        mov = float(df.iloc[-1].get("move_idx", 0.0)) if len(df) else 0.0
        mv_status = "still" if mov < 0.01 else ("slight motion" if mov < MOVE_GATE else "moving")
        c1, c2 = st.columns(2)
        c1.metric("Movement (last 2s)", f"{mov:.3f}", help=f"0 = still; ≥{MOVE_GATE:.2f} = noticeable motion")
        c1.caption(f"You're **{mv_status}**. High motion samples are de-emphasized in the chart.")
        if baseline.get("ratio") is not None:
            b_ratio = float(baseline["ratio"]); b_std = float(baseline.get("std_ratio") or 0.0); b_n = int(baseline.get("n") or 0)
            c2.metric("EEG baseline (α/β)", f"{b_ratio:.3f}")
            c2.caption("Lower α/β than baseline ⇒ **engaged**; higher ⇒ **relaxed**.")
            hr_b = baseline.get("hr_bpm")
            if hr_b: st.caption(f"Heart-rate baseline: **{hr_b:.0f} bpm** (auto-learned).")
            st.caption(f"Baseline samples: **{b_n}**, spread σ ≈ **{b_std:.3f}**.")
        else:
            st.info("No EEG baseline yet. Click **Start 15s Calibration**, or keep the headset still for ~20 s and we’ll auto-learn it.")

        # Recap bullets here (compact)
        recap = latest_action(acts, "recap_last_minute")
        if recap and len(df) and baseline.get("ratio"):
            st.markdown("<div class='section-subtitle'>Recap (last 60s)</div>", unsafe_allow_html=True)
            bullets = build_recap_bullets(df, float(baseline["ratio"]), float(recap.get("ts", df.iloc[-1]["ts"])))
            st.markdown(f"- {bullets[0]}\n- {bullets[1]}\n- {bullets[2]}")

        # Intervention effect shown ONCE here
        if acts and len(df) and baseline.get("ratio"):
            last_any = max(acts, key=lambda x: float(x.get("ts", 0)))
            delta = effect_around(df, float(last_any.get("ts", 0)), float(baseline["ratio"]))
            if delta is not None:
                st.caption(f"Intervention effect (last action): **{delta:+.1f} focus** over 20s window.")

# MID — Cardio + Recent Actions (latest per type, compact)
with mid:
    st.markdown("##### Cardio")
    hr_b = baseline.get("hr_bpm")
    hr_now = float(df.iloc[-1].get("hr_bpm_smooth", df.iloc[-1].get("hr_bpm", -1))) if len(df) else -1
    hrv_now = float(df.iloc[-1].get("hrv_rmssd_ms", -1)) if len(df) else -1
    m1, m2 = st.columns(2)
    m1.metric("Heart Rate", f"{hr_now:.0f} bpm" if hr_now > 0 else "—")
    m2.metric("HRV (RMSSD)", f"{hrv_now:.0f} ms" if hrv_now > 0 else "—")
    if hr_b: st.caption(f"HR baseline: {hr_b:.0f} bpm (auto-learned)")

    # Compact recent actions list
    st.markdown("<div class='section-subtitle'>Recent actions (latest per type)</div>", unsafe_allow_html=True)
    latest = latest_actions_by_type(acts)  # only focus_mode_on, breathing_coach, recap_last_minute
    st.markdown("<div class='recent-compact'>", unsafe_allow_html=True)
    if not latest:
        st.markdown("<div class='meta'>No interventions yet.</div>", unsafe_allow_html=True)
    else:
        for a in latest:
            t = a.get("type", "action"); when = fmt_reltime(float(a.get("ts", 0)))
            title = {"recap_last_minute":"Recap","breathing_coach":"Breathing Coach","focus_mode_on":"Focus Mode"}.get(t, t)
            st.markdown(f"<div class='title'>{title} · {when}</div>", unsafe_allow_html=True)
            why = a.get("why") or {}
            meta = (f"Δ%={why.get('delta_pct',0):.1f} · ratio={why.get('ratio',0):.3f} · "
                    f"baseline={why.get('baseline_ratio',0):.3f} · HR={why.get('hr_bpm','—')} · mov={why.get('move_idx','—')}")
            st.markdown(f"<div class='meta'>{meta}</div>", unsafe_allow_html=True)
            st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT — Timelines (Altair), PPG badge
with right:
    st.markdown("##### Timelines")
    if len(df):
        alpha_chart = alt.Chart(df).mark_line().encode(
            x=alt.X("t:T", title="Time"),
            y=alt.Y("ratio_smooth:Q", title="α/β (smoothed)"),
            tooltip=[alt.Tooltip("t:T", title="Time"),
                     alt.Tooltip("ratio_smooth:Q", title="α/β(s)"),
                     alt.Tooltip("move_idx:Q", title="motion")]
        ).properties(height=220)
        st.altair_chart(alpha_chart, use_container_width=True)

        if baseline.get("ratio"):
            b = float(baseline["ratio"]) or 1.0
            st.caption(f"EEG thresholds: mild<{0.85*b:.3f}, severe<{0.70*b:.3f}, relax>{1.10*b:.3f}")

        hr_series = None; hr_title = ""
        if "hr_bpm_smooth" in df.columns and df["hr_bpm_smooth"].notna().any():
            hr_series, hr_title = "hr_bpm_smooth", "Heart rate (smoothed, bpm)"
        elif "hr_bpm" in df.columns and df["hr_bpm"].replace(0, np.nan).notna().any():
            hr_series, hr_title = "hr_bpm", "Heart rate (bpm)"

        if hr_series:
            hr_chart = alt.Chart(df).mark_line().encode(
                x=alt.X("t:T", title="Time"),
                y=alt.Y(f"{hr_series}:Q", title=hr_title),
                tooltip=[alt.Tooltip("t:T", title="Time"),
                         alt.Tooltip(f"{hr_series}:Q", title="HR(bpm)")]
            ).properties(height=220)
            st.altair_chart(hr_chart, use_container_width=True)

            q = compute_ppg_quality(df); render_quality_badge(q)
            hr_count = int(df.get("hr_bpm", pd.Series(dtype=float)).replace(0, np.nan).notna().sum()) if "hr_bpm" in df.columns else 0
            st.caption(f"HR samples in window: {hr_count} • Motion gate threshold: move_idx ≥ {MOVE_GATE:.02f}")
        else:
            st.info("No heart-rate samples in the current window. Enable **Send PPG** in Mind Monitor and keep the app in the foreground.")
    else:
        st.info("Waiting for data…")

# ---------- Auto refresh ----------
if auto_refresh:
    time.sleep(POLL_SEC)
    st.rerun()

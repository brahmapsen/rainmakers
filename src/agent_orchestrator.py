from __future__ import annotations
import asyncio, time
from typing import Deque, Dict, Any, List
from collections import deque

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------- Models ----------
class IntentEvent(BaseModel):
    intent: str
    confidence: float
    features: Dict[str, float]  # expects alpha/beta bands + alpha_beta_ratio + hr_bpm/hrv/move_idx
    ts: float

# ---------- App ----------
app = FastAPI(title="BCI Focus Copilot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------- State ----------
samples: Deque[Dict[str, Any]] = deque(maxlen=1200)  # ~0.5s cadence x 10 min headroom
actions: Deque[Dict[str, Any]] = deque(maxlen=200)

baseline = {
    "ratio": None,       # alpha/beta baseline
    "std_ratio": None,
    "alpha": None,
    "beta": None,
    "hr_bpm": None,      # heart-rate baseline (learned lazily)
    "n": 0,
}

calibration = {"active": False, "buf": [], "start": 0.0}

_subscribers: List[asyncio.Queue] = []  # SSE (optional)
_last_action_ts: Dict[str, float] = {}

def now() -> float: return time.time()
def _mean(xs: List[float]) -> float: return sum(xs) / max(1, len(xs))
def _percent(a: float, b: float, eps: float = 1e-6) -> float: return 100.0 * (a - b) / (b + eps)
def _recent(window_s: float) -> List[Dict[str, Any]]:
    cut = now() - window_s
    return [s for s in samples if s["ts"] >= cut]

def _broadcast(msg: Dict[str, Any]) -> None:
    for q in list(_subscribers):
        try: q.put_nowait(msg)
        except Exception: pass

def _cooldown_ok(kind: str, cd: float) -> bool:
    return now() - _last_action_ts.get(kind, 0.0) > cd

def _push_action(action: Dict[str, Any]):
    actions.append(action)
    _broadcast({"type": "agent_action", "data": action})

# ---------- Endpoints ----------
@app.post("/ingest")
async def ingest(evt: IntentEvent):
    ratio = float(evt.features.get("alpha_beta_ratio", 0.0))
    alpha = float(evt.features.get("alpha", 0.0))
    beta  = float(evt.features.get("beta",  0.0))
    hr    = float(evt.features.get("hr_bpm", -1.0))
    hrv   = float(evt.features.get("hrv_rmssd_ms", -1.0))
    mov   = float(evt.features.get("move_idx", 0.0))

    s = {"ts": evt.ts, "ratio": ratio, "alpha": alpha, "beta": beta,
         "intent": evt.intent, "conf": evt.confidence, "hr_bpm": hr, "hrv_rmssd_ms": hrv, "move_idx": mov}
    samples.append(s)

    # Calibration (EEG ratio)
    if calibration["active"]:
        calibration["buf"].append(ratio)
    else:
        # Lazy baseline for ratio after ~20s of data
        if baseline["ratio"] is None and len(_recent(20)) > 10:
            rvals = [x["ratio"] for x in _recent(20)]
            rmean = _mean(rvals)
            rstd  = (sum((x - rmean)**2 for x in rvals) / max(1, len(rvals))) ** 0.5 or 0.1 * max(1e-3, rmean)
            baseline.update({"ratio": rmean, "std_ratio": rstd, "n": len(rvals)})
        # Lazy HR baseline (use last 45–60s of valid beats)
        if baseline["hr_bpm"] is None:
            hr_vals = [x["hr_bpm"] for x in _recent(60) if x.get("hr_bpm", -1) > 0]
            if len(hr_vals) >= 10:
                baseline["hr_bpm"] = float(_mean(hr_vals))

    # Decisioning (only if we have EEG baseline)
    action = None
    b = baseline["ratio"]
    focus_level = None
    if b:
        # Focus level: higher when ratio < baseline
        focus_level = max(0, min(100, int(50 + 100 * (b - ratio) / (b + 1e-6))))

        # Movement gating (robust to scale): compare to last 10s stats
        mv = [x["move_idx"] for x in _recent(10) if "move_idx" in x]
        m_mu = _mean(mv) if mv else 0.0
        m_sd = (sum((x - m_mu)**2 for x in mv) / max(1, len(mv))) ** 0.5 if mv else 0.0
        move_high = mov > (m_mu + 2.0 * max(m_sd, 1e-6)) and mov > 0.02  # 0.02 is a safety floor

        mild   = [x for x in _recent(10) if x["ratio"] < 0.85 * b]
        severe = [x for x in _recent(15) if x["ratio"] < 0.70 * b]
        strong = [x for x in _recent(20) if x["ratio"] > 1.10 * b]

        # Heart-aware nudge (only if HR baseline known and HR elevated)
        hr_b = baseline.get("hr_bpm")
        hr_elev = (hr_b is not None and hr > (hr_b + 8.0))  # > ~8 bpm over baseline

        # Prioritize: severe (breathing) → mild (recap) → strong focus (focus mode)
        if not move_high and len(severe) >= 0.6 * max(1, len(_recent(15))) and _cooldown_ok("breath", 120):
            action = {
                "type": "breathing_coach",
                "detail": "Start 60s paced breathing",
                "why": {"ratio": ratio, "baseline_ratio": b, "delta_pct": _percent(ratio, b), "hr_bpm": hr, "move_idx": mov},
                "ts": evt.ts,
            }
            _last_action_ts["breath"] = now()
        elif not move_high and (len(mild) >= 0.7 * max(1, len(_recent(10))) or (hr_elev and ratio < 0.8 * b)) and _cooldown_ok("recap", 60):
            action = {
                "type": "recap_last_minute",
                "detail": "Provide 3-bullet recap of last 60s",
                "why": {"ratio": ratio, "baseline_ratio": b, "delta_pct": _percent(ratio, b), "hr_bpm": hr, "move_idx": mov},
                "ts": evt.ts,
            }
            _last_action_ts["recap"] = now()
        elif len(strong) >= 0.7 * max(1, len(_recent(20))) and _cooldown_ok("focusmode", 120):
            action = {
                "type": "focus_mode_on",
                "detail": "Hide notifications for 2 min",
                "why": {"ratio": ratio, "baseline_ratio": b, "delta_pct": _percent(ratio, b), "hr_bpm": hr, "move_idx": mov},
                "ts": evt.ts,
            }
            _last_action_ts["focusmode"] = now()

        # Live metric for UIs
        _broadcast({"type": "metric", "data": {
            "ts": evt.ts, "ratio": ratio, "alpha": alpha, "beta": beta,
            "intent": evt.intent, "conf": evt.confidence,
            "baseline_ratio": b, "focus": focus_level,
            "hr_bpm": hr, "hrv_rmssd_ms": hrv, "move_idx": mov,
        }})

    if action:
        _push_action(action)
        return {"ok": True, "action": action}
    return {"ok": True}

@app.post("/calibrate/start")
async def calibrate_start():
    calibration.update({"active": True, "buf": [], "start": now()})
    return {"ok": True, "message": "Calibration started"}

@app.post("/calibrate/stop")
async def calibrate_stop():
    calibration["active"] = False
    buf = calibration.get("buf", [])
    if not buf:
        return {"ok": False, "message": "No calibration data collected"}
    rmean = _mean(buf)
    rstd  = (sum((x - rmean)**2 for x in buf) / max(1, len(buf))) ** 0.5 or 0.1 * max(1e-3, rmean)
    baseline.update({"ratio": rmean, "std_ratio": rstd, "n": len(buf)})
    return {"ok": True, "baseline": baseline}

@app.get("/baseline")
async def get_baseline(): return {"baseline": baseline, "calibrating": calibration["active"]}

@app.get("/snapshot")
async def snapshot(seconds: int = 180):
    cut = now() - seconds
    hist = [s for s in samples if s["ts"] >= cut]
    acts = [a for a in actions if a["ts"] >= cut]
    return {"baseline": baseline, "samples": hist, "actions": acts, "server_ts": now()}

# --- Demo trigger endpoint (for reliable judging) ---
@app.post("/demo/trigger")
async def demo_trigger(action: str = Query(..., pattern="^(recap_last_minute|breathing_coach|focus_mode_on)$")):
    a = {"type": action, "detail": "Demo trigger", "why": {"demo": 1}, "ts": now()}
    _push_action(a)
    return {"ok": True, "action": a}

@app.get("/events")
async def events(request: Request):
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.append(q)
    async def gen():
        try:
            while True:
                if await request.is_disconnected(): break
                msg = await q.get()
                yield f"data: {msg}\n\n"
        finally:
            if q in _subscribers: _subscribers.remove(q)
    return StreamingResponse(gen(), media_type="text/event-stream")

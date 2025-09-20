# agents/muse_bci/agent.py
import os
import time
import math
import json
import threading
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from scipy.signal import welch
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

"""
Muse 2 BCI Agent
- OSC listener for Mind Monitor (EEG/PPG/ACC/GYRO)
- α/β ratio, HR (bpm), HRV (RMSSD), motion index
- Baseline calibration + simple intent/arbitration
- HTTP interface: /meta, /invoke, /snapshot, /events (SSE)

Quick start:
  uv pip install -r agents/muse/requirements.txt
  uv run uvicorn agents.muse.main:app --port 8001 --reload

Point Mind Monitor OSC target to: <laptop-ip> : 5000
"""

# ---------- Config ----------
OSC_PORT = int(os.getenv("OSC_PORT", "5000"))
FS_EEG   = float(os.getenv("FS_EEG", "256"))     # Mind Monitor default is 256 Hz
WINDOW_SEC = int(os.getenv("WINDOW_SEC", "180")) # how much history we keep (seconds)
CAL_SEC    = int(os.getenv("CAL_SEC", "15"))
EMA_ALPHA  = float(os.getenv("EMA_ALPHA", "0.30"))
MOVE_GATE  = float(os.getenv("MOVE_GATE", "0.03"))

ALPHA_BAND = (8.0, 13.0)
BETA_BAND  = (13.0, 30.0)

# ---------- Data containers ----------
class Ring:
    def __init__(self, maxlen: int):
        self.buf: Deque = deque(maxlen=maxlen)
    def push(self, x): self.buf.append(x)
    def list(self):   return list(self.buf)
    def __len__(self): return len(self.buf)

def now_ts() -> float:
    return time.time()

# ---------- OSC receiver ----------
class MuseOSCReceiver:
    """
    Collects packets from Mind Monitor:
      /muse/eeg -> 4 floats (TP9, AF7, AF8, TP10)
      /muse/acc -> 3 floats (x,y,z)
      /muse/gyro -> 3 floats (x,y,z)
      /muse/ppg* -> raw/filtered PPG (optional)
      /muse/elements/heart_rate -> HR bpm
      /muse/elements/rrinterval -> RR ms (if available)
    """
    def __init__(self, port: int = OSC_PORT, window_sec: int = WINDOW_SEC):
        self.port = port

        # raw buffers
        self.eeg_tp9: Deque[float] = deque(maxlen=int(FS_EEG*window_sec))
        self.eeg_af7: Deque[float] = deque(maxlen=int(FS_EEG*window_sec))
        self.eeg_af8: Deque[float] = deque(maxlen=int(FS_EEG*window_sec))
        self.eeg_tp10: Deque[float] = deque(maxlen=int(FS_EEG*window_sec))

        self.acc: Ring   = Ring(maxlen=window_sec*64)   # ~64 Hz acc (varies)
        self.gyro: Ring  = Ring(maxlen=window_sec*64)
        self.ppg: Ring   = Ring(maxlen=window_sec*64)
        self.hr: Ring    = Ring(maxlen=window_sec*4)
        self.rr: Ring    = Ring(maxlen=window_sec*8)

        # derived timeline
        self.samples: Ring = Ring(maxlen=window_sec*2)  # ~2 Hz summarized

        # baseline
        self.baseline_ratio: Optional[float] = None
        self.baseline_std: Optional[float]   = None
        self.baseline_n: int = 0
        self.hr_baseline: Optional[float] = None

        # tracking
        self._ema_ratio: Optional[float] = None
        self._server: Optional[ThreadingOSCUDPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # action log (only last of each type kept by UI, but we keep history)
        self.actions: List[Dict] = []

    # ----- spectral helpers -----
    def _bandpower(self, data: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
        if data.size < int(fs * 1.0):  # need at least 1s
            return np.nan
        f, pxx = welch(data, fs=fs, nperseg=min(512, data.size))
        lo, hi = band
        mask = (f >= lo) & (f <= hi)
        if mask.sum() == 0:
            return np.nan
        return float(np.trapz(pxx[mask], f[mask]))

    def _compute_motion(self) -> float:
        # Use rolling RMS of derivative of acc magnitude (~0.5s)
        if len(self.acc) < 3: return 0.0
        arr = np.array([a["mag"] for a in self.acc.list()[-32:]], dtype=float)
        if arr.size < 3: return 0.0
        d = np.diff(arr)
        rms = float(np.sqrt(np.mean(d*d)))
        # normalize roughly to small numbers; MOVE_GATE is tuned against this
        return float(rms)

    def _compute_hrv_rmssd(self) -> Optional[float]:
        rr = [x["rr_ms"] for x in self.rr.list() if x.get("rr_ms")]
        if len(rr) < 3: return None
        d = np.diff(rr)
        return float(np.sqrt(np.mean(d*d)))

    # ----- OSC handlers -----
    def _on_eeg(self, addr, *args):
        # args: [TP9, AF7, AF8, TP10] (floats)
        ts = now_ts()
        vals = [float(x) for x in args[:4]]
        tp9, af7, af8, tp10 = vals
        self.eeg_tp9.append(tp9)
        self.eeg_af7.append(af7)
        self.eeg_af8.append(af8)
        self.eeg_tp10.append(tp10)

        # quick α/β using frontal channels (AF7/AF8)
        af = np.array(self.eeg_af7, dtype=float)
        bf = np.array(self.eeg_af8, dtype=float)
        # concatenate L/R for a little robustness
        if af.size and bf.size:
            x = np.vstack([af[-1024:], bf[-1024:]]).mean(axis=0)  # last ~4s at 256 Hz
        else:
            return

        alpha = self._bandpower(x, FS_EEG, ALPHA_BAND)
        beta  = self._bandpower(x, FS_EEG, BETA_BAND)
        if math.isnan(alpha) or math.isnan(beta):
            return
        ratio = alpha / (beta + 1e-9)

        # EMA smoothing
        if self._ema_ratio is None:
            self._ema_ratio = ratio
        else:
            self._ema_ratio = EMA_ALPHA*ratio + (1-EMA_ALPHA)*self._ema_ratio

        # infer HR/HRV/motion snapshot
        hr_bpm = self.hr.list()[-1]["bpm"] if len(self.hr) > 0 else None
        hrv = self._compute_hrv_rmssd()
        mov = self._compute_motion()

        # baseline auto-learn (if not calibrated, gently adapt)
        if self.baseline_ratio is None and len(self.samples) > 30:
            # median over last 30 samples (~15s) as a starting baseline
            med = float(np.nanmedian([s["ratio"] for s in self.samples.list()[-30:]]))
            if math.isfinite(med):
                self.baseline_ratio = med
                self.baseline_std = float(np.nanstd([s["ratio"] for s in self.samples.list()[-30:]]))
                self.baseline_n = 30

        # intent gating
        intent = None
        conf = 0.5
        if self.baseline_ratio:
            b = self.baseline_ratio
            r = self._ema_ratio
            # lower ratio -> more focus
            if r < 0.70*b:
                intent, conf = "engaged", 0.8
            elif r < 0.90*b:
                intent, conf = "engaged", 0.6
            elif r > 1.10*b:
                intent, conf = "relaxed", 0.6
            else:
                intent, conf = "neutral", 0.5

        # record summarized sample at ~2 Hz
        if (len(self.samples) == 0) or (ts - self.samples.list()[-1]["ts"] >= 0.5):
            self.samples.push({
                "ts": ts,
                "ratio": float(self._ema_ratio),
                "alpha": float(alpha),
                "beta": float(beta),
                "conf": conf,
                "hr_bpm": hr_bpm if hr_bpm else None,
                "hrv_rmssd_ms": hrv if hrv else None,
                "move_idx": float(mov),
                "intent": intent
            })

    def _on_acc(self, addr, *args):
        ts = now_ts()
        x, y, z = [float(a) for a in args[:3]]
        mag = math.sqrt(x*x + y*y + z*z)
        self.acc.push({"ts": ts, "x": x, "y": y, "z": z, "mag": mag})

    def _on_gyro(self, addr, *args):
        ts = now_ts()
        x, y, z = [float(a) for a in args[:3]]
        self.gyro.push({"ts": ts, "x": x, "y": y, "z": z})

    def _on_ppg_any(self, addr, *args):
        ts = now_ts()
        self.ppg.push({"ts": ts, "addr": addr, "values": [float(a) for a in args]})

    def _on_hr(self, addr, *args):
        ts = now_ts()
        bpm = float(args[0])
        self.hr.push({"ts": ts, "bpm": bpm})
        # gentle HR baseline learn
        if self.hr_baseline is None:
            self.hr_baseline = bpm
        else:
            self.hr_baseline = 0.98*self.hr_baseline + 0.02*bpm

    def _on_rr(self, addr, *args):
        ts = now_ts()
        rr_ms = float(args[0])  # milliseconds between beats
        self.rr.push({"ts": ts, "rr_ms": rr_ms})

    # ----- server control -----
    def start(self, bind_ip: str = "0.0.0.0"):
        if self._server:
            return
        disp = Dispatcher()
        disp.map("/muse/eeg", self._on_eeg)
        disp.map("/muse/acc", self._on_acc)
        disp.map("/muse/gyro", self._on_gyro)
        disp.map("/muse/ppg*", self._on_ppg_any)  # raw/filtered etc.
        disp.map("/muse/elements/heart_rate", self._on_hr)
        disp.map("/muse/elements/rrinterval", self._on_rr)

        self._server = ThreadingOSCUDPServer((bind_ip, self.port), disp)
        self._stop.clear()
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        self._server = None
        self._thread = None
        self._stop.set()

    def calibrate_start(self):
        # mark the time; the UI will call calibrate_stop after CAL_SEC
        self._cal_start = now_ts()

    def calibrate_stop(self):
        # use last CAL_SEC seconds of samples
        t0 = now_ts() - CAL_SEC
        win = [s["ratio"] for s in self.samples.list() if s["ts"] >= t0 and math.isfinite(s["ratio"])]
        if len(win) >= 10:
            self.baseline_ratio = float(np.median(win))
            self.baseline_std = float(np.std(win))
            self.baseline_n = len(win)
        return {
            "baseline": {
                "ratio": self.baseline_ratio,
                "std_ratio": self.baseline_std,
                "n": self.baseline_n,
                "hr_bpm": self.hr_baseline
            }
        }

    def snapshot(self, seconds: int = WINDOW_SEC) -> Dict:
        t0 = now_ts() - seconds
        samples = [s for s in self.samples.list() if s["ts"] >= t0]
        return {
            "baseline": {
                "ratio": self.baseline_ratio,
                "std_ratio": self.baseline_std,
                "n": self.baseline_n,
                "hr_bpm": self.hr_baseline,
            },
            "samples": samples,
            "actions": self.actions[-50:],  # history cap
        }

    def demo_trigger(self, action: str):
        ts = now_ts()
        detail = ""
        if action == "breathing_coach":
            detail = "Guide 4-6 breathing for 60s"
        elif action == "recap_last_minute":
            detail = "Summarize last 60s (signals)"
        elif action == "focus_mode_on":
            detail = "Silence notifications for 2 minutes"
        self.actions.append({
            "ts": ts,
            "type": action,
            "detail": detail,
            "why": {
                "ratio": (self.samples.list()[-1]["ratio"] if len(self.samples) else 0.0),
                "baseline_ratio": self.baseline_ratio or 0.0,
                "hr_bpm": (self.hr.list()[-1]["bpm"] if len(self.hr) else None),
                "move_idx": self._compute_motion(),
                "delta_pct": 0.0
            }
        })


receiver = MuseOSCReceiver()

# ---------- FastAPI ----------
app = FastAPI(title="Muse 2 BCI Agent")

class InvokeBody(BaseModel):
    name: str
    params: Optional[dict] = None

@app.get("/meta")
def meta():
    return {
        "name": "muse2_bci",
        "display_name": "Muse 2 BCI Agent",
        "version": "0.1.0",
        "description": "Reads EEG/PPG/ACC/GYRO from Mind Monitor (OSC), computes α/β, HR, HRV, motion, and exposes focus/relax intent.",
        "endpoints": {
            "invoke": "/invoke",
            "snapshot": "/snapshot",
            "events": "/events"
        },
        "tools": [
            {"name": "start", "description": "Start OSC listener", "input_schema": {"type":"object","properties":{"bind_ip":{"type":"string"}}}},
            {"name": "stop", "description": "Stop OSC listener"},
            {"name": "snapshot", "description": "Return recent samples/baseline/actions", "input_schema":{"type":"object","properties":{"seconds":{"type":"integer"}}}},
            {"name": "calibrate_start", "description": f"Begin {CAL_SEC}s baseline capture"},
            {"name": "calibrate_stop",  "description": "Compute baseline from last window"},
            {"name": "demo_trigger", "description":"Force an intervention", "input_schema":{"type":"object","properties":{"action":{"type":"string","enum":["recap_last_minute","breathing_coach","focus_mode_on"]}}}}
        ]
    }

@app.post("/invoke")
def invoke(body: InvokeBody):
    name = body.name
    p = body.params or {}
    try:
        if name == "start":
            receiver.start(bind_ip=p.get("bind_ip","0.0.0.0"))
            return {"ok": True, "msg": f"Listening on udp://0.0.0.0:{OSC_PORT}"}
        if name == "stop":
            receiver.stop(); return {"ok": True}
        if name == "snapshot":
            seconds = int(p.get("seconds", WINDOW_SEC))
            return receiver.snapshot(seconds=seconds)
        if name == "calibrate_start":
            receiver.calibrate_start(); return {"ok": True, "msg": f"Collecting {CAL_SEC}s baseline"}
        if name == "calibrate_stop":
            return receiver.calibrate_stop()
        if name == "demo_trigger":
            action = p.get("action")
            receiver.demo_trigger(str(action))
            return {"ok": True}
        return JSONResponse({"error": f"unknown tool '{name}'"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": repr(e)}, status_code=500)

@app.get("/snapshot")
def snapshot(seconds: int = WINDOW_SEC):
    return receiver.snapshot(seconds=seconds)

# Simple SSE stream (optional)
@app.get("/events")
async def events(request: Request):
    async def gen():
        last_ts = 0.0
        while True:
            if await request.is_disconnected():
                break
            snap = receiver.snapshot(seconds=30)
            smp = snap["samples"]
            if smp:
                cur_ts = smp[-1]["ts"]
                if cur_ts != last_ts:
                    last_ts = cur_ts
                    yield f"data: {json.dumps(snap)}\n\n"
            await asyncio.sleep(1.0)
    import asyncio
    return PlainTextResponse(gen(), media_type="text/event-stream")

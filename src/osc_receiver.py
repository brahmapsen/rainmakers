# Mind Monitor OSC -> features -> agent (EEG + optional PPG & IMU), with raw debug prints
import asyncio, os, time
import numpy as np
import httpx
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer
from features import extract_bandpowers  # your Welch bandpower helper

# -------- Config --------
OSC_PORT  = int(os.getenv("OSC_PORT", 5000))
AGENT_URL = os.getenv("AGENT_URL", "http://127.0.0.1:8001/ingest")

FS_EEG    = int(os.getenv("EEG_FS", 256))   # Muse raw EEG ~256 Hz
FS_PPG    = int(os.getenv("PPG_FS", 64))    # PPG ~64 Hz (if enabled in Mind Monitor)
FS_IMU    = int(os.getenv("IMU_FS", 52))    # accel/gyro ~52 Hz

WIN_SEC   = float(os.getenv("WIN_SEC", 2.0))
STEP_SEC  = float(os.getenv("STEP_SEC", 0.5))

# Raw-print controls
DEBUG_RAW_LIMIT = int(os.getenv("DEBUG_RAW_LIMIT", "8"))   # print first N packets per stream
DEBUG_SNAPSHOT  = os.getenv("DEBUG_SNAPSHOT", "0") != "0"  # print latest raw snapshot every step

# -------- Buffers --------
# EEG: 4 channels (TP9, AF7, AF8, TP10)
eeg_buf = np.zeros((4, int(FS_EEG * WIN_SEC * 2)), dtype=np.float32)
eeg_idx = 0

# PPG: ~10s ring
ppg_buf = np.zeros(int(FS_PPG * 10), dtype=np.float32)
ppg_idx = 0

# IMU: ~10s rings
acc_buf  = np.zeros((3, int(FS_IMU * 10)), dtype=np.float32); acc_idx  = 0
gyro_buf = np.zeros((3, int(FS_IMU * 10)), dtype=np.float32); gyro_idx = 0

# Keep last raw samples for snapshot printing
last_eeg  = None  # np.array([TP9,AF7,AF8,TP10])
last_ppg  = None  # float
last_acc  = None  # np.array([ax,ay,az])
last_gyro = None  # np.array([gx,gy,gz])

# Per-stream debug counters
_dbg_eeg = 0
_dbg_ppg = 0
_dbg_acc = 0
_dbg_gyro= 0

# -------- Handlers --------
def eeg_handler(addr, *vals):
    """ /muse/eeg -> (TP9, AF7, AF8, TP10, [AUX]) """
    global eeg_idx, eeg_buf, last_eeg, _dbg_eeg
    if len(vals) < 4:
        return
    pos = eeg_idx % eeg_buf.shape[1]
    v = np.array(vals[:4], dtype=np.float32)
    eeg_buf[:, pos] = v
    eeg_idx += 1
    last_eeg = v
    # Raw debug prints (first N packets only)
    if _dbg_eeg < DEBUG_RAW_LIMIT:
        # Show in a convenient reading order: AF7, AF8, TP9, TP10
        print(f"[RAW][EEG] AF7={v[1]:.1f} AF8={v[2]:.1f} TP9={v[0]:.1f} TP10={v[3]:.1f}")
        _dbg_eeg += 1
        if _dbg_eeg == DEBUG_RAW_LIMIT:
            print("[RAW][EEG] (silencing further raw prints; set DEBUG_RAW_LIMIT to change)")

def ppg_handler(addr, *vals):
    """ /muse/ppg (take first value) """
    global ppg_idx, ppg_buf, last_ppg, _dbg_ppg
    if not vals:
        return
    p = float(vals[0])
    ppg_buf[ppg_idx % ppg_buf.size] = p
    ppg_idx += 1
    last_ppg = p
    if _dbg_ppg < DEBUG_RAW_LIMIT:
        print(f"[RAW][PPG] ppg={p:.1f}")
        _dbg_ppg += 1
        if _dbg_ppg == DEBUG_RAW_LIMIT:
            print("[RAW][PPG] (silencing further raw prints; set DEBUG_RAW_LIMIT to change)")

def acc_handler(addr, *vals):
    """ /muse/acc -> (x,y,z) """
    global acc_idx, acc_buf, last_acc, _dbg_acc
    if len(vals) < 3:
        return
    pos = acc_idx % acc_buf.shape[1]
    a = np.array(vals[:3], dtype=np.float32)
    acc_buf[:, pos] = a
    acc_idx += 1
    last_acc = a
    if _dbg_acc < DEBUG_RAW_LIMIT:
        print(f"[RAW][ACC] ax={a[0]:.3f} ay={a[1]:.3f} az={a[2]:.3f}")
        _dbg_acc += 1
        if _dbg_acc == DEBUG_RAW_LIMIT:
            print("[RAW][ACC] (silencing further raw prints; set DEBUG_RAW_LIMIT to change)")

def gyro_handler(addr, *vals):
    """ /muse/gyro -> (x,y,z) """
    global gyro_idx, gyro_buf, last_gyro, _dbg_gyro
    if len(vals) < 3:
        return
    pos = gyro_idx % gyro_buf.shape[1]
    g = np.array(vals[:3], dtype=np.float32)
    gyro_buf[:, pos] = g
    gyro_idx += 1
    last_gyro = g
    if _dbg_gyro < DEBUG_RAW_LIMIT:
        print(f"[RAW][GYRO] gx={g[0]:.3f} gy={g[1]:.3f} gz={g[2]:.3f}")
        _dbg_gyro += 1
        if _dbg_gyro == DEBUG_RAW_LIMIT:
            print("[RAW][GYRO] (silencing further raw prints; set DEBUG_RAW_LIMIT to change)")

# -------- Helpers for features --------
def _movement_index_from_acc(last_seconds=2.0) -> float:
    if acc_idx <= FS_IMU * last_seconds:
        return 0.0
    end = acc_idx % acc_buf.shape[1]
    start = (end - int(FS_IMU * last_seconds)) % acc_buf.shape[1]
    win = np.hstack([acc_buf[:, start:], acc_buf[:, :end]]) if start > end else acc_buf[:, start:end]
    mag = np.linalg.norm(win, axis=0)
    return float(np.mean(np.abs(np.diff(mag))))

def _hr_hrv_from_ppg(window_seconds=8.0):
    """Very crude HR/HRV from PPG peaks; motion-sensitive."""
    if ppg_idx <= FS_PPG * window_seconds:
        return -1.0, -1.0
    end = ppg_idx % ppg_buf.size
    start = (end - int(FS_PPG * window_seconds)) % ppg_buf.size
    win = np.hstack([ppg_buf[start:], ppg_buf[:end]]) if start > end else ppg_buf[start:end]
    win = win - np.mean(win)
    peaks = (win[1:-1] > win[:-2]) & (win[1:-1] > win[2:])
    pk = np.nonzero(peaks)[0] + 1
    if len(pk) < 4:
        return -1.0, -1.0
    rr = np.diff(pk) / FS_PPG
    rr = rr[(rr > 0.3) & (rr < 2.0)]  # 30–200 bpm plausible
    if len(rr) < 3:
        return -1.0, -1.0
    hr_bpm = float(60.0 / np.median(rr))
    hrv_rmssd_ms = float(np.sqrt(np.mean(np.square(np.diff(rr)))) * 1000.0)
    return hr_bpm, hrv_rmssd_ms

# -------- Periodic compute & print --------
async def periodic():
    last = 0.0
    async with httpx.AsyncClient(timeout=3.0) as client:
        while True:
            await asyncio.sleep(0.05)
            tnow = time.time()
            if tnow - last < STEP_SEC:
                continue
            last = tnow

            # need at least one analysis window of EEG
            need = int(FS_EEG * WIN_SEC)
            if eeg_idx < need:
                continue

            # take last WIN_SEC of EEG
            end = eeg_idx % eeg_buf.shape[1]
            start = (end - need) % eeg_buf.shape[1]
            win_eeg = (np.hstack([eeg_buf[:, start:], eeg_buf[:, :end]])
                       if start > end else eeg_buf[:, start:end])

            feats = extract_bandpowers(win_eeg, FS_EEG)  # expects alpha, beta, alpha_beta_ratio
            ratio = float(feats["alpha_beta_ratio"])
            intent = "relaxed" if ratio > 1.0 else "engaged"
            conf = float(min(0.95, max(0.55,
                        abs(feats["alpha"] - feats["beta"]) / (feats["alpha"] + feats["beta"] + 1e-6))))

            move_idx = _movement_index_from_acc(2.0)
            hr_bpm, hrv_rmssd = _hr_hrv_from_ppg(8.0)

            # POST to agent
            payload = {
                "intent": intent,
                "confidence": conf,
                "features": {**feats,
                             "move_idx": float(move_idx),
                             "hr_bpm": float(hr_bpm),
                             "hrv_rmssd_ms": float(hrv_rmssd)},
                "ts": float(tnow),
            }
            try:
                await client.post(AGENT_URL, json=payload)
            except Exception as e:
                print("[osc_receiver] POST failed:", e)

            # -------- Console display --------
            # Optional raw snapshot (most recent values)
            raw_snapshot = ""
            if DEBUG_SNAPSHOT:
                if last_eeg is not None:
                    raw_snapshot += (f" | EEG(AF7={last_eeg[1]:.1f},AF8={last_eeg[2]:.1f},"
                                     f"TP9={last_eeg[0]:.1f},TP10={last_eeg[3]:.1f})")
                if last_ppg is not None:
                    raw_snapshot += f" | PPG({last_ppg:.1f})"
                if last_acc is not None:
                    raw_snapshot += f" | ACC(ax={last_acc[0]:.3f},ay={last_acc[1]:.3f},az={last_acc[2]:.3f})"
                if last_gyro is not None:
                    raw_snapshot += f" | GYRO(gx={last_gyro[0]:.3f},gy={last_gyro[1]:.3f},gz={last_gyro[2]:.3f})"

            # Single concise status line per step
            print(
                f"[osc_receiver] {time.strftime('%H:%M:%S')} "
                f"intent={intent:7s}  α/β={ratio:0.2f}  α={feats['alpha']:6.1f}  β={feats['beta']:6.1f}  "
                f"mov={move_idx:0.3f}  HR={(hr_bpm if hr_bpm>0 else float('nan')):>5.1f}  "
                f"HRV={(hrv_rmssd if hrv_rmssd>0 else float('nan')):>5.0f}ms  conf={conf:0.2f}"
                f"{raw_snapshot}"
            )

# -------- Main --------
async def main():
    loop = asyncio.get_event_loop()
    disp = Dispatcher()
    # EEG (both forms)
    disp.map("/muse/eeg", eeg_handler)
    disp.map("/muse/eeg/*", eeg_handler)
    # Extras (enable in Mind Monitor settings)
    disp.map("/muse/ppg",  ppg_handler);  disp.map("/muse/ppg/*",  ppg_handler)
    disp.map("/muse/acc",  acc_handler);  disp.map("/muse/acc/*",  acc_handler)
    disp.map("/muse/gyro", gyro_handler); disp.map("/muse/gyro/*", gyro_handler)

    server = AsyncIOOSCUDPServer(("0.0.0.0", OSC_PORT), disp, loop)
    transport, protocol = await server.create_serve_endpoint()
    print(f"[osc_receiver] Listening on udp://0.0.0.0:{OSC_PORT}")
    try:
        await periodic()
    finally:
        transport.close()

if __name__ == "__main__":
    asyncio.run(main())

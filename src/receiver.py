# Subscribe to LSL EEG, compute bandpowers every 0.5s window, classify, POST to agent
import os, time, json
import numpy as np
import httpx
from pylsl import StreamInlet, resolve_byprop, resolve_streams
from features import extract_bandpowers

STREAM_NAME = os.getenv("STREAM_NAME", "SYNTH-EEG")
AGENT_URL = os.getenv("AGENT_URL", "http://127.0.0.1:8001/ingest")
WIN_SEC = float(os.getenv("WIN_SEC", 2.0))
STEP_SEC = float(os.getenv("STEP_SEC", 0.5))

print(f"[receiver] Resolving LSL stream name={STREAM_NAME}...")
streams = resolve_byprop('name', STREAM_NAME, timeout=5)
if not streams:
    print("Resolving stream")
    streams = [s for s in resolve_streams() if s.name() == STREAM_NAME]
if not streams:
    raise RuntimeError(f"No LSL stream named {STREAM_NAME} found. Start synth_lsl.py first.")

inlet = StreamInlet(streams[0], max_buflen=60)
fs = inlet.info().nominal_srate()
ch = inlet.info().channel_count()
print(f"[receiver] Connected: {ch} channels @ {fs} Hz")

buf = np.zeros((ch, int(WIN_SEC*fs*2)), dtype=np.float32) # rolling buffer (2× window)
write_idx = 0

client = httpx.Client(timeout=3.0)

last_ts = time.time()

while True:
    samples, timestamps = inlet.pull_chunk(timeout=1.0)
    if not samples:
        continue
    arr = np.array(samples, dtype=np.float32).T # shape (n_ch, n_samples)
    n = arr.shape[1]
    if write_idx + n <= buf.shape[1]:
        buf[:, write_idx:write_idx+n] = arr
        write_idx += n
    else:
        # wrap
        first = buf.shape[1] - write_idx
        buf[:, write_idx:] = arr[:, :first]
        buf[:, :n-first] = arr[:, first:]
        write_idx = (write_idx + n) % buf.shape[1]

    # every STEP_SEC, evaluate last WIN_SEC
    now = time.time()
    if now - last_ts < STEP_SEC:
        continue
    last_ts = now

    # slice the most recent window
    end = write_idx
    start = (end - int(WIN_SEC*fs)) % buf.shape[1]
    if start < end:
        win = buf[:, start:end]
    else:
        win = np.concatenate([buf[:, start:], buf[:, :end]], axis=1)

    feats = extract_bandpowers(win, fs)
    # naive intent
    intent = "relaxed" if feats["alpha_beta_ratio"] > 1.0 else "engaged"
    conf = float(min(0.95, max(0.55, abs(feats["alpha"] - feats["beta"]) / (feats["alpha"] + feats["beta"] + 1e-6))))

    payload = {
        "intent": intent,
        "confidence": conf,
        "features": feats,
        "ts": time.time(),
    }
    try:
        client.post(AGENT_URL, json=payload)
        print(f"[receiver] intent={intent:7s} conf={conf:.2f} α/β={feats['alpha_beta_ratio']:.2f}")
    except Exception as e:
        print("[receiver] POST failed:", e)
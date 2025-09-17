# Synthetic EEG → LSL stream with HTTP control (relaxed vs engaged)
# Synthetic EEG → LSL stream with HTTP control (relaxed vs engaged)
import threading, time, math, os
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
from fastapi import FastAPI
import uvicorn

FS = int(os.getenv("FS", 256)) # sample rate
N_CH = int(os.getenv("N_CH", 8)) # channels
NAME = os.getenv("STREAM_NAME", "SYNTH-EEG")
TYPE = "EEG"
STATE = {"mode": os.getenv("MODE", "relaxed")} # relaxed | engaged

# Channel labels (8 typical positions)
CHAN_LABELS = ["CP3","C3","F5","PO3","PO4","F6","C4","CP4"][:N_CH]

# Create LSL outlet
info = StreamInfo(NAME, TYPE, N_CH, FS, 'float32', 'synth-eeg-001')
chns = info.desc().append_child("channels")
for label in CHAN_LABELS:
    ch = chns.append_child("channel")
    ch.append_child_value("label", label)
    ch.append_child_value("unit", "uV")
    ch.append_child_value("type", "EEG")
outlet = StreamOutlet(info, chunk_size=FS//8, max_buffered=FS*10)

# Signal generator
class SynthEEG:
    def __init__(self, fs, n_ch):
        self.fs = fs
        self.n_ch = n_ch
        self.t = 0
        self.dt = 1.0/fs
        # Random phases per channel
        self.phase_alpha = np.random.rand(n_ch) * 2*np.pi
        self.phase_beta = np.random.rand(n_ch) * 2*np.pi

    def step_block(self, block=32):
        global STATE
        t_vec = self.t + np.arange(block) * self.dt
        # Modes: relaxed → strong alpha (~10 Hz); engaged → strong beta (~20 Hz)
        if STATE["mode"] == "relaxed":
            A_alpha, A_beta = 12.0, 4.0
        else:
            A_alpha, A_beta = 4.0, 12.0
        # Base sinusoids per channel with slight frequency jitter
        alpha_f = 10.0 + np.random.randn(self.n_ch)*0.1
        beta_f = 20.0 + np.random.randn(self.n_ch)*0.2
        sig = []
        for ch in range(self.n_ch):
            a = A_alpha * np.sin(2*np.pi*alpha_f[ch]*t_vec + self.phase_alpha[ch])
            b = A_beta * np.sin(2*np.pi*beta_f[ch]*t_vec + self.phase_beta[ch])
            noise = np.random.randn(block) * 1.5
            line = 0.3*np.sin(2*np.pi*60*t_vec) # mild line noise
            sig.append(a + b + noise + line)
        self.t += block*self.dt
        return np.array(sig, dtype=np.float32) # shape (n_ch, block)


synth = SynthEEG(FS, N_CH)

# Streaming thread
_running = True

def stream_loop():
    while _running:
        chunk = synth.step_block(block=32)
        # LSL expects samples as list-of-channels; push_chunk supports 2D (n_ch x n_samp)
        outlet.push_chunk(chunk.tolist(), local_clock())
        time.sleep(32/FS * 0.9)

thr = threading.Thread(target=stream_loop, daemon=True)
thr.start()

# HTTP control API
app = FastAPI()

@app.get("/state")
def get_state():
    return {"mode": STATE["mode"]}

@app.post("/state/{mode}")
def set_state(mode: str):
    mode = mode.lower()
    if mode not in ("relaxed", "engaged"):
        return {"ok": False, "error": "mode must be relaxed|engaged"}
    STATE["mode"] = mode
    return {"ok": True, "mode": mode}

if __name__ == "__main__":
    print(f"[synth_lsl] Streaming {NAME} ({N_CH} ch @ {FS} Hz). Control at POST /state/<relaxed|engaged>.")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SYNTH_PORT", 7000)))
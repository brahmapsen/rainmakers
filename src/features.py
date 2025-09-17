## src/features.py
import numpy as np
from scipy.signal import welch

def bandpower(f, Pxx, fmin, fmax):
    idx = (f >= fmin) & (f <= fmax)
    if not np.any(idx):
        return 0.0
    return float(np.trapz(Pxx[idx], f[idx]))

def extract_bandpowers(eeg_win, fs):
    """eeg_win: (n_ch, n_samples) → returns dict of bandpowers averaged across channels."""
    bands = [(1,4,'delta'), (4,8,'theta'), (8,13,'alpha'), (13,30,'beta'), (30,45,'gamma')]
    per_ch = []
    for ch in range(eeg_win.shape[0]):
        f, Pxx = welch(eeg_win[ch], fs=fs, nperseg=min(256, eeg_win.shape[1]))
        per_ch.append({name: bandpower(f,Pxx,lo,hi) for lo,hi,name in bands})
    # aggregate across channels
    agg = {k: float(np.mean([d[k] for d in per_ch])) for k in per_ch[0].keys()}
    # convenience ratios
    agg["alpha_beta_ratio"] = agg["alpha"] / (agg["beta"] + 1e-6)
    return agg

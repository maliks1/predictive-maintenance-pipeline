import numpy as np
import pandas as pd

# Constants from Notebook
CHANNEL_NAMES = ["smc", "smd", "vib_table", "vib_spindle", "ae_table", "ae_spindle"]
SIGNAL_LENGTH = 9000
FEATURE_CLIP = 1_000_000.0
EPS = 1e-6
CHANNEL_STATS = [
    "mean", "std", "rms", "abs_mean", "peak", "p2p", 
    "crest", "energy", "skew", "kurt", "zero_cross"
]

def safe_float(value):
    try:
        value = float(value)
        if not np.isfinite(value):
            return np.nan
        return float(np.clip(value, -FEATURE_CLIP, FEATURE_CLIP))
    except Exception:
        return np.nan

def clean_features(X):
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.clip(lower=-FEATURE_CLIP, upper=FEATURE_CLIP)
        return X.astype(np.float64)
    
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    X = np.clip(X, -FEATURE_CLIP, FEATURE_CLIP)
    return X

def compute_channel_features(sig, prefix):
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    sig = sig[np.isfinite(sig)]
    feats = {}

    if len(sig) < 10:
        for stat in CHANNEL_STATS:
            feats[f"{prefix}_{stat}"] = np.nan
        return feats

    sig = np.clip(sig, -FEATURE_CLIP, FEATURE_CLIP)
    
    mean = float(np.mean(sig))
    std = float(np.std(sig))
    rms = float(np.sqrt(np.mean(sig**2)))
    abs_mean = float(np.mean(np.abs(sig)))
    peak = float(np.max(np.abs(sig)))
    p2p = float(np.max(sig) - np.min(sig))
    crest = peak / rms if rms > EPS else 0.0
    energy = float(np.mean(sig**2))

    if std > EPS:
        z = (sig - mean) / std
        skew = float(np.mean(z**3))
        kurt = float(np.mean(z**4) - 3.0)
    else:
        skew = 0.0
        kurt = 0.0

    zero_cross = float(np.sum(np.diff(np.sign(sig - mean)) != 0))

    values = [mean, std, rms, abs_mean, peak, p2p, crest, energy, skew, kurt, zero_cross]
    for stat, val in zip(CHANNEL_STATS, values):
        feats[f"{prefix}_{stat}"] = safe_float(val)

    return feats

def extract_features_from_signals(signals, metadata=None):
    if len(signals) != len(CHANNEL_NAMES):
        raise ValueError(f"Number of signals must be {len(CHANNEL_NAMES)}, but received {len(signals)}")

    feats = {}
    for ch, sig in zip(CHANNEL_NAMES, signals):
        feats.update(compute_channel_features(sig, ch))

    metadata = metadata or {}
    feats["DOC"] = safe_float(metadata.get("DOC", np.nan))
    feats["Feed"] = safe_float(metadata.get("Feed", np.nan))
    feats["Material"] = safe_float(metadata.get("Material", np.nan))

    return feats
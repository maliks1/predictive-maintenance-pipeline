"""
Data loader for daily sensor data.

Expects a .mat file with the same structure as the training dataset
(data/mill.mat): a struct array named "mill" where each run contains,
in order:
    0: Case, 1: Run, 2: VB (target, unused here), 3: Time,
    4: DOC, 5: Feed, 6: Material,
    7-12: sensor signals (smc, smd, vib_table, vib_spindle, ae_table, ae_spindle)
"""

from pathlib import Path

import numpy as np
import scipy.io as sio

CASE_INDEX = 0
RUN_INDEX = 1
DOC_INDEX = 4
FEED_INDEX = 5
MATERIAL_INDEX = 6
SENSOR_INDICES = [7, 8, 9, 10, 11, 12]
SIGNAL_LENGTH = 9000


def to_scalar(value):
    """Extract scalar value from a MATLAB field. Returns NaN on failure."""
    try:
        arr = np.asarray(value).squeeze()
        if arr.size == 0:
            return np.nan
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return np.nan


def to_signal(value, length=SIGNAL_LENGTH):
    """Convert a MATLAB field to a 1D signal with fixed length."""
    try:
        sig = np.asarray(value, dtype=np.float64).squeeze().reshape(-1)
    except (TypeError, ValueError):
        return None

    if sig.size == 0:
        return None

    if sig.size > length:
        sig = sig[:length]
    elif sig.size < length:
        sig = np.pad(sig, (0, length - sig.size), mode="constant", constant_values=0.0)

    if not np.all(np.isfinite(sig)):
        return None

    return sig


def load_daily_sensor_data(path):
    """
    Load daily sensor data from a MATLAB .mat file.

    Args:
        path: path to a .mat file containing a "mill" struct array.

    Returns:
        signals: list of 6 numpy arrays
            (smc, smd, vib_table, vib_spindle, ae_table, ae_spindle)
        metadata: dict with keys DOC, Feed, Material, run_id
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sensor data file not found: {path}")

    mat_file = sio.loadmat(str(path))
    mat_key_map = {k.strip(): k for k in mat_file}

    if "mill" not in mat_key_map:
        raise KeyError(
            f"Field 'mill' not found in {path}. Available keys: {list(mat_file.keys())}"
        )

    runs = mat_file[mat_key_map["mill"]][0]
    if len(runs) == 0:
        raise ValueError(f"No runs found in {path}")

    # Use the most recent run (last entry) as today's data.
    run = runs[-1]

    signals = []
    for idx in SENSOR_INDICES:
        sig = to_signal(run[idx])
        if sig is None:
            raise ValueError(f"Invalid/empty signal at index {idx} in {path}")
        signals.append(sig)

    case_id = to_scalar(run[CASE_INDEX])
    run_id = to_scalar(run[RUN_INDEX])
    doc = to_scalar(run[DOC_INDEX])
    feed = to_scalar(run[FEED_INDEX])
    material = to_scalar(run[MATERIAL_INDEX])

    case_txt = str(int(case_id)) if np.isfinite(case_id) else "NA"
    run_txt = str(int(run_id)) if np.isfinite(run_id) else "NA"

    metadata = {
        "DOC": doc,
        "Feed": feed,
        "Material": material,
        "run_id": f"{case_txt}_{run_txt}",
    }

    return signals, metadata

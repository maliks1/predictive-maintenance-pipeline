# main_pipeline.py
"""
Automated Predictive Maintenance Pipeline
Inference harian + reporting + email alert.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

# Modul eksternal (sudah dibuat terpisah)
from data_loader import load_daily_sensor_data
from feature_extraction import extract_features_from_signals
from pdf_report import generate_pdf_report
from email_alert import send_email_alert

# ==================================================
# 1. LOGGING
# ==================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main_pipeline")

# ==================================================
# 2. CONFIGURATION
# ==================================================
# Path (bisa di-override via env)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "notebooks/models/rf_tool_wear_model.pkl"))
DATA_PATH = Path(os.getenv("DATA_PATH", "data/daily/mill.mat"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", "reports"))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Email (wajib dari env / GitHub Secrets)
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO")
ALERT_SUBJECT_PREFIX = os.getenv("ALERT_SUBJECT_PREFIX", "[CRITICAL] Tool Wear Alert")

# ==================================================
# 3. LOAD MODEL
# ==================================================
def load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    bundle = joblib.load(path)
    required = [
        "regressor", "classifier", "feature_columns",
        "channel_names", "signal_length",
        "threshold_mm", "alert_prob_threshold",
    ]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise KeyError(f"Model bundle tidak lengkap. Missing: {missing}")
    log.info(f"Model berhasil dimuat dari {path}")
    return bundle

# ==================================================
# 4. FEATURE PREPARATION
# ==================================================
def prepare_feature_row(features: dict, feature_columns: list) -> pd.DataFrame:
    """Samakan urutan & nama fitur dengan saat training."""
    df = pd.DataFrame([features])
    # Tambahkan kolom yang hilang (NaN)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    # Buang kolom ekstra
    df = df[feature_columns]
    # Bersihkan inf
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.astype(np.float64)

# ==================================================
# 5. INFERENCE
# ==================================================
def run_inference(X: pd.DataFrame, bundle: dict) -> dict:
    regressor = bundle["regressor"]
    classifier = bundle["classifier"]
    threshold_mm = bundle["threshold_mm"]
    alert_prob = bundle["alert_prob_threshold"]

    predicted_vb = float(regressor.predict(X)[0])

    prob_failure = np.nan
    if classifier is not None:
        prob_failure = float(classifier.predict_proba(X)[0][1])

    is_critical_vb = predicted_vb > threshold_mm
    is_high_prob = (not np.isnan(prob_failure)) and (prob_failure >= alert_prob)
    is_alert = is_critical_vb or is_high_prob

    return {
        "predicted_vb_mm": round(predicted_vb, 4),
        "failure_probability": round(prob_failure, 4) if not np.isnan(prob_failure) else None,
        "threshold_mm": threshold_mm,
        "alert_prob_threshold": alert_prob,
        "is_critical_vb": bool(is_critical_vb),
        "is_high_prob": bool(is_high_prob),
        "is_alert": bool(is_alert),
    }

# ==================================================
# 6. STATUS DECISION
# ==================================================
def determine_status(result: dict) -> str:
    if result["is_alert"]:
        return "CRITICAL"
    prob = result["failure_probability"]
    if prob is not None and prob >= result["alert_prob_threshold"] - 0.10:
        return "WARNING"
    return "NORMAL"

# ==================================================
# 7. MAIN PIPELINE
# ==================================================
def main() -> int:
    log.info("=" * 50)
    log.info("Predictive Maintenance Pipeline - START")
    log.info("=" * 50)

    timestamp = datetime.now()
    run_summary = {
        "timestamp": timestamp.isoformat(),
        "data_path": str(DATA_PATH),
        "model_path": str(MODEL_PATH),
        "status": "UNKNOWN",
        "alert_sent": False,
        "pdf_path": None,
        "error": None,
    }

    try:
        # 1. Load model
        bundle = load_model(MODEL_PATH)

        # 2. Load data sensor harian
        log.info(f"Memuat data sensor: {DATA_PATH}")
        signals, metadata = load_daily_sensor_data(DATA_PATH)
        log.info(f"Berhasil memuat {len(signals)} channel sensor")

        # 3. Ekstrak fitur
        features = extract_features_from_signals(signals, metadata)
        X = prepare_feature_row(features, bundle["feature_columns"])
        log.info(f"Fitur siap: {X.shape[1]} kolom")

        # 4. Inference
        result = run_inference(X, bundle)
        status = determine_status(result)
        result["status"] = status
        result["run_id"] = metadata.get("run_id", "unknown")
        result["timestamp"] = timestamp.isoformat()

        log.info(f"Predicted VB : {result['predicted_vb_mm']} mm")
        log.info(f"Failure Prob : {result['failure_probability']}")
        log.info(f"Status       : {status}")

        # 5. Jika critical / warning -> generate PDF & kirim email
        if status in ("CRITICAL", "WARNING"):
            pdf_path = generate_pdf_report(result)
            result["pdf_path"] = str(pdf_path)
            log.info(f"Laporan PDF dibuat: {pdf_path}")

            if SMTP_EMAIL and SMTP_PASSWORD and ALERT_EMAIL_TO:
                subject = f"{ALERT_SUBJECT_PREFIX} - {status} - {timestamp:%Y-%m-%d %H:%M}"
                body = (
                    f"Status Mesin: {status}\n"
                    f"Timestamp: {timestamp.isoformat()}\n"
                    f"Predicted VB: {result['predicted_vb_mm']} mm "
                    f"(threshold: {result['threshold_mm']} mm)\n"
                    f"Failure Probability: {result['failure_probability']}\n\n"
                    f"Detail lengkap terlampir pada PDF."
                )
                send_email_alert(subject, body, pdf_path)
                run_summary["alert_sent"] = True
                log.info("Email alert berhasil dikirim.")
            else:
                log.warning(
                    "SMTP_EMAIL / SMTP_PASSWORD / ALERT_EMAIL_TO belum diset. "
                    "Email alert dilewati."
                )
        else:
            log.info("Status NORMAL, tidak perlu alert.")

        run_summary.update({
            "status": status,
            "predicted_vb_mm": result["predicted_vb_mm"],
            "failure_probability": result["failure_probability"],
            "pdf_path": result.get("pdf_path"),
        })

    except Exception as e:
        log.exception(f"Pipeline gagal: {e}")
        run_summary["status"] = "FAILED"
        run_summary["error"] = str(e)

    # 6. Simpan ringkasan eksekusi
    summary_path = LOG_DIR / f"run_{timestamp:%Y%m%d_%H%M%S}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, default=str)
    log.info(f"Ringkasan disimpan: {summary_path}")

    log.info("=" * 50)
    log.info("Predictive Maintenance Pipeline - END")
    log.info("=" * 50)

    # Exit code: 0 sukses, 1 jika ada error
    return 1 if run_summary["status"] == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
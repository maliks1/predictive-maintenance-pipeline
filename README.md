# Predictive Maintenance Pipeline

A machine learning-based predictive maintenance system for CNC milling machines that predicts tool wear (VB - flank wear) in real time and triggers alerts when the tool requires replacement.

## Overview

This pipeline monitors 6 sensor channels from a milling machine, extracts statistical features from vibration, acoustic emission, and spindle current signals, and uses trained Random Forest models to predict tool wear and assess failure probability. When predictions exceed safety thresholds, the system automatically generates a PDF report and sends an email alert.

## Features

- **Real-time Tool Wear Prediction** — Predicts flank wear (VB) in millimeters using a Random Forest Regressor
- **Failure Probability Estimation** — Classifies damage risk using a Random Forest Classifier
- **Multi-channel Signal Processing** — Extracts 6 statistical features per channel (std, rms, peak, crest, kurtosis, zero-crossing rate)
- **Automated Alerting** — Sends email notifications with PDF reports when wear or failure probability exceeds configured thresholds
- **Daily Inference Pipeline** — Batch processing of daily sensor data via GitHub Actions (manual trigger; cron schedule available but commented out)
- **Model Versioning** — Trained models exported with full metrics and metadata

## Dataset

This project uses the **NASA Milling Wear Dataset**, a public dataset collected from CNC milling experiments to study tool wear behavior. The dataset contains time-series sensor measurements recorded during milling operations with varying conditions.

📎 **Source**: [NASA Milling Wear Dataset (data.nasa.gov)](https://data.nasa.gov/dataset/milling-wear)

### Dataset Contents

| Component | Description |
|-----------|-------------|
| **Sensor Channels** | 6 channels: Spindle Motor Current, Spindle Motor Drive, Table Vibration, Spindle Vibration, Acoustic Emission (Table), Acoustic Emission (Spindle) |
| **Sampling Rate** | High-frequency acquisitions per machining run |
| **Target Variable** | Flank wear (VB) measured in millimeters at regular intervals |
| **Experimental Runs** | Multiple cases with different cutting conditions and tool wear progression |
| **File Format** | MATLAB `.mat` file containing signal arrays and measurement labels |

### Usage Notes

- The dataset is publicly available for research and educational purposes
- Raw signals are preprocessed and segmented into daily batches for inference
- Feature extraction computes 6 statistical descriptors per channel (36 signal features) plus 3 metadata features (DOC, Feed, Material) = **39 features total**
- Tool wear measurements serve as ground truth for model training and evaluation

## Architecture

### 🔧 Model Training (Notebook)

```mermaid
flowchart LR
    A[📦 NASA Milling Dataset<br/>mill.mat] --> B[🔍 Feature Extraction<br/>39 features: 36 signal + 3 metadata]
    B --> C[🔀 GroupShuffleSplit<br/>by Case_Run]
    C --> D1[🌲 RandomForestRegressor<br/>Predict VB mm]
    C --> D2[🌲 RandomForestClassifier<br/>Failure Probability]

    D1 --> E1[📊 MAE · RMSE · R²<br/>Precision · Recall · F1]
    D2 --> E2[📊 AUC · Precision<br/>Recall · F1]

    E1 --> F[💾 Export Bundle<br/>random_forest_model.pkl]
    E2 --> F
    
    style A fill:#4a90e2,stroke:#2c5282,color:#fff
    style F fill:#48bb78,stroke:#276749,color:#fff
    style D1 fill:#ed8936,stroke:#c05621,color:#fff
    style D2 fill:#ed8936,stroke:#c05621,color:#fff
```

### ⚙️ Daily Inference Pipeline (Production)

```mermaid
flowchart TD
    START([⚙️ GitHub Actions<br/>Manual Trigger]) --> LOAD[📥 Load Model Bundle<br/>random_forest_model.pkl]
    LOAD --> DATA[📡 Load Daily Sensor Data<br/>6 channel × 9000 samples]
    DATA --> FE[🔍 Extract Features<br/>39 statistical features]
    FE --> PRED[🤖 Inference<br/>Regressor + Classifier]
    
    PRED --> DECISION{🎯 Status Decision<br/>VB > threshold_mm<br/>OR Prob ≥ alert_prob_threshold}
    
    DECISION -->|VB ≤ threshold & Prob < alert_prob| NORMAL[✅ NORMAL<br/>Log only]
    DECISION -->|VB > threshold OR Prob ≥ alert_prob| ALERT[🚨 ALERT<br/>Tool requires replacement]
    
    ALERT --> REPORT[📄 Generate PDF Report]
    
    REPORT --> EMAIL[📧 Send Email Alert<br/>SMTP Gmail + Attachment]
    EMAIL --> DONE([✔️ Pipeline Complete<br/>Save run_summary.json])
    NORMAL --> DONE
    
    style START fill:#6b46c1,stroke:#44337a,color:#fff
    style NORMAL fill:#48bb78,stroke:#276749,color:#fff
    style ALERT fill:#e53e3e,stroke:#9b2c2c,color:#fff
    style DONE fill:#48bb78,stroke:#276749,color:#fff
    style DECISION fill:#ed8936,stroke:#c05621,color:#fff
```

### 🔐 Secret Management

```mermaid
flowchart LR
    subgraph GitHub["🔐 GitHub Repository"]
        SECRETS[GitHub Secrets<br/>SMTP_EMAIL<br/>SMTP_PASSWORD<br/>ALERT_EMAIL_TO]
    end
    
    subgraph Actions["⚙️ GitHub Actions Runner"]
        ENV[Environment Variables]
        PIPELINE[main_pipeline.py]
    end
    
    subgraph External["🌐 External Services"]
        GMAIL[📧 Gmail SMTP<br/>port 587]
        TEAM[👥 Maintenance Team]
    end
    
    SECRETS -->|injected at runtime| ENV
    ENV --> PIPELINE
    PIPELINE -->|authenticated| GMAIL
    GMAIL -->|alert + PDF| TEAM
    
    style SECRETS fill:#2d3748,stroke:#1a202c,color:#fff
    style GMAIL fill:#e53e3e,stroke:#9b2c2c,color:#fff
    style TEAM fill:#4a90e2,stroke:#2c5282,color:#fff
```

## Project Structure

```
predictive-maintenance-pipeline/
├── main_pipeline.py          # Main inference pipeline orchestrator
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   ├── mill.mat              # MATLAB training dataset (6 sensor channels)
│   └── daily/                # Directory for daily sensor data files
├── notebooks/
│   ├── model_training.ipynb  # Model training & evaluation notebook
│   └── models/
│       └── random_forest_model.pkl  # Trained model bundle (exported)
├── src/
│   ├── alert_system.py        # Email alert system
│   ├── data_loader.py         # MATLAB .mat data loading
│   ├── feature_extraction.py  # Signal feature extraction functions
│   └── pdf_generator.py       # PDF report generation
├── reports/                  # Generated PDF alert reports
├── logs/                     # Pipeline logs & run summaries
└── .github/workflows/
    └── daily_pipeline.yml    # GitHub Actions cron workflow
```

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

```bash
# Clone or navigate to the project directory
cd predictive-maintenance-pipeline

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `pandas` | Data manipulation |
| `scipy` | MATLAB file I/O (`scipy.io`) |
| `scikit-learn` | Random Forest models, metrics |
| `joblib` | Model serialization |
| `fpdf2` | PDF report generation |
| `jupyter` / `ipykernel` | Notebook execution |

## Configuration

### Environment Variables

The following environment variables must be set before running the pipeline:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to trained model file | `notebooks/models/random_forest_model.pkl` |
| `DATA_PATH` | Path to daily sensor data | `data/daily/mill.mat` |
| `REPORT_DIR` | Output directory for PDF reports | `reports` |
| `LOG_DIR` | Output directory for logs & summaries | `logs` |
| `SMTP_EMAIL` | Sender Gmail address | _(required)_ |
| `SMTP_PASSWORD` | Gmail App Password | _(required)_ |
| `ALERT_EMAIL_TO` | Recipient email address | _(required)_ |
| `ALERT_SUBJECT_PREFIX` | Email subject prefix | `[ALERT] Tool Wear Alert` |

### Setting Up Email Alerts

1. Enable 2-Step Verification on your Google Account
2. Generate an [App Password](https://myaccount.google.com/apppasswords)
3. Set the environment variables:

```powershell
# Windows PowerShell
$env:SMTP_EMAIL = "your-email@gmail.com"
$env:SMTP_PASSWORD = "your-app-password"
$env:ALERT_EMAIL_TO = "recipient@company.com"
```

```bash
# Linux/Mac
export SMTP_EMAIL="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
export ALERT_EMAIL_TO="recipient@company.com"
```

## Usage

### Training the Model

Run the Jupyter Notebook to train and evaluate the model:

```bash
# Activate virtual environment
.venv\Scripts\activate

# Launch Jupyter
jupyter notebook notebooks/model_training.ipynb
```

Execute all cells sequentially. The trained model will be saved to `notebooks/models/random_forest_model.pkl`.

### Running the Pipeline

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run the pipeline
python main_pipeline.py
```

The pipeline will:
1. Load the trained model
2. Process daily sensor data (most recent run from the .mat file)
3. Extract 39 features (36 signal + 3 metadata)
4. Predict tool wear and failure probability
5. Determine machine status (NORMAL / ALERT)
6. Generate a PDF report and send an email alert if ALERT status

### Log Output

Logs are written to both console and `logs/pipeline.log`. Run summaries are saved as `logs/run_YYYYMMDD_HHMMSS.json`:

```
2026-09-05 08:00:00 [INFO] ==================================================
2026-09-05 08:00:00 [INFO] Predictive Maintenance Pipeline - START
2026-09-05 08:00:00 [INFO] Model successfully loaded from notebooks/models/random_forest_model.pkl
2026-09-05 08:00:01 [INFO] Machine Status: ALERT
2026-09-05 08:00:01 [INFO] Alert email successfully sent.
```

## Model Details

### Sensor Channels (6 input signals)

| Channel | Name | Description |
|---------|------|-------------|
| 0 | `smc` | Spindle Motor Current |
| 1 | `smd` | Spindle Motor Drive |
| 2 | `vib_table` | Table Vibration |
| 3 | `vib_spindle` | Spindle Vibration |
| 4 | `ae_table` | Acoustic Emission (Table) |
| 5 | `ae_spindle` | Acoustic Emission (Spindle) |

### Extracted Features (6 per channel = 36 signal + 3 metadata = 39 total)

Each signal channel produces these statistical features:

| Feature | Formula | Significance |
|---------|---------|--------------|
| `std` | Standard deviation | Signal variability |
| `rms` | Root mean square | Power of the signal |
| `peak` | Maximum absolute value | Peak stress indicator |
| `crest` | peak / RMS | Impulse detection |
| `kurt` | Excess kurtosis | Tail heaviness / outliers |
| `zero_cross` | Zero-crossing rate | Frequency content |

### Thresholds

Thresholds are configured at training time and stored in the model bundle. They can be adjusted in the notebook before retraining.

| Parameter | Notebook Default | Meaning |
|-----------|-----------------|---------|
| `threshold_mm` | 0.4 mm | Tool replacement wear limit |
| `alert_prob_threshold` | 0.5 (50%) | Failure probability alert level |

### Status Logic (Binary)

| Status | Condition | Action |
|--------|-----------|--------|
| **NORMAL** | Wear below threshold AND low failure probability | No action |
| **ALERT** | Wear > threshold_mm OR failure probability ≥ alert_prob_threshold | Stop machine, replace tool, send alert |

## Workflow

### Training Pipeline (`model_training.ipynb`)

```
Cell 1: Import & Config          → Libraries, constants, data paths
Cell 2: Helper Functions          → to_scalar, to_signal, safe_float, clean_features
Cell 3: Feature Extraction        → compute_channel_features, extract_features_from_signals
Cell 4: Load Dataset              → Parse MATLAB .mat, extract features from all runs
Cell 5: Data Split                → GroupShuffleSplit (80/20), column alignment
Cell 6: Train Regressor           → RandomForestRegressor (n=700, depth=14)
Cell 7: Regression Evaluation     → MAE, RMSE, R²
Cell 8: Alert Threshold Eval      → Precision, Recall, F1 at 0.4 mm
Cell 9: Train Classifier          → RandomForestClassifier for failure probability
Cell 10: Export Model             → Save bundle with metrics to .pkl
```

### Inference Pipeline (`main_pipeline.py`)

```
Step 1: Load Model               → Validate bundle completeness
Step 2: Load Daily Data          → Read sensor signals from .mat file
Step 3: Extract Features         → 39 features from 6 channels + metadata
Step 4: Predict                  → Regressor → VB (mm), Classifier → probability
Step 5: Status Decision          → NORMAL / ALERT
Step 6: Generate Report          → PDF with prediction details (if ALERT)
Step 7: Send Alert               → Email with PDF attachment (if ALERT)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: mill.mat not found` | Ensure `data/mill.mat` exists or set `DATA_PATH` env var |
| `Model not found` | Run `model_training.ipynb` first to generate the model file |
| `Email sending failed` | Verify SMTP credentials; use App Password, not regular password |
| `No valid samples extracted` | Check MATLAB file format; ensure correct field names |
| `Infinity in features` | Signals contain non-finite values; check raw sensor data |

## License

This project is for educational and research purposes.

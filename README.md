# SURYOKTI

SURYOKTI is a research-oriented solar-PV intelligence platform. It combines weather-aware power prediction, daylight-aware operational anomaly detection, visual-fault classification research, and maintenance triage.

## Current project status

| Component | Status | What it does |
|---|---|---|
| Energy prediction | Implemented and evaluated | Estimates PV AC power from weather and solar-time features. |
| Anomaly detection | Implemented and under active validation | Combines physical rules, Isolation Forest novelty scoring, and optional forecast residuals. |
| CNN visual fault classification | Architecture and class taxonomy defined | Training/inference artefacts are not yet included as a completed production component. |
| Dashboard | Under implementation | Intended to present telemetry, events, evidence, and maintenance priority. |

## System flow

```text
Telemetry + weather
        |
        +--> Power prediction --> expected output
        |
        +--> Anomaly detection --> data-quality / novelty / underperformance events
                                         |
                           optional visual inspection (CNN)
                                         |
                              maintenance triage and work orders
```

The anomaly pipeline deliberately treats an alert as an **inspection candidate**, not a confirmed hardware failure, unless supported by maintenance, SCADA, or field-inspection evidence.

## Repository layout

```text
anomaly_detection/            Canonical daylight-aware anomaly pipeline and tests
energy_prediction/            Original prediction workflow and inference tools
energy_prediction_romania/    Chronologically evaluated two-site implementation
energy_prediction_adaptive/   Schema-adaptive prediction pipeline
cnn_fault_detection/          Visual-fault taxonomy and architecture reference
integration/                  Prediction-to-anomaly-to-maintenance orchestration
maintenance/                  Prioritisation and work-order logic
datasets/                     Dataset documentation and lightweight metadata
documentation/                Technical audits, methodology, and provenance
experiments/                  Isolated regional and reconstruction research
```

`Not Nec/`, raw third-party data, duplicate archives, cached environments, generated plots, and retrainable binary model artefacts are intentionally excluded from Git. They remain local and are documented through provenance files rather than committed as opaque copies.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Main commands

```powershell
# Canonical anomaly assessment; supply a forecast column to activate residual screening.
python anomaly_detection/run_pipeline.py --input <telemetry.csv> --output-dir <output-dir> --prediction-column <forecast_column>

# End-to-end demonstration: forecast -> anomaly events -> maintenance triage.
python integration/end_to_end_workflow/end_to_end_demo.py

# Two-site chronological forecasting implementation.
python energy_prediction_romania/run_pipeline.py

# Synthetic-fault regression benchmark (not a claim of field accuracy).
python energy_prediction_romania/run_synthetic_fault_validation.py --site all
```

## Verification

```powershell
python -m unittest discover -s anomaly_detection/tests -v
python energy_prediction_romania/tests/test_preprocessing.py
python energy_prediction_romania/tests/test_model.py
python energy_prediction_adaptive/tests/test_validation_splits.py
```

The GitHub Actions workflow repeats the core source and test checks on every pull request and push to `main`.

## Evaluation principles

- Use chronological train/validation/test partitions for time-series prediction.
- Report daytime metrics separately from all-hours metrics.
- Fit anomaly calibration only on historical, clean daylight data.
- Do not call a statistical anomaly a verified equipment fault without ground-truth evidence.
- Keep synthetic-fault tests separate from real-world performance claims.

See [documentation](documentation/README.md), [anomaly detection](anomaly_detection/README.md), and [the Romania implementation](energy_prediction_romania/README.md) for component-level details.

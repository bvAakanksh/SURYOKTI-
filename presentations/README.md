# SURYOKTI: Presentations & Slide Decks

This directory contains Microsoft PowerPoint (`.pptx`) presentation decks and automated python slide generation scripts documenting the progress, methodology, model selection, and fine-tuning results of the SURYOKTI project.

---

## 1. Directory Structure

```text
presentations/
├── README.md                                    # Presentations index and usage guide (this file)
│
├── progress/                                    # Historical & milestone progress slide decks
│   └── Presentation update - 2nd SEPT.pptx      # September progress presentation deck
│
├── final/                                       # Final presentations and defense decks
│   └── SURYOKTI_Model_Selection_and_FineTuning.pptx # Comprehensive model selection, fine-tuning, and results deck
│
└── scripts/                                     # Automated slide deck generation tools
    └── generate_presentation.py                 # python-pptx generator building the complete presentation
```

---

## 2. Generating Presentations Programmatically

The slide decks can be regenerated or customized programmatically using `python-pptx`:

```powershell
.venv\Scripts\python.exe presentations/scripts/generate_presentation.py
```

### Presentation Outline in `SURYOKTI_Model_Selection_and_FineTuning.pptx`:
1. **Title & Executive Summary:** Project scope, physics-informed forecasting, and anomaly detection.
2. **Problem Statement:** Operational challenges in solar PV forecasting and premature anomaly labeling.
3. **Dataset Provenance & Preprocessing:** Distinguishing synthetic demo datasets from real physical plant telemetry.
4. **Feature Engineering & Solar Physics:** Solar zenith angles, cyclical hour/month encodings, and thermal stress.
5. **Model Architectures & Cross-Validation:** RidgeCV, Random Forest, XGBoost, and Stacking Ensembles.
6. **Experimental Results & Parity Analysis:** Holdout scores, K-Fold stability, and TimeSeries CV generalization.
7. **Daylight-Calibrated Anomaly Detection:** MAD thresholding, daylight gating, and operational fault separation.
8. **Summary & Future Roadmap:** Regional transferability, CNN defect classification, and automated maintenance dispatch.


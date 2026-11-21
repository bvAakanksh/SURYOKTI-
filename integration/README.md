# SURYOKTI: Integration & End-to-End Orchestration Module

The **Integration Module** connects forecasting, daylight residual screening, and maintenance triage. It deliberately preserves an event as `Unclassified` until an independent inspection system supplies a verified fault type.

---

## 1. Directory Structure

```text
integration/
├── README.md                                    # Integration architecture guide (this file)
│
├── prediction_anomaly_integration/              # Fusion of prediction residuals and anomaly flags
│   └── integrated_pipeline.py                   # Joint model evaluator tracking real-time performance
│
└── end_to_end_workflow/                         # Master plant orchestration
    └── end_to_end_demo.py                       # Single-script execution: telemetry -> forecast -> anomaly -> work order
```

---

## 2. Integrated Pipeline Flow

```text
  [ Raw Telemetry ] ──► [ Feature Engineering ] ──► [ Energy Prediction Model ]
                                                             │
                                                             ▼ (P_pred)
  [ Actual Power ] ───────────────────────────────► [ Residual Calculation ]
                                                             │
                                                             ▼
                                                    [ Anomaly Detection ]
                                                    (Daylight MAD Gated)
                                                             │
                                                             ▼ (Operational Anomaly?)
                                                    [ Maintenance Engine ]
                                                    (Priority & Verification Work Orders)
```

---

## 3. Running the Integrated End-to-End Demo

Execute the complete end-to-end pipeline demonstration:
```powershell
.venv\Scripts\python.exe integration/end_to_end_workflow/end_to_end_demo.py
```
This runs real predictions on telemetry data, screens daytime residual deficits, and issues prioritized field-verification work orders. It does not simulate visual fault classifications.


# SURYOKTI: Intelligent Maintenance Decision Engine

The **Maintenance Module** bridges analytical insights (power forecasts, anomaly detections, and visual defect classifications) into actionable, cost-optimized operational maintenance schedules and work orders.

---

## 1. Directory Structure

```text
maintenance/
├── README.md                                    # Maintenance methodology and workflow guide (this file)
│
├── scheduling/                                  # Predictive maintenance scheduling algorithms
│   └── schedule_maintenance.py                 # Weather-aware and generation-loss scheduling engine
│
├── prioritization/                              # Loss-weighted severity ranking
│   └── fault_priority_matrix.py                 # Multi-factor priority matrix (Loss, Safety, Severity)
│
└── recommendation_logic/                        # Actionable O&M work order generator
    └── actionable_recommendations.py            # Diagnostic rule engine dispatching field work orders
```

---

## 2. Maintenance Decision Workflow

```text
  ┌─────────────────────────────────────────────────────────────┐
  │                        INPUT SIGNALS                        │
  │  - Energy Prediction: Expected Output P_expected (kW)       │
  │  - Anomaly Detection: Power Deficit ΔP = P_exp - P_act (kW) │
  │  - CNN Vision: Surface Defect Class (e.g., Dusty, Bird-drop)│
  │  - Weather Forecast: Upcoming rain probability / wind       │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 PRIORITY SCORING ENGINE                     │
  │  Priority Score = 0.40 * Financial Loss ($/day)            │
  │                 + 0.35 * Safety / Hardware Risk (0-10)      │
  │                 + 0.25 * Anomaly Persistence (Hours)        │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │               ACTIONABLE DISPATCH ASSIGNMENT                │
  │  - Priority 1 (CRITICAL): Physical / Electrical Damage      │
  │  - Priority 2 (HIGH): Persistent Bird-drop (Hotspot Risk)   │
  │  - Priority 3 (MEDIUM): Severe Dust Accumulation            │
  │  - Priority 4 (LOW): Minor Sensor Calibration Drift         │
  └─────────────────────────────────────────────────────────────┘
```

---

## 3. Prioritization Scoring Formula

$$\text{Priority Index} = w_{\text{loss}} \cdot \tilde{L}_{\text{financial}} + w_{\text{safety}} \cdot S_{\text{risk}} + w_{\text{persist}} \cdot \tilde{T}_{\text{duration}}$$

Where:
* $\tilde{L}_{\text{financial}} = \Delta P (\text{kW}) \times \text{Peak Sun Hours} \times \text{Tariff Rate } (\$/\text{kWh})$ (normalized to $[0, 1]$)
* $S_{\text{risk}} \in [0, 1]$ represents physical/fire hazard:
  - `Physical-Damage` = $1.0$
  - `Electrical-damage` = $0.9$
  - `Bird-drop` (hotspot risk) = $0.7$
  - `Dusty` = $0.3$
  - `Normal / Sensor drift` = $0.1$
* $\tilde{T}_{\text{duration}}$ is the continuous hours the fault condition has persisted.

---

## 4. Execution Examples

### Calculate Fault Priority Rankings
```powershell
.venv\Scripts\python.exe maintenance/prioritization/fault_priority_matrix.py
```

### Generate Actionable Work Orders
```powershell
.venv\Scripts\python.exe maintenance/recommendation_logic/actionable_recommendations.py
```


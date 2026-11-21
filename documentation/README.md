# SURYOKTI: Technical Documentation & Audit Reports

This directory houses the formal scientific documentation, technical audit evaluations, decision memos, mathematical methodology notes, and academic provenance records for the SURYOKTI project.

---

## 1. Directory Structure

```text
documentation/
├── README.md                                    # Documentation guide and index (this file)
│
├── project_reports/                             # Formal investigation and decision memos
│   ├── target_reconstruction_report.md          # Multi-inverter channel decomposition & candidate analysis
│   ├── target_reconstruction_report.json        # Machine-readable candidate metrics and channel stats
│   └── maui_decision_memo.md                    # Executive decision memo on plant capacity & evidence
│
├── technical_audits/                            # Independent technical validation reports
│   ├── maui_technical_audit.md                  # Comprehensive audit of timezones, inverters, and CV
│   └── maui_technical_audit.json                # Audit verification numbers and metric checks
│
├── methodology/                                 # Mathematical formulas and physics specifications
│   └── solar_physics_and_metrics.md             # Detailed equations for solar angles, metrics, and MAD
│
└── license/                                     # Data licensing and academic attribution
    └── license_and_provenance_note.txt          # Open data terms, Mendeley citation, and file hashes
```

---

## 2. Key Document Summaries

### A. [`maui_technical_audit.md`](technical_audits/maui_technical_audit.md)
* **Purpose:** Independent scientific audit of initial regional adaptation claims on NREL PVDAQ System 2105.
* **Key Findings:**
  1. Verified UTC to HST (-13h) alignment against solar noon.
  2. Identified multi-channel inverter overlap and potential double counting in raw logger channels.
  3. Recomputed holdout metrics and audited time-series cross validation splits.
  4. Formally recommended treating plant-level target reconstruction as UNVERIFIED pending electrical schematics.

### B. [`target_reconstruction_report.md`](project_reports/target_reconstruction_report.md)
* **Purpose:** Rigorous channel-by-channel electrical decomposition of 11 inverter telemetry feeds (`inv_01` through `inv_11`).
* **Key Findings:**
  - Evaluated 5 competing candidate target aggregations.
  - Demonstrated that Candidate 1 (234 kW) double-counts sub-combiners.
  - Identified Candidate 5 (`inv_04 + inv_10`, 86.85 kW peak) as a plausible 110 kW DC plant interpretation with ILR = 1.267, while maintaining the UNVERIFIED status.

### C. [`maui_decision_memo.md`](project_reports/maui_decision_memo.md)
* **Purpose:** Concise executive decision memorandum establishing what is proven by data versus what remains uncertain, setting prerequisites before proceeding to cross-region generalization.


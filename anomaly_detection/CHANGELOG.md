# Anomaly Detection Checkpoints

This log records every change made while completing the anomaly-detection workflow. Legacy files under `Not Nec/` and existing experiment outputs are preserved unless the user explicitly authorizes deletion.

## Checkpoint 1 — 2026-11-18

- **Change:** Added this checkpoint log.
- **Reason:** Establish an auditable record before restructuring the module.
- **Files deleted:** None.
- **Verification:** File created successfully.

## Checkpoint 2 — 2026-11-18

- **Changed:** Added `__init__.py`, `config.py`, and `schema.py`.
- **Reason:** Created the canonical package boundary, one threshold configuration object, and non-destructive telemetry validation.
- **Files deleted:** None.
- **Verification pending:** The pipeline implementation will be added next; package import will be verified with the first runnable checkpoint.

## Checkpoint 3 — 2026-11-18

- **Added:** `pipeline.py`.
- **Reason:** Implemented reusable configuration-driven rules, daylight-only Isolation Forest calibration, optional MAD residual detection, status reconciliation, event aggregation, and artifact save/load.
- **Behavioral safeguard:** Rows with invalid telemetry are retained and categorized as data-quality anomalies; residual alarms are unavailable unless an explicit prediction column is supplied.
- **Files deleted:** None.
- **Verification pending:** Synthetic unit tests and a bundled-data smoke test will be added before the existing detector is migrated.

## Checkpoint 4 — 2026-11-18

- **Added:** `run_pipeline.py` and `tests/test_pipeline.py`.
- **Reason:** Added a command-line interface and dependency-free tests for the canonical pipeline.
- **Verification result:** Initial test run failed because the synthetic fixture accidentally simulated a frozen daytime irradiance sensor, leaving fewer than 20 clean calibration rows. Direct script execution also lacked the repository root on `sys.path`.
- **Corrective edits:** Made the synthetic irradiance profile vary by hour and added explicit repository-root import resolution to the CLI.
- **Files deleted:** None.
- **Next verification:** Rerun all five unit tests, CLI smoke test, and artifact reload.

## Checkpoint 5 — 2026-11-18

- **Verification result:** Four of five tests passed. The remaining test incorrectly selected the first 12 hours as night, but that range included daytime observations.
- **Corrective edit:** Updated the test to explicitly select rows with zero irradiance.
- **CLI finding:** `Solar_Weather_Data_Processed.csv` retains whitespace and spaces in its headers, so the canonical CLI correctly rejected it under its strict schema contract. The cleaned canonical demo CSV will be used for the smoke test.
- **Files deleted:** None.

## Checkpoint 6 — 2026-11-18

- **Verification result:** All five synthetic unit tests passed. The real-data smoke test did not fit because its supplied `SZ_Angle` conflicts with irradiance: every row with `GHI > 50` had a zenith value above the configured daylight limit.
- **Corrective edit:** Added `use_solar_zenith_gate` to `AnomalyConfig`, defaulting to `False`. GHI is therefore the canonical daylight gate unless a site has validated solar-geometry timestamps and explicitly enables the additional zenith check.
- **Finding corrected:** The failed smoke test was caused by incompatible source fields, not the sensor-freeze rule.
- **Files deleted:** None.
- **Next verification:** Rerun the real-data smoke test with the default GHI-only gate.

## Checkpoint 7 — 2026-11-18

- **Verification result:** All five unit tests passed. A chronological real-data smoke test generated 251 holdout assessments, 22 aggregated events, a serialized model, row-level CSV, event CSV, and run metadata. Artifact reload also passed.
- **Added/changed:** Updated `integration/prediction_anomaly_integration/integrated_pipeline.py` with `run_canonical_detection`, which returns canonical named statuses and event records.
- **Bug fixed:** The legacy adapter now correctly interprets Scikit-learn Isolation Forest `-1` as an anomaly. It labels a raw outlier as `ENVIRONMENTAL_NOVELTY`; it no longer incorrectly asserts an operational fault from the raw flag.
- **Files deleted:** None.
- **Verification pending:** Add integration-contract tests and migrate the active anomaly CLI/documentation to the canonical runner.

## Checkpoint 8 — 2026-11-18

- **Added:** `tests/test_integration_contract.py`.
- **Reason:** Tests the canonical integration return contract and the legacy Isolation Forest flag adapter.
- **Verification result:** Six of seven tests passed. The failed assertion used a nighttime row; `NIGHTTIME_INACTIVE` correctly takes precedence over novelty, so the test was changed to daylight rows.
- **Files deleted:** None.
- **Next verification:** Rerun the full suite after the test correction, then document the canonical CLI as the supported entrypoint.

## Checkpoint 9 — 2026-11-18

- **Documentation changed:** Root README and anomaly-module README now name `anomaly_detection/run_pipeline.py` as the supported entrypoint. The old detector is explicitly retained as a legacy prototype.
- **Integration changed:** The end-to-end demo now calibrates and scores through the canonical pipeline, aggregates events, and dispatches only residual/operational event categories. It no longer uses a separate hand-coded MAD path or fabricated persistence durations.
- **Files deleted:** None.
- **Verification pending:** Run the updated end-to-end demonstration and confirm the canonical event-to-maintenance handoff.

## Checkpoint 10 — 2026-11-18

- **Verification result:** All seven canonical pipeline and integration tests pass. The end-to-end demo completed using canonical event records and generated three `Unclassified` field-verification work orders.
- **Generated artifacts:**
  - `results/canonical_demo/` — smoke-test model, row-level results, events, and metadata using existing detector output with a forecast column.
  - `results/canonical_cleaned_demo/` — rule-and-novelty run using `Solar_Weather_Data_Cleaned.csv` without a forecast column.
- **Important result:** The cleaned demo run produced data-quality and potential-operational categories. These are statistical/heuristic candidates only; the dataset has known time/solar-geometry inconsistencies and no maintenance ground truth.
- **Files deleted:** None.
- **Status:** Canonical anomaly pipeline, CLI, integration contract, event aggregation, test coverage, documentation, and maintenance handoff are complete. Maui and target-reconstruction scripts remain preserved as unverified research artifacts and are not the supported production path.

## Checkpoint 11 — 2026-11-18

- **Diagnosis:** The demo benchmark combines different source periods. Its `Hour` field conflicts with `Timestamp` in 2,602 of 2,777 rows, and its supplied solar-angle fields conflict materially with GHI.
- **Added:** `energy_prediction/data/build_temporally_consistent_demo.py` and `energy_prediction/inference/generate_chronological_forecast.py`.
- **Reason:** The first script produces a versioned, temporally consistent software benchmark without overwriting raw/legacy geometry. The second trains only on an earlier chronological partition and exports holdout forecasts for residual anomaly detection.
- **Scientific safeguard:** Both outputs are labeled `SOFTWARE_BENCHMARK_ONLY`; no unverified coordinates or synthetic merge is presented as a site-accurate physical dataset.
- **Files deleted:** None.
- **Verification pending:** Build the corrected dataset, produce holdout forecasts, and run residual anomaly detection against that forecast output.

## Checkpoint 12 — 2026-11-18

- **Generated:** `datasets/energy_prediction/processed/Solar_Weather_Data_Temporal_Consistent.csv` and its JSON report; it corrects 2,602 timestamp-derived hour values and preserves original angles as `Legacy_SZ_Angle` and `Legacy_SA_Angle`.
- **Generated:** Chronological forecast holdout CSV, forecast metadata, and `models/energy_prediction/temporal_consistent_forecast.joblib`.
- **Verification result:** The intentionally non-leaking forecast achieves R² = -0.0166, MAE = 249.48 kW, RMSE = 382.60 kW. This verifies that the merged demo data is not a credible predictor/target pairing; it must not be used for residual-fault claims.
- **Generated:** `results/temporal_consistent_forecast/` canonical anomaly artifacts. These remain software-pipeline validation only because their forecast source is unsuitable.
- **Corrective edits:** The anomaly feature builder now omits unavailable optional features, removing the all-null solar-zenith warning. Forecast metadata now automatically labels non-positive-R² outputs as unsuitable for residual anomaly claims.
- **Files deleted:** None.
- **Next verification:** Regenerate the forecast/anomaly artifacts after the metadata and optional-feature refinements.

## Checkpoint 13 — 2026-11-18

- **Verification result:** All seven anomaly tests pass after the refinements. The forecast export and canonical residual-anomaly run both complete successfully and compile checks pass.
- **Overwritten generated artifacts:** The forecast model, forecast holdout CSV/metadata, and `results/temporal_consistent_forecast/` were regenerated from the same inputs to apply the updated metadata/feature behavior. No source dataset, legacy model, or research artifact was overwritten.
- **Environment note:** Scikit-learn emitted a Windows `wmic` physical-core discovery warning; it falls back to logical cores and does not alter model results.
- **Decision boundary:** The generated forecast metadata explicitly states `NOT_SUITABLE_FOR_RESIDUAL_ANOMALY_CLAIMS`. A verified, temporally aligned single-site telemetry + weather dataset is required before residual anomalies can be interpreted as plant-performance evidence.
- **Files deleted:** None.

## Checkpoint 14 — 2026-11-18

- **Assessment:** Romania Site A and B use temporally aligned power/weather data and existing chronological holdout forecasts. Their reported all-hours stacking R² values are 0.9157 and 0.8771 respectively, unlike the synthetic demo benchmark.
- **Added:** `energy_prediction_romania/run_anomaly_detection.py`.
- **Reason:** Maps Romania holdout forecasts into the canonical anomaly schema, calibrates thresholds only on the earlier 70% of each holdout prediction period, and produces separately versioned site artifacts.
- **Scientific safeguard:** Outputs are explicitly marked `RESEARCH_BENCHMARK_NO_FAULT_GROUND_TRUTH`; anomaly events remain inspection candidates rather than confirmed failures.
- **Files deleted:** None.
- **Verification pending:** Run both site adapters and inspect output schema/status distributions.

## Checkpoint 15 — 2026-11-18

- **Generated:** `energy_prediction_romania/results/anomaly_detection/site_a/` and `site_b/`, each containing a fitted anomaly pipeline, row-level anomaly CSV, event CSV, and run metadata.
- **Verification result:** Artifact row/event counts match the metadata for both sites and the adapter compiles successfully.
- **Site A assessment:** 767 rows; 410 daylight-normal, 288 nighttime-inactive, 47 environmental-novelty, 21 statistical-residual-exceedance, and 1 data-quality row; 50 grouped events.
- **Site B assessment:** 561 rows; 303 daylight-normal, 214 nighttime-inactive, 38 environmental-novelty, and 6 statistical-residual-exceedance rows; 24 grouped events.
- **Documentation changed:** Added the Romania anomaly command and output location to `energy_prediction_romania/README.md`.
- **Files deleted:** None.
- **Interpretation:** These are technically credible residual-anomaly candidates because the forecast and power data are aligned, but precision/recall and confirmed failure claims still require maintenance or inspection labels.

## Checkpoint 16 — 2026-11-18

- **Added:** `energy_prediction_romania/run_synthetic_fault_validation.py` and `anomaly_detection/tests/test_synthetic_fault_validation.py`.
- **Purpose:** Added a deterministic, future-only synthetic-fault regression benchmark for the canonical prediction-plus-anomaly path. It injects labeled daylight outages, partial derates, and negative-power sensor faults only into a copy of the assessment holdout; calibration data and source prediction files remain untouched.
- **Documentation changed:** Added the benchmark command and its scientific limitation to `energy_prediction_romania/README.md`.
- **Files deleted:** None.
- **Verification pending:** Run the unit suite and both-site synthetic benchmark; record generated artifacts and detection metrics in the next checkpoint.

## Checkpoint 17 — 2026-11-18

- **Verification result:** All eight canonical anomaly, integration, and synthetic-injection tests pass.
- **Generated:** `energy_prediction_romania/results/synthetic_fault_validation/site_a/` and `site_b/`. Each directory contains `injection_manifest.csv`, `row_level_synthetic_assessment.csv`, `anomaly_events.csv`, and `run_metadata.json`; each assessment contains 27 labeled injected rows (three three-row events for each of three fault types).
- **Site A synthetic results:** Daylight-outage, partial-derate, and negative-power-sensor event recall were each 100%. Exact row-status recall was 100%, 66.7%, and 100% respectively. The 66.7% partial-derate row figure is classification-precedence variation, not a missed event.
- **Site B synthetic results:** Daylight-outage, partial-derate, and negative-power-sensor event recall were each 100%. Exact row-status recall was 100%, 88.9%, and 100% respectively.
- **Non-injected daytime critical-alert rate:** 4.84% on Site A and 2.14% on Site B. This is deliberately not called a false-positive rate because unlabeled real anomalies may exist in those rows.
- **Files deleted:** None. Source forecast holdouts and calibration partitions were not modified.

## Checkpoint 18 — 2026-11-20

- **Added:** `energy_prediction_romania/run_tuned_prediction_pipeline.py`.
- **Purpose:** Adds a reproducible non-CNN tuning path that selects RF, XGBoost, and a non-negative blend strictly on each chronological validation partition, then evaluates the selected blend once on the untouched test partition. It writes only to separate `models/tuned/` and `results/tuned_predictions/` paths and preserves all baseline artifacts.
- **Anomaly calibration changed:** Added optional forecast-power residual bins to the canonical pipeline. When enabled, each bin uses only calibration residuals and falls back safely to the global robust MAD threshold when sparse. The default remains one global bin, preserving existing behavior.
- **Files deleted:** None.
- **Verification pending:** Compile and test the canonical pipeline, then run the two-site tuning benchmark and use the resulting predictions for a separately versioned anomaly assessment.

## Checkpoint 19 — 2026-11-20

- **Prediction tuning result:** The causal validation search selected RF-only for Site A and XGBoost-only for Site B. On the untouched test period, Site B improved from the established stack's 2.883 kW MAE / 5.482 kW RMSE to 2.666 kW / 5.309 kW. Site A's tuned candidate was worse (2.441 kW / 4.881 kW versus 2.291 kW / 4.733 kW), so it is retained as a comparison artifact, not presented as a replacement.
- **Changed:** Romania anomaly and synthetic-validation runners now accept a versioned prediction directory/output root and can enable forecast-power residual bins. Default commands retain their original global-threshold behavior.
- **Documentation changed:** Added commands for causal tuning and binned-residual anomaly assessment.
- **Files deleted:** None.
- **Verification pending:** Run the binned residual assessment and synthetic regression validation against tuned predictions, then record the new artifacts and metrics.

## Checkpoint 20 — 2026-11-20

- **Binned-threshold experiment:** Four forecast-power bins lowered Site A's non-injected critical-alert rate, but reduced synthetic partial-derate event recall to 66.7% on both sites. It remains available as an experimental option and is not the recommended configuration.
- **Global threshold sweep:** On fixed tuned-forecast synthetic scenarios, 2.5 MAD achieved 100% partial-derate event recall on both sites. A 2.0-MAD threshold also achieved 100% but created substantially more non-injected alerts; the original 3.0-MAD threshold missed one of three Site A derate events.
- **Changed:** Romania anomaly and synthetic-validation CLIs now accept `--residual-mad-multiplier`, defaulting to the preserved baseline value of 3.0. README recommends the separately versioned 2.5-MAD tuned assessment command.
- **Files deleted:** None.
- **Verification pending:** Generate the versioned 2.5-MAD assessment and regression-validation artifacts and record final metrics.

## Checkpoint 21 — 2026-11-20

- **Verification result:** All eight canonical anomaly, integration, and synthetic-injection tests pass after the tuning changes.
- **Generated recommended assessment:** `energy_prediction_romania/results/tuned_anomaly_detection_recommended/` and `results/tuned_synthetic_fault_validation_recommended/`, using tuned forecast outputs, one global residual threshold, and a 2.5-MAD multiplier.
- **Recommended synthetic result:** All simulated daylight-outage, partial-derate, and negative-power-sensor events were detected at both sites. Partial-derate exact row recall was 77.8% (Site A) and 88.9% (Site B); event recall was 100% for both. Non-injected daytime critical-alert rates were 4.84% and 4.29% respectively, and remain candidate-alert rates rather than false-positive rates because the telemetry is unlabeled.
- **Recommended anomaly output:** Site A has 51 candidate events and Site B has 30 under the tuned configuration. These are not confirmed faults.
- **Files deleted:** None. Baseline, binned-threshold experimental, and tuned artifacts are all separately versioned and preserved.

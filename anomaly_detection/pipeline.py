"""Reusable, daylight-aware solar-PV anomaly detection pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import AnomalyConfig
from .schema import prepare_telemetry


STATUS_NIGHT = "NIGHTTIME_INACTIVE"
STATUS_NORMAL = "DAYLIGHT_NORMAL"
STATUS_DATA_QUALITY = "DATA_QUALITY_ANOMALY"
STATUS_OPERATIONAL = "POTENTIAL_OPERATIONAL_ANOMALY"
STATUS_RESIDUAL = "STATISTICAL_RESIDUAL_EXCEEDANCE"
STATUS_NOVELTY = "ENVIRONMENTAL_NOVELTY"


class AnomalyPipeline:
    """Fit on historical telemetry and score new observations without leakage.

    A forecast column is optional. When provided at both fit and score time, the
    pipeline calibrates a daylight-only MAD threshold for underperformance. In
    its absence, the output remains a rules-and-novelty assessment rather than
    pretending to identify a performance deficit.
    """

    def __init__(self, config: AnomalyConfig | None = None):
        self.config = config or AnomalyConfig()
        self.model_: Pipeline | None = None
        self.feature_columns_: list[str] = []
        self.residual_median_kw_: float | None = None
        self.residual_sigma_kw_: float | None = None
        self.residual_underperformance_threshold_kw_: float | None = None
        self.residual_prediction_bin_edges_: list[float] = []
        self.residual_underperformance_thresholds_by_bin_: list[float] = []
        self.contamination_: float | None = None

    def _daylight_mask(self, frame: pd.DataFrame) -> pd.Series:
        config = self.config
        daylight = frame[config.irradiance_column] > config.daylight_irradiance_wm2
        if config.use_solar_zenith_gate and config.solar_zenith_column in frame.columns:
            daylight &= frame[config.solar_zenith_column] < config.maximum_solar_zenith_deg
        return daylight.fillna(False)

    def _night_mask(self, frame: pd.DataFrame) -> pd.Series:
        config = self.config
        night = frame[config.irradiance_column] <= config.maximum_night_irradiance_wm2
        if config.use_solar_zenith_gate and config.solar_zenith_column in frame.columns:
            night |= frame[config.solar_zenith_column] >= config.maximum_solar_zenith_deg
        return night.fillna(False)

    def _rules(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Evaluate deterministic quality and physical-consistency rules."""
        config = self.config
        flags = pd.DataFrame(index=frame.index)
        power = frame[config.power_column]
        ghi = frame[config.irradiance_column]
        daylight = self._daylight_mask(frame)
        night = self._night_mask(frame)

        flags["Invalid_Timestamp"] = frame[config.timestamp_column].isna()
        flags["Missing_Core_Telemetry"] = power.isna() | ghi.isna()
        flags["Negative_Irradiance"] = ghi < 0
        flags["Extreme_Irradiance"] = ghi > config.maximum_irradiance_wm2
        flags["Negative_Power"] = power < 0
        flags["Night_Generation"] = night & (power > config.nighttime_generation_kw)
        flags["Daylight_Zero_Output"] = (
            daylight
            & (ghi >= config.daytime_zero_irradiance_wm2)
            & (power <= config.daytime_zero_power_kw)
        )

        if config.temperature_column in frame.columns:
            temperature = frame[config.temperature_column]
            flags["Extreme_Temperature"] = (
                (temperature < config.minimum_temperature_c)
                | (temperature > config.maximum_temperature_c)
            )
        else:
            flags["Extreme_Temperature"] = False

        if config.wind_speed_column in frame.columns:
            flags["Extreme_Wind"] = frame[config.wind_speed_column] > config.maximum_wind_speed
        else:
            flags["Extreme_Wind"] = False

        flags["Sensor_Freeze_GHI"] = daylight & ghi.diff().abs().eq(0)
        return flags.fillna(False).astype(bool)

    def _feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Create a stable novelty-feature matrix from telemetry available post-event."""
        config = self.config
        timestamp = frame[config.timestamp_column]
        features = pd.DataFrame(index=frame.index)
        features["irradiance_wm2"] = frame[config.irradiance_column]
        features["observed_power_kw"] = frame[config.power_column]
        features["hour_sin"] = np.sin(2 * np.pi * timestamp.dt.hour / 24.0)
        features["hour_cos"] = np.cos(2 * np.pi * timestamp.dt.hour / 24.0)
        for output_name, source_column in (
            ("temperature_c", config.temperature_column),
            ("wind_speed", config.wind_speed_column),
            ("solar_zenith_deg", config.solar_zenith_column),
        ):
            if source_column in frame.columns:
                features[output_name] = frame[source_column]
        return features

    def fit(self, reference: pd.DataFrame, prediction_column: str | None = None) -> "AnomalyPipeline":
        """Calibrate novelty and optional residual thresholds on historical data."""
        frame = prepare_telemetry(reference, self.config)
        flags = self._rules(frame)
        daylight = self._daylight_mask(frame)
        quality_issue = flags.drop(columns=["Daylight_Zero_Output"]).any(axis=1)
        eligible = daylight & ~quality_issue & ~flags["Daylight_Zero_Output"]
        if eligible.sum() < 20:
            raise ValueError("At least 20 clean daylight observations are required to fit anomaly detection.")

        self.feature_columns_ = self._feature_frame(frame).columns.tolist()
        rule_rate = float(flags.any(axis=1).mean())
        self.contamination_ = float(np.clip(
            rule_rate,
            self.config.isolation_forest_min_contamination,
            self.config.isolation_forest_max_contamination,
        ))
        self.model_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("isolation_forest", IsolationForest(
                    n_estimators=self.config.isolation_forest_estimators,
                    contamination=self.contamination_,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )),
            ]
        )
        self.model_.fit(self._feature_frame(frame).loc[eligible, self.feature_columns_])

        self.residual_median_kw_ = None
        self.residual_sigma_kw_ = None
        self.residual_underperformance_threshold_kw_ = None
        self.residual_prediction_bin_edges_ = []
        self.residual_underperformance_thresholds_by_bin_ = []
        if prediction_column is not None:
            if prediction_column not in frame.columns:
                raise ValueError(f"Prediction column '{prediction_column}' is not present in calibration data.")
            predicted = pd.to_numeric(frame[prediction_column], errors="coerce")
            residuals = frame[self.config.power_column] - predicted
            calibration = residuals.loc[eligible & residuals.notna()]
            if len(calibration) < 20:
                raise ValueError("At least 20 clean daylight residuals are required for MAD calibration.")
            self.residual_median_kw_ = float(calibration.median())
            mad = float((calibration - self.residual_median_kw_).abs().median())
            self.residual_sigma_kw_ = 1.4826 * mad
            self.residual_underperformance_threshold_kw_ = (
                self.residual_median_kw_
                - self.config.residual_mad_multiplier * self.residual_sigma_kw_
            )
            # Forecast error generally grows with expected generation.  Optional
            # bins calibrate separate robust thresholds without looking at future
            # scored data; sparse bins safely fall back to the global threshold.
            if self.config.residual_prediction_bins > 1:
                calibration_prediction = predicted.loc[calibration.index]
                quantiles = np.linspace(0, 1, self.config.residual_prediction_bins + 1)[1:-1]
                edges = np.unique(np.quantile(calibration_prediction, quantiles)).tolist()
                self.residual_prediction_bin_edges_ = [float(edge) for edge in edges]
                bin_ids = np.digitize(calibration_prediction, self.residual_prediction_bin_edges_)
                thresholds = []
                for bin_id in range(len(self.residual_prediction_bin_edges_) + 1):
                    bin_residuals = calibration.loc[bin_ids == bin_id]
                    if len(bin_residuals) < self.config.residual_min_bin_samples:
                        thresholds.append(self.residual_underperformance_threshold_kw_)
                        continue
                    median = float(bin_residuals.median())
                    sigma = 1.4826 * float((bin_residuals - median).abs().median())
                    thresholds.append(median - self.config.residual_mad_multiplier * sigma)
                self.residual_underperformance_thresholds_by_bin_ = thresholds
        return self

    def score(self, telemetry: pd.DataFrame, prediction_column: str | None = None) -> pd.DataFrame:
        """Return row-level rules, anomaly status, and optional residual evidence."""
        if self.model_ is None:
            raise RuntimeError("Fit the pipeline before scoring telemetry.")
        frame = prepare_telemetry(telemetry, self.config)
        flags = self._rules(frame)
        daylight = self._daylight_mask(frame)
        night = self._night_mask(frame)
        quality_issue = flags.drop(columns=["Daylight_Zero_Output"]).any(axis=1)
        novelty_eligible = daylight & ~quality_issue & ~flags["Daylight_Zero_Output"]

        output = frame.copy()
        output["Novelty_Score"] = np.nan
        output["IsolationForest_Flag"] = False
        if novelty_eligible.any():
            features = self._feature_frame(frame).reindex(columns=self.feature_columns_)
            output.loc[novelty_eligible, "Novelty_Score"] = self.model_.decision_function(
                features.loc[novelty_eligible]
            )
            output.loc[novelty_eligible, "IsolationForest_Flag"] = (
                self.model_.predict(features.loc[novelty_eligible]) == -1
            )

        output["Residual_kW"] = np.nan
        output["Power_Deficit_kW"] = np.nan
        output["Residual_Threshold_kW"] = self.residual_underperformance_threshold_kw_
        residual_alarm = pd.Series(False, index=frame.index)
        if prediction_column is not None:
            if prediction_column not in frame.columns:
                raise ValueError(f"Prediction column '{prediction_column}' is not present in scored data.")
            predicted = pd.to_numeric(frame[prediction_column], errors="coerce")
            residuals = frame[self.config.power_column] - predicted
            output["Residual_kW"] = residuals
            output["Power_Deficit_kW"] = np.maximum(0.0, predicted - frame[self.config.power_column])
            if self.residual_underperformance_threshold_kw_ is not None:
                thresholds = pd.Series(self.residual_underperformance_threshold_kw_, index=frame.index)
                if self.residual_underperformance_thresholds_by_bin_:
                    bin_ids = np.digitize(predicted, self.residual_prediction_bin_edges_)
                    thresholds = pd.Series(
                        np.asarray(self.residual_underperformance_thresholds_by_bin_)[bin_ids],
                        index=frame.index,
                    )
                output["Residual_Threshold_kW"] = thresholds
                residual_alarm = daylight & (residuals < thresholds)

        status = np.full(len(frame), STATUS_NORMAL, dtype=object)
        status[night.to_numpy()] = STATUS_NIGHT
        status[output["IsolationForest_Flag"].to_numpy()] = STATUS_NOVELTY
        status[residual_alarm.to_numpy()] = STATUS_RESIDUAL
        status[flags["Daylight_Zero_Output"].to_numpy()] = STATUS_OPERATIONAL
        status[quality_issue.to_numpy()] = STATUS_DATA_QUALITY
        output["Anomaly_Status"] = status
        output["Data_Quality_Flags"] = flags.apply(
            lambda row: ",".join(row.index[row].tolist()), axis=1
        )
        output["Is_Daylight"] = daylight
        return output

    def aggregate_events(self, scored: pd.DataFrame) -> pd.DataFrame:
        """Merge contiguous non-normal rows into dispatch-ready anomaly events."""
        config = self.config
        required = {config.timestamp_column, "Anomaly_Status"}
        if not required.issubset(scored.columns):
            raise ValueError("Scored telemetry must contain timestamp and Anomaly_Status columns.")
        active = scored.loc[
            ~scored["Anomaly_Status"].isin([STATUS_NORMAL, STATUS_NIGHT])
        ].copy()
        columns = [
            "Event_ID", "Anomaly_Status", "Start_Time", "End_Time", "Observation_Count",
            "Duration_Hours", "Peak_Deficit_kW", "Estimated_Energy_Loss_kWh", "Data_Quality_Flags",
        ]
        if active.empty:
            return pd.DataFrame(columns=columns)

        active = active.sort_values(config.timestamp_column).reset_index(drop=True)
        gap_hours = active[config.timestamp_column].diff().dt.total_seconds().div(3600)
        event_break = (
            active["Anomaly_Status"].ne(active["Anomaly_Status"].shift())
            | gap_hours.gt(config.event_gap_hours)
        )
        active["_event_group"] = event_break.cumsum()
        events = []
        for event_number, (_, group) in enumerate(active.groupby("_event_group"), start=1):
            start = group[config.timestamp_column].min()
            end = group[config.timestamp_column].max()
            duration = max((end - start).total_seconds() / 3600, 0.0)
            deficit = pd.to_numeric(group.get("Power_Deficit_kW"), errors="coerce")
            flags = sorted({flag for value in group["Data_Quality_Flags"] for flag in value.split(",") if flag})
            events.append({
                "Event_ID": f"ANOM-{start.strftime('%Y%m%dT%H%M%SZ')}-{event_number:04d}",
                "Anomaly_Status": group["Anomaly_Status"].iloc[0],
                "Start_Time": start,
                "End_Time": end,
                "Observation_Count": int(len(group)),
                "Duration_Hours": round(duration, 3),
                "Peak_Deficit_kW": round(float(deficit.max()) if deficit.notna().any() else 0.0, 3),
                "Estimated_Energy_Loss_kWh": round(float(deficit.fillna(0).sum()), 3),
                "Data_Quality_Flags": ",".join(flags),
            })
        return pd.DataFrame(events, columns=columns)

    def save(self, path: str | Path) -> None:
        """Persist this fitted pipeline with its configuration and calibration."""
        if self.model_ is None:
            raise RuntimeError("Only a fitted pipeline can be saved.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @staticmethod
    def load(path: str | Path) -> "AnomalyPipeline":
        """Load a pipeline saved with :meth:`save`."""
        pipeline = joblib.load(path)
        if not isinstance(pipeline, AnomalyPipeline):
            raise TypeError("Artifact does not contain an AnomalyPipeline.")
        return pipeline

# ============================================================
# HYBRID MODEL PLOTS
# Meta Model: Random Forest + XGBoost -> Linear Regression
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model  import LinearRegression
from sklearn.ensemble      import RandomForestRegressor, StackingRegressor
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
from xgboost               import XGBRegressor

# ── Output directory ──────────────────────────────────────────────────────────
PLOT_DIR = os.path.dirname(os.path.abspath(__file__))   # saves into plots/energy_prediction/
os.makedirs(PLOT_DIR, exist_ok=True)
PROJECT_ROOT = os.path.abspath(os.path.join(PLOT_DIR, "..", ".."))

# ── Colour palette ────────────────────────────────────────────────────────────
C_RF    = "#2196F3"   # blue   – Random Forest
C_XGB   = "#FF9800"   # orange – XGBoost
C_META  = "#4CAF50"   # green  – Hybrid / Meta
C_ACTUAL= "#9C27B0"   # purple – Actual values
C_LR    = "#F44336"   # red    – Linear Regression (baseline)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor"  : "#F9F9F9",
    "axes.grid"       : True,
    "grid.alpha"      : 0.4,
    "font.family"     : "DejaVu Sans",
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 11,
})


# ============================================================
# 1. LOAD DATA
# ============================================================

candidates = [
    os.path.join(PROJECT_ROOT, "datasets", "energy_prediction", "demo_datasets", "Solar_Weather_Data_Processed.csv"),
    os.path.join(PROJECT_ROOT, "Solar Weather Processed Data", "Solar_Weather_Data_Processed.csv"),
    os.path.join(PROJECT_ROOT, "datasets", "energy_prediction", "processed", "Solar_Weather_Data_Cleaned.csv"),
    os.path.join(PROJECT_ROOT, "Data Preprocessing", "Solar_Weather_Data_Cleaned.csv"),
]
DATA_PATH = next((c for c in candidates if os.path.exists(c)), candidates[0])

print(f"Loading dataset from:\n  {DATA_PATH}\n")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.replace(" ", "_")

features = [
    "GHI", "Temperature", "Relative_Humidity",
    "Wind_Speed", "Wind_Direction",
    "Hour", "Day", "Month", "SZ_Angle", "SA_Angle"
]
target = "AC_Output"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 2. TRAIN MODELS
# ============================================================
print("Training Linear Regression ...")
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("Training Random Forest ...")
model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

print("Training XGBoost ...")
model_xgb = XGBRegressor(
    n_estimators=100, learning_rate=0.05,
    max_depth=4, random_state=42, n_jobs=-1
)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Training Hybrid Meta Model (RF + XGBoost -> Linear Regression) ...")
model_meta = StackingRegressor(
    estimators=[
        ("random_forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("xgboost",       XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4,
                                       random_state=42, n_jobs=-1)),
    ],
    final_estimator=LinearRegression(),
    cv=kf,
    n_jobs=-1
)
model_meta.fit(X_train, y_train)
y_pred_meta = model_meta.predict(X_test)

print("All models trained.\n")

# ── Convenience metrics helper ─────────────────────────────────────────────
def metrics(y_true, y_pred):
    return {
        "MAE" : mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2"  : r2_score(y_true, y_pred),
    }

m_lr   = metrics(y_test, y_pred_lr)
m_rf   = metrics(y_test, y_pred_rf)
m_xgb  = metrics(y_test, y_pred_xgb)
m_meta = metrics(y_test, y_pred_meta)


# ============================================================
# PLOT 1 — Actual vs Predicted (Hybrid)
# ============================================================
print("Generating Plot 1: Actual vs Predicted (Hybrid) ...")
fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(y_test, y_pred_meta, alpha=0.45, s=18,
           color=C_META, label="Hybrid (RF + XGB)", zorder=3)

lim = [min(y_test.min(), y_pred_meta.min()) - 0.5,
       max(y_test.max(), y_pred_meta.max()) + 0.5]
ax.plot(lim, lim, "k--", linewidth=1.5, label="Perfect Prediction")
ax.set_xlim(lim); ax.set_ylim(lim)

ax.set_xlabel("Actual AC Output (kW)")
ax.set_ylabel("Predicted AC Output (kW)")
ax.set_title("Actual vs Predicted — Hybrid Meta Model (RF + XGBoost)")
ax.legend()
ax.text(0.04, 0.92,
        f"R² = {m_meta['R2']:.4f}\nMAE = {m_meta['MAE']:.4f} kW\nRMSE = {m_meta['RMSE']:.4f} kW",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

plt.tight_layout()
path1 = os.path.join(PLOT_DIR, "hybrid_actual_vs_predicted.png")
plt.savefig(path1, dpi=150)
plt.close()
print(f"  Saved -> {path1}")


# ============================================================
# PLOT 2 — Residuals (Hybrid)
# ============================================================
print("Generating Plot 2: Residuals ...")
residuals_meta = y_test.values - y_pred_meta

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 2a  Residual scatter
axes[0].scatter(y_pred_meta, residuals_meta, alpha=0.4, s=16,
                color=C_META, zorder=3)
axes[0].axhline(0, color="black", linewidth=1.2, linestyle="--")
axes[0].set_xlabel("Predicted AC Output (kW)")
axes[0].set_ylabel("Residual (Actual − Predicted)")
axes[0].set_title("Residual Plot — Hybrid Meta Model")

# 2b  Residual histogram
axes[1].hist(residuals_meta, bins=50, color=C_META, edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="black", linewidth=1.2, linestyle="--")
axes[1].set_xlabel("Residual (kW)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution — Hybrid Meta Model")
axes[1].text(0.96, 0.92, f"Mean = {residuals_meta.mean():.4f}\nStd  = {residuals_meta.std():.4f}",
             transform=axes[1].transAxes, ha="right", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

plt.tight_layout()
path2 = os.path.join(PLOT_DIR, "hybrid_residuals.png")
plt.savefig(path2, dpi=150)
plt.close()
print(f"  Saved -> {path2}")


# ============================================================
# PLOT 3 — Model Comparison: MAE / RMSE / R²
# ============================================================
print("Generating Plot 3: Model Comparison Bar Charts ...")

model_names = ["Linear\nRegression", "Random\nForest", "XGBoost", "Hybrid\n(RF+XGB)"]
mae_vals  = [m_lr["MAE"],  m_rf["MAE"],  m_xgb["MAE"],  m_meta["MAE"]]
rmse_vals = [m_lr["RMSE"], m_rf["RMSE"], m_xgb["RMSE"], m_meta["RMSE"]]
r2_vals   = [m_lr["R2"],   m_rf["R2"],   m_xgb["R2"],   m_meta["R2"]]
colors    = [C_LR, C_RF, C_XGB, C_META]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Comparison — Holdout Test Set", fontsize=14, fontweight="bold", y=1.01)

for ax, vals, ylabel, title, best_fn in [
    (axes[0], mae_vals,  "MAE (kW)",  "Mean Absolute Error v",    min),
    (axes[1], rmse_vals, "RMSE (kW)", "Root Mean Squared Error v", min),
    (axes[2], r2_vals,   "R²",        "R² Score ^",               max),
]:
    bars = ax.bar(model_names, vals, color=colors, edgecolor="white", linewidth=0.8)
    # highlight best bar
    best_val = best_fn(vals)
    for bar, val in zip(bars, vals):
        if val == best_val:
            bar.set_edgecolor("black")
            bar.set_linewidth(2)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.18)

plt.tight_layout()
path3 = os.path.join(PLOT_DIR, "hybrid_model_comparison.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> {path3}")


# ============================================================
# PLOT 4 — Feature Importance (RF + XGBoost base models)
# ============================================================
print("Generating Plot 4: Feature Importance ...")

rf_imp  = model_rf.feature_importances_
xgb_imp = model_xgb.feature_importances_

imp_df = pd.DataFrame({
    "Feature"      : features,
    "Random Forest": rf_imp,
    "XGBoost"      : xgb_imp,
}).sort_values("Random Forest", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Feature Importance — Hybrid Model Base Learners", fontsize=14, fontweight="bold")

for ax, col, color, title in [
    (axes[0], "Random Forest", C_RF,  "Random Forest Feature Importance"),
    (axes[1], "XGBoost",       C_XGB, "XGBoost Feature Importance"),
]:
    ax.barh(imp_df["Feature"], imp_df[col], color=color, edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    for i, (feat, val) in enumerate(zip(imp_df["Feature"], imp_df[col])):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, imp_df[col].max() * 1.25)

plt.tight_layout()
path4 = os.path.join(PLOT_DIR, "hybrid_feature_importance.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> {path4}")


# ============================================================
# PLOT 5 — K-Fold Cross-Validation R² per Fold (all models)
# ============================================================
print("Generating Plot 5: K-Fold CV R² per Fold ...")

cv_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest"    : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost"          : XGBRegressor(n_estimators=100, learning_rate=0.05,
                                       max_depth=4, random_state=42, n_jobs=-1),
    "Hybrid (RF+XGB)"  : StackingRegressor(
        estimators=[
            ("rf",  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=100, learning_rate=0.05,
                                  max_depth=4, random_state=42, n_jobs=-1)),
        ],
        final_estimator=LinearRegression(), cv=kf, n_jobs=-1
    ),
}
cv_colors = [C_LR, C_RF, C_XGB, C_META]

fig, ax = plt.subplots(figsize=(10, 5))
fold_labels = [f"Fold {i+1}" for i in range(5)]
x = np.arange(5)
width = 0.2

for idx, (name, mdl) in enumerate(cv_models.items()):
    scores = cross_validate(mdl, X, y, cv=kf, scoring="r2", n_jobs=-1)
    fold_r2 = scores["test_score"]
    bars = ax.bar(x + idx * width, fold_r2, width=width,
                  label=name, color=cv_colors[idx], alpha=0.85, edgecolor="white")

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(fold_labels)
ax.set_ylabel("R² Score")
ax.set_title("5-Fold Cross-Validation R² per Fold")
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0, 1.05)

plt.tight_layout()
path5 = os.path.join(PLOT_DIR, "hybrid_kfold_r2_per_fold.png")
plt.savefig(path5, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> {path5}")


# ============================================================
# PLOT 6 — Hybrid: Actual vs Predicted (time-series sample)
# ============================================================
print("Generating Plot 6: Time-Series Sample (first 150 test points) ...")

n = 150
idx_range = range(n)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(idx_range, y_test.values[:n],   color=C_ACTUAL, linewidth=1.5,
        label="Actual AC Output", zorder=4)
ax.plot(idx_range, y_pred_meta[:n],     color=C_META,   linewidth=1.5,
        linestyle="--", label="Hybrid Predicted", zorder=3, alpha=0.9)
ax.fill_between(idx_range,
                y_test.values[:n], y_pred_meta[:n],
                alpha=0.15, color=C_META, label="Error Region")

ax.set_xlabel("Sample Index (Test Set)")
ax.set_ylabel("AC Output (kW)")
ax.set_title("Hybrid Meta Model — Actual vs Predicted (First 150 Test Samples)")
ax.legend(fontsize=10)

plt.tight_layout()
path6 = os.path.join(PLOT_DIR, "hybrid_timeseries_sample.png")
plt.savefig(path6, dpi=150)
plt.close()
print(f"  Saved -> {path6}")


# ============================================================
# PLOT 7 — Overfitting Check: Train R² vs Test R²
# ============================================================
print("Generating Plot 7: Overfitting Check ...")

train_r2 = {
    "Linear\nRegression": r2_score(y_train, model_lr.predict(X_train)),
    "Random\nForest"    : r2_score(y_train, model_rf.predict(X_train)),
    "XGBoost"           : r2_score(y_train, model_xgb.predict(X_train)),
    "Hybrid\n(RF+XGB)"  : r2_score(y_train, model_meta.predict(X_train)),
}
test_r2 = {
    "Linear\nRegression": m_lr["R2"],
    "Random\nForest"    : m_rf["R2"],
    "XGBoost"           : m_xgb["R2"],
    "Hybrid\n(RF+XGB)"  : m_meta["R2"],
}

names_ov = list(train_r2.keys())
x_ov     = np.arange(len(names_ov))
w        = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x_ov - w/2, list(train_r2.values()), width=w,
            label="Training R²", color=[C_LR, C_RF, C_XGB, C_META], alpha=0.6)
b2 = ax.bar(x_ov + w/2, list(test_r2.values()),  width=w,
            label="Test R²",     color=[C_LR, C_RF, C_XGB, C_META], alpha=1.0)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=8)

ax.set_xticks(x_ov)
ax.set_xticklabels(names_ov)
ax.set_ylabel("R² Score")
ax.set_title("Overfitting Check — Training R² vs Test R²")
ax.set_ylim(0, 1.12)
ax.legend()

plt.tight_layout()
path7 = os.path.join(PLOT_DIR, "hybrid_overfitting_check.png")
plt.savefig(path7, dpi=150)
plt.close()
print(f"  Saved -> {path7}")


# ============================================================
# PLOT 8 — Parity Plot: All 4 models side-by-side
# ============================================================
print("Generating Plot 8: Parity Plot (all 4 models) ...")

pred_dict = {
    "Linear Regression" : (y_pred_lr,   C_LR),
    "Random Forest"     : (y_pred_rf,   C_RF),
    "XGBoost"           : (y_pred_xgb,  C_XGB),
    "Hybrid (RF+XGB)"   : (y_pred_meta, C_META),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Parity Plots — All Models vs Hybrid", fontsize=14, fontweight="bold")

for ax, (name, (preds, col)) in zip(axes.flatten(), pred_dict.items()):
    m = metrics(y_test, preds)
    lim = [min(y_test.min(), preds.min()) - 0.3,
           max(y_test.max(), preds.max()) + 0.3]
    ax.scatter(y_test, preds, alpha=0.35, s=14, color=col)
    ax.plot(lim, lim, "k--", linewidth=1.2, label="Perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual AC Output (kW)")
    ax.set_ylabel("Predicted AC Output (kW)")
    ax.set_title(name)
    ax.text(0.04, 0.90,
            f"R²={m['R2']:.4f}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
path8 = os.path.join(PLOT_DIR, "hybrid_parity_all_models.png")
plt.savefig(path8, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> {path8}")


# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 60)
print("ALL HYBRID MODEL PLOTS SAVED TO:")
print(f"  {PLOT_DIR}")
print("=" * 60)
plots_saved = [
    ("hybrid_actual_vs_predicted.png",  "Actual vs Predicted scatter"),
    ("hybrid_residuals.png",            "Residual scatter + histogram"),
    ("hybrid_model_comparison.png",     "MAE / RMSE / R² bar chart"),
    ("hybrid_feature_importance.png",   "RF & XGBoost feature importance"),
    ("hybrid_kfold_r2_per_fold.png",    "5-Fold CV R² per fold"),
    ("hybrid_timeseries_sample.png",    "Time-series sample (150 pts)"),
    ("hybrid_overfitting_check.png",    "Train vs Test R² overfitting check"),
    ("hybrid_parity_all_models.png",    "Parity plots — all 4 models"),
]
for fname, desc in plots_saved:
    print(f"  * {fname:<42} — {desc}")


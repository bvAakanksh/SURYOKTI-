# ============================================================
# METRIC COMPARISON — traintest2.py vs train_test.py models
# ============================================================
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_validate
from sklearn.linear_model   import LinearRegression
from sklearn.ensemble       import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
from sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score
from xgboost                import XGBRegressor

from pathlib import Path

def _find_dataset():
    candidates = [
        Path(__file__).resolve().parents[2] / "datasets" / "energy_prediction" / "demo_datasets" / "Solar_Weather_Data_Processed.csv",
        Path("datasets/energy_prediction/demo_datasets/Solar_Weather_Data_Processed.csv"),
        Path(__file__).resolve().parents[2] / "Solar Weather Processed Data" / "Solar_Weather_Data_Processed.csv",
        Path("Solar Weather Processed Data/Solar_Weather_Data_Processed.csv"),
        Path(r"C:\VITB\Curated Solar Weather and PV AC Power Output Datas\Solar Weather Processed Data\Solar_Weather_Data_Processed.csv"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return r"Solar Weather Processed Data\Solar_Weather_Data_Processed.csv"

DATA = _find_dataset()
df   = pd.read_csv(DATA)
df.columns = df.columns.str.strip().str.replace(" ", "_")

FEATURES = ["GHI","Temperature","Relative_Humidity","Wind_Speed",
            "Wind_Direction","Hour","Day","Month","SZ_Angle","SA_Angle"]
TARGET = "AC_Output"

X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf   = KFold(n_splits=5, shuffle=True, random_state=42)
tscv = TimeSeriesSplit(n_splits=5)

def met(y_true, y_pred):
    return dict(MAE=mean_absolute_error(y_true,y_pred),
                RMSE=np.sqrt(mean_squared_error(y_true,y_pred)),
                R2=r2_score(y_true,y_pred))

def cv_met(model, X, y, cv):
    s = cross_validate(model, X, y, cv=cv,
                       scoring={"MAE":"neg_mean_absolute_error",
                                "RMSE":"neg_root_mean_squared_error","R2":"r2"},
                       n_jobs=-1)
    return dict(CV_MAE=-s["test_MAE"].mean(), CV_RMSE=-s["test_RMSE"].mean(),
                CV_R2=s["test_R2"].mean(),
                CV_R2_std=s["test_R2"].std())

# ── Build models ───────────────────────────────────────────────────────────
model_lr  = LinearRegression()
model_rf  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4,
                          random_state=42, n_jobs=-1)

# traintest2.py hybrid: XGB + Neural Network
model_nn  = Pipeline([("sc", StandardScaler()),
                       ("nn", MLPRegressor(hidden_layer_sizes=(64,32),
                                           activation="relu", solver="adam",
                                           max_iter=1000, random_state=42))])
hybrid_xgb_nn = StackingRegressor(
    estimators=[("xgb", XGBRegressor(n_estimators=100,learning_rate=0.05,
                                      max_depth=4,random_state=42)),
                ("nn", model_nn)],
    final_estimator=LinearRegression(), cv=5)

# train_test.py hybrid: RF + XGB
hybrid_rf_xgb = StackingRegressor(
    estimators=[("rf",  RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)),
                ("xgb", XGBRegressor(n_estimators=100,learning_rate=0.05,
                                      max_depth=4,random_state=42,n_jobs=-1))],
    final_estimator=LinearRegression(), cv=kf, n_jobs=-1)

ALL_MODELS = {
    "Linear Regression"          : model_lr,
    "Random Forest"              : model_rf,
    "XGBoost"                    : model_xgb,
    "Hybrid: XGB+NN (traintest2)": hybrid_xgb_nn,
    "Hybrid: RF+XGB (train_test)" : hybrid_rf_xgb,
}

rows = []
for name, mdl in ALL_MODELS.items():
    print(f"  Training: {name} ...")
    mdl.fit(X_train, y_train)
    y_pred      = mdl.predict(X_test)
    y_pred_tr   = mdl.predict(X_train)
    hm  = met(y_test, y_pred)
    trm = met(y_train, y_pred_tr)

    print(f"    KFold CV ...")
    kf_m = cv_met(mdl, X, y, kf)

    print(f"    TimeSeriesCV ...")
    ts_m = cv_met(mdl, X, y, tscv)

    rows.append({
        "Model"       : name,
        "Holdout MAE" : hm["MAE"],
        "Holdout RMSE": hm["RMSE"],
        "Holdout R2"  : hm["R2"],
        "Train R2"    : trm["R2"],
        "R2 Gap"      : trm["R2"] - hm["R2"],
        "KFold CV R2" : kf_m["CV_R2"],
        "KFold R2 Std": kf_m["CV_R2_std"],
        "TS CV R2"    : ts_m["CV_R2"],
        "TS R2 Std"   : ts_m["CV_R2_std"],
    })

df_res = pd.DataFrame(rows).set_index("Model")
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.5f}".format)

print("\n\n" + "="*80)
print("FULL METRIC COMPARISON TABLE")
print("="*80)
print(df_res.T.to_string())

print("\n\n" + "="*80)
print("LONG-TERM STABILITY RANKING (by TS CV R2 + lowest R2 Gap)")
print("="*80)
df_res["LT_Score"] = df_res["TS CV R2"] - df_res["R2 Gap"] - df_res["TS R2 Std"]
print(df_res[["TS CV R2","TS R2 Std","R2 Gap","LT_Score"]].sort_values("LT_Score", ascending=False).round(5).to_string())


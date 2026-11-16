# Benchmark step B: train on a window of months; step C: predict one batch.
import sys, time, resource, json, glob
import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
mode, files, batch = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
def rss(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
def feats(df):
    X = pd.DataFrame({
        "month": df.FlightDate.dt.month.astype("int8"), "dow": df.FlightDate.dt.dayofweek.astype("int8"),
        "dom": df.FlightDate.dt.day.astype("int8"),
        "airline": df.Reporting_Airline.astype(str).map(hash).mod(1000).astype("int16"),
        "origin": df.Origin.astype(str).map(hash).mod(5000).astype("int16"),
        "dest": df.Dest.astype(str).map(hash).mod(5000).astype("int16"),
        "dep": df.CRSDepTime.astype("float32"), "arr": df.CRSArrTime.astype("float32"),
        "elapsed": df.CRSElapsedTime, "dist": df.Distance})
    return X
t0 = time.time()
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df[df.ArrDel15.isin([0.0, 1.0])]
X, y = feats(df), df.ArrDel15.astype("int8").values
n = len(df); del df
t_load = time.time() - t0; r_load = rss()
t1 = time.time()
if mode == "hgb":
    mdl = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42, early_stopping=False)
else:
    mdl = LogisticRegression(max_iter=200)
    X = X.fillna(0)
mdl.fit(X, y)
t_fit = time.time() - t1
b = pd.read_parquet(batch); Xb = feats(b)
if mode != "hgb": Xb = Xb.fillna(0)
t2 = time.time(); p = mdl.predict_proba(Xb)[:, 1]; t_pred = time.time() - t2
print(json.dumps({"model": mode, "train_months": len(files), "train_rows": n, "load_sec": round(t_load,1), "rss_after_load_MB": r_load,
  "fit_sec": round(t_fit,1), "predict_rows": len(Xb), "predict_sec": round(t_pred,2), "peak_rss_MB": rss()}))

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import config


class MainsMultiLabelPredictor:
    def __init__(
        self,
        data_path: str,
        model_path: str,
        threshold: float = 0.3,
        top_k: int = 10,
    ):
        self.data_path = os.path.abspath(data_path)
        self.model_path = model_path
        self.threshold = float(threshold)
        self.top_k = int(top_k)

        self.scaler = joblib.load("scaler.pkl")
        self.feature_cols = joblib.load("columns.pkl")
        self.devices = joblib.load("devices.pkl")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

        self.df = self._load_table()
        self.ts_col = self._resolve_timestamp_column()
        self._prepare_ranges()
        self.power_col = self._resolve_power_column()

    def robust_convert(self, val) -> float:
        if val is None:
            return 0.0
        if isinstance(val, (float, int, np.floating, np.integer)):
            v = float(val)
            if np.isfinite(v):
                return v
            return 0.0
        s = str(val).strip()
        if not s:
            return 0.0
        s = s.replace(",", ".").replace("CAP", "").replace("IND", "").strip()
        try:
            v = float(s)
            if np.isfinite(v):
                return v
            return 0.0
        except Exception:
            return 0.0

    def normalize_colname(self, x):
        return str(x).strip().lower()

    def _load_table(self):
        if self.data_path.lower().endswith(".csv"):
            return pd.read_csv(self.data_path)
        return pd.read_excel(self.data_path, engine="openpyxl")

    def _resolve_timestamp_column(self):
        cols_norm = {self.normalize_colname(c): c for c in self.df.columns}
        ts_col = cols_norm.get("timestamp", None)
        if ts_col is None:
            raise RuntimeError("Brak kolumny Timestamp w pliku zbiorczym.")
        return ts_col

    def normalize_ts_string(self, s: str) -> str:
        if s is None:
            return ""
        x = str(s).strip()
        x = x.replace("/", ".")
        x = re.sub(r"\s+", " ", x)
        return x

    def parse_dt(self, x):
        s = self.normalize_ts_string(x)
        if not s:
            return pd.NaT
        return pd.to_datetime(s, errors="coerce", dayfirst=True)

    def split_timestamp_range(self, ts):
        s = self.normalize_ts_string(ts)
        if not s:
            return (pd.NaT, pd.NaT)
        if "-" not in s:
            t = self.parse_dt(s)
            return (t, t)
        left, right = s.split("-", 1)
        t1 = self.parse_dt(left.strip())
        t2 = self.parse_dt(right.strip())
        return (t1, t2)

    def _prepare_ranges(self):
        df = self.df.copy()
        df["_t_start"], df["_t_end"] = zip(*df[self.ts_col].map(self.split_timestamp_range))
        df = df.dropna(subset=["_t_start"]).reset_index(drop=True)
        self.df = df

    def _resolve_power_column(self):
        candidates = [
            "Active Power 1 (Cycle) [W]",
            "Active Power Phase 1 Average",
            "Active Power Phase 1",
            "Active Power",
        ]
        cols_norm = {self.normalize_colname(c): c for c in self.df.columns}
        for cand in candidates:
            key = self.normalize_colname(cand)
            if key in cols_norm:
                return cols_norm[key]
        return None

    def _row_to_features(self, row: pd.Series):
        vals = []
        for c in self.feature_cols:
            v = row[c] if c in row.index else 0.0
            vals.append(self.robust_convert(v))
        x = np.array([vals], dtype=np.float32)
        xs = self.scaler.transform(x)
        return xs

    def _select_row_for_timestamp(self, t: pd.Timestamp):
        m = (self.df["_t_start"] <= t) & (t <= self.df["_t_end"])
        if not m.any():
            return None
        cands = self.df.loc[m].copy()
        dur = (cands["_t_end"] - cands["_t_start"]).dt.total_seconds()
        cands["_dur"] = dur.fillna(0.0)
        row = cands.sort_values(["_dur", "_t_start"], ascending=[True, True]).iloc[0]
        return row

    def predict_at(self, timestamp_str: str, threshold: float | None = None, top_k: int | None = None):
        t = self.parse_dt(timestamp_str)
        if pd.isna(t):
            return {"status": "BAD_TS"}

        row = self._select_row_for_timestamp(t)
        if row is None:
            return {"status": "NOT_FOUND"}

        xs = self._row_to_features(row)
        probs = self.model.predict(xs, verbose=0)[0].astype(float)

        preds = [(self.devices[i], float(probs[i])) for i in range(len(self.devices))]
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)

        thr = self.threshold if threshold is None else float(threshold)
        k = self.top_k if top_k is None else int(top_k)

        active = [(d, p) for d, p in preds_sorted if p >= thr]
        top = preds_sorted[: max(1, k)]

        power_w = float("nan")
        if self.power_col is not None and self.power_col in row.index:
            power_w = self.robust_convert(row[self.power_col])

        return {
            "status": "OK",
            "matched_window": f"{row['_t_start']} - {row['_t_end']}",
            "power_w": power_w,
            "active": active,
            "top": top,
        }


def main():
    print("=" * 60)
    print("   NILM PREDICTOR - MAINS MULTI-LABEL")
    print("=" * 60)

    try:
        predictor = MainsMultiLabelPredictor(
            data_path=config.PLIK_ZBIORCZY,
            model_path=config.MODEL_NAME,
            threshold=0.3,
            top_k=10,
        )
        print("Załadowano model, scaler i dane.\n")
    except Exception as e:
        print(f"Błąd inicjalizacji: {e}")
        return

    while True:
        ts = input("Podaj datę (DD.MM.YYYY HH:MM:SS) [q-wyjscie]: ").strip()
        if ts.lower() == "q":
            break

        r = predictor.predict_at(ts)

        if r["status"] == "BAD_TS":
            print("Niepoprawny format daty.")
            print("=" * 60)
            continue

        if r["status"] == "NOT_FOUND":
            print("Nie znaleziono okna, w którym timestamp mieści się w przedziale.")
            print("=" * 60)
            continue

        print("-" * 40)
        print(f"Dopasowane okno: {r['matched_window']}")
        if not np.isnan(r["power_w"]):
            print(f"Active Power:     {r['power_w']:.2f} W")
        print("-" * 40)

        if r["active"]:
            print("AKTYWNE URZĄDZENIA (>= próg):")
            for d, p in r["active"]:
                print(f" - {d.upper():22s} {p * 100:6.2f}%")
        else:
            print("AKTYWNE URZĄDZENIA: brak (poniżej progu)")

        print("TOP:")
        for d, p in r["top"]:
            print(f" - {d.upper():22s} {p * 100:6.2f}%")

        print("=" * 60)


if __name__ == "__main__":
    main()

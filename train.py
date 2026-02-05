import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config


@tf.keras.utils.register_keras_serializable(package="nilm")
class WeightedBCE(tf.keras.losses.Loss):
    def __init__(self, pos_weights, reduction=tf.keras.losses.Reduction.AUTO, name="WeightedBCE"):
        super().__init__(reduction=reduction, name=name)
        self.pos_weights_list = [float(x) for x in pos_weights]
        self.pos_weights = tf.constant(self.pos_weights_list, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        loss_pos = -y_true * tf.math.log(y_pred) * self.pos_weights
        loss_neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(loss_pos + loss_neg)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"pos_weights": self.pos_weights_list})
        return cfg

    @classmethod
    def from_config(cls, config):
        pos_weights = config.pop("pos_weights")
        return cls(pos_weights=pos_weights, **config)


class MainsMultiLabelTrainer:
    def __init__(
        self,
        data_path: str,
        model_path: str,
        target_cols: list[str],
        test_size: float = 0.2,
        random_state: int = 42,
        batch_size: int = 64,
        epochs: int = 120,
        patience: int = 12,
    ):
        self.data_path = os.path.abspath(data_path)
        self.model_path = model_path
        self.target_cols = target_cols
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)

        self.scaler_path = "scaler.pkl"
        self.columns_path = "columns.pkl"
        self.devices_path = "devices.pkl"

        self.df = None
        self.feature_cols = None
        self.device_cols = None
        self.model = None
        self.scaler = None

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

    def load_table(self):
        if self.data_path.lower().endswith(".csv"):
            df = pd.read_csv(self.data_path)
        else:
            df = pd.read_excel(self.data_path, engine="openpyxl")
        self.df = df

    def validate_and_select_columns(self):
        cols_norm = {self.normalize_colname(c): c for c in self.df.columns}

        ts_col = cols_norm.get("timestamp", None)
        if ts_col is None:
            raise RuntimeError("Brak kolumny Timestamp w pliku zbiorczym.")

        device_cols = []
        for c in self.target_cols:
            key = self.normalize_colname(c)
            if key in cols_norm:
                device_cols.append(cols_norm[key])

        if not device_cols:
            raise RuntimeError("Nie znaleziono żadnej kolumny etykiet z TARGET_COLS w pliku zbiorczym.")

        drop_cols = set(device_cols + [ts_col])
        for c in self.df.columns:
            if self.normalize_colname(c) == "surname":
                drop_cols.add(c)

        feature_cols = [c for c in self.df.columns if c not in drop_cols]

        if not feature_cols:
            raise RuntimeError("Brak kolumn cech po odjęciu etykiet i pól meta.")

        self.ts_col = ts_col
        self.device_cols = device_cols
        self.feature_cols = feature_cols

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

    def preprocess(self):
        df = self.df.copy()

        df["_t_start"], df["_t_end"] = zip(*df[self.ts_col].map(self.split_timestamp_range))
        df = df.dropna(subset=["_t_start"]).reset_index(drop=True)

        for c in self.device_cols:
            df[c] = df[c].map(self.robust_convert).astype(np.float32)
            df[c] = (df[c] > 0.5).astype(np.float32)

        X_df = df[self.feature_cols].copy()
        for c in X_df.columns:
            X_df[c] = X_df[c].map(self.robust_convert)
        X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        Y = df[self.device_cols].values.astype(np.float32)
        X = X_df.values.astype(np.float32)

        self.df = df
        return X, Y

    def compute_pos_weights(self, y: np.ndarray) -> np.ndarray:
        eps = 1e-6
        pos = y.sum(axis=0).astype(np.float32)
        neg = (y.shape[0] - pos).astype(np.float32)
        w = neg / (pos + eps)
        w = np.clip(w, 1.0, 50.0)
        return w

    def build_model(self, n_features: int, n_labels: int, pos_weights: np.ndarray):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(n_labels, activation="sigmoid"),
        ])

        model.compile(
            optimizer="adam",
            loss=WeightedBCE(pos_weights),
            metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
        )
        self.model = model

    def save_artifacts(self, scaler: StandardScaler):
        joblib.dump(scaler, self.scaler_path)
        joblib.dump(self.feature_cols, self.columns_path)
        joblib.dump(self.device_cols, self.devices_path)

    def train(self):
        self.load_table()
        self.validate_and_select_columns()
        X, Y = self.preprocess()

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        self.scaler = scaler

        self.save_artifacts(scaler)

        pos_w = self.compute_pos_weights(y_train)
        self.build_model(X_train_s.shape[1], y_train.shape[1], pos_w)

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
        )

        self.model.fit(
            X_train_s,
            y_train,
            validation_data=(X_test_s, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[es],
            verbose=2,
        )

        self.model.save(self.model_path)

def main():
    trainer = MainsMultiLabelTrainer(
        data_path=config.PLIK_ZBIORCZY,
        model_path=config.MODEL_NAME,
        target_cols=list(getattr(config, "TARGET_COLS", [])),
        test_size=0.2,
        random_state=42,
        batch_size=64,
        epochs=120,
        patience=12,
    )
    trainer.train()


if __name__ == "__main__":
    main()

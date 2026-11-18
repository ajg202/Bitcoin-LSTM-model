"""
LSTM time-series forecasting (TensorFlow / Keras).

PREDICT_MODE (set in the CONFIG section):
  "volatility" : predicts how big the price moves will be over the next HORIZON steps
                 (log realized volatility). Volatility clusters, so this is far more
                 predictable than direction. Useful for risk sizing, stop distances, filters.
                 Also compares the LSTM with two baselines ("Persist" = same as lately, and a
                 Ridge linear model on the same features) and prints its bias, so you can see
                 whether the neural net really adds value.
N_SEEDS trains several LSTMs and averages them, and a "seed check" shows how consistent they are.
Volatility mode also runs a volatility-targeting backtest: position size = target volatility / forecast volatility.
  "returns"    : predicts the % change over HORIZON steps and converts it to a price.
                 Includes a fee-aware backtest so you can see if any edge survives trading costs.
  "price"      : predicts the raw value directly (only suitable for data in a stable range).

Features: your raw columns, plus optional technical indicators (returns, rolling volatility,
RSI, MACD, moving-average ratios, range, volume z-score) and time-of-day features.

HOW TO USE
----------
1. pip install tensorflow pandas numpy scikit-learn matplotlib joblib
2. Edit the CONFIG section below (file path + column names).
3. Run:  python lstm_template.py

GPU: TensorFlow only uses an NVIDIA GPU automatically, and on Windows only inside WSL2 (native Windows
runs on the CPU for TensorFlow 2.11+). In WSL2:  pip install "tensorflow[and-cuda]"
The script prints at start-up whether a GPU was found. On a GPU, try BATCH_SIZE = 256 or 512.

Everything below the CONFIG section works without changes.
"""

import json
import os
import random
import webbrowser

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

#=============================================================================
#CONFIG  <-- EDIT THIS SECTION
#=============================================================================
CSV_PATH = "btc_15m_data_2018_to_2025.csv"          #path to dataset (.csv)

DATE_COL = "Open time"              #time column used for sorting; set to None if you don't have one
FEATURE_COLS = [                    #raw input columns (numeric, NOT timestamps)
    "Open", "High", "Low", "Volume",
    "Quote asset volume", "Number of trades",
    "Taker buy base asset volume", "Taker buy quote asset volume",
]
TARGET_COL = "Close"                #price column to forecast (must be > 0 unless PREDICT_MODE = "price")

#optional columns used only to build indicators.
HIGH_COL = "High"
LOW_COL = "Low"
VOLUME_COL = "Volume"

PREDICT_MODE = "volatility"         #"volatility", "returns" or "price"

USE_RAW_FEATURES = True             #include FEATURE_COLS (as step-to-step changes unless mode = "price")
ADD_INDICATORS = True               #add engineered features (RSI, MACD, rolling volatility, ...)
ADD_TIME_FEATURES = True            #add hour-of-day / day-of-week (needs DATE_COL)
INCLUDE_TARGET_AS_FEATURE = True    #include TARGET_COL among the raw features

LOOKBACK = 60                       #how many past steps the model sees per sample
HORIZON = 12                        #steps ahead. Volatility mode: use 10+ (single steps are too noisy).
                                    #must be smaller than LOOKBACK in volatility mode.

VOL_TARGET = "relative"             #volatility mode only:
                                    #"relative": predict future volatility RELATIVE to recent volatility
                                    #(a log ratio). Removes level shifts between periods.
                                    #"absolute": predict the volatility level directly.

CALIBRATE_BIAS = True               #volatility mode only: measure each model's average error on the VALIDATION
                                    #set and subtract it from its predictions (fixes a constant over/under-
                                    #prediction). Test results stay clean; Val numbers become optimistic.

FEE = 0.001                         #backtests (returns and volatility modes): fee per side (0.001 = 0.1%)
BT_MAX_EXPOSURE = 1.0               #volatility backtest: largest position size (1.0 = no leverage; rest is cash)
REPLAY_HTML = True                  #volatility mode: write replay.html, an animated bar-by-bar replay of the test period
REPLAY_BARS = None                  #replay only the last N bars of the test period (None = all of it)
REPLAY_OPEN_BROWSER = True          #try to open replay.html automatically when the run finishes

BT_TUNE_ON_VALIDATION = True        #volatility backtest: choose how often to resize and the no-trade band on the
                                    #VALIDATION period (fewer trades = lower fees). The test period is never used for this.
BT_REBALANCE_EVERY = 1              #used when tuning is off: decide the position every this many blocks of HORIZON steps
BT_REBALANCE_BAND = 0.10            #used when tuning is off: only resize if the position size would change by more than this

TRAIN_FRAC = 0.70                   #chronological split: train / validation / test
VAL_FRAC = 0.15                     #(test = whatever is left)

LSTM_UNITS = [64, 32]               #one entry per stacked LSTM layer
DROPOUT = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 100                        #early stopping will usually end training sooner
BATCH_SIZE = 64                     #on a GPU, 256-512 is usually much faster (re-check your results if you change it)
SEED = 42                           #random seed of the first model; extra models use SEED+1, SEED+2, ...
N_SEEDS = 5                         #train this many LSTMs and average their predictions (an "ensemble").
                                    #More stable than a single run. Training time scales with this; 1 = single model.

OUTPUT_DIR = "lstm_output"          #model, scalers and plots are saved here
#=============================================================================

MODE = PREDICT_MODE
EPS = 1e-8


#----------------------------------------------------------------------------
#Setup / loading
#----------------------------------------------------------------------------
def report_device() -> None:
    """Prints whether TensorFlow can see a GPU, and stops it from grabbing all GPU memory up front."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("Device: no GPU found by TensorFlow -> training on the CPU (see the GPU note at the top of this file).")
        return
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass                                             #GPU already initialised; harmless
    print(f"Device: TensorFlow sees {len(gpus)} GPU(s): {[g.name for g in gpus]} -> training will use the GPU")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def validate_config() -> None:
    if MODE not in ("volatility", "returns", "price"):
        raise ValueError('PREDICT_MODE must be "volatility", "returns" or "price"')
    if int(N_SEEDS) < 1:
        raise ValueError("N_SEEDS must be at least 1")
    if not (USE_RAW_FEATURES or ADD_INDICATORS or (ADD_TIME_FEATURES and DATE_COL)):
        raise ValueError("No input features selected. Enable USE_RAW_FEATURES or ADD_INDICATORS.")
    if MODE == "volatility":
        if VOL_TARGET not in ("relative", "absolute"):
            raise ValueError('VOL_TARGET must be "relative" or "absolute"')
        if LOOKBACK <= HORIZON:
            raise ValueError("In volatility mode LOOKBACK must be larger than HORIZON.")
        if HORIZON < 5:
            print("WARNING: HORIZON < 5 makes volatility very noisy. Try 10 or more.")


def get_feature_cols() -> list:
    cols = list(FEATURE_COLS)
    if INCLUDE_TARGET_AS_FEATURE and TARGET_COL not in cols:
        cols.append(TARGET_COL)
    return cols


def parse_dates(col: pd.Series) -> pd.Series:
    """Handles date strings and epoch timestamps in seconds, milliseconds or microseconds."""
    if pd.api.types.is_numeric_dtype(col):
        m = col.abs().max()
        unit = "us" if m > 1e14 else "ms" if m > 1e11 else "s"
        return pd.to_datetime(col, unit=unit)
    return pd.to_datetime(col)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)

    if DATE_COL:
        if DATE_COL not in df.columns:
            raise ValueError(f"DATE_COL '{DATE_COL}' not found. Available: {list(df.columns)}")
        df[DATE_COL] = parse_dates(df[DATE_COL])
        df = df.sort_values(DATE_COL).reset_index(drop=True)

    needed = [TARGET_COL]
    if USE_RAW_FEATURES:
        needed += get_feature_cols()
    if ADD_INDICATORS:
        needed += [c for c in (HIGH_COL, LOW_COL, VOLUME_COL) if c]
    needed = list(dict.fromkeys(needed))                     #unique, keep order

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}\nAvailable: {list(df.columns)}")

    df = df[([DATE_COL] if DATE_COL else []) + needed].copy()
    df[needed] = df[needed].apply(pd.to_numeric, errors="coerce")

    #catch columns that aren't numeric (e.g. timestamps or text) before they wipe out every row
    all_nan = [c for c in needed if df[c].isna().all()]
    if all_nan:
        raise ValueError(
            f"These columns have no numeric values: {all_nan}\n"
            "They are probably timestamps or text. Remove them from FEATURE_COLS "
            "(use a time column as DATE_COL instead)."
        )

    df[needed] = df[needed].interpolate(limit_direction="both")
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after cleaning. Check your column names and CSV contents.")

    if (MODE != "price" or ADD_INDICATORS) and (df[TARGET_COL] <= 0).any():
        raise ValueError(f"'{TARGET_COL}' has zero/negative values, so returns can't be computed. "
                         'Use PREDICT_MODE = "price" and ADD_INDICATORS = False instead.')
    return df


#----------------------------------------------------------------------------
#Features (every feature at row t uses only rows <= t: no look-ahead)
#----------------------------------------------------------------------------
def make_stationary(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Each column -> step-to-step change (log-change if non-negative, else plain difference)."""
    out = pd.DataFrame(index=feat_df.index)
    for c in feat_df.columns:
        s = feat_df[c]
        out[c] = np.log1p(s).diff() if s.min() >= 0 else s.diff()
    return out


def indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df[TARGET_COL].astype(float)
    logp = np.log(close)
    ret = logp.diff()
    f = pd.DataFrame(index=df.index)

    for k in (1, 5, 15):                                      #momentum over several lags
        f[f"ret_{k}"] = logp.diff(k)
    for w in (10, 30, 100):                                   #rolling volatility (log scale)
        f[f"logvol_{w}"] = np.log(ret.rolling(w).std() + EPS)
    for w in (20, 50):                                        #distance from moving averages
        f[f"ma_ratio_{w}"] = close / close.rolling(w).mean() - 1

    delta = close.diff()                                      #RSI(14), scaled to 0..1
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = (100 - 100 / (1 + gain / (loss + EPS))) / 100

    macd = (close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()) / close
    f["macd"] = macd
    f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    if HIGH_COL and LOW_COL:
        f["log_range"] = np.log((df[HIGH_COL] - df[LOW_COL]).clip(lower=0) / close + EPS)
    if VOLUME_COL:
        lv = np.log1p(df[VOLUME_COL].clip(lower=0))
        f["volume_z"] = (lv - lv.rolling(50).mean()) / (lv.rolling(50).std() + EPS)
    return f


def time_features(dates: pd.Series) -> pd.DataFrame:
    hour = dates.dt.hour + dates.dt.minute / 60.0
    dow = dates.dt.dayofweek
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
    }, index=dates.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    if USE_RAW_FEATURES:
        raw = df[get_feature_cols()]
        parts.append(raw if MODE == "price" else make_stationary(raw))
    if ADD_INDICATORS:
        parts.append(indicator_features(df))
    if ADD_TIME_FEATURES and DATE_COL:
        parts.append(time_features(df[DATE_COL]))
    return pd.concat(parts, axis=1)


#----------------------------------------------------------------------------
#Data preparation
#----------------------------------------------------------------------------
def build_sequences(features: np.ndarray, y_scaled: np.ndarray, lookback: int, horizon: int):
    """
    Window ending at row e uses rows e-lookback+1 ... e and predicts the label at row e+horizon.
    Returns X (samples, lookback, n_features), y (samples,), and e (window-end row of each sample).
    """
    X, y, ends = [], [], []
    for e in range(lookback - 1, len(features) - horizon):
        X.append(features[e - lookback + 1 : e + 1])
        y.append(y_scaled[e + horizon])
        ends.append(e)
    return np.array(X), np.array(y), np.array(ends)


def prepare_data() -> dict:
    validate_config()
    H = HORIZON
    df = load_data()
    prices = df[TARGET_COL].values.astype("float64")

    steps_per_year, step_sec = None, None                    #bars per year (crypto trades 24/7), for annualising
    if DATE_COL:
        step_sec = df[DATE_COL].diff().dt.total_seconds().median()
        if step_sec and step_sec > 0:
            steps_per_year = 365.25 * 24 * 3600 / step_sec

    feat_df = build_features(df).replace([np.inf, -np.inf], np.nan)
    ok = feat_df.notna().all(axis=1).to_numpy()
    if not ok.any():
        raise ValueError("No row has a complete set of features (too few rows for the indicator windows?).")
    start = int(np.argmax(ok))                               #skip warm-up rows of rolling windows
    feat_df = feat_df.iloc[start:].fillna(0.0)
    prices = prices[start:]
    times = ((df[DATE_COL] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()[start:].astype("int64")
             if DATE_COL else None)                          #epoch seconds of each row (for the replay)
    feat_names = list(feat_df.columns)
    feat_raw = feat_df.to_numpy(dtype="float64")
    n = len(prices)

    #Raw label per row t (what the model should output when t is the *target* row):
    #   price      : P[t]
    #   returns    : log(P[t] / P[t-H])                    (H-step return ending at t)
    #   volatility : log( sqrt(mean(r[t-H+1..t]^2)) )      (realized volatility over the H steps ending at t)
    #(volatility + VOL_TARGET "relative": label = level[t] - level[t-H], i.e. the log ratio of the vol over the
    #next H steps to the vol over the previous H steps, where level[t] is the log volatility above)
    y_raw = np.full(n, np.nan)
    level = None                                             #volatility mode: log realized vol of each row
    y_start = H                                              #first row that has a valid label
    if MODE == "price":
        y_raw[H:] = prices[H:]
    elif MODE == "returns":
        y_raw[H:] = np.log(prices[H:] / prices[:-H])
    else:
        r = np.full(n, np.nan)
        r[1:] = np.log(prices[1:] / prices[:-1])
        rv = np.sqrt(pd.Series(r ** 2).rolling(H).mean().to_numpy())
        level = np.full(n, np.nan)
        level[H:] = np.log(rv[H:] + EPS)
        if VOL_TARGET == "relative":
            y_start = 2 * H
            y_raw[y_start:] = level[y_start:] - level[H:-H]
        else:
            y_raw[H:] = level[H:]

    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    if train_end <= y_start + LOOKBACK:
        raise ValueError("Not enough rows for the chosen LOOKBACK/HORIZON. Use more data or smaller values.")

    #Scale (fit on TRAIN rows only to avoid leakage)
    Scaler = MinMaxScaler if MODE == "price" else StandardScaler
    feat_scaler = Scaler().fit(feat_raw[:train_end])
    targ_scaler = Scaler().fit(y_raw[y_start:train_end].reshape(-1, 1))
    feat_scaled = feat_scaler.transform(feat_raw)
    if MODE != "price":
        feat_scaled = np.clip(feat_scaled, -5, 5)            #tame extreme outliers
    y_scaled = np.full(n, np.nan)
    y_scaled[y_start:] = targ_scaler.transform(y_raw[y_start:].reshape(-1, 1)).ravel()

    X, y, ends = build_sequences(feat_scaled, y_scaled, LOOKBACK, H)
    t_idx = ends + H                                         #split by the row the label falls on
    tr = t_idx < train_end
    va = (t_idx >= train_end) & (t_idx < val_end)
    te = t_idx >= val_end
    if min(tr.sum(), va.sum(), te.sum()) == 0:
        raise ValueError("A split is empty. Use more data, a smaller LOOKBACK, or adjust the split fractions.")

    return dict(prices=prices, y_raw=y_raw, level=level, y_start=y_start, steps_per_year=steps_per_year,
                times=times, step_seconds=step_sec,
                feat_scaled=feat_scaled, feat_names=feat_names,
                X=X, y=y, ends=ends, tr=tr, va=va, te=te, train_end=train_end, warmup=start,
                feat_scaler=feat_scaler, targ_scaler=targ_scaler)


#----------------------------------------------------------------------------
#Model
#----------------------------------------------------------------------------
def build_model(lookback: int, n_features: int) -> Sequential:
    model = Sequential()
    model.add(Input(shape=(lookback, n_features)))
    for i, units in enumerate(LSTM_UNITS):
        is_last = i == len(LSTM_UNITS) - 1
        model.add(LSTM(units, return_sequences=not is_last))
        model.add(Dropout(DROPOUT))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    #Huber loss is less thrown off by the occasional huge return spike
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss="huber" if MODE == "returns" else "mse", metrics=["mae"])
    return model


class RidgeBaseline:
    """Linear benchmark on the same inputs: last-step features + window averages. Mimics Keras' predict()."""

    @staticmethod
    def _flat(X: np.ndarray) -> np.ndarray:
        return np.concatenate([X[:, -1, :], X.mean(axis=1)], axis=1)

    def fit(self, X_tr, y_tr, X_va, y_va) -> "RidgeBaseline":
        best = None
        for alpha in (0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6):         #pick regularisation on the validation set
            m = Ridge(alpha=alpha).fit(self._flat(X_tr), y_tr)
            err = mean_squared_error(y_va, m.predict(self._flat(X_va)))
            if best is None or err < best[0]:
                best = (err, alpha, m)
        _, self.alpha, self.model = best
        return self

    def predict(self, X, verbose=0) -> np.ndarray:
        return self.model.predict(self._flat(X)).reshape(-1, 1)


class EnsembleModel:
    """Averages the predictions of several trained Keras models. Mimics Keras' predict()."""

    def __init__(self, models: list):
        self.models = models

    def predict(self, X, verbose=0) -> np.ndarray:
        return np.mean([m.predict(X, verbose=verbose) for m in self.models], axis=0)


def predict_split(D: dict, model, mask) -> dict:
    e = D["ends"][mask]
    t = e + HORIZON
    pred = D["targ_scaler"].inverse_transform(model.predict(D["X"][mask], verbose=0)).ravel()
    out = {}
    if MODE == "volatility":
        lvl = D["level"]
        out["true_logvol"] = lvl[t]
        out["persist_logvol"] = lvl[e]                       #vol over the last H steps ("same as lately")
        out["pred_logvol"] = lvl[e] + pred if VOL_TARGET == "relative" else pred
        out["e"] = e                                                 #row where each forecast is made
    else:
        out["true_price"] = D["prices"][t]
        out["base_price"] = D["prices"][e]                   #last price the model saw
        if MODE == "returns":
            out["true_ret"] = D["y_raw"][t]
            out["pred_ret"] = pred
            out["pred_price"] = D["prices"][e] * np.exp(pred)
        else:
            out["pred_price"] = pred
    return out


#----------------------------------------------------------------------------
#Evaluation
#----------------------------------------------------------------------------
def seed_check(D: dict, models: list, va, te, ridge_rmse: float, ens_rmse: float) -> None:
    """Volatility mode: how consistent are the individual LSTMs, and how often does each beat Ridge?"""
    def test_rmse(m) -> float:
        v, t = predict_split(D, m, va), predict_split(D, m, te)
        shift = float(np.mean(v["pred_logvol"] - v["true_logvol"])) if CALIBRATE_BIAS else 0.0
        return float(np.sqrt(np.mean((t["pred_logvol"] - shift - t["true_logvol"]) ** 2)))

    print(f"\n=== Seed check: each LSTM on its own, calibrated the same way (Ridge test RMSE = {ridge_rmse:.4f}) ===")
    rmses = []
    for i, m in enumerate(models):
        r = test_rmse(m)
        rmses.append(r)
        print(f"  seed {SEED + i:<5} test RMSE {r:.4f}   vs Ridge {1 - r / ridge_rmse:+.1%}")
    wins = sum(r < ridge_rmse for r in rmses)
    print(f"  Beats Ridge in {wins} of {len(models)} seeds | single-model spread {min(rmses):.4f} to {max(rmses):.4f} "
          f"(mean {np.mean(rmses):.4f})")
    print(f"  Ensemble of all {len(models)}: test RMSE {ens_rmse:.4f}   vs Ridge {1 - ens_rmse / ridge_rmse:+.1%}")


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray, bias: bool = False) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    tail = f"  Bias={np.mean(y_pred - y_true):+.4f}" if bias else ""
    print(f"{name:<8} MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}{tail}")


def run_backtest(true_ret: np.ndarray, pred_ret: np.ndarray) -> dict:
    """Long/short on the sign of the predicted return, with fees. Trades don't overlap (every HORIZON steps)."""
    idx = np.arange(0, len(true_ret), HORIZON)
    tr, pr = true_ret[idx], pred_ret[idx]
    min_edge = 2 * FEE                                       #only trade if predicted move beats a round trip
    strategies = {
        "Long/short": np.sign(pr),
        f"Filtered (|pred|>{min_edge:.2%})": np.sign(pr) * (np.abs(pr) > min_edge),
    }
    curves = {"Buy & hold": np.cumsum(tr)}
    print(f"\n=== Backtest on test set (fee {FEE:.3%} per side, {len(idx)} non-overlapping {HORIZON}-step trades) ===")
    print(f"{'Strategy':<26}{'Gross':>10}{'Net':>10}{'Trades':>9}{'Hit rate':>10}{'Fees':>9}")
    for name, pos in strategies.items():
        turnover = np.abs(np.diff(pos, prepend=0.0))         #a long->short flip pays two sides
        gross = pos * tr
        net = gross - FEE * turnover
        active = pos != 0
        hit = f"{np.mean(gross[active] > 0):.1%}" if active.any() else "n/a"
        print(f"{name:<26}{np.expm1(gross.sum()):>10.1%}{np.expm1(net.sum()):>10.1%}"
              f"{int((turnover > 0).sum()):>9d}{hit:>10}{FEE * turnover.sum():>9.1%}")
        curves[name + " (gross)"] = np.cumsum(gross)
        curves[name + " (net)"] = np.cumsum(net)
    print(f"{'Buy & hold':<26}{np.expm1(tr.sum()):>10.1%}")
    return curves


def simulate_position(desired: np.ndarray, period_ret: np.ndarray, band: float):
    """Long-only. The position is only resized when the desired size moves by more than `band`; fee on each change."""
    m = len(desired)
    w, cost = np.zeros(m), np.zeros(m)
    cur, n_reb = 0.0, 0
    for i in range(m):
        if i == 0 or abs(desired[i] - cur) > band:
            cost[i] = FEE * abs(desired[i] - cur)
            cur = desired[i]
            n_reb += 1
        w[i] = cur
    return w * period_ret - cost, w, n_reb, float(cost.sum())


def perf_stats(R: np.ndarray, periods_per_year) -> dict:
    ppy = periods_per_year or 1.0
    eq = np.cumprod(1 + R)
    path = np.r_[1.0, eq]
    sd = R.std(ddof=1)
    return {"total": eq[-1] - 1, "vol": sd * np.sqrt(ppy), "equity": eq,
            "sharpe": R.mean() / sd * np.sqrt(ppy) if sd > 0 else float("nan"),
            "mdd": float((path / np.maximum.accumulate(path) - 1).min())}


def _forecasts(split: str, res: dict, ridge_res: dict = None) -> dict:
    """Log-volatility forecast of every strategy for one split ("Val" or "Test")."""
    r = res[split]
    out = {"LSTM": r["pred_logvol"]}
    if ridge_res is not None:
        out["Ridge"] = ridge_res[split]["pred_logvol"]
    out["Persist"] = r["persist_logvol"]
    out["Oracle*"] = r["true_logvol"]                        #true future vol: an upper bound, not tradable
    return out


def simulate_strategies(D: dict, res: dict, ridge_res: dict, split: str, every: int, band: float) -> dict:
    """
    Volatility targeting on one split. Every `every` blocks of HORIZON steps, size the position as
    min(cap, target vol / forecast vol) and hold it until the next decision. Each forecast's target is the
    median of its own forecasts on the VALIDATION set, so no test information is used.
    """
    P = D["prices"]
    step = every * HORIZON
    e = res[split]["e"]
    idx = np.arange(0, len(e), step)
    idx = idx[e[idx] + step <= len(P) - 1]                   #drop a last, incomplete holding period
    period = P[e[idx] + step] / P[e[idx]] - 1                #simple return of each holding period
    fc, fc_val = _forecasts(split, res, ridge_res), _forecasts("Val", res, ridge_res)
    strat = {}
    for name, f in fc.items():
        target = float(np.median(np.exp(fc_val[name])))
        desired = np.minimum(BT_MAX_EXPOSURE, target / np.exp(f[idx]))
        R, w, n_reb, fees = simulate_position(desired, period, band)
        strat[name] = {"R": R, "w": w, "n_reb": n_reb, "fees": fees, "gross": w * period}
    ppy = D["steps_per_year"] / step if D.get("steps_per_year") else None
    return {"strat": strat, "period": period, "step": step, "ppy": ppy}


def tune_rebalancing(D: dict, res: dict, ridge_res: dict = None):
    """Picks how often to decide and how wide the no-trade band is, using the VALIDATION period only."""
    print("\n=== Choosing the resize frequency on the VALIDATION period (test data is not used) ===")
    print(f"'every' = decide the position every k blocks of {HORIZON} steps; 'band' = smallest position change that is traded.")
    print(f"{'every':>6}{'band':>7}{'Net':>9}{'Sharpe':>8}{'Resizes':>9}{'Fees':>8}")
    best = None
    for k in (1, 4, 16):
        for b in (0.10, 0.25):
            sim = simulate_strategies(D, res, ridge_res, "Val", k, b)
            d = sim["strat"]["LSTM"]
            st = perf_stats(d["R"], sim["ppy"])
            print(f"{k:>6}{b:>7.0%}{st['total']:>9.1%}{st['sharpe']:>8.2f}{d['n_reb']:>9d}{d['fees']:>8.1%}")
            score = np.nan_to_num(st["sharpe"], nan=-9.0)
            if best is None or score > best[0]:
                best = (score, k, b)
    sim = simulate_strategies(D, res, ridge_res, "Val", 1, 0.10)
    bh = sim["period"].copy()
    bh[0] -= FEE
    print(f"(buy & hold on the validation period: Sharpe {perf_stats(bh, sim['ppy'])['sharpe']:.2f})")
    print(f"-> chosen by validation Sharpe: decide every {best[1]} block(s), band {best[2]:.0%}")
    return best[1], best[2]


def run_vol_backtest(D: dict, res: dict, ridge_res: dict = None) -> dict:
    every, band = BT_REBALANCE_EVERY, BT_REBALANCE_BAND
    if BT_TUNE_ON_VALIDATION:
        every, band = tune_rebalancing(D, res, ridge_res)
    sim = simulate_strategies(D, res, ridge_res, "Test", every, band)
    strat, period, step, ppy = sim["strat"], sim["period"], sim["step"], sim["ppy"]

    def constant(level: float):                              #hold a fixed fraction; pay the entry fee once
        R = level * period
        gross = R.copy()
        R[0] -= FEE * level
        return R, gross

    c_level = float(strat["LSTM"]["w"].mean())               #same average exposure as the LSTM strategy
    rows = []
    for name, level in (("Buy & hold", 1.0), (f"Const {c_level:.2f}", c_level)):
        R, g = constant(level)
        rows.append((name, R, g, level, 1, FEE * level))
    for n, d in strat.items():
        rows.append((n if n != "LSTM" else "LSTM vol-target", d["R"], d["gross"], float(d["w"].mean()),
                     d["n_reb"], d["fees"]))

    unit = "annualised" if ppy else "per period (no DATE_COL, so not annualised)"
    print(f"\n=== Volatility-targeting backtest on the test set ({len(period)} decisions, one every {step} steps) ===")
    print(f"Long-only; position = min({BT_MAX_EXPOSURE:g}, target vol / forecast vol), resized only if it moves by more than "
          f"{band:.0%}; fee {FEE:.3%} per side; cash earns 0%. Vol and Sharpe are {unit}.")
    print(f"{'Strategy':<17}{'Gross':>9}{'Net':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>8}{'AvgExp':>8}{'Resizes':>9}{'Fees':>7}")
    stats, curves = {}, {}
    for name, R, g, avg_exp, n_reb, fees in rows:
        st = perf_stats(R, ppy)
        stats[name], curves[name] = st, st["equity"]
        print(f"{name:<17}{np.prod(1 + g) - 1:>9.1%}{st['total']:>9.1%}{st['vol']:>8.1%}{st['sharpe']:>8.2f}"
              f"{st['mdd']:>8.1%}{avg_exp:>8.2f}{n_reb:>9d}{fees:>7.1%}")
    print("* Oracle uses the true future volatility. It is only an upper bound and cannot be traded.")

    l, c, p = stats["LSTM vol-target"], stats[rows[1][0]], stats["Persist"]
    def vs(a: dict, b: dict) -> str:
        dv, dd = a["vol"] - b["vol"], a["mdd"] - b["mdd"]
        return (f"Sharpe {a['sharpe'] - b['sharpe']:+.2f} | volatility {abs(dv) * 100:.1f} pts {'lower' if dv < 0 else 'higher'} | "
                f"max drawdown {abs(dd) * 100:.1f} pts {'shallower' if dd > 0 else 'deeper'} | "
                f"net return {(a['total'] - b['total']) * 100:+.1f} pts")

    print(f"\nLSTM vs constant exposure of the same size: {vs(l, c)}")
    print(f"LSTM vs Persist forecast:                   {vs(l, p)}")
    if ppy:
        years = len(period) / ppy
        se = np.sqrt((1 + 0.5 * stats["Buy & hold"]["sharpe"] ** 2) / years)
        print(f"The test period is about {years:.2f} years, so each Sharpe ratio is only known to roughly +/-{se:.1f}. "
              "Small differences between rows are not reliable.")
    if stats["Oracle*"]["sharpe"] < stats["Buy & hold"]["sharpe"]:
        print("-> Even with perfect knowledge of future volatility (Oracle), this rule does not beat buy & hold after fees. "
              "So fees and the rule are the limit here, not forecast accuracy.")
    print("-> Holding less always cuts risk. Judge the forecast by comparing the LSTM row with 'Const' (same average "
          "exposure, no forecasting) and with Persist. Compare 'Gross' and 'Net' to see how much the fees cost.")
    print("-> This is a research backtest, not trading advice.")
    return {"curves": curves, "exposure": strat["LSTM"]["w"], "step": step, "every": every, "band": band,
            "R": {n: d["R"] for n, d in strat.items()}}


REPLAY_TEMPLATE = r'''<!doctype html><html><head><meta charset="utf-8"><title>Model replay</title>
<style>
:root{--bg:#fff;--fg:#1c1c1c;--mut:#6b7280;--grid:#e5e7eb;--price:#111827;--pred:#e8590c;--act:#1c7ed6;--ridge:#2f9e44;--pers:#868e96;--b1:rgba(232,89,12,.28);--b2:rgba(232,89,12,.12)}
@media(prefers-color-scheme:dark){:root{--bg:#15171c;--fg:#e8eaed;--mut:#9aa0a6;--grid:#2a2e35;--price:#f1f3f5;--pred:#ffa94d;--act:#74c0fc;--ridge:#69db7c;--pers:#adb5bd;--b1:rgba(255,169,77,.3);--b2:rgba(255,169,77,.13)}}
body{margin:0;padding:12px;font:14px system-ui,sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:16px;margin:0 0 4px}small{color:var(--mut)}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0}
button,select{font:inherit;padding:4px 10px}#sl{flex:1;min-width:200px}
canvas{width:100%;display:block}
#info{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:4px 16px;margin-top:8px}
#info div{color:var(--mut)}#info b{color:var(--fg);font-weight:600}
.k{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:2px}
</style></head><body>
<h1>Bitcoin volatility model: replay on unseen test data</h1>
<small>One step = one bar. Orange = forecast made at that moment (the cone is the expected price range for the next bars). Blue = what actually happened, drawn only once that window has finished. This data was never used for training or tuning.</small>
<div class="bar"><button id="play">Play</button><button id="rst">Restart</button>
Speed <select id="spd"><option value="4">4 bars/s</option><option value="15">15</option><option value="40" selected>40</option><option value="120">120</option><option value="400">400</option></select>
View <select id="win"><option value="150">150 bars</option><option value="400" selected>400 bars</option><option value="1000">1000 bars</option></select>
<label><input type="checkbox" id="cr" checked> Ridge</label><label><input type="checkbox" id="cp"> Persist</label>
<input type="range" id="sl" min="0" value="0"></div>
<canvas id="c1"></canvas><canvas id="c2"></canvas><div id="info"></div>
<script>
const D=__DATA__,n=D.p.length,H=D.H,$=id=>document.getElementById(id);
const css=getComputedStyle(document.documentElement),col=k=>css.getPropertyValue(k).trim();
const lg=a=>a.map(v=>Math.log(Math.max(v,1e-6)));
const LP=lg(D.pred),LA=lg(D.actual),LS=lg(D.persist),LR=D.ridge?lg(D.ridge):null;
const cum=L=>{const c=new Float64Array(n+1);for(let j=0;j<n;j++)c[j+1]=c[j]+Math.abs(L[j]-LA[j]);return c};
const cP=cum(LP),cS=cum(LS),cR=LR?cum(LR):null,cH=new Float64Array(n+1);
for(let j=0;j<n;j++)cH[j+1]=cH[j]+(Math.abs(D.ret[j])<=D.pred[j]/100*Math.sqrt(H)?1:0);
const fmt=j=>D.t?new Date(D.t[j]*1e3).toISOString().slice(0,16).replace("T"," ")+" UTC":"bar "+j;
const money=v=>"$"+v.toLocaleString(undefined,{maximumFractionDigits:0});
let pos=0,playing=false,last=0,lastI=-1;$("sl").max=n-1;
function fit(c,h){const r=devicePixelRatio||1,w=c.clientWidth;c.width=w*r;c.height=h*r;c.style.height=h+"px";const x=c.getContext("2d");x.setTransform(r,0,0,r,0,0);return[x,w,h]}
function axes(x,m,w,h,lo,hi,fy,a,i,X){x.strokeStyle=col("--grid");x.fillStyle=col("--mut");x.font="11px system-ui";x.textAlign="right";
 for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4,y=m.t+(h-m.t-m.b)*(1-k/4);x.beginPath();x.moveTo(m.l,y);x.lineTo(w-m.r,y);x.stroke();x.fillText(fy(v),m.l-4,y+4)}
 x.textAlign="center";for(let k=0;k<=4;k++){const j=Math.round(a+(i-a)*k/4);x.fillText(fmt(j).slice(D.t?5:0),Math.min(Math.max(X(j),m.l+40),w-m.r-40),h-5)}}
function draw(i){
 const W=+$("win").value,a=Math.max(0,i-W+1),span=W+H,m={l:62,r:10,t:8,b:20},v=D.pred[i]/100;
 let[x,w,h]=fit($("c1"),300);const X=j=>m.l+(j-a)/span*(w-m.l-m.r);
 let lo=1e18,hi=0;for(let j=a;j<=i;j++){lo=Math.min(lo,D.p[j]);hi=Math.max(hi,D.p[j])}
 lo=Math.min(lo,D.p[i]*Math.exp(-2*v*Math.sqrt(H)))*.995;hi=Math.max(hi,D.p[i]*Math.exp(2*v*Math.sqrt(H)))*1.005;
 const Y=p=>m.t+(h-m.t-m.b)*(1-(p-lo)/(hi-lo));axes(x,m,w,h,lo,hi,money,a,i,X);
 const cone=(s,c)=>{x.beginPath();for(let k=0;k<=H;k++)x.lineTo(X(i+k),Y(D.p[i]*Math.exp(s*v*Math.sqrt(k))));for(let k=H;k>=0;k--)x.lineTo(X(i+k),Y(D.p[i]*Math.exp(-s*v*Math.sqrt(k))));x.closePath();x.fillStyle=c;x.fill()};
 cone(2,col("--b2"));cone(1,col("--b1"));
 x.beginPath();for(let j=a;j<=i;j++)x.lineTo(X(j),Y(D.p[j]));x.strokeStyle=col("--price");x.lineWidth=1.6;x.stroke();
 x.beginPath();x.arc(X(i),Y(D.p[i]),4,0,7);x.fillStyle=col("--price");x.fill();
 // volatility panel
 [x,w,h]=fit($("c2"),210);let mx=0;const vis=(arr,to)=>{for(let j=a;j<=to;j++)mx=Math.max(mx,arr[j])};
 vis(D.pred,i);vis(D.actual,i-H);if($("cp").checked)vis(D.persist,i);if(D.ridge&&$("cr").checked)vis(D.ridge,i);mx*=1.15;
 const Y2=q=>m.t+(h-m.t-m.b)*(1-q/mx);axes(x,m,w,h,0,mx,q=>q.toFixed(2)+"%",a,i,X);
 const ln=(arr,to,c,lw,dash)=>{if(to<a)return;x.beginPath();for(let j=a;j<=to;j++)x.lineTo(X(j),Y2(arr[j]));x.strokeStyle=c;x.lineWidth=lw;x.setLineDash(dash||[]);x.stroke();x.setLineDash([])};
 if($("cp").checked)ln(D.persist,i,col("--pers"),1,[3,3]);if(D.ridge&&$("cr").checked)ln(D.ridge,i,col("--ridge"),1.2);
 ln(D.actual,i-H,col("--act"),1.8);ln(D.pred,i,col("--pred"),2.2);
 x.beginPath();x.moveTo(X(i),Y2(D.pred[i]));x.lineTo(X(i+H),Y2(D.pred[i]));x.strokeStyle=col("--pred");x.setLineDash([5,4]);x.lineWidth=2;x.stroke();x.setLineDash([]);
 x.strokeStyle=col("--mut");x.beginPath();x.moveTo(X(i),m.t);x.lineTo(X(i),h-m.b);x.stroke();
 // readout
 const c=Math.max(0,i-H+1),j=i-H,hl=D.p[i]*Math.exp(-v*Math.sqrt(H)),hh=D.p[i]*Math.exp(v*Math.sqrt(H));
 const k=(c_,t)=>`<span class="k" style="background:${col(c_)}"></span>${t}`;
 $("info").innerHTML=`<div>Time <b>${fmt(i)}</b></div><div>Price <b>${money(D.p[i])}</b></div>
 <div>${k("--pred","Forecast vol, next "+H+" bars")} <b>${D.pred[i].toFixed(3)}%</b>/bar</div>
 <div>Expected 1&sigma; range <b>${money(hl)} to ${money(hh)}</b></div>
 <div>${k("--act","Last finished window")} <b>${j>=0?D.actual[j].toFixed(3)+"%":"n/a"}</b> (forecast was ${j>=0?D.pred[j].toFixed(3)+"%":"n/a"})</div>
 <div>Windows scored so far <b>${c}</b></div>
 <div>Avg log error, model <b>${c?(cP[c]/c).toFixed(3):"n/a"}</b>${D.ridge&&c?" | Ridge <b>"+(cR[c]/c).toFixed(3)+"</b>":""}${c?" | Persist <b>"+(cS[c]/c).toFixed(3)+"</b>":""}</div>
 <div>Price stayed inside the 1&sigma; range <b>${c?(100*cH[c]/c).toFixed(0)+"%":"n/a"}</b> of the time (about 68% if well calibrated)</div>`}
function frame(t){if(playing){pos=Math.min(n-1,pos+(t-last)/1e3*(+$("spd").value));if(pos>=n-1){playing=false;$("play").textContent="Play"}}
 last=t;const i=Math.floor(pos);if(i!==lastI){lastI=i;$("sl").value=i;draw(i)}requestAnimationFrame(frame)}
const redo=()=>{lastI=-1};
$("play").onclick=()=>{if(pos>=n-1)pos=0;playing=!playing;$("play").textContent=playing?"Pause":"Play"};
$("rst").onclick=()=>{pos=0;redo()};$("sl").oninput=e=>{pos=+e.target.value;redo()};
["win","cr","cp"].forEach(id=>$(id).onchange=redo);addEventListener("resize",redo);
addEventListener("keydown",e=>{if(e.code=="Space"){e.preventDefault();$("play").click()}});
requestAnimationFrame(frame);
</script></body></html>
'''


def export_replay(D: dict, res: dict, ridge_res: dict, path: str) -> None:
    """Writes one self-contained HTML file that replays the test period bar by bar (no internet needed)."""
    t, H, P = res["Test"], HORIZON, D["prices"]
    keep = slice(-int(REPLAY_BARS), None) if REPLAY_BARS else slice(None)
    e = t["e"][keep]
    pct = lambda lv: [round(float(v), 4) for v in np.exp(lv[keep]) * 100]     #log-vol -> % per bar
    data = {
        "H": H,
        "p": [round(float(v), 2) for v in P[e]],                              #price when each forecast is made
        "pred": pct(t["pred_logvol"]),
        "persist": pct(t["persist_logvol"]),
        "actual": pct(t["true_logvol"]),                                      #realized vol over the next H bars
        "ret": [round(float(v), 6) for v in np.log(P[e + H] / P[e])],         #actual return over the next H bars
        "ridge": pct(ridge_res["Test"]["pred_logvol"]) if ridge_res else None,
        "t": [int(v) for v in D["times"][e]] if D.get("times") is not None else None,
    }
    html = REPLAY_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReplay written: {os.path.abspath(path)}  ({len(e)} bars, {os.path.getsize(path) / 1e6:.1f} MB). "
          "Open it in a browser and press Play (space bar also works).")
    if REPLAY_OPEN_BROWSER:
        try:
            webbrowser.open("file://" + os.path.abspath(path))
        except Exception:
            pass


def evaluate(D: dict, res: dict, ridge_res: dict = None) -> dict:
    """Prints metrics for the chosen mode. Returns extra objects needed for plotting."""
    t = res["Test"]
    extra = {}

    if MODE == "volatility":
        print(f"\n=== Volatility forecast: log realized volatility over the next {HORIZON} steps "
              f"(target: {VOL_TARGET}) ===")
        print("Bias = average (predicted - actual) in log units; 0 is ideal, negative = under-predicting.")
        for name, r in res.items():
            report(name, r["true_logvol"], r["pred_logvol"], bias=True)
        print("--- baselines (test set) ---")
        report("Persist", t["true_logvol"], t["persist_logvol"], bias=True)
        if ridge_res is not None:
            report("Ridge", t["true_logvol"], ridge_res["Test"]["pred_logvol"], bias=True)
        mean_base = np.full_like(t["true_logvol"], D["level"][HORIZON:D["train_end"]].mean())
        report("Mean", t["true_logvol"], mean_base, bias=True)

        rmse = lambda a: np.sqrt(mean_squared_error(t["true_logvol"], a))
        rm, rp = rmse(t["pred_logvol"]), rmse(t["persist_logvol"])
        corr = np.corrcoef(t["true_logvol"], t["pred_logvol"])[0, 1]
        bias = float(np.mean(t["pred_logvol"] - t["true_logvol"]))
        v = lambda x: np.exp(x) * 100
        print(f"\nCorrelation of predicted vs actual volatility (test): {corr:.3f}")
        print(f"Bias (test): {bias:+.3f} -> predictions are typically {np.exp(bias):.2f}x the actual volatility "
              f"(average per-step vol: actual {v(t['true_logvol']).mean():.3f}%, predicted {v(t['pred_logvol']).mean():.3f}%)")
        print(f"RMSE vs 'Persist' baseline: {1 - rm / rp:+.1%}  (positive = the LSTM beats 'volatility stays as it was lately')")
        if ridge_res is not None:
            gain = 1 - rm / rmse(ridge_res["Test"]["pred_logvol"])
            print(f"RMSE vs 'Ridge' baseline:   {gain:+.1%}  (positive = the LSTM beats a simple linear model)")
            if gain > 0.02:
                verdict = "The LSTM adds value over a linear model on the same features."
            elif gain > -0.02:
                verdict = ("The LSTM is no better than the linear model. Ridge is simpler, faster and easier to "
                           "trust, so prefer it unless the LSTM improves with tuning.")
            else:
                verdict = ("A linear model beats the LSTM here. Common causes: overfitting (try fewer features, smaller "
                           "LSTM_UNITS, more dropout) or a volatility level shift between periods (check Bias above and "
                           "compare VOL_TARGET = \"relative\" vs \"absolute\").")
            print("->", verdict)
        print("-> Volatility is sticky, so 'Persist' is a strong baseline: only a clear win over it AND over Ridge is meaningful.")
        extra["vol_bt"] = run_vol_backtest(D, res, ridge_res)
        return extra

    print("\n=== Price accuracy (original units) ===")
    for name, r in res.items():
        report(name, r["true_price"], r["pred_price"])
    report("Naive", t["true_price"], t["base_price"])           #baseline: "the price won't change"

    if MODE == "returns":
        print(f"\n=== Return accuracy (log-returns over {HORIZON} step(s)) ===")
        for name, r in res.items():
            report(name, r["true_ret"], r["pred_ret"])
        report("Zero", t["true_ret"], np.zeros_like(t["true_ret"]))

        true_dir = np.sign(t["true_ret"])
        dir_acc = np.mean(np.sign(t["pred_ret"]) == true_dir)
        up = np.mean(true_dir > 0)
        print(f"\nDirection accuracy (test): {dir_acc:.1%}")
        print(f"Always-guess-the-majority-direction baseline: {max(up, 1 - up):.1%}  (up moves: {up:.1%})")
        print("-> Return R2 near/below 0 and direction accuracy near the baseline means the model "
              "found little or no predictable signal in these features.")
        extra["curves"] = run_backtest(t["true_ret"], t["pred_ret"])
    return extra


#----------------------------------------------------------------------------
#Main
#----------------------------------------------------------------------------
def main() -> None:
    report_device()
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    H = HORIZON

    D = prepare_data()
    X, tr, va, te = D["X"], D["tr"], D["va"], D["te"]
    vol_note = f" | Vol target: {VOL_TARGET}" if MODE == "volatility" else ""
    print(f"Mode: {MODE}{vol_note} | Horizon: {H} | Lookback: {LOOKBACK} | Rows used: {len(D['prices'])} "
          f"(skipped {D['warmup']} warm-up rows)")
    print(f"{len(D['feat_names'])} features: {D['feat_names']}")
    print(f"Samples -> train: {tr.sum()}, val: {va.sum()}, test: {te.sum()}")

    models, histories = [], []
    for i in range(int(N_SEEDS)):
        seed = SEED + i
        set_seed(seed)
        m = build_model(LOOKBACK, X.shape[2])
        if i == 0:
            m.summary()
        print(f"\n=== Training LSTM {i + 1} of {int(N_SEEDS)} (seed {seed}) ===")
        h = m.fit(
            X[tr], D["y"][tr],
            validation_data=(X[va], D["y"][va]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,           #keep temporal order
            callbacks=[              #fresh callbacks for every model
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
            ],
            verbose=1,
        )
        models.append(m)
        histories.append(h)
        print(f"Seed {seed}: best validation loss {min(h.history['val_loss']):.4f} "
              f"after {len(h.history['val_loss'])} epochs")
    model = EnsembleModel(models)            #all predictions below are the average of the members

    res = {"Train": predict_split(D, model, tr), "Val": predict_split(D, model, va),
           "Test": predict_split(D, model, te)}
    ridge, ridge_res = None, None
    shifts = {"lstm": 0.0, "ridge": 0.0}
    if MODE == "volatility":                                 #linear benchmark on the same inputs
        ridge = RidgeBaseline().fit(X[tr], D["y"][tr], X[va], D["y"][va])
        print(f"Ridge baseline fitted (alpha={ridge.alpha:g})")
        ridge_res = {"Val": predict_split(D, ridge, va), "Test": predict_split(D, ridge, te)}
        if CALIBRATE_BIAS:
            test_rmse = lambda d: float(np.sqrt(np.mean((d["pred_logvol"] - d["true_logvol"]) ** 2)))
            print("\nBias calibration: subtracting each model's average error on the VALIDATION set")
            for key, rs in (("lstm", res), ("ridge", ridge_res)):
                shift = float(np.mean(rs["Val"]["pred_logvol"] - rs["Val"]["true_logvol"]))
                before = test_rmse(rs["Test"])
                for d in rs.values():
                    d["pred_logvol"] = d["pred_logvol"] - shift
                shifts[key] = shift
                print(f"  {key.upper():<6} shift {shift:+.4f}   test RMSE {before:.4f} -> {test_rmse(rs['Test']):.4f}")
            print("  (Val rows below are optimistic since the shift was measured there; Test is the clean number.)")
        if int(N_SEEDS) > 1:
            rm = lambda d: float(np.sqrt(np.mean((d["pred_logvol"] - d["true_logvol"]) ** 2)))
            seed_check(D, models, va, te, rm(ridge_res["Test"]), rm(res["Test"]))
    extra = evaluate(D, res, ridge_res)
    t = res["Test"]

    #---- Plots --------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    for i, h in enumerate(histories):
        plt.plot(h.history["loss"], color="tab:blue", alpha=0.6, label="train loss" if i == 0 else None)
        plt.plot(h.history["val_loss"], color="tab:orange", alpha=0.6, label="val loss" if i == 0 else None)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150)

    if MODE == "volatility":
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(np.exp(t["true_logvol"]) * 100, label="Actual", alpha=0.7)
        ax.plot(np.exp(t["persist_logvol"]) * 100, label="Persist baseline", alpha=0.5)
        if ridge_res is not None:
            ax.plot(np.exp(ridge_res["Test"]["pred_logvol"]) * 100, label="Ridge", alpha=0.6)
        ax.plot(np.exp(t["pred_logvol"]) * 100, label="LSTM")
        ax.set_title(f"Test set: per-step volatility over the next {H} steps (%)")
        ax.set_xlabel("Test sample"); ax.set_ylabel("Volatility (%)"); ax.legend()
    else:
        n_plots = 2 if MODE == "returns" else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4.5 * n_plots), squeeze=False)
        ax = axes[0, 0]
        ax.plot(t["true_price"], label="Actual")
        ax.plot(t["pred_price"], label="Predicted")
        ax.set_title(f"Test set: {TARGET_COL} (horizon={H})"); ax.set_ylabel(TARGET_COL); ax.legend()
        if MODE == "returns":
            ax = axes[1, 0]
            ax.plot(t["true_ret"], label="Actual return", alpha=0.7)
            ax.plot(t["pred_ret"], label="Predicted return")
            ax.set_title("Test set: log-returns"); ax.set_xlabel("Test sample"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "test_predictions.png"), dpi=150)

    if "curves" in extra:
        plt.figure(figsize=(12, 5))
        for name, c in extra["curves"].items():
            plt.plot(c, label=name, linestyle="--" if "gross" in name else "-", alpha=0.85)
        plt.title("Backtest: cumulative log-return on the test set"); plt.xlabel(f"Trade (every {H} steps)")
        plt.ylabel("Cumulative log-return"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "backtest.png"), dpi=150)

    if "vol_bt" in extra:
        bt = extra["vol_bt"]
        fig, (a, b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        for name, eq in bt["curves"].items():
            a.plot(eq, label=name, linestyle="--" if name.endswith("*") else "-", alpha=0.85)
        a.set_yscale("log"); a.set_ylabel("Equity (start = 1, log scale)"); a.legend()
        a.set_title("Volatility-targeting backtest on the test set (net of fees)")
        b.plot(bt["exposure"], color="tab:green")
        b.set_ylabel("LSTM exposure"); b.set_xlabel(f"Decision (one every {bt['step']} steps)"); b.set_ylim(0, BT_MAX_EXPOSURE * 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "vol_backtest.png"), dpi=150)

    if MODE == "volatility" and REPLAY_HTML:
        export_replay(D, res, ridge_res, os.path.join(OUTPUT_DIR, "replay.html"))

    #---- Save artifacts -----------------------------------------------------
    for i, m in enumerate(models):
        m.save(os.path.join(OUTPUT_DIR, f"lstm_model_seed{SEED + i}.keras"))
    joblib.dump(D["feat_scaler"], os.path.join(OUTPUT_DIR, "feature_scaler.pkl"))
    joblib.dump(D["targ_scaler"], os.path.join(OUTPUT_DIR, "target_scaler.pkl"))
    joblib.dump({"mode": MODE, "lookback": LOOKBACK, "horizon": H, "features": D["feat_names"]},
                os.path.join(OUTPUT_DIR, "config.pkl"))

    #---- Forecast beyond the end of the data --------------------------------
    last_window = D["feat_scaled"][-LOOKBACK:].reshape(1, LOOKBACK, -1)
    pred = D["targ_scaler"].inverse_transform(model.predict(last_window, verbose=0))[0, 0]
    last_price = D["prices"][-1]
    print(f"\nLast actual {TARGET_COL}: {last_price:.4f}")
    if MODE == "volatility":
        lvl_now = D["level"][-1]
        to_vol = (lambda p: np.exp(lvl_now + p)) if VOL_TARGET == "relative" else np.exp
        v_pred, v_now = to_vol(pred - shifts["lstm"]), np.exp(lvl_now)
        print(f"Predicted per-step volatility over the next {H} steps: {v_pred:.3%} "
              f"(about {v_pred * np.sqrt(H):.2%} over the whole horizon, if moves are independent)")
        if ridge is not None:
            r_pred = D["targ_scaler"].inverse_transform(ridge.predict(last_window))[0, 0]
            print(f"Ridge baseline forecast: {to_vol(r_pred - shifts['ridge']):.3%}")
        print(f"Realized per-step volatility over the last {H} steps: {v_now:.3%}")
    elif MODE == "returns":
        print(f"Forecast {TARGET_COL}, {H} step(s) ahead: {last_price * np.exp(pred):.4f} "
              f"({np.expm1(pred):+.3%} predicted change)")
    else:
        print(f"Forecast {TARGET_COL}, {H} step(s) ahead: {pred:.4f}")
    print(f"Saved model, scalers and plots to ./{OUTPUT_DIR}/")

    plt.show()


if __name__ == "__main__":
    main()
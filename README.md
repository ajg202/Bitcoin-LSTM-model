# Bitcoin Volatility Forecasting with LSTM Neural Networks

A time-series forecasting project that uses an ensemble of LSTM networks to predict how large Bitcoin's price moves will be over the next 3 hours (12 bars of 15 minutes), then tests whether that forecast is useful for trading.

> **Research project, not financial advice.** Nothing here is a trading recommendation. See [Limitations](#limitations).

## Key findings

**The forecasting works.** On a later, unseen time period (about 1.08 years of 15-minute data):

| Model | Test RMSE (log volatility) | R² | vs. LSTM ensemble |
|---|---|---|---|
| **LSTM ensemble (5 seeds)** | **0.4126** | **0.492** | n/a |
| Ridge regression (same features) | 0.4404 | 0.421 | LSTM 6.3% lower RMSE |
| Persistence ("same as the last 12 bars") | 0.5121 | 0.218 | LSTM 19.4% lower RMSE |
| Historical average | 0.6205 | -0.149 | n/a |

- Predicted and actual volatility have a correlation of 0.719.
- Every one of the 5 random seeds beat Ridge on its own (by 4.5% to 6.6%).
- The 1σ price range implied by the forecast contained the actual outcome about 70% of the time (about 68% would be expected if well calibrated).
- **Caveat:** the win over Ridge depends on a simple bias correction fitted on validation data. Without it, the ensemble scored 0.4496 and lost to Ridge (0.4396).

**Better forecasts did not turn into a better trading strategy.** A long-only volatility-targeting backtest (position size = target volatility / forecast volatility, with a 0.1% fee per side) gave:

| Strategy | Net return | Sharpe | Max drawdown |
|---|---|---|---|
| Buy and hold | 64.2% | 1.11 | -30.5% |
| Constant 76% exposure (no forecasting) | 50.0% | 1.11 | -23.2% |
| **LSTM volatility targeting** | 18.8% | 0.61 | -27.7% |
| Ridge volatility targeting | 13.2% | 0.49 | n/a |
| Persistence volatility targeting | 0.3% | 0.21 | -34.2% |
| Oracle (perfect knowledge of future volatility) | 43.3% | 1.20 | -20.8% |

- The LSTM strategy ranked ahead of the Ridge and persistence strategies, but lost to simply holding a constant 76% of Bitcoin.
- Even the oracle barely beat constant exposure on Sharpe and still trailed on net return, so the limit here is the trading rule and costs, not forecast accuracy.
- With about one year of data, each Sharpe ratio is only known to roughly ±1.2. Small differences between rows are not reliable.
- Transaction costs mattered: a first version that resized its position every 3 hours paid about 31% of capital in fees. Deciding less often (chosen on validation data only) cut that to about 6%.

**Earlier experiments with the same data:** predicting the raw price failed because the model cannot output prices above the range it trained on (its forecast flattened near the training maximum while Bitcoin kept rising). Predicting returns, using only the raw columns in that early version, found essentially no signal (direction accuracy 51.9% against a 50.3% majority baseline).

## How it works

1. **Load and clean** the CSV: sort by time, convert columns to numbers, fill small gaps.
2. **Build features** from each row using only past data:
   - raw columns as step-to-step changes,
   - technical indicators: multi-lag returns, rolling volatility (10, 30 and 100 bars), RSI, MACD, moving-average distance, high-low range, volume z-score,
   - hour-of-day and day-of-week.
3. **Split chronologically** into 70% train, 15% validation and 15% test, with no shuffling.
4. **Create the label**: realized volatility over the next 12 bars, expressed as a log ratio to the volatility over the previous 12 bars (`VOL_TARGET = "relative"`).
5. **Train** 5 stacked-LSTM models with different random seeds, using early stopping, and average their predictions.
6. **Compare against baselines**: persistence, Ridge regression, and the historical average.
7. **Calibrate**: remove each model's average error measured on the validation set.
8. **Backtest** the volatility-targeting strategy on the test period, and write an animated replay.

### Safeguards against fooling yourself

- Scalers are fitted on training rows only.
- Every feature at time *t* uses only data up to *t* (checked by recomputing features on truncated data).
- The backtest's trading frequency and no-trade band are chosen on the validation period, never the test period.
- The backtest is net of fees, shows gross and net returns side by side, and includes a constant-exposure benchmark and an oracle upper bound for context.
- The data handling and backtest arithmetic were verified against independent recalculations from raw prices.

## Quick start

**1. Install**

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib joblib
```

**2. Get data.** The project was built on the Kaggle dataset
[Bitcoin Historical Datasets 2018-2024](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024)
(15-minute candles with Binance-style columns). Check the dataset page for its license and exact date coverage. Any CSV with a time column and a price column can work.

**3. Edit the `CONFIG` section** at the top of `lstm_template.py`: at minimum `CSV_PATH`, and if your column names differ, `DATE_COL`, `FEATURE_COLS` and `TARGET_COL`.

**4. Run**

```bash
python lstm_template.py
```

Training 5 models takes a while on a CPU. For a quick test, set `N_SEEDS = 1`.

### GPU

TensorFlow uses an NVIDIA GPU automatically when it can see one, and the script prints a `Device:` line at start-up. On Windows, TensorFlow 2.11 and later only supports GPUs through WSL2 (inside WSL2: `pip install "tensorflow[and-cuda]"`). On a GPU, `BATCH_SIZE = 256` or `512` is usually much faster, but changing it changes the results slightly.

## Configuration

All settings live in the `CONFIG` block at the top of `lstm_template.py`.

| Setting | Default | What it does |
|---|---|---|
| `CSV_PATH` | `your_data.csv` | Path to your dataset |
| `DATE_COL` / `FEATURE_COLS` / `TARGET_COL` | Binance-style names | Time column, raw input columns, price column |
| `PREDICT_MODE` | `"volatility"` | `"volatility"`, `"returns"` or `"price"` |
| `LOOKBACK` / `HORIZON` | 60 / 12 | Past bars the model sees / bars ahead to forecast |
| `VOL_TARGET` | `"relative"` | Predict volatility as a ratio to recent volatility, or as an absolute level |
| `CALIBRATE_BIAS` | `True` | Remove average error measured on the validation set |
| `N_SEEDS` | 5 | Number of LSTMs averaged in the ensemble |
| `LSTM_UNITS`, `DROPOUT`, `LEARNING_RATE`, `EPOCHS`, `BATCH_SIZE` | `[64, 32]`, 0.2, 1e-3, 100, 64 | Network and training settings (early stopping ends training sooner) |
| `FEE` | 0.001 | Backtest fee per side (0.1%) |
| `BT_MAX_EXPOSURE` | 1.0 | Largest backtest position (1.0 = no leverage) |
| `BT_TUNE_ON_VALIDATION` | `True` | Choose trading frequency and no-trade band using validation data |
| `REPLAY_HTML`, `REPLAY_BARS`, `REPLAY_OPEN_BROWSER` | `True`, `None`, `True` | Control the replay file |
| `TRAIN_FRAC` / `VAL_FRAC` | 0.70 / 0.15 | Chronological split (test is the remainder) |
| `SEED` | 42 | Seed of the first model (others use `SEED+1`, ...) |

### Modes

- **`volatility`** (default, the focus of this project): forecasts the size of price moves, compares against persistence and Ridge, and runs the volatility-targeting backtest and replay.
- **`returns`**: forecasts the percentage change and converts it back to a price. Reports direction accuracy and runs a fee-aware long/short backtest. In early experiments it found little or no signal.
- **`price`**: forecasts the raw price. Only suitable for data that stays in a stable range.

## Output

Everything is saved to `lstm_output/`:

| File | Contents |
|---|---|
| `replay.html` | Animated replay of the test period (volatility mode) |
| `test_predictions.png` | Predicted vs actual on the test set |
| `loss_curve.png` | Training and validation loss for every seed |
| `vol_backtest.png` | Equity curves and exposure of the backtest (volatility mode) |
| `backtest.png` | Long/short backtest (returns mode only) |
| `lstm_model_seed{N}.keras` | One trained model per seed |
| `feature_scaler.pkl`, `target_scaler.pkl`, `config.pkl` | Scalers, plus mode, lookback, horizon and feature names |

### The replay

`replay.html` is a single offline file. Open it in a browser and press Play (or the space bar). The price advances one bar at a time, and the model's forecast updates alongside it.

- **Black line:** the actual Bitcoin price.
- **Orange cone (top chart):** the expected price range for the next 12 bars. The darker inner layer is ±1σ and the lighter outer layer is ±2σ.
- **Orange line (bottom chart):** the volatility forecast made at each moment. The dashed extension is the forecast for the upcoming window.
- **Blue line:** the actual realized volatility for the same window, drawn only after those 12 bars have finished, so it trails the orange line.
- **Green line:** the Ridge baseline. **Grey dotted line:** the persistence baseline (off by default).

The readout panel shows running error scores for each model and how often the price stayed inside the 1σ range.

### Reading the console output

- **Bias** is the average (predicted minus actual) in log units. 0 is ideal, and negative means under-predicting.
- **"RMSE vs Persist / Ridge"** shows how much better the LSTM is than each baseline. Positive is better.
- **Seed check** shows each LSTM's score on its own and how many beat Ridge. Prefer a result where the spread between seeds is smaller than the gap to the baseline.
- **Backtest table:** compare the LSTM row with the "Const" row (same average exposure, no forecasting), not just with buy and hold. Holding less always reduces risk.

## Limitations

- **One asset, one market regime, one test period.** The test year was a strong uptrend, which favors strategies that stay invested.
- **The test period was reused during development.** It was never trained on, but results were inspected repeatedly while design choices (the relative target, the bias correction) were made. Treat the numbers as promising research. A fresh out-of-sample check on newer data is the real test.
- **The bias correction is a judgment call** made after seeing results. The correction value itself is computed only from validation data.
- **The backtest is simple:** flat fees, no slippage, no funding costs, cash earns nothing, long-only. It is a research tool, not a trading system.
- **Baselines are simple.** Ridge and persistence are used for comparison. The standard volatility benchmarks in the literature (GARCH and HAR-RV) are not included yet.
- **No walk-forward evaluation:** the model is trained once and tested on one later period.

## Ideas for future work

- Fresh-data check: run the unchanged model on newer data.
- Add HAR-RV and GARCH baselines.
- Scheduled-event features known in advance (central-bank meetings, inflation releases, options expiries) and a data-driven time-of-day volatility profile.
- News-based features (attention and sentiment) from a public crypto-news dataset.
- Walk-forward evaluation across several periods, including bear and sideways markets.

## Project structure

```
lstm_template.py     # the whole pipeline: data, features, models, baselines, backtest, replay
README.md
your_data.csv        # not included: download it yourself
lstm_output/         # created on each run
```

## Acknowledgements

Data: [Bitcoin Historical Datasets 2018-2024](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024) on Kaggle. Please check its license before redistributing the data.

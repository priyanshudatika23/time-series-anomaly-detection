# Time-Series Anomaly Detection Engine

A three-tier anomaly-detection pipeline for financial time-series — **statistical → classic ML → deep learning** — built under a **leakage-free, time-based evaluation**.

Detects flash crashes, volume spikes, and slow structural shifts in daily equity data using three escalating detectors that mirror the evolution of time-series modelling, then contrasts where each one succeeds and fails.

---

## Overview

Anomaly detection on market data is genuinely hard: in a volatile asset, *large moves are partly normal*, which makes "anomaly" hard to define. This project builds three detectors of increasing sophistication and treats the **engineering trade-off between them** as the central result rather than chasing a single accuracy number.

## Architecture

| Tier | Detector | Family | Strength | Blind spot |
|------|----------|--------|----------|-----------|
| 1 | Trailing Z-Score | Statistical | Sharp single-day spikes, near-zero compute | Single feature only |
| 2 | Isolation Forest | Classic ML | Multivariate outliers | Ignores temporal order |
| 3 | LSTM Autoencoder | Deep learning | Slow structural shifts (memory) | Needs lots of data + tuning |

## Key design decisions

- **Leakage-free evaluation.** Time-based train/test split, **every scaler fit on the training window only**, and causal (trailing) statistics throughout — no future information leaks into the test set.
- **Stationary features for the deep tier.** The LSTM is fed returns, rolling volatility, and a volume **z-score** — never raw log-volume, which trends upward over the years and would make the autoencoder flag "a newer regime" instead of true anomalies.
- **Normal-only training.** The autoencoder trains on a mostly-normal window, so genuine anomalies produce high reconstruction error instead of being learned away.
- **Real validation.** Synthetic anomalies are injected at known dates to compute actual precision/recall, then cross-checked qualitatively against known market events.

## Results

*Daily equity data (~2,450 points, ~10 years), 70/30 time split, synthetic-injection validation with ±1-day tolerance.*

| Tier | Precision | Recall | F1 | Real-data flag rate |
|------|-----------|--------|----|--------------------|
| **Isolation Forest** | **88%** | **58%** | **0.70** | 2.2% |
| Z-Score | 60% | 50% | 0.55 | 0.6% |
| LSTM Autoencoder | 25% | 42% | 0.31 | 0.4% |

- The **Isolation Forest** was the strongest tier; its eight most-anomalous days were *all* in the **March–April 2020 COVID crash** — a clean qualitative confirmation against a real-world event.
- **Debugging win:** the LSTM initially flagged 5.5% of test days because raw log-volume's upward trend pushed test inputs outside the training range and inflated reconstruction error. Substituting a **stationary volume z-score** cut the spurious flag rate to **0.4%** and restored valid precision/recall.

> The validation injects *point* anomalies (single-day spikes), which inherently favour the instantaneous detectors (Z-Score, Isolation Forest). The LSTM targets *structural* shifts that unfold over a sequence, so its point-spike score understates its intended role — a planned next step is to inject sustained anomalies to test exactly that.

## Project structure

```
.
├── anomaly_detection.ipynb   # main notebook — run top to bottom
├── requirements.txt          # dependencies (all preinstalled in Colab except yfinance/plotly)
├── README.md
└── LICENSE
```

## Getting started

### Run in Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com).
2. **File → Upload notebook** → select `anomaly_detection.ipynb`.
3. **Runtime → Run all.**

Only the data-download cell needs internet; TensorFlow, scikit-learn, and pandas are already in Colab.

### Run locally

```bash
git clone https://github.com/<your-username>/time-series-anomaly-detection.git
cd time-series-anomaly-detection
pip install -r requirements.txt
jupyter notebook anomaly_detection.ipynb
```

The asset, history length, and all hyperparameters live in the **`CONFIG`** cell near the top (default ticker `RELIANCE.NS`, 10 years of daily data). A long daily history matters — the LSTM tier needs thousands of points, so avoid young tickers.

## How it works

The notebook is organised into four phases:

1. **Data** — pull years of daily price/volume via `yfinance`; handle gaps without breaking the even time spacing the models assume.
2. **Features & preprocessing** — daily returns, 20-day rolling volatility, log-volume, and a stationary volume z-score; time-based split; train-only scaling (StandardScaler for the classic tiers, MinMax for the LSTM).
3. **Three tiers** — trailing z-score, Isolation Forest, and an LSTM autoencoder over overlapping 10-day sequences thresholded at the 99th percentile of training reconstruction error.
4. **Validation & visualization** — synthetic anomaly injection for precision/recall, an interactive master chart with per-detector markers, and an agreement analysis across the three methods.

## Limitations & future work

- Validation injects point anomalies only; adding sustained/structural anomalies would properly showcase the LSTM tier.
- Single asset — a natural extension is a cross-sectional panel of equities.
- Thresholds (z = 3, 99th percentile, 3% contamination) are reasonable, defensible defaults rather than per-asset tuned values.

## License

Released under the MIT License — see [LICENSE](LICENSE).

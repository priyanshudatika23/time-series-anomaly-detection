# Time-Series Anomaly Detection Engine

A four-tier anomaly-detection pipeline for financial time-series — **statistical → classic ML → deep learning → consensus ensemble** — built under a **leakage-free, time-based evaluation**.

Detects flash crashes and volume spikes in daily equity data using escalating detectors that mirror the evolution of time-series modelling, then **fuses them into a consensus ensemble that outperforms every individual tier** — and reports honestly where each one succeeds and fails, including a deep-learning tier that (on this asset) did *not* beat the classical detectors.

---

## Overview

Anomaly detection on market data is genuinely hard: in a volatile asset, *large moves are partly normal*, which makes "anomaly" hard to define. This project builds detectors of increasing sophistication, treats the **engineering trade-off between them** as a central result, and combines them into a **consensus ensemble** whose complementary blind spots yield the best precision/recall balance in the pipeline. It also does something most portfolio projects don't: it evaluates the deep tier rigorously and **reports a negative result honestly** rather than overselling it.

## Architecture

| Tier | Detector | Family | Role | Blind spot |
|------|----------|--------|------|-----------|
| 1 | Robust Trailing Z-Score | Statistical | Sharp single-day spikes, near-zero compute | Single feature only |
| 2 | Isolation Forest | Classic ML | Multivariate outliers | Ignores temporal order |
| 3 | LSTM Autoencoder | Deep learning | Learns normal multi-day sequence dynamics | **Did not beat Tiers 1–2 in validation on this asset** |
| 4 | **Consensus Ensemble** | Combination | **Best precision/recall balance — beats every tier on F1** | Defined only where the tiers overlap |

## Key design decisions

- **Leakage-free evaluation.** Time-based train/test split, **every scaler fit on the training window only**, and causal (trailing) statistics throughout — no future information leaks into the test set.
- **Robust, strictly-causal z-score.** The baseline judges each day against the **prior** window only (`.shift(1)`) using a **median/MAD** score, so a spike can't inflate the very statistic used to detect it — this recovers spikes a naive gaussian z-score masks.
- **Stationary features for the deep tier.** The LSTM is fed returns, rolling volatility, and a volume **z-score** — never raw log-volume, which trends upward over the years and would make the autoencoder flag "a newer regime" instead of true anomalies. (Diagnosing and fixing that non-stationarity dropped the LSTM's spurious flag rate from 5.5% to ~1%.)
- **Contamination-based LSTM threshold.** The textbook 99th-percentile-of-*training*-error cutoff fails here for an instructive reason: the training window spans the ultra-volatile COVID crash, which inflates train error so much that the cutoff sits above every calmer test-era error and flags nothing. Thresholding by contamination (top ~1% most-unreconstructable windows) keeps the tier alive.
- **Consensus ensemble.** A day is flagged only when **≥2 of the 3 tiers agree**. Because the tiers make *different* mistakes, requiring agreement cancels each one's idiosyncratic false positives — lifting precision while holding recall.
- **Real validation.** Synthetic anomalies of **graduated magnitude (3–6σ)** are injected at known dates to compute actual precision/recall, then cross-checked qualitatively against known market events. The notebook **auto-prints a ready-to-paste résumé bullet** with the numbers from your run.

## Results

*`RELIANCE.NS`, ~2,450 daily points (2016–2026), 70/30 time split, synthetic-injection validation (12 anomalies at graduated 3–6σ) with ±1-day tolerance. Figures are printed by the notebook on every run and ranked by F1.*

| Tier | Precision | Recall | F1 |
|------|:---------:|:------:|:---:|
| **Consensus Ensemble (≥2 votes)** | **86%** | **100%** | **0.92** |
| Isolation Forest | 48% | 100% | 0.65 |
| Robust Z-Score | 48% | 100% | 0.65 |
| LSTM Autoencoder | 50% | 17% | 0.25 |

- **The ensemble earns its place.** Every individual tier recovers the injected anomalies (100% recall) but fires on ordinary volatile days too (~48% precision). Requiring **≥2 tiers to agree** lifts precision to **86%** while holding recall at 100%, so the ensemble's **F1 of 0.92 beats every single detector (0.65)** — a direct demonstration of *why* you combine models. Recall sits at 100% because a ≥3σ move with a volume surge is a genuinely unambiguous event on this asset; the meaningful differentiator here is **precision**, and that is what the ensemble improves.
- **Qualitative confirmation.** Isolation Forest's eight most-anomalous real days fall *entirely* within the **March–April 2020 COVID crash** (2020-03-23, 03-25, 04-07, 04-22, …) — clean evidence the detector finds genuine events, not noise.
- **Honest finding on the deep tier.** The LSTM autoencoder did **not** outperform the classical detectors on this asset. It recovered only ~17% of injected point anomalies, and a controlled slow-drift probe went undetected — autoencoders reconstruct low-variance mean-shifts easily, and the COVID-stretched feature scaling desensitises the reconstruction threshold. It does occupy a *distinct* niche on the real test window (its flags are almost all its own, clustering into multi-day stretches neither other tier caught), but that edge is unverified against labelled events. The effective production detector here is the **robust-z + Isolation Forest ensemble**; the deep tier is retained for architectural completeness and as an honestly-reported comparison.
- **Two engineering wins baked in.** (1) The LSTM initially flagged the whole recent era because raw log-volume trends upward; switching to a stationary volume z-score cut its spurious flag rate from 5.5% to ~1%. (2) An early validation *multiplied* returns by a constant, leaving near-zero-return days non-anomalous; injecting genuine **k-σ shocks** fixed the measurement.

### Résumé bullet

> *Built a four-tier time-series anomaly-detection pipeline (robust rolling z-score, Isolation Forest, LSTM autoencoder, and a consensus ensemble) on ~2,450 days of Reliance equity data; the consensus ensemble achieved **100% recall at 86% precision (F1 0.92)** under a leakage-free, time-based evaluation with synthetic anomaly injection — outperforming every individual detector.*

## Project structure

```
.
├── anomaly_detection.ipynb   # main notebook — run top to bottom
├── requirements.txt          # dependencies (all preinstalled in Colab except yfinance/plotly)
├── README.md
├── .gitignore
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
3. **Four tiers** — a robust (median/MAD), strictly-causal z-score; a 300-tree Isolation Forest; an LSTM autoencoder over overlapping 10-day sequences flagged by a contamination threshold; and a **consensus ensemble** that flags a day when ≥2 tiers agree.
4. **Validation & visualization** — graduated synthetic anomaly injection for precision/recall (ranked by F1, with an auto-generated résumé bullet), an interactive master chart that rings the consensus days, and an agreement analysis across all methods.

## Limitations & future work

- **Deep tier didn't pay off here.** A slow-drift probe confirmed the autoencoder doesn't catch smooth structural shifts on this asset (mean-shifts are the anomaly type autoencoders reconstruct most easily). Making it competitive would likely require training on a non-COVID "normal" window (so the feature scaling isn't stretched) or targeting variance/dynamics changes rather than level shifts.
- **Validation injects point anomalies**, which favour the instantaneous detectors; a labelled event dataset would allow evaluating structural detection properly.
- **Single asset** — a natural extension is a cross-sectional panel of equities.
- **The ensemble uses a simple majority vote**; a weighted or learned combiner (e.g. stacking) is a natural next step once labelled events are available.
- Thresholds (robust z ≈ 3.5, LSTM top-1% contamination, 3% IF contamination, ≥2 votes) are reasonable, defensible defaults rather than per-asset tuned values.

## License

Released under the MIT License — see [LICENSE](LICENSE).

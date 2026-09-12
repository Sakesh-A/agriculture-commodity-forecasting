# System Architecture & Design

This document details the system design, mathematical formulations, data contracts, and pipeline architecture for the agricultural commodity price forecasting framework.

---

## 1. System Overview

This framework is a multi-modal machine learning system designed to forecast price dynamics (multi-horizon log-returns and realized volatility) for key agricultural commodities traded on the Chicago Board of Trade (CBOT / CME Group), focusing on **Corn (`ZC`)**, **Soybeans (`ZS`)**, and **Wheat (`ZW`)**.

The platform ingests high-frequency market prices, macroeconomic indicators, gridded geospatial weather data, and unstructured USDA fundamental reports to generate probabilistic forecasts using a progression of models from tree baselines to Temporal Fusion Transformers (TFT).

### 1.1 Contract Specifications & Delivery Cycles
Agricultural commodity markets trade via standardized futures contracts with discrete expirations tied to planting, pollination, and harvest cycles.

#### Active Delivery Months
- **Corn (ZC):** March (`H`), May (`K`), July (`N`), September (`U`), December (`Z`).
  - *New Crop Anchor:* December (`Z`) reflects post-harvest supply.
  - *Old Crop Anchor:* July (`N`) reflects late-season inventory depletion.
- **Soybeans (ZS):** January (`F`), March (`H`), May (`K`), July (`N`), August (`Q`), September (`U`), November (`X`).
  - *New Crop Anchor:* November (`X`).
- **Wheat (ZW):** March (`H`), May (`K`), July (`N`), September (`U`), December (`Z`).

### 1.2 Continuous Contract Construction & Roll Mechanics
Individual contract expirations create step-jump discontinuities when concatenated naively.

#### Roll Schedule
- Contracts are rolled on the **5th trading day prior to the First Notice Date (FND)**, matching the liquidity transition window from front-month ($C_1$) to deferred-month ($C_2$).

#### Panama Ratio Adjustment (Multiplicative)
To ensure continuous price series do not inject false return shocks or generate negative historical prices:
$$\text{Adjustment Factor } \alpha_k = \prod_{i=k}^{K} \frac{P_{\tau_i}^{(C_{i+1})}}{P_{\tau_i}^{(C_i)}}$$
$$\widetilde{P}_t = P_t \times \alpha_k \quad \text{for } t \in [\tau_{k-1}, \tau_k)$$

Multiplicative adjustment guarantees that historical log-returns $r_t = \ln(\widetilde{P}_t / \widetilde{P}_{t-1})$ equal the true tradeable contract returns on non-roll days.

### 1.3 Term Structure & Fundamental Spread Signals
The shape of the futures forward curve serves as a primary feature, reflecting storage economics and physical availability:

1. **Calendar Spread (Roll Basis):**
   $$S_{t}^{(1, 2)} = P_{t}^{(C_2)} - P_{t}^{(C_1)}$$
   - $S_t > 0$ (**Contango**): Supply abundance, storage cost dominant.
   - $S_t < 0$ (**Backwardation**): Physical tightness, high convenience yield.
2. **Old Crop / New Crop Spread:**
   - Corn: July ($N$) vs. December ($Z$) spread ($\text{Spread}_{t}^{N-Z} = P_{t}^{(N)} - P_{t}^{(Z)}$).
   - Soybeans: July ($N$) vs. November ($X$) spread.
3. **Crush & Processing Spreads (Exogenous):**
   - Soybean Processing Margin: $\text{Crush} = P_{\text{Meal}} \times 0.022 + P_{\text{Oil}} \times 11 - P_{\text{Soybeans}}$.
   - Corn Ethanol Linkage: Ratio and spread of Corn (`ZC`) against WTI Crude Oil (`CL`).

---

## 2. Problem Formulation & Mathematical Objectives

### 2.1 Stationarity & Time-Series Regularity
Raw commodity prices $P_t$ violate weak-form stationarity due to time-varying means $\mathbb{E}[P_t]$ and explosive variances $\text{Var}(P_t) = t\sigma^2$. To ensure consistent generalization across distinct market regimes, all price series undergo logarithmic differentiation.

#### Stationarity Validation Protocol
Every candidate input series $X_t$ must pass a dual hypothesis verification:
1. **Augmented Dickey-Fuller (ADF) Test:** Reject null hypothesis of unit root ($p < 0.05$).
2. **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test:** Fail to reject null hypothesis of stationarity ($p > 0.05$).

Series failing ADF testing are differenced of order $d=1$:
$$\Delta X_t = X_t - X_{t-1}$$

### 2.2 Prediction Targets & Forecast Horizons
Let $\widetilde{P}_t$ denote the Panama-adjusted continuous front-month price. The framework models multi-horizon forward log returns across three operationally distinct horizons:
- **Daily Horizon:** $h = 1$ trading day (Immediate liquidity & news reaction)
- **Weekly Horizon:** $h = 5$ trading days (Weekly USDA Crop Progress cycle)
- **Monthly Horizon:** $h = 20$ trading days (Monthly WASDE balance sheet re-estimation)

#### Forward Log-Return Target
$$r_{t, h} = \ln\left(\frac{\widetilde{P}_{t+h}}{\widetilde{P}_t}\right) = \ln(\widetilde{P}_{t+h}) - \ln(\widetilde{P}_t)$$

#### Forward Realized Volatility Target
$$\sigma_{t, h} = \sqrt{\frac{252}{h} \sum_{i=1}^h r_{t+i, 1}^2}$$

### 2.3 Probabilistic Objective & Quantile Pinball Loss
Agricultural return distributions exhibit pronounced non-Gaussian fat tails and skewness. Models optimize conditional quantiles $\mathcal{Q} = \{0.10, 0.50, 0.90\}$ rather than conditional means.

#### Pinball Loss Formulation
For target $y$ and quantile prediction $\hat{y}_q$ at quantile $q \in (0, 1)$:
$$\mathcal{L}_q(y, \hat{y}_q) = \max\left( q (y - \hat{y}_q), (1 - q)(\hat{y}_q - y) \right)$$

#### Multi-Horizon Joint Objective
$$\mathcal{L}_{\text{total}} = \frac{1}{|\mathcal{Q}| \cdot |\mathcal{H}|} \sum_{q \in \mathcal{Q}} \sum_{h \in \mathcal{H}} \mathcal{L}_q\left(r_{t, h}, \hat{r}_{t, h}^{(q)}\right)$$
where $\mathcal{H} = \{1, 5, 20\}$ and $\mathcal{Q} = \{0.10, 0.50, 0.90\}$.

---

## 3. Data Ingestion & Alignment Pipeline

The forecasting pipeline ingests multi-frequency, asynchronous data sources into a unified daily trading decision grid indexed at market close ($t$). To prevent lookahead bias and feature leakage, all ingestion modules enforce point-in-time (PIT) publication timestamps rather than observation period dates.

```
+-----------------------------------------------------------------------------+
|                         RAW ASYNCHRONOUS INGESTION                          |
|                                                                             |
|  [Daily Market]           [Macro & Energy]        [Weekly Crop Progress]    |
|  CBOT (ZC, ZS, ZW)        FRED (WTI, DXY, TNX)    USDA QuickStats           |
|  Close: 1:20 PM CT        Close: 4:00 PM ET       Mon 3:00 PM CT (Post-mkt) |
|         │                        │                        │                 |
|         │                        │                        │                 |
|  [Monthly WASDE]          [Weather Observables]   [Text / News Feeds]       |
|  USDA Balance Sheets      Open-Meteo / NOAA GFS   WASDE Summaries / News    |
|  12:00 PM ET Midday       End-of-day Grids        Asynchronous Streams      |
+---------┬────────────────────────┼────────────────────────┬-----------------+
          │                        │                        │
          ▼                        ▼                        ▼
+-----------------------------------------------------------------------------+
|                      POINT-IN-TIME ALIGNMENT ENGINE                         |
|                                                                             |
|  1. Daily Trading Calendar Base Index (NYSE / CME Grain Trading Days)       |
|  2. Lag Rule: If T_release > 1:20 PM CT on day t  -->  Available day t+1    |
|  3. State Preservation: Last Observation Carried Forward (LOCF)             |
|  4. Shock & Decay Encoding: ΔX_t = X_t - X_{t-prior},  τ_t = t - t_release  |
+--------------------------------------┬--------------------------------------+
                                       ▼
+-----------------------------------------------------------------------------+
|                       UNIFIED MULTI-MODAL DAILY TENSOR                      |
|                                                                             |
|  - Continuous Market Features: X_market[t] in R^d_m                         |
|  - Exogenous Macro Signals:    X_macro[t]  in R^d_e                         |
|  - Tabular USDA Signals:       X_usda[t]   in R^d_u                         |
|  - Weather Embeddings:         X_weath[t]  in R^d_w                         |
|  - Unstructured Embeddings:    X_text[t]   in R^d_txt                       |
+-----------------------------------------------------------------------------+
```

### 3.1 Data Source Registry & Ingestion Protocols

| Domain | Ingestion Module | Source / API | Native Cadence | Assets / Tickers | Primary Features |
|---|---|---|---|---|---|
| **Market** | `src/data/market.py` | `yfinance` / CME | Daily | `ZC=F`, `ZS=F`, `ZW=F` | Open, High, Low, Close, Volume, Basis Spreads |
| **Macro** | `src/data/macro.py` | `fredapi` | Daily | `DCOILWTICO`, `DTWEXBGS`, `DGS10` | WTI Spot, Trade-Weighted Dollar, 10Y Yield |
| **USDA Progress** | `src/data/usda.py` | USDA QuickStats API | Weekly (Mon) | US Corn & Soybeans | `% Planted`, `% Emerged`, `% Good/Excellent` |
| **USDA WASDE** | `src/data/usda.py` | USDA ERS / Scraping | Monthly (~10th) | Global / US Grain Balances | Harvested Acres, Yield, Ending Stocks, $S/U$ |
| **Weather (Stage 1)** | `src/data/weather.py` | Open-Meteo Historical / NOAA | Daily | US Corn Belt Stations | $T_{\max}$, $T_{\min}$, Precipitation, Soil Moisture |
| **Weather (Stage 2)** | `src/data/weather_spatial.py` | ERA5 / NOAA GFS Grids | Daily (Raster) | 0.1° Midwest Grids | 2D Multichannel Tensors: Temp, Soil, Precip |
| **News & Text** | `src/data/news.py` | USDA PDF Reports / News API | Event-driven | WASDE Executive Summaries | FinBERT 768-dim embeddings, Sentiment Scores |

### 3.2 Point-in-Time (PIT) Alignment Engine & Leakage Prevention

To ensure no future information leaks into historical simulation rows, all ingested records are aligned to the CME Agricultural Trading Day grid.

#### CBOT Session Timing
- Day trading session closes at **1:20 PM Central Time (CT)**.
- Any signal published after 1:20 PM CT is strictly barred from trading row $t$ and is first made available on row $t+1$.

#### Explicit Alignment Rules
1. **USDA Crop Progress Reports:**
   - Published weekly on Mondays at **3:00 PM CT** (1 hour and 40 minutes post-market close).
   - *Alignment Rule:* $\text{Available Date} = t+1$ (Tuesday Market Session). Tagging this data to Monday's close represents invalid lookahead conditioning.
2. **Monthly WASDE Reports:**
   - Published between the 9th and 12th of each month at **12:00 PM Eastern Time (11:00 AM CT)**.
   - *Alignment Rule:* Released mid-session. To prevent intraday volatility distortions across pure daily EOD models, WASDE features are ingested on day $t$ close for post-release horizon evaluation, or staged to $t+1$ for conservative execution simulations.
3. **Macroeconomic Vintage Consistency:**
   - Economic indicators subject to periodic government revision (e.g., quarterly GDP or preliminary crop estimates) must retain their first-release value. No post-hoc revised figures may overwrite historical observation rows.

### 3.3 Asynchronous Merging: Shock and Information Decay Formulation

Forward-filling sparse weekly/monthly data directly creates stale feature artifacts. To convey both the updated level and its receding information freshness to transformer attention mechanisms, the alignment pipeline computes two auxiliary variables for every lower-frequency stream:

1. **Information Surprise / Delta ($\Delta S_t$):**
   $$\Delta S_t = S_{\tau} - S_{\tau - 1} \quad \forall t \in [\tau_k, \tau_{k+1})$$
   where $\tau$ indexes discrete report release dates. Captures the shock magnitude of newly printed balance sheet revisions.
2. **Elapsed Freshness Counter ($\Delta \tau_t$):**
   $$\Delta \tau_t = t - \tau_k \quad \text{for } t \ge \tau_k$$
   An integer counter tracking trading days elapsed since the latest release. Provides positional decay context to temporal attention layers.

### 3.4 Geospatial Weather Staged Architecture

#### Stage 1: Production-Weighted Tabular Aggregation
To maintain minimal latency and avoid heavy raster storage during early development, Stage 1 aggregates meteorological station data over the five primary production states representing $>60\%$ of US Corn and Soybean production: **Iowa (IA), Illinois (IL), Nebraska (NE), Minnesota (MN), and Indiana (IN)**.

$$\bar{W}_t = \sum_{i=1}^{5} \omega_i \cdot W_{i, t} \quad \text{where } \sum_{i=1}^5 \omega_i = 1$$
Normalized production weights $\omega_i$ are calibrated from the preceding 3-year average USDA NASS production totals:
- Iowa ($\omega_{\text{IA}} = 0.31$)
- Illinois ($\omega_{\text{IL}} = 0.28$)
- Nebraska ($\omega_{\text{NE}} = 0.18$)
- Minnesota ($\omega_{\text{MN}} = 0.13$)
- Indiana ($\omega_{\text{IN}} = 0.10$)

#### Stage 2: Spatio-Temporal Grids & Latent Projection
Stage 2 transitions from tabular scalar averages to dense 2D weather tensors across the bounding box $[36^\circ\text{N} - 45^\circ\text{N}, 80^\circ\text{W} - 104^\circ\text{W}]$:
- Input Grid: $\mathbf{X}_t^{\text{weather}} \in \mathbb{R}^{C \times H \times W}$ where channels $C = \{\text{T2M}_{\max}, \text{T2M}_{\min}, \text{Precip}, \text{SoilMoisture}\}$.
- A static agricultural cropland mask $\mathbf{M} \in \mathbb{R}^{H \times W}$ filters out non-productive pixels.
- A 2D Convolutional Backbone (ResNet-18 or ConvLSTM) encodes the spatial grid into a fixed embedding vector:
  $$\mathbf{z}_t^{\text{spatial}} = \text{Encoder}_{\text{CNN}}(\mathbf{X}_t^{\text{weather}} \odot \mathbf{M}) \in \mathbb{R}^{d_{\text{weather}}}$$
Maintaining the vector interface $\mathbf{z}_t \in \mathbb{R}^{d}$ ensures the downstream sequence models (LightGBM / TFT) remain decoupled from spatial preprocessing logic.

---

## 4. Feature Engineering

The feature engineering layer transforms raw, heterogeneous data streams into scale-invariant, stationary features. Every engineered feature is strictly indexed by its point-in-time publication timestamp to guarantee zero lookahead leakage.

---

### 4.1 Market Microstructure, Volatility & Memory Preservation

Raw close-to-close returns discard valuable intraday price dispersion, while standard integer differencing erases long-term economic memory.

#### 1. Range-Based Volatility Estimators
- **Parkinson Volatility (Intraday Extreme Range):**
  $$\sigma_{\text{Parkinson}, t} = \sqrt{\frac{1}{4 \ln 2} \left(\ln \frac{H_t}{L_t}\right)^2}$$
- **Garman-Klass Volatility (High-Efficiency OHLC Estimator):**
  $$\sigma_{\text{GK}, t} = \sqrt{0.5 \left(\ln \frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1)\left(\ln \frac{C_t}{O_t}\right)^2}$$
- **Realized Volatility Windows:**
  $$\sigma_{t, W} = \sqrt{\frac{252}{W} \sum_{i=0}^{W-1} r_{t-i, 1}^2} \quad \text{for } W \in \{5, 20, 60\}$$

#### 2. Long-Memory Preservation via Fractional Differentiation
To achieve weak stationarity without completely erasing multi-month price memory (as happens with integer differencing $d=1$), continuous series undergo fractional differencing:
$$(1 - B)^d X_t = \sum_{k=0}^\infty \omega_k X_{t-k}$$
where the weights are computed iteratively:
$$\omega_0 = 1, \quad \omega_k = -\omega_{k-1} \frac{d - k + 1}{k}$$
The order $d^* \in (0, 1)$ is chosen via grid search as the minimum value that satisfies both ADF rejection ($p < 0.05$) and KPSS acceptance ($p > 0.05$).

#### 3. Momentum & Liquidity Ratios
- Multi-lag log returns: $r_{t-k, 1} = \ln(\widetilde{P}_{t-k} / \widetilde{P}_{t-k-1})$ for $k \in \{0, 1, 2, 3, 4, 9, 19\}$.
- Moving average distance: $d_{\text{SMA}, t}^{(W)} = (\widetilde{P}_t - \text{SMA}_t(W)) / \text{SMA}_t(W)$ for $W \in \{20, 50, 200\}$.
- Volume ratio: $V_{\text{ratio}, t} = V_t / (\frac{1}{20}\sum_{i=0}^{19} V_{t-i})$.

---

### 4.2 Term Structure, Cross-Commodity & Capital Positioning

#### 1. Annualized Roll Yield (Forward Curve Slope)
$$y_t^{\text{roll}} = \frac{\ln(P_t^{(C_2)}) - \ln(P_t^{(C_1)})}{\Delta T_{1, 2}}$$
- $y_t^{\text{roll}} > 0 \implies$ Contango (storage dominant, physical surplus).
- $y_t^{\text{roll}} < 0 \implies$ Backwardation (convenience yield dominant, physical deficit).

#### 2. Agricultural Processing & Parity Spreads
- **Soybean Board Crush Margin:**
  $$\text{Margin}_{\text{crush}, t} = \left(0.022 \times P_{t}^{\text{Meal}} + 11.0 \times P_{t}^{\text{Oil}}\right) - P_{t}^{\text{Soybeans}}$$
- **Corn-to-Ethanol Energy Ratio:**
  $$R_{\text{energy}, t} = \frac{P_{t}^{\text{Corn}}}{P_{t}^{\text{WTI Crude}}}$$

#### 3. Market Positioning: CFTC Commitments of Traders (COT)
Ingested weekly on Friday close (data reflecting Tuesday):
- **Speculator Net Positioning:**
  $$\text{NetPos}_t = \text{Long}_{\text{non-comm}, t} - \text{Short}_{\text{non-comm}, t}$$
- **COT 3-Year Percentile Index (Crowdedness Oscillator):**
  $$\text{COT\_Index}_t = \frac{\text{NetPos}_t - \min_{W}(\text{NetPos})}{\max_{W}(\text{NetPos}) - \min_{W}(\text{NetPos})} \in [0, 1] \quad (W = 156 \text{ weeks})$$

---

### 4.3 Agronomic, Climate & Global Counter-Seasonal Features

#### 1. Thermal Time & Biological Stress Indices
- **Growing Degree Days (GDD):**
  $$\text{GDD}_t = \max\left(0, \frac{T_{\max}^* + T_{\min}^*}{2} - T_{\text{base}}\right)$$
  where $T_{\text{base}} = 50^\circ\text{F}$, $T_{\max}^* = \min(T_{\max}, 86^\circ\text{F})$, $T_{\min}^* = \max(T_{\min}, 50^\circ\text{F})$.
- **Extreme Degree Days (EDD - Summer Heat Stress):**
  $$\text{EDD}_t = \max\left(0, T_{\max} - 93^\circ\text{F}\right)$$
- **Cumulative Season GDD Anomaly:**
  $$\Delta \text{GDD}_{\text{season}, t} = \sum_{k=t_0}^t \text{GDD}_k - \overline{\text{GDD}}_{t_0:t}^{\text{historical}}$$

#### 2. Biological Seasonality & Harmonic Calendars
Agricultural dynamics operate on strict annual solar cycles. Calendar days are mapped to continuous circular coordinates:
$$x_{\sin, t} = \sin\left(\frac{2\pi \cdot \text{DOY}_t}{365.25}\right), \quad x_{\cos, t} = \cos\left(\frac{2\pi \cdot \text{DOY}_t}{365.25}\right)$$
- **Biological Growth Regime Flags:** Binary masks marking Midwest Planting (Apr-May), Critical Pollination/Silking (July), Pod Fill (Aug), and Harvest (Sept-Nov).

#### 3. Global Counter-Seasonal Supply (South America & Teleconnections)
- **Brazilian Real FX (`USDBRL`):** Currency depreciation incentivizes aggressive Brazilian farmer selling and acreage expansion.
- **Oceanic Niño Index (ENSO / SST Anomaly):** Sea surface temperature anomalies in Niño 3.4 region tracking La Niña (drought risk in US Plains and southern South America) vs. El Niño.

---

### 4.4 Fundamental Balance Sheet Shocks & Text Vectorization

#### 1. Tabular Balance Sheet & Progress Signals
- **WASDE Ending Stocks Revision Delta:**
  $$\Delta \text{Stocks}_t = \frac{\text{Stocks}_t^{\text{WASDE}} - \text{Stocks}_{t-1}^{\text{WASDE}}}{\text{Stocks}_{t-1}^{\text{WASDE}}}$$
- **WASDE Stocks-to-Use ($S/U$) Shift:**
  $$\Delta S/U_t = (S/U)_t - (S/U)_{t-1}$$
- **Weekly Crop Progress Delta:**
  $$\Delta \text{GE}_t = \text{GE}_t - \text{GE}_{t-1}$$

#### 2. Unstructured Text Embeddings (FinBERT)
Narrative summaries in USDA WASDE reports and global export inspection releases are embedded using domain-adapted transformers:
$$\mathbf{e}_t = \text{FinBERT}_{\text{pooler}}(\text{Text}_t) \in \mathbb{R}^{768}$$
$$\mathbf{s}_t = \text{Softmax}\left(\mathbf{W}\mathbf{e}_t + \mathbf{b}\right) = [p_{\text{positive}}, p_{\text{neutral}}, p_{\text{negative}}]^\top \in \mathbb{R}^3$$

---

## 5. Model Architectures

The framework evaluates a progression of models ranging from classical econometric benchmarks to multi-horizon deep sequence transformers, culminating in multi-modal cross-attention fusion.

---

### 5.1 Baseline Suite: Econometric & Tabular Ensembles

#### 1. ARMA-GARCH Econometric Benchmark
To establish benchmark volatility forecasting and test for conditional heteroskedasticity:
- **Conditional Mean (ARMA):**
  $$r_t = \mu + \sum_{i=1}^p \phi_i r_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t$$
- **Conditional Volatility (GARCH(1,1)):**
  $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
  Serves as the analytical yardstick for forward realized volatility $\sigma_{t, h}$.

#### 2. Multi-Quantile LightGBM
Gradient boosted decision trees trained on engineered tabular lags:
- Separate models trained for each horizon $h \in \{1, 5, 20\}$ and quantile $q \in \{0.10, 0.50, 0.90\}$.
- Objective: Piecewise linear Pinball Loss.
- Hyperparameters optimized via Bayesian search: `max_depth`, `learning_rate`, `num_leaves`, `colsample_bytree`, `subsample`.

---

### 5.2 Deep Sequence Modeling: Temporal Fusion Transformer (TFT)

The primary deep learning architecture is the Temporal Fusion Transformer (Lim et al., 2021), designed specifically for heterogeneous multi-horizon time-series forecasting.

#### 1. Input Variable Typology
TFT decomposes the feature space into three strictly isolated functional categories:
1. **Static Metadata ($\mathbf{s}$):** Invariant entity descriptors (e.g., commodity identifier `ZC` vs. `ZS`, exchange code).
2. **Observed Past Inputs ($\mathbf{x}_t$):** Dynamic time-series known only up to time $t$ (e.g., past returns, realized volatility, GDD anomalies, COT positioning, WASDE balance sheet deltas).
3. **Known Future Inputs ($\mathbf{z}_{t+k}$):** Deterministic signals known across both past and forecast horizons (e.g., harmonic calendar embeddings $x_{\sin}, x_{\cos}$, day of week, scheduled USDA report release dates).

#### 2. Variable Selection Networks (VSN)
Financial time series suffer from low signal-to-noise ratios. VSN applies instance-level feature gating:
$$\mathbf{v}_t = \text{Softmax}\left(\text{GRN}_{v}(\mathbf{\xi}_t)\right)$$
$$\widetilde{\mathbf{\xi}}_t = \sum_{j=1}^{M} v_{t, j} \cdot \text{GRN}_{\xi}^{(j)}(\mathbf{\xi}_{t, j})$$
where $v_{t, j}$ represents the dynamically learned importance weight of feature $j$ at timestamp $t$.

#### 3. Temporal Self-Attention & Sequence Layers
- **Local Context:** A bidirectional LSTM encoder/decoder processes the selected features to inject inductive bias for sequential locality.
- **Long-Range Dependencies:** Multi-head interpretable attention attends across past lookback windows:
  $$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right)$$
  Attention weights are shared across heads, producing an interpretable scalar temporal attribution score for each lookback step.

#### 4. Multi-Horizon Quantile Prediction Head
The decoder outputs simultaneous forward return predictions across horizons $\mathcal{H} = \{1, 5, 20\}$ and quantiles $\mathcal{Q} = \{0.10, 0.50, 0.90\}$ in a single forward pass:
$$\hat{\mathbf{y}}_{t, h} = \mathbf{W}_q \cdot \text{GRN}_{\text{out}}(\mathbf{\psi}_{t+h}) + \mathbf{b}_q$$
optimized via the joint multi-quantile loss $\mathcal{L}_{\text{total}}$ defined in Section 2.3.

---

### 5.3 Multi-Modal Cross-Attention Extension

To incorporate unstructured news and gridded weather rasters into the TFT backbone:
1. **Text Stream:** FinBERT embeddings $\mathbf{e}_t^{\text{text}} \in \mathbb{R}^{768}$ pass through a dimension-matching linear projection: $\mathbf{u}_t^{\text{text}} = \mathbf{W}_t \mathbf{e}_t^{\text{text}} \in \mathbb{R}^{d_{\text{model}}}$.
2. **Spatial Weather Stream:** 2D CNN spatial embedding $\mathbf{z}_t^{\text{spatial}} \in \mathbb{R}^{d_{\text{weather}}}$ projected to $\mathbb{R}^{d_{\text{model}}}$.
3. **Cross-Attention Bottleneck:** An auxiliary cross-attention layer attends between the temporal sequence states and the multi-modal embeddings prior to the final quantile decoding head:
   $$\text{CrossAttn}(\mathbf{H}_{\text{time}}, \mathbf{H}_{\text{modal}}) = \text{Softmax}\left(\frac{\mathbf{Q}_{\text{time}} \mathbf{K}_{\text{modal}}^\top}{\sqrt{d}}\right) \mathbf{V}_{\text{modal}}$$

---

## 6. Validation & Backtesting Strategy

_Document cross-validation splits, leakage prevention (purging/embargoing), and simulation mechanics._
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

_Document data sources, ingestion protocols, update frequencies, and point-in-time alignment strategies._

---

## 4. Feature Engineering

_Document technical, fundamental, macroeconomic, and meteorological feature sets._

---

## 5. Model Architectures

_Document model topologies, baselines, deep learning architectures, and ensemble strategies._

---

## 6. Validation & Backtesting Strategy

_Document cross-validation splits, leakage prevention (purging/embargoing), and simulation mechanics._
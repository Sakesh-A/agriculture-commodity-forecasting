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

_Define the forecasting formulation (e.g., target definitions, forecast horizons, loss functions, and probabilistic formulations)._

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
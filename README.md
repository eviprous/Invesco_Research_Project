# Invesco_Research_Project
# Empirical Study of Investor Preferences

This repository contains the research code for an empirical study on **investor preferences in equity markets**, developed as a collaboration between **Boston University (MSMFT)** and **Invesco**.

## Motivation

Market prices reflect more than just fundamentals. While traditional asset pricing models focus on risk, cash flows, and expected returns, real-world markets are also shaped by **how investors choose to allocate capital**, given their constraints, incentives, and behavioral biases.

These allocation choices can lead to:
- market concentration,
- crowding into popular assets,
- deviations from fundamental risk-based pricing.

The goal of this project is to **separate the informational (risk-based) component of the market from the preference-driven component**, and to study how investor preferences influence portfolio construction and asset pricing dynamics.

## Big Picture Idea

We decompose the market into two conceptually distinct parts:

1. **A Preference-Neutral Component**  
   A portfolio designed to be indifferent to investor preferences, reflecting balanced exposure to market risk rather than investor demand.

2. **A Preference Component**  
   A self-financing portfolio that captures deviations between the market portfolio and the preference-neutral portfolio, interpreted as returns driven by investor preferences.

This framework allows us to isolate and study the role of investor preferences independently from traditional risk exposures.

## Approach Overview

The core idea is implemented in three steps:

- Construct **preference-neutral portfolios** that treat assets indifferently under different definitions of neutrality (size, risk, or market exposure).
- Compare these portfolios to the **capital-weighted market portfolio**.
- Define the difference between the two as a **preference portfolio**, representing preference-driven capital allocation.

By structuring the market in this way, we can analyze how preference-driven behavior evolves over time and how it interacts with market conditions.

## Data

The analysis uses a long historical sample of U.S. equities:

- **Universe**: S&P 500 constituents
- **Data source**: WRDS / CRSP
- **Sample period**: January 1970 – December 2024
- Monthly returns, market capitalizations, and index membership history
- Fama–French five-factor and momentum factors for systematic risk controls

Index membership is handled dynamically to ensure only firms included in the S&P 500 at each point in time are used.

## Repository Philosophy

This repository is structured with a clear separation of concerns:

- **Core logic** (data processing, portfolio construction, models) lives in Python modules.
- **Notebooks** are used only for presentation, visualization, and discussion.
- All analysis is reproducible and modular.

Results and empirical findings are intentionally omitted from this README and documented separately in analysis notebooks and research outputs.

---

**Authors**  
Evi Prousanidou, Santiago Tolivia Diaz, Zuqing Wei

Boston University MSMFT & Invesco

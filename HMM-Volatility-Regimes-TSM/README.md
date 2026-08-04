# Volatility-Regime Identification and Forecasting in TSM

This independent quantitative-finance project investigates whether a Gaussian
Hidden Markov Model can identify persistent volatility regimes in the US-listed
shares of Taiwan Semiconductor Manufacturing Company, traded under the ticker
TSM.

## Project overview

The project:

- constructs a multivariate set of return, volatility and market-risk features;
- estimates several Gaussian Hidden Markov Model specifications;
- identifies four economically interpretable volatility regimes;
- evaluates regime stability on a chronologically separated test sample;
- compares general and regime-conditioned EWMA and GARCH models;
- assesses forecasting performance using RMSE, MAE and QLIKE.

## Main result

The HMM identified persistent and economically interpretable market regimes.
However, conditioning EWMA and GARCH parameters directly on the decoded regime
did not improve out-of-sample point-forecast accuracy. The results therefore
support the HMM primarily as a market-state identification and risk-
interpretation tool.

## Repository structure

- `report/` – full project report
- `notebooks/` – data preparation, HMM estimation and forecasting analysis
- `figures/` – selected project figures
- `requirements.txt` – required Python packages

## Data

Daily TSM and VIX data are downloaded using Yahoo Finance. The scripts retrieve
the data directly, so raw market data are not permanently stored in this
repository.

## Disclaimer

This project is for educational and research purposes only and does not
constitute investment advice.

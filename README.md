# Sathwik Devadiga — Flask Market Risk Portfolio

This version converts the static portfolio into a Flask application with a live market-risk backend.

## What is dynamic

- Market prices fetched from Yahoo Finance via `yfinance`
- User-defined tickers and portfolio weights
- Historical VaR
- Parametric VaR
- Monte Carlo VaR
- Expected Shortfall
- Annualized volatility
- Maximum drawdown
- Correlation matrix
- Position risk contribution
- Historical VaR backtesting
- Interactive Plotly charts
- Quote refresh endpoint

## Run

1. Install Python 3.11 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   pip install -r requirements.txt

4. Start Flask:

   python app.py

5. Open:

   http://127.0.0.1:5000

## Important

Yahoo Finance prices can be delayed depending on exchange and instrument. This is appropriate for a portfolio project and risk analytics demo, but should not be presented as exchange-grade real-time data or used for trading execution.

For production deployment, use Gunicorn/Waitress and a proper market-data provider if you need true streaming or licensed real-time data.


## Progress indicator
The VaR / ES engine now runs as a background job. The browser polls Flask every 500 ms and displays real backend progress through validation, market-data download, VaR/ES calculation, Monte Carlo simulation, backtesting, chart preparation, and latest-price retrieval.


## Portfolio table input

The dashboard now hides raw Yahoo Finance tickers from users. Users choose a company name from a table and enter its weight. The browser converts company names to ticker symbols before calling Flask.

The table also:
- supports adding/removing holdings
- validates that total weight equals 100%
- prevents duplicate stocks
- keeps the backend request format unchanged


## Chart fix for larger portfolios

Plotly charts now use controlled tick counts, percentage formatting, automatic margins,
shortened NSE ticker labels, date-axis formatting, dynamic chart heights, and Plotly.react().
This prevents labels and axes from overlapping as more stocks are added.


## Connected Project 02 — Index Futures Hedging

The portfolio now follows a connected workflow:

Project 01:
Portfolio construction → VaR / ES → risk analytics → backtesting

Project 02:
Current exposure → portfolio beta vs NIFTY → hedge ratio → futures hedge sizing →
recalculated beta / volatility / 99% VaR / ES → NIFTY stress P&L comparison

Portfolio holdings, weights, portfolio value and history setting are stored in browser
localStorage when Project 01 is calculated, so Project 02 loads them automatically.

The futures contract count is calculated as:

hedge_notional = portfolio_beta × portfolio_value × hedge_ratio

theoretical_contracts = hedge_notional / (NIFTY_level × lot_size)

The hedge-ratio slider uses the historical hedge return approximation:

hedged_portfolio_return = portfolio_return - (portfolio_beta × hedge_ratio × NIFTY_return)

The lot size is configurable in the UI because exchange contract specifications can change.


## Hedging workflow v2

- Project 01 and Project 02 now use the same browser-stored portfolio state.
- Editing holdings or weights in Project 02 persists back to Project 01.
- Project 01 loads the saved portfolio instead of resetting to defaults.
- NIFTY futures lot size is fixed at 65 in this project.
- Users can input an integer number of futures contracts.
- Contract count automatically adjusts the hedge-ratio slider.
- Moving the hedge-ratio slider automatically updates the implied contract count.
- Before-vs-after comparison now includes both 95% and 99% Historical VaR.


## Project 03 — Credit Risk Analysis

Independent credit-risk project added to the portfolio.

Flow: Borrower Data → Credit Assessment → PD → LGD → EAD → Expected Loss

Includes Logistic Regression PD modelling, a nonlinear Gradient Boosting benchmark, borrower-level PD, risk grade, credit-score style output, model-driver explanation, LGD/EAD assumptions, Expected Loss, ROC-AUC, KS statistic, confusion matrix, calibration, and portfolio-level credit exposure / Expected Loss.

The model uses a deterministic synthetic borrower dataset so the app can run locally without exposing real borrower data.

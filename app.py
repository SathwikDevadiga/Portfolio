from flask import Flask, render_template, request, jsonify
import math
import uuid
import threading
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

from credit_model import credit_engine

app = Flask(__name__)

# In-memory progress store for local/demo use.
risk_jobs = {}
hedge_jobs = {}


PAGE_ROUTES = {
    "/": ("index.html", "home"),
    "/about": ("about.html", "about"),
    "/skills": ("skills.html", "skills"),
    "/projects": ("projects.html", "projects"),
    "/dashboard": ("dashboard.html", "dashboard"),
    "/research": ("research.html", "research"),
    "/resume": ("resume.html", "resume"),
    "/contact": ("contact.html", "contact"),
    "/projects/var": ("project-var.html", "project_var"),
    "/projects/hedging": ("project-hedging.html", "project_hedging"),
    "/projects/credit-risk": ("project-credit-risk.html", "project_credit_risk"),
}

def make_view(template):
    def view():
        return render_template(template)
    return view

for path, (template, endpoint) in PAGE_ROUTES.items():
    app.add_url_rule(path, endpoint, make_view(template))

def parse_list(value):
    return [x.strip() for x in str(value).split(",") if x.strip()]

def get_close_prices(tickers, period="2y"):
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError("No market data returned. Check ticker symbols.")

    # yfinance returns either single-level or MultiIndex columns depending on tickers/version.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", axis=1, level=1)
        else:
            raise ValueError("Close prices were not found in market-data response.")
    else:
        if "Close" not in raw.columns:
            raise ValueError("Close prices were not found in market-data response.")
        close = raw[["Close"]]
        close.columns = [tickers[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    # Keep requested ticker order when possible.
    available = [t for t in tickers if t in close.columns]
    if not available:
        raise ValueError("None of the requested tickers returned usable prices.")
    close = close[available].dropna(how="all").ffill().dropna()
    if len(close) < 60:
        raise ValueError("Not enough historical observations. Try a longer period or different tickers.")
    return close

def get_quotes(tickers):
    quotes = []
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="5d", interval="1d", auto_adjust=True)
            if hist.empty:
                continue
            close = hist["Close"].dropna()
            price = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) > 1 else price
            change_pct = ((price / prev) - 1) * 100 if prev else 0.0
            currency = ""
            try:
                currency = tk.fast_info.get("currency", "") or ""
            except Exception:
                pass
            quotes.append({
                "ticker": ticker,
                "price": price,
                "change_pct": float(change_pct),
                "currency": currency
            })
        except Exception:
            continue
    return quotes

@app.get("/api/quotes")
def api_quotes():
    try:
        tickers = parse_list(request.args.get("tickers", "RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS"))
        if not tickers:
            raise ValueError("Enter at least one ticker.")
        return jsonify({
            "quotes": get_quotes(tickers),
            "updated_at": datetime.now().strftime("%d %b %Y, %H:%M:%S")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def update_progress(job_id, percent, message):
    job = risk_jobs.get(job_id)
    if job:
        job["progress"] = int(percent)
        job["message"] = message


def calculate_risk_job(job_id, data):
    try:
        update_progress(job_id, 5, "Reading portfolio...")

        tickers = parse_list(data.get("tickers", ""))
        weights = np.array([float(x) for x in parse_list(data.get("weights", ""))], dtype=float)
        portfolio_value = float(data.get("portfolio_value", 1000000))
        confidence = float(data.get("confidence", 0.95))
        horizon = int(data.get("horizon", 1))
        period = str(data.get("period", "2y"))

        update_progress(job_id, 10, "Validating portfolio...")

        if not tickers:
            raise ValueError("Enter at least one stock.")
        if len(tickers) != len(weights):
            raise ValueError("Number of weights must equal number of stocks.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")
        if weights.sum() <= 0:
            raise ValueError("Portfolio weights must sum to a positive number.")
        if not (0.5 < confidence < 1.0):
            raise ValueError("Confidence must be between 0.5 and 1.")
        if horizon < 1 or horizon > 10:
            raise ValueError("Horizon must be between 1 and 10 days.")

        weights = weights / weights.sum()

        update_progress(job_id, 20, "Fetching historical market data...")
        prices = get_close_prices(tickers, period=period)

        update_progress(job_id, 35, "Processing price history...")
        available = list(prices.columns)
        idx = [tickers.index(t) for t in available]
        weights = weights[idx]
        weights = weights / weights.sum()
        tickers = available

        returns = prices.pct_change(fill_method=None)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")

        if len(returns) < 60:
            raise ValueError(
                "Not enough common price history across the selected stocks. "
                "Try removing a recently listed stock or use a longer history period."
            )

        portfolio_returns = returns.dot(weights)
        portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()
        alpha = 1 - confidence

        update_progress(job_id, 45, "Calculating portfolio returns...")

        update_progress(job_id, 52, "Calculating Historical VaR and ES...")
        hist_q = float(np.quantile(portfolio_returns, alpha))
        hist_var_1d = max(0.0, -hist_q * portfolio_value)
        tail = portfolio_returns[portfolio_returns <= hist_q]
        es_1d = max(0.0, -float(tail.mean()) * portfolio_value) if len(tail) else hist_var_1d

        update_progress(job_id, 60, "Calculating Parametric VaR...")
        mu = float(portfolio_returns.mean())
        sigma = float(portfolio_returns.std(ddof=1))
        z = float(norm.ppf(alpha))
        param_return_q = mu + z * sigma
        param_var_1d = max(0.0, -param_return_q * portfolio_value)

        hist_var = hist_var_1d * math.sqrt(horizon)
        es = es_1d * math.sqrt(horizon)
        param_var = param_var_1d * math.sqrt(horizon)

        update_progress(job_id, 68, "Running Monte Carlo simulation...")
        rng = np.random.default_rng(42)
        sims = rng.normal(mu * horizon, sigma * math.sqrt(horizon), 10000)
        mc_q = float(np.quantile(sims, alpha))
        mc_var = max(0.0, -mc_q * portfolio_value)

        update_progress(job_id, 75, "Calculating portfolio risk metrics...")
        annualized_vol = sigma * math.sqrt(252)
        growth = (1 + portfolio_returns).cumprod()
        drawdown = growth / growth.cummax() - 1
        max_drawdown = float(drawdown.min())

        update_progress(job_id, 80, "Calculating risk contribution...")
        cov = returns.cov().values
        port_var = float(weights.T @ cov @ weights)
        mrc = cov @ weights
        contrib = weights * mrc
        contrib_pct = contrib / port_var if port_var != 0 else np.zeros_like(contrib)

        update_progress(job_id, 85, "Backtesting VaR model...")
        rolling_window = min(250, max(60, len(portfolio_returns) // 2))
        exceptions = 0
        tested = 0

        for i in range(rolling_window, len(portfolio_returns)):
            window = portfolio_returns.iloc[i - rolling_window:i]
            q = np.quantile(window, alpha)
            actual = portfolio_returns.iloc[i]
            tested += 1
            if actual < q:
                exceptions += 1

        expected = tested * alpha
        lower = max(0, expected - 2 * math.sqrt(max(expected, 1)))
        upper = expected + 2 * math.sqrt(max(expected, 1))
        bt_result = "PASS" if lower <= exceptions <= upper else "REVIEW"

        update_progress(job_id, 92, "Preparing correlation matrix...")
        corr = returns.corr()

        update_progress(job_id, 96, "Fetching latest prices...")
        quotes = get_quotes(tickers)

        result = {
            "metrics": {
                "portfolio_value": portfolio_value,
                "historical_var": hist_var,
                "parametric_var": param_var,
                "monte_carlo_var": mc_var,
                "expected_shortfall": es,
                "annualized_volatility": annualized_vol,
                "max_drawdown": max_drawdown,
            },
            "distribution": [round(float(x), 8) for x in portfolio_returns.tail(750).values],
            "var_cutoff": hist_q,
            "risk_contribution": {
                "labels": tickers,
                "values": [round(float(x), 8) for x in contrib_pct]
            },
            "correlation": {
                "labels": list(corr.columns),
                "values": [[round(float(v), 6) for v in row] for row in corr.values]
            },
            "performance": {
                "dates": [d.strftime("%Y-%m-%d") for d in growth.index],
                "values": [round(float(v), 8) for v in growth.values]
            },
            "backtest": {
                "observations": tested,
                "expected_exceptions": expected,
                "actual_exceptions": exceptions,
                "result": bt_result
            },
            "quotes": quotes,
            "updated_at": datetime.now().strftime("%d %b %Y, %H:%M:%S")
        }

        risk_jobs[job_id]["result"] = result
        risk_jobs[job_id]["status"] = "complete"
        update_progress(job_id, 100, "Risk analysis complete")

    except Exception as e:
        job = risk_jobs.get(job_id)
        if job:
            job["status"] = "error"
            job["error"] = str(e)
            job["message"] = "Calculation failed"


@app.post("/api/risk/start")
def start_risk():
    try:
        data = request.get_json(force=True)
        job_id = str(uuid.uuid4())
        risk_jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "message": "Starting risk analysis...",
            "result": None,
            "error": None
        }

        thread = threading.Thread(target=calculate_risk_job, args=(job_id, data), daemon=True)
        thread.start()
        return jsonify({"job_id": job_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/risk/progress/<job_id>")
def risk_progress(job_id):
    job = risk_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)



def update_hedge_progress(job_id, percent, message):
    job = hedge_jobs.get(job_id)
    if job:
        job["progress"] = int(percent)
        job["message"] = message


def calculate_hedge_job(job_id, data):
    try:
        update_hedge_progress(job_id, 5, "Reading portfolio from Project 1...")

        tickers = parse_list(data.get("tickers", ""))
        weights = np.array(
            [float(x) for x in parse_list(data.get("weights", ""))],
            dtype=float
        )
        portfolio_value = float(data.get("portfolio_value", 1000000))
        period = str(data.get("period", "2y"))
        lot_size = 65
        benchmark = "^NSEI"

        if not tickers:
            raise ValueError("No Project 1 portfolio was found.")
        if len(tickers) != len(weights):
            raise ValueError("Portfolio holdings and weights do not match.")
        if weights.sum() <= 0:
            raise ValueError("Portfolio weights are invalid.")
        if lot_size <= 0:
            raise ValueError("Futures lot size must be positive.")

        weights = weights / weights.sum()

        update_hedge_progress(job_id, 20, "Fetching portfolio and NIFTY history...")
        all_tickers = tickers + [benchmark]
        prices = get_close_prices(all_tickers, period=period)

        if benchmark not in prices.columns:
            raise ValueError("NIFTY 50 benchmark data could not be loaded.")

        available_stocks = [t for t in tickers if t in prices.columns]
        if not available_stocks:
            raise ValueError("No usable stock data was returned.")

        idx = [tickers.index(t) for t in available_stocks]
        weights = weights[idx]
        weights = weights / weights.sum()
        tickers = available_stocks

        update_hedge_progress(job_id, 40, "Aligning portfolio and index returns...")
        stock_prices = prices[tickers]
        index_price = prices[benchmark]

        stock_returns = stock_prices.pct_change(fill_method=None)
        index_returns = index_price.pct_change(fill_method=None).rename("NIFTY")

        combined = stock_returns.copy()
        combined["NIFTY"] = index_returns
        combined = combined.replace([np.inf, -np.inf], np.nan).dropna(how="any")

        if len(combined) < 60:
            raise ValueError(
                "Not enough common history between the portfolio and NIFTY. "
                "Try a longer historical period."
            )

        portfolio_returns = combined[tickers].dot(weights)
        nifty_returns = combined["NIFTY"]

        update_hedge_progress(job_id, 60, "Calculating portfolio beta...")
        index_variance = float(np.var(nifty_returns, ddof=1))
        if index_variance == 0:
            raise ValueError("NIFTY return variance is zero; beta cannot be calculated.")

        beta = float(np.cov(portfolio_returns, nifty_returns, ddof=1)[0, 1] / index_variance)

        update_hedge_progress(job_id, 72, "Calculating current portfolio risk...")
        confidence = 0.99
        alpha = 1 - confidence

        def historical_metrics(series):
            arr = np.asarray(series, dtype=float)
            q = float(np.quantile(arr, alpha))
            var_amt = max(0.0, -q * portfolio_value)
            tail = arr[arr <= q]
            es_amt = max(0.0, -float(tail.mean()) * portfolio_value) if len(tail) else var_amt
            vol = float(np.std(arr, ddof=1) * math.sqrt(252))
            return var_amt, es_amt, vol

        before_var, before_es, before_vol = historical_metrics(portfolio_returns)

        latest_index = float(index_price.dropna().iloc[-1])
        contract_notional = latest_index * lot_size

        update_hedge_progress(job_id, 84, "Building futures hedge model...")
        full_hedge_notional = beta * portfolio_value
        theoretical_contracts = (
            full_hedge_notional / contract_notional
            if contract_notional != 0
            else 0.0
        )
        rounded_contracts = int(round(abs(theoretical_contracts)))
        hedge_direction = "SHORT" if beta >= 0 else "LONG"

        # Full analytical hedge for the comparison baseline.
        full_hedged_returns = portfolio_returns - beta * nifty_returns
        after_var, after_es, after_vol = historical_metrics(full_hedged_returns)
        after_beta = float(
            np.cov(full_hedged_returns, nifty_returns, ddof=1)[0, 1] / index_variance
        )

        stress_shocks = [-0.02, -0.05, -0.10, 0.05]

        update_hedge_progress(job_id, 94, "Preparing hedge scenarios...")
        result = {
            "portfolio": {
                "tickers": tickers,
                "weights": [round(float(w), 8) for w in weights],
                "portfolio_value": portfolio_value,
                "period": period,
            },
            "hedge": {
                "benchmark": "NIFTY 50",
                "benchmark_ticker": benchmark,
                "index_level": latest_index,
                "lot_size": lot_size,
                "contract_notional": contract_notional,
                "portfolio_beta": beta,
                "full_hedge_notional": full_hedge_notional,
                "theoretical_contracts": theoretical_contracts,
                "rounded_contracts": rounded_contracts,
                "direction": hedge_direction,
            },
            "before": {
                "beta": beta,
                "volatility": before_vol,
                "var_99": before_var,
                "es_99": before_es,
            },
            "full_hedge": {
                "beta": after_beta,
                "volatility": after_vol,
                "var_99": after_var,
                "es_99": after_es,
            },
            "series": {
                "portfolio_returns": [round(float(x), 8) for x in portfolio_returns.values],
                "index_returns": [round(float(x), 8) for x in nifty_returns.values],
            },
            "stress_shocks": stress_shocks,
            "updated_at": datetime.now().strftime("%d %b %Y, %H:%M:%S")
        }

        hedge_jobs[job_id]["result"] = result
        hedge_jobs[job_id]["status"] = "complete"
        update_hedge_progress(job_id, 100, "Hedge analysis ready")

    except Exception as e:
        job = hedge_jobs.get(job_id)
        if job:
            job["status"] = "error"
            job["error"] = str(e)
            job["message"] = "Hedge analysis failed"


@app.post("/api/hedge/start")
def start_hedge():
    try:
        data = request.get_json(force=True)
        job_id = str(uuid.uuid4())

        hedge_jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "message": "Starting hedge analysis...",
            "result": None,
            "error": None
        }

        thread = threading.Thread(
            target=calculate_hedge_job,
            args=(job_id, data),
            daemon=True
        )
        thread.start()

        return jsonify({"job_id": job_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/hedge/progress/<job_id>")
def hedge_progress(job_id):
    job = hedge_jobs.get(job_id)

    if not job:
        return jsonify({"error": "Hedge job not found"}), 404

    return jsonify(job)


@app.post("/api/credit-risk/assess")
def credit_risk_assess():
    try:
        data = request.get_json(force=True)
        borrower = {
            "annual_income": float(data.get("annual_income", 1200000)),
            "loan_amount": float(data.get("loan_amount", 500000)),
            "loan_tenure_years": float(data.get("loan_tenure_years", 5)),
            "credit_history_years": float(data.get("credit_history_years", 6)),
            "existing_emi": float(data.get("existing_emi", 18000)),
            "credit_utilization": float(data.get("credit_utilization", 0.35)),
            "delinquencies_12m": float(data.get("delinquencies_12m", 0)),
            "age": float(data.get("age", 31)),
            "employment_type": str(data.get("employment_type", "Salaried")),
            "home_ownership": str(data.get("home_ownership", "Rented")),
            "loan_purpose": str(data.get("loan_purpose", "Personal")),
        }
        lgd = float(data.get("lgd", 0.45))
        ead = float(data.get("ead", borrower["loan_amount"]))
        if borrower["annual_income"] <= 0: raise ValueError("Annual income must be positive.")
        if borrower["loan_amount"] <= 0: raise ValueError("Loan amount must be positive.")
        if ead <= 0: raise ValueError("EAD must be positive.")
        if not 0 <= lgd <= 1: raise ValueError("LGD must be between 0 and 100%.")
        if not 0 <= borrower["credit_utilization"] <= 1: raise ValueError("Credit utilization must be between 0 and 100%.")
        if borrower["age"] < 18: raise ValueError("Borrower age must be at least 18.")
        return jsonify(credit_engine.assess(borrower, lgd, ead))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/api/credit-risk/portfolio")
def credit_risk_portfolio():
    try:
        return jsonify(credit_engine.portfolio_summary())
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/api/credit-risk/model-metrics")
def credit_risk_model_metrics():
    return jsonify(credit_engine.metrics)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

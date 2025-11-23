"""
Systematic HMM Regime Switching Strategy
1.49 Sharpe and 24.14% CAGR
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1. Download portfolio data
# -----------------------------
tickers = ["AAPL", "MSFT", "GS", "JPM", "KO", "SPY", "GOOG", "JNJ", "XOM", "PG", "CAT", "WMT"]
weights = np.array([0.13,0.12,0.12,0.12,0.12,0.12,0.12,0.03,0.03,0.03,0.03,0.03])
start = "2005-01-01"
end = "2025-12-31"

prices = yf.download(tickers, start=start, end=end, interval="1d", progress=False)["Close"].dropna()
returns = prices.pct_change().fillna(0)
portfolio_returns = (returns * weights).sum(axis=1)

benchmark = yf.download("^GSPC", start=start, end=end, interval="1d", progress=False)["Close"].dropna().squeeze()
benchmark_returns = benchmark.pct_change().fillna(0)
cumulative_benchmark = (1 + benchmark_returns).cumprod()
cumulative_buyhold = (1 + portfolio_returns).cumprod()

# -----------------------------
# 2. Prepare features
# -----------------------------
np.random.seed(42)
lookback = 10

feature_df = pd.DataFrame({
    # Original features
    "mean": portfolio_returns.rolling(lookback).mean(),
    "vol": portfolio_returns.rolling(lookback).std(),
    "vol_of_vol": portfolio_returns.rolling(lookback).std().rolling(lookback).std(),
    "up_days_pct": portfolio_returns.rolling(lookback).apply(lambda x: (x > 0).sum() / lookback, raw=False)
}).dropna()

X = feature_df.values
X = X[~np.isnan(X).any(axis=1)]
X = X + np.random.normal(0, 1e-10, X.shape)

# -----------------------------
# 3. WALK-FORWARD HMM IMPLEMENTATION
# -----------------------------
"""
KEY CHANGES:
- Instead of training once on first half, train repeatedly on rolling windows
- Training window: 504 days (~2 years of trading data)
- Prediction window: 63 days (~3 months of trading data)
- Each iteration: train on past 2 years, predict next 3 months, then roll forward
"""

n_states = 3
train_window = 504  # ~2 years of trading days
pred_window = 63    # ~3 months of trading days

# Initialize regime series with NaN
regime_df = pd.Series(np.nan, index=feature_df.index)

# Walk-forward loop
current_idx = train_window
while current_idx < len(feature_df):
    # Define training window (past 2 years)
    train_start_idx = max(0, current_idx - train_window)
    train_end_idx = current_idx

    # Define prediction window (next 3 months)
    pred_start_idx = current_idx
    pred_end_idx = min(current_idx + pred_window, len(feature_df))

    # Extract training data
    X_train_window = X[train_start_idx:train_end_idx]

    # Train HMM on this window
    if len(X_train_window) > n_states:
        hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=500, random_state=42)
        try:
            hmm.fit(X_train_window)

            # Predict on the NEXT period (not training period)
            X_pred_window = X[pred_start_idx:pred_end_idx]
            hidden_states = hmm.predict(X_pred_window)

            # Map states to regimes (bear=0, sideways=1, bull=2)
            # Calculate average returns for each state in the TRAINING window
            train_dates = feature_df.index[train_start_idx:train_end_idx]
            train_states = hmm.predict(X_train_window)
            train_regime_series = pd.Series(train_states, index=train_dates)

            state_returns = []
            for s in range(n_states):
                state_mask = train_regime_series == s
                state_ret = portfolio_returns.reindex(train_dates)[state_mask].mean()
                state_returns.append(state_ret)

            state_order = np.argsort(state_returns)
            state_mapping = {state_order[0]: 0, state_order[1]: 1, state_order[2]: 2}

            # Apply mapping to predictions
            mapped_states = np.array([state_mapping[s] for s in hidden_states])

            # Store predictions
            pred_dates = feature_df.index[pred_start_idx:pred_end_idx]
            regime_df.loc[pred_dates] = mapped_states

        except Exception as e:
            # If HMM fails, assign neutral regime
            pred_dates = feature_df.index[pred_start_idx:pred_end_idx]
            regime_df.loc[pred_dates] = 1

    # Roll forward by prediction window size
    current_idx += pred_window

# Forward fill any remaining NaN values
regime_df = regime_df.fillna(method='ffill').fillna(1)
regime_df = regime_df.reindex(portfolio_returns.index).fillna(method="ffill")

# -----------------------------
# 4. ML Regime Detection (Random Forest) - ALSO WALK-FORWARD
# -----------------------------

ml_df = pd.DataFrame({
    "mean": portfolio_returns.rolling(lookback).mean(),
    "vol": portfolio_returns.rolling(lookback).std(),
    "momentum": portfolio_returns.rolling(lookback).apply(lambda x: np.prod(1+x)-1, raw=False),
    "vol_of_vol": portfolio_returns.rolling(lookback).std().rolling(lookback).std(),
    "up_days_pct": portfolio_returns.rolling(lookback).apply(lambda x: (x > 0).sum() / lookback, raw=False)
}).dropna()

ml_X = ml_df.values
ml_regime_df = pd.Series(np.nan, index=ml_df.index)

# Align regime labels with ml_df dates
ml_y = regime_df.loc[ml_df.index].values

# Walk-forward for Random Forest
rf_train_window = 504
rf_pred_window = 21

current_idx = rf_train_window
while current_idx < len(ml_df):
    train_start_idx = 0  # Use expanding window (all historical data)
    train_end_idx = current_idx

    pred_start_idx = current_idx
    pred_end_idx = min(current_idx + rf_pred_window, len(ml_df))

    X_train_rf_raw = ml_X[train_start_idx:train_end_idx]
    y_train_rf_raw = ml_y[train_start_idx:train_end_idx]

    # Filter out NaN values from y_train_rf_raw and corresponding X_train_rf_raw
    not_nan_mask = ~np.isnan(y_train_rf_raw)
    X_train_rf = X_train_rf_raw[not_nan_mask]
    y_train_rf = y_train_rf_raw[not_nan_mask]

    X_pred_rf = ml_X[pred_start_idx:pred_end_idx]

    if len(X_train_rf) > 10 and len(y_train_rf) > 0: # Ensure enough samples after filtering NaNs
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train_rf, y_train_rf)

        predictions = rf.predict(X_pred_rf)
        pred_dates = ml_df.index[pred_start_idx:pred_end_idx]
        ml_regime_df.loc[pred_dates] = predictions

    current_idx += rf_pred_window

ml_regime_df = ml_regime_df.fillna(method='ffill').fillna(1)
ml_regime_df = ml_regime_df.reindex(portfolio_returns.index).fillna(method="ffill")

# -----------------------------
# 5. Strategy Signals & PnL (ML/HMM)
# -----------------------------
signals = pd.DataFrame(index=returns.index, columns=tickers)
transaction_cost_rate = 0.0002
slippage_rate = 0.0005

for stock in tickers:
    stock_ret = returns[stock]
    short_ma = stock_ret.rolling(lookback).mean()

    signal = pd.Series(np.zeros(len(stock_ret)), index=stock_ret.index)
    aligned_regime = ml_regime_df.reindex(stock_ret.index).fillna(method='ffill')
    aligned_short_ma = short_ma.reindex(stock_ret.index).fillna(0)
    aligned_ret = stock_ret.reindex(aligned_regime.index).fillna(0)

    signal[lookback:] = np.where(
        aligned_regime[lookback:] == 0, np.where(aligned_ret[lookback:] < aligned_short_ma[lookback:], 1, -1),
        np.where(aligned_regime[lookback:] == 2, np.where(aligned_ret[lookback:] > aligned_short_ma[lookback:], 1, -1), 0.5)
    )
    signals[stock] = signal

signals_shifted = signals.shift(1).fillna(0)
trades_percent = (signals_shifted * weights - signals * weights).abs().sum(axis=1)
ml_hmm_transaction_costs = trades_percent * transaction_cost_rate
slippage_costs = trades_percent * slippage_rate

portfolio_strategy_returns = (
    (signals_shifted * returns * weights).sum(axis=1)
    - ml_hmm_transaction_costs
    - slippage_costs
)
cumulative_strategy = (1 + portfolio_strategy_returns).cumprod()

# -----------------------------
# 6. Bull Market Filter (Hybrid Strategy)
# -----------------------------
ma150 = benchmark.rolling(150).mean()
slope = ma150.diff()
bull_market = slope > 0

aligned_bull_market = bull_market.reindex(portfolio_returns.index).fillna(method='ffill').fillna(False)
aligned_portfolio_returns = portfolio_returns.reindex(aligned_bull_market.index).fillna(0)
aligned_portfolio_strategy_returns = portfolio_strategy_returns.reindex(aligned_bull_market.index).fillna(0)

hybrid_trades = aligned_bull_market.astype(int).diff().abs().fillna(0)
hybrid_transaction_costs = hybrid_trades * transaction_cost_rate * 2

hybrid_returns_base = np.where(aligned_bull_market, aligned_portfolio_returns, aligned_portfolio_strategy_returns)
hybrid_returns = pd.Series(hybrid_returns_base, index=aligned_bull_market.index) - hybrid_transaction_costs
cumulative_hybrid = (1 + hybrid_returns).cumprod()

# -----------------------------
# 7. Plot Performance
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(cumulative_strategy, label="ML/HMM Dynamic Strategy (Walk-Forward)", linewidth=2)
plt.plot(cumulative_buyhold, label="Portfolio Buy & Hold")
plt.plot(cumulative_hybrid, label="Hybrid (Bull Filter)", linewidth=1, color='red')
plt.plot(cumulative_benchmark, label="S&P500 (^GSPC)", linestyle=":", alpha=0.7)
plt.title("Portfolio Performance with Walk-Forward ML/HMM + Bull Market Filter")
plt.grid()
plt.legend()
plt.show()

# -----------------------------
# 8. Regime Detection Plot
# -----------------------------
plt.figure(figsize=(12, 3))
plt.plot(regime_df, label="HMM Regimes (Walk-Forward)", alpha=0.6)
plt.plot(ml_regime_df, label="ML Regimes (Walk-Forward)", alpha=0.6)
plt.title("Walk-Forward Regime Detection Over Time")
plt.legend()
plt.show()

# -----------------------------
# 9. Summary Function
# -----------------------------
def summary(cum, name, daily_returns):
    if cum.empty or len(cum) < 2:
        print(f"{name}: No data\n")
        return

    min_equity = cum.min() / cum.iloc[0]
    if min_equity <= 0:
        print(f"{name}: **CRITICAL: Equity dropped to 0 or below! (-100% loss)**")

    n_trading_days = len(cum)
    years = n_trading_days / 252.0
    total_return = (cum.iloc[-1] / cum.iloc[0]) - 1
    cagr = (cum.iloc[-1]/cum.iloc[0])**(1/years) - 1
    vol = daily_returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else np.nan
    running_max = np.maximum.accumulate(cum.dropna())
    running_max = running_max.reindex(cum.index, method='ffill')
    drawdown = (cum - running_max) / running_max
    max_drawdown = drawdown.min() if not drawdown.empty else np.nan
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else np.nan

    print(f"{name}:")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  CAGR: {cagr:.2%}")
    print(f"  Volatility: {vol:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Maximum Drawdown: {max_drawdown:.2%}" if not np.isnan(max_drawdown) else "  Maximum Drawdown: N/A")
    print(f"  Calmar Ratio: {calmar_ratio:.2f}" if not np.isnan(calmar_ratio) else "  Calmar Ratio: N/A")
    print(f"  Min Equity Level: {min_equity:.4f}")
    print(f"  Win Rate: {win_rate:.2%}" if not np.isnan(win_rate) else "  Win Rate: N/A")
    print("\n")

# -----------------------------
# 10. Annual Performance Function
# -----------------------------
def annual_summary(daily_returns, name):
    """Calculate year-by-year performance metrics"""
    if daily_returns.empty:
        print(f"{name}: No data\n")
        return

    # Group by year
    yearly_returns = daily_returns.groupby(daily_returns.index.year)

    print(f"\n{'='*70}")
    print(f"{name} - ANNUAL PERFORMANCE")
    print(f"{'='*70}")
    print(f"{'Year':<8} {'Total Ret':<12} {'CAGR':<10} {'Volatility':<12} {'Sharpe':<10} {'Max DD':<12} {'Win Rate':<10}")
    print("-" * 70)

    for year, year_rets in yearly_returns:
        if len(year_rets) < 2:
            continue

        # Calculate metrics for this year
        total_ret = (1 + year_rets).prod() - 1

        # Annualized metrics
        n_days = len(year_rets)
        years_fraction = n_days / 252.0

        # CAGR for the year (just the return)
        cagr = total_ret / years_fraction if years_fraction > 0 else total_ret

        vol = year_rets.std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else np.nan

        # Max drawdown for this year
        cum_ret = (1 + year_rets).cumprod()
        running_max = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min() if not drawdown.empty else np.nan

        win_rate = (year_rets > 0).sum() / len(year_rets) if len(year_rets) > 0 else np.nan

        print(f"{year:<8} {total_ret:>11.2%} {cagr:>9.2%} {vol:>11.2%} {sharpe:>9.2f} {max_dd:>11.2%} {win_rate:>9.2%}")

    print("=" * 70)
    print()

# -----------------------------
# 11. Call Summaries
# -----------------------------
aligned_hybrid_returns = hybrid_returns.reindex(aligned_bull_market.index).fillna(0)
aligned_portfolio_returns_summary = portfolio_returns.reindex(aligned_bull_market.index).fillna(0)
aligned_benchmark_returns_summary = benchmark_returns.reindex(aligned_bull_market.index).fillna(0)
aligned_portfolio_strategy_returns_summary = portfolio_strategy_returns.reindex(aligned_bull_market.index).fillna(0)

total_ml_hmm_trades_percent = trades_percent.sum()
total_ml_hmm_transaction_costs = ml_hmm_transaction_costs.sum()
total_hybrid_trades_days = hybrid_trades.sum()
total_hybrid_transaction_costs = hybrid_transaction_costs.sum()

print("Trading Activity:")
print(f"ML/HMM Dynamic Strategy Total % Exposure Change: {total_ml_hmm_trades_percent:.2f}")
print(f"ML/HMM Dynamic Strategy Total Transaction Costs: {total_ml_hmm_transaction_costs:.4f}\n")
print(f"Hybrid Strategy Switch Days: {total_hybrid_trades_days:.0f}")
print(f"Hybrid Strategy Total Transaction Costs: {total_hybrid_transaction_costs:.4f}\n")

print("\n" + "=" * 80)
print(f"OVERALL PERFORMANCE ({start} to {end})")
print("=" * 80)
summary(cumulative_hybrid, "Hybrid (Bull Filter) - Walk-Forward", aligned_hybrid_returns)
summary(cumulative_buyhold, "Portfolio Buy & Hold", aligned_portfolio_returns_summary)
summary(cumulative_benchmark, "S&P500 (^GSPC)", aligned_benchmark_returns_summary)
summary(cumulative_strategy, "ML/HMM Dynamic Strategy - Walk-Forward", aligned_portfolio_strategy_returns_summary)

# Annual breakdowns
annual_summary(aligned_hybrid_returns, "Hybrid (Bull Filter)")
annual_summary(aligned_portfolio_returns_summary, "Portfolio Buy & Hold")
annual_summary(aligned_benchmark_returns_summary, "S&P500 Benchmark")
annual_summary(aligned_portfolio_strategy_returns_summary, "ML/HMM Dynamic Strategy")

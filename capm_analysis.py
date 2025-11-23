"""
CAPM analysis of hybrid strategy vs S&P 500
Daily Alpha:        0.000728
Annual Alpha:       18.3528%
Beta:               0.4584
R-squared:          0.2937
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ====================================================================
# CAPM ANALYSIS
# ====================================================================
# Calculates alpha and beta of Hybrid Strategy vs S&P 500 benchmark

# Assign names to the Series before merging
hybrid_returns.name = 'hybrid_returns'
benchmark_returns.name = 'benchmark_returns'

# Align the indices of hybrid_returns and benchmark returns
aligned_returns = pd.merge(
    hybrid_returns,
    benchmark_returns,  # S&P 500 benchmark (correct for CAPM)
    left_index=True,
    right_index=True,
    suffixes=('_hybrid', '_benchmark')
).dropna()

# Check if aligned_returns is empty
if aligned_returns.empty:
    print("\nNo overlapping dates with valid returns for hybrid strategy and benchmark.")
else:
    # Prepare data for CAPM regression
    x = aligned_returns['benchmark_returns'].values.reshape(-1, 1)
    y = aligned_returns['hybrid_returns'].values

    # Perform Linear Regression (CAPM)
    # Model: R_strategy = alpha + beta * R_market
    model = LinearRegression()
    model.fit(x, y)

    # Extract CAPM metrics
    alpha = model.intercept_      # Daily alpha
    beta = model.coef_[0]         # Beta (systematic risk)
    r2 = model.score(x, y)        # R-squared (explanatory power)

    # Annualize alpha (252 trading days per year)
    alpha_annual = alpha * 252

    # Print CAPM metrics
    print("\n" + "="*70)
    print("CAPM ANALYSIS (Hybrid Strategy vs S&P 500)")
    print("="*70)
    print(f"Daily Alpha:        {alpha:.6f}")
    print(f"Annual Alpha:       {alpha_annual:.4%}")
    print(f"Beta:               {beta:.4f}")
    print(f"R-squared:          {r2:.4f}")
    print("="*70)
    print("\nInterpretation:")
    print(f"  - Alpha: Strategy generates {alpha_annual:.2%} excess return per year")
    print(f"  - Beta:  Strategy has {beta:.1%} of market's systematic risk")
    print(f"  - R²:    {r2:.1%} of returns explained by market movements")
    print("="*70 + "\n")

    # Plot CAPM Regression
    plt.figure(figsize=(10, 6))
    plt.scatter(aligned_returns['benchmark_returns'], 
                aligned_returns['hybrid_returns'], 
                alpha=0.3, 
                label='Daily Returns')
    plt.plot(aligned_returns['benchmark_returns'], 
             model.predict(x), 
             color='red', 
             linewidth=2,
             label=f'CAPM Fit (α={alpha_annual:.2%}, β={beta:.2f})')
    plt.xlabel('S&P 500 Daily Returns', fontsize=12)
    plt.ylabel('Hybrid Strategy Daily Returns', fontsize=12)
    plt.title('CAPM Regression: Hybrid Strategy vs S&P 500', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.show()

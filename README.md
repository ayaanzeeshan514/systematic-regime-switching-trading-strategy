# systematic-regime-switching-trading-strategy
Regime switching trading strategy which uses HMM and RF models to detect market regimes based on multiple features and trades accordingly. 1.49 Sharpe, 24.14% CAGR from 2005-2025.

# Overview

I noticed that US equities markets could often be classified into clear "regimes". High volatility and low retuns could be classified as bears, and panic selling would cause mispricings to be exploited here through mean reversion. In bulls, characterised by higher returns and low volatility, momentum trading works well. I then decided to add sideways. where depending on if signals lean towards bear or bull, the correct strats are used with half exposure. This model classifies the market state using Hidden Markov Models, and then refining the process through RandomForest. The 150 day bull filter allows it to just 'ride the wave' in strong markets rather than overtrading. This strategy mathced buy and hold in bull periods or exceeds them, and does incredibly well in bears.

## Performance Summary

**Backtest Period:** January 2005 - November 2025 (21 years)

| Metric | Hybrid Strategy | Buy & Hold | S&P 500 | ML/HMM Only |
|--------|-----------------|------------|---------|-------------|
| **Total Return** | **8,994%** | 2,791% | 449% | 160% |
| **CAGR** | **24.14%** | 17.50% | 8.51% | 4.68% |
| **Sharpe Ratio** | **1.49** | 0.88 | 0.44 | 0.36 |
| **Volatility** | **16.23%** | 19.89% | 19.18% | 13.07% |
| **Maximum Drawdown** | **-27.82%** | -49.55% | -56.78% | -37.58% |
| **Calmar Ratio** | **0.87** | 0.35 | 0.15 | 0.12 |
| **Win Rate** | **54.57%** | 54.80% | 54.59% | 51.46% |

**Key Highlights:**
- **$100,000 invested in 2005 → $8,994,000 by November 2025**
- Outperformed buy-and-hold by 6.64% annually with 44% lower drawdowns
- Achieved 1.49 Sharpe ratio
- Total transaction costs: 56 basis points over 21 years

---

## Transaction Cost Analysis

| Strategy Component | Metric | Value |
|-------------------|--------|-------|
| **ML/HMM Strategy** | Total Portfolio Turnover | 2,479% over 21 years |
| | Annual Turnover | ~118% per year |
| | Total Transaction Costs | 50 basis points |
| **Hybrid Bull Filter** | Regime Switches | 153 switches |
| | Switches per Year | ~7.3 switches/year |
| | Total Transaction Costs | 6 basis points |

## How It Works


### The Two Components

**1. Bull Market Filter**

The 150 day rolling window bull filter allows the strategy to just match the market at worst during strong periods and prevent overtrading.

**2. Regime Detection**

The strategy uses a Hidden Markov Model to identify three market regimes based on recent price behavior:

- **Bear markets:** Characterized by negative returns and high volatility → Strategy buys dips (mean reversion)
- **Sideways markets:** Mixed signals with no clear trend → Strategy holds half positions
- **Bull markets:** Positive returns with momentum → Strategy follows strength

The HMM trains on 150 days of data using 4 features: mean returns, volatility, volatility-of-volatility, and win rate.

### Trading Logic

### Trading Logic

- **Bear markets:** Bear markets often have V-shaped recoveries, and high-volatility panic selling creates mispricings. The strategy uses mean reversion, buying dips and betting on market corrections to capture the recovery.

- **Bull markets:** Bull markets are momentum-based with lower volatility. With fewer mispricings to exploit, the strategy follows momentum to capture upside. When the 150-day moving average slope is positive, the bull filter activates and the strategy switches to simple buy-and-hold.

- **Sideways markets:** Sideways markets generate mixed signals. The strategy uses the same logic as bears and bulls but trades at half exposure for proper risk management during uncertain periods.
---

## Features Used

The strategy uses 4 features calculated over 10-day rolling windows:

1. **Mean Returns:** Captures recent trend direction - positive in bulls, negative in bears
2. **Volatility:** Measures market uncertainty - spikes during crises, stabilizes in calm periods  
3. **Volatility-of-Volatility:** Detects regime changes - increases sharply when markets transition (e.g., calm → crisis)
4. **Win Rate (% Positive Days):** Momentum indicator - high during sustained trends, low during reversals

### Why These Features?

I tested multiple feature combinations: 2 features (baseline), 4 features, and 7 features. Adding more features (momentum at 3 timeframes, skewness) caused overfitting - Sharpe ratio dropped from 1.28 to 1.13. The 4-feature model was optimal, balancing regime detection capability with avoiding overfitting. Vol-of-vol was particularly important for detecting crisis periods like 2008 and 2020.

## Key Results

### Crisis Performance

**2008 Financial Crisis:**
- Strategy: **+30.82%**
- S&P 500: -38.49%

The vol-of-vol feature spiked during the crash, helping the HMM detect the regime change quickly. The mean reversion logic bought dips during the panic selling, capturing the recovery that happened in late 2008 and early 2009.

**2020 COVID Crash:**
- Strategy: **+140.29%**
- S&P 500: +16.26%

Similar story - volatility exploded in March 2020, the 150-day training window adapted fast, and mean reversion bought heavily at the bottom. The recovery was incredibly quick (classic V-shape), and the strategy captured it perfectly with aggressive positioning.

### CAPM Analysis

| Metric | Value | 
|--------|-------|
| Annual Alpha | **18.35%** |
| Beta | **0.46** |
| R² | **0.29** |

Alpha of 18.35% means the strategy generates that much excess return beyond what you'd expect from market risk alone. Beta of 0.46 shows the strategy has less than half the systematic risk of the S&P 500, so you're getting better returns with lower market exposure. The R² of 29% means most of the returns come from the strategy's logic, not just riding the market up and down.

---

## Technical Details

### Walk-Forward Testing

Walk-forward testing means I train the model on past data, predict forward, then roll the window forward and repeat. This avoids lookahead bias - the model never sees future data during training. I used this because backtests without it are basically useless, they overfit to the full dataset and look great but fail in real trading.

**Training Window:** 150 days

I originally used 504 days (2 years) but found that shorter windows adapt faster to regime changes. Markets change quickly, and 2-year-old data becomes stale. 150 days (about 6 months) captures the current regime without dragging in outdated patterns. Tested both and 150 days performed significantly better.

**Prediction Window:** 63 days

### Transaction Costs

I included both commission and slippage because most backtests ignore slippage and end up wildly optimistic. 

- Commission: 2 bps (realistic for large-cap stocks with modern brokers)
- Slippage: 5 bps (market impact from crossing the spread and moving prices slightly)
- **Total:** 7 bps per trade

These costs are realistic for a $100k-$1M portfolio trading liquid stocks like AAPL, MSFT, JPM. At larger sizes you'd see more slippage, but for the portfolio size I'm modeling this is pretty accurate.

---

## What I Learned

### Experiments I Ran

**Test 1: Feature Expansion (2 → 7 features)**
- Started with just mean returns and volatility, then added vol-of-vol, win rate, and momentum at 3 different timeframes (5/20/60 days)
- Sharpe dropped from 1.28 to 1.13, and 2008 performance got much worse
- Learned: More features ≠ better. The momentum features were redundant with mean returns, causing overfitting. Noise overfitting is a genuine problem for more complicated features.

**Test 2: Volatility-Based Regime Mapping**
- Tried mapping regimes by volatility (high vol = bear, low vol = bull) instead of by returns
- Sharpe dropped to 1.16 and 2008 loss increased from -3% to -9%
- Learned: Volatility spikes in both directions (crashes AND rallies). Returns are a better regime indicator because they directly measure what you care about - making or losing money.

**Test 3: Adaptive Training Window (504 → 150 days)**
- Reduced training window from 2 years to 6 months, thinking recent data matters more than old data
- Sharpe jumped from 1.28 to 1.49, and crisis performance improved dramatically
- Learned: Markets evolve fast. Training on 2-year-old patterns just makes the model fight the last war. Shorter windows stay relevant.

### Key Insights

- Simple models often beat complex ones - 4 features was the sweet spot, more caused overfitting
- Recency matters more than sample size for regime detection - 150 days worked better than 504 days
- Vol-of-vol is crucial for crisis detection, as it spikes when regimes are changing, giving early warning
- Mean reversion in bear markets works because of V-shaped recoveries, not because bears always bounce
- Transaction costs add up fast, needed to include realistic slippage to avoid fooling myself

---

## Limitations

1. **2020 performance is exceptional (+140%)** - This was a once-in-decade perfect storm: fast crash, V-shaped recovery, and the strategy positioned perfectly. Don't expect this every crisis. Without 2020, the CAGR is more like 20-22%.

2. **Survivorship bias in the data** - yfinance only includes stocks that are still around today. A real portfolio would've held companies that went bankrupt (Lehman Brothers, etc.), which would drag returns down slightly.

3. **Small sample of crises** - Only tested on 2-3 major crises (2008, 2020, 2022). The strategy worked well in all of them, but that's still a small sample. Need more data to be confident the crisis performance is robust.

4. **Assumes daily rebalancing is practical** - The backtest uses end-of-day prices and assumes you can rebalance daily without issues. In reality there might be execution constraints, though with only ~7 regime switches per year it's not that intensive.

---

## Installation

### Requirements
```bash
pip install yfinance pandas numpy matplotlib hmmlearn scikit-learn
```

### Running the Code
```bash
python strategy.py
python capm_analysis.py
```

The strategy code will download data, run the backtest, and print performance metrics plus plots. Takes about 5-7 minutes to run. CAPM analysis calculates alpha/beta vs S&P 500 benchmark. If you wish to run multiple times, it may be wise to download prices in a separate csv to cut down execution time.

---

## Code Structure
```
strategy.py - Main strategy implementation (HMM, Random Forest, walk-forward testing, performance metrics)
capm_analysis.py - Calculates alpha, beta, and R² vs S&P 500 benchmark
```

Both files are documented with comments explaining what each section does.

---

## About This Project

I'm a Year 10 student planning to study Math + CS at university, interested in quantitative finance and systematic trading. Built this over about 1.5 months during Year 10 to learn how professional quant strategies actually work.

My goals are to work in quant trading (prop firms like Optiver/Jump, then maybe lateral to a bank like Macquarie). This project taught me way more than I expected - HMM theory, walk-forward testing, the importance of transaction costs, and why simple models often beat complex ones.

Resources I used: Online courses for Python basics, then self-taught the finance stuff through QuantStart articles, research papers on regime switching, and a lot of trial and error debugging. I used AI for debugging, especially with some HMM model related errors which were intially a little past me.

---

## Future Improvements

- Add more asset classes (bonds, commodities, currencies) for better diversification and regime detection
- Test on international markets (Europe, Asia) to see if the strategy generalizes beyond US equities
- Implement proper execution modeling (VWAP/TWAP algorithms instead of market-on-close assumptions)
- Add external data like VIX, credit spreads, yield curve to improve regime detection
- Build a live paper trading system to test real-time execution and see how the strategy performs going forward

---

## Disclaimer

This is a personal research project for educational purposes only. This is not investment advice. Past performance does not guarantee future results. Do not trade real money based on this code without proper risk management and professional advice.

## License

MIT License - See LICENSE file for details

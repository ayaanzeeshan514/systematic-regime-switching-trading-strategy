# systematic-regime-switching-trading-strategy
Regime switching trading strategy which uses HMM and RF models to detect market regimes based on multiple features and trades accordingly. 1.49 Sharpe, 24.14% CAGR from 2005-2025.

# Overview

Initially, I was using mean reverson across the whole period, but I realised it only worked when market were volatile. I noticed that US equities markets could often be classified into clear "regimes".

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
- **$100,000 invested in 2005 → $9,094,480 by November 2025**
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

[EXPLAIN IN YOUR OWN WORDS: What does your strategy do? Keep it simple - explain like you're telling a friend]

### The Two Components

**1. Bull Market Filter**

[EXPLAIN: How does the 200-day moving average filter work? When does it switch strategies?]

**2. Regime Detection**

[EXPLAIN: What are the three regimes (bear/sideways/bull)? How does HMM detect them?]

### Trading Logic

[EXPLAIN: What does the strategy do in each regime?]
- **Bear markets:** [YOUR EXPLANATION]
- **Bull markets:** [YOUR EXPLANATION]
- **Sideways markets:** [YOUR EXPLANATION]

---

## Features Used

[LIST YOUR 4 FEATURES AND EXPLAIN WHY YOU CHOSE THEM:]

1. **[FEATURE 1 NAME]:** [Why is this useful?]
2. **[FEATURE 2 NAME]:** [Why is this useful?]
3. **[FEATURE 3 NAME]:** [Why is this useful?]
4. **[FEATURE 4 NAME]:** [Why is this useful?]

### Why These Features?

[WRITE 2-3 SENTENCES: Why 4 features? What did you test?]

---

## Key Results

### Crisis Performance

**2008 Financial Crisis:**
- Strategy: **[YOUR %]**
- S&P 500: -38.49%

[WRITE 1-2 SENTENCES: Why did it perform well?]

**2020 COVID Crash:**
- Strategy: **[YOUR %]**
- S&P 500: +16.26%

[WRITE 1-2 SENTENCES: Why did it perform well?]

### CAPM Analysis

| Metric | Value | 
|--------|-------|
| Annual Alpha | **[YOUR %]** |
| Beta | **[YOUR NUMBER]** |
| R² | **[YOUR NUMBER]** |

[WRITE 2-3 SENTENCES: What do these numbers mean? Why does alpha matter?]

---

## Technical Details

### Walk-Forward Testing

[EXPLAIN IN YOUR OWN WORDS: What is walk-forward testing? Why did you use it?]

**Training Window:** [YOUR NUMBER] days

[WRITE 1-2 SENTENCES: Why this window size? What did you test?]

**Prediction Window:** [YOUR NUMBER] days

### Transaction Costs

[EXPLAIN: What costs did you include? Why are they realistic?]

- Commission: [NUMBER] bps
- Slippage: [NUMBER] bps
- **Total:** [NUMBER] bps per trade

---

## What I Learned

### Experiments I Ran

[DESCRIBE YOUR TESTING PROCESS IN YOUR OWN WORDS]

**Test 1: [NAME OF TEST]**
- [What did you try?]
- [What happened?]
- [What did you learn?]

**Test 2: [NAME OF TEST]**
- [What did you try?]
- [What happened?]
- [What did you learn?]

**Test 3: [NAME OF TEST]**
- [What did you try?]
- [What happened?]
- [What did you learn?]

### Key Insights

[BULLET POINTS: What are the 3-5 most important things you discovered?]

- [INSIGHT 1]
- [INSIGHT 2]
- [INSIGHT 3]

---

## Limitations

[BE HONEST: What are the problems or limitations with this strategy?]

1. **[LIMITATION 1]:** [Explain why this matters]
2. **[LIMITATION 2]:** [Explain why this matters]
3. **[LIMITATION 3]:** [Explain why this matters]

---

## Installation

### Requirements
```bash
pip install yfinance pandas numpy matplotlib hmmlearn scikit-learn
```

### Running the Code
```bash
python strategy.py
```

[OPTIONAL: Add 1-2 sentences about what outputs to expect]

---

## Code Structure

[BRIEFLY DESCRIBE YOUR CODE ORGANIZATION - can be simple]
```
strategy.py - [What's in this file?]
capm_analysis.py - [What's in this file? Or delete if you combined it]
```

---

## About This Project

[WRITE 3-5 SENTENCES ABOUT YOURSELF AND WHY YOU BUILT THIS:]

- Who are you?
- Why did you build this?
- What are your goals?
- How long did it take?
- What resources did you use to learn?

---

## Future Improvements

[LIST 3-5 IDEAS: What would you add/change if you had more time?]

---

## Disclaimer

This is a personal research project for educational purposes only. This is not investment advice. Past performance does not guarantee future results. Do not trade real money based on this code without proper risk management and professional advice.

---

## Contact

[OPTIONAL - only if you want to share:]
- Email: [YOUR EMAIL]
- LinkedIn: [YOUR LINKEDIN]
- GitHub: [YOUR GITHUB USERNAME]

---

## License

MIT License - See LICENSE file for details

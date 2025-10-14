# Manual Sharpe Ratio Calculation Guide

## Overview

This guide explains how to manually calculate the Sharpe ratio in your Time-LLM cryptocurrency trading project. The Sharpe ratio is a measure of risk-adjusted return that helps you evaluate the performance of your trading strategies.

## What is the Sharpe Ratio?

The Sharpe ratio measures the excess return per unit of risk:

```
Sharpe Ratio = (Mean Return - Risk-Free Rate) / Standard Deviation of Returns
```

For annualized Sharpe ratio:
```
Annualized Sharpe = √N × (Mean Return - Risk-Free Rate per Period) / Std Dev of Returns

Where N = number of periods per year:
- Daily data: N = 252 (trading days)
- Weekly data: N = 52
- Monthly data: N = 12
```

## Implementation in Your Code

### 1. **In Backtesting (`backtest.py`)**

I've added manual Sharpe ratio calculation that automatically activates when Backtrader's analyzer returns 0 or None:

```python
# The manual calculation is integrated into:
- run_strategy()          # For single strategy runs
- optimize_strategy()     # For parameter optimization
- walk_forward_optimization()  # For walk-forward analysis
```

**How it works:**
1. First tries to use Backtrader's built-in SharpeRatio analyzer
2. If that returns 0 or None, automatically calculates manually
3. Uses portfolio values to compute periodic returns
4. Applies the standard Sharpe ratio formula

### 2. **Standalone Calculator (`sharpe_calculator.py`)**

A complete module for calculating Sharpe ratios from various data sources:

```python
from backtesting.sharpe_calculator import sharpe_ratio_from_returns

# Calculate from returns list
returns = [0.01, -0.005, 0.02, 0.015, -0.01]
sharpe = sharpe_ratio_from_returns(
    returns, 
    risk_free_rate=0.0,
    periods_per_year=252  # For daily data
)
print(f"Sharpe ratio: {sharpe:.4f}")
```

## Available Functions

### Main Functions

1. **`sharpe_ratio_from_returns()`**
   - Input: List/array of periodic returns
   - Best for: When you already have calculated returns
   - Example: `[0.01, -0.005, 0.02, ...]`

2. **`sharpe_ratio_from_prices()`**
   - Input: List/array of prices
   - Best for: When you have price time series
   - Example: `[100, 101, 100.5, 102, ...]`

3. **`sharpe_ratio_from_dataframe()`**
   - Input: Pandas DataFrame with returns column
   - Best for: Working with CSV/DataFrame data
   - Example: Your inference CSV files

4. **`sortino_ratio()`**
   - Similar to Sharpe but only penalizes downside volatility
   - Often preferred for crypto trading

5. **`information_ratio()`**
   - Measures performance relative to a benchmark
   - Useful for comparing against buy-and-hold

## Usage Examples

### Example 1: Calculate from Your Inference CSV

```python
import pandas as pd
from backtesting.sharpe_calculator import sharpe_ratio_from_dataframe

# Load your inference results
df = pd.read_csv("backtesting/data/inference_returns_w.csv")

# Calculate Sharpe from actual returns
sharpe = sharpe_ratio_from_dataframe(
    df, 
    return_column='returns',
    risk_free_rate=0.0,
    periods_per_year=52  # Weekly data
)

print(f"Sharpe ratio: {sharpe:.4f}")
```

### Example 2: Calculate from Backtest Results

```python
from backtesting.backtest import BacktestRunner

# Run your backtest
runner = BacktestRunner("path/to/data.csv")
runner.run_strategy('SimpleAI')

# The manual Sharpe will be calculated automatically if needed
summary = runner.create_summary_table()
print(summary)
```

### Example 3: Compare Multiple Strategies

```python
from backtesting.sharpe_calculator import sharpe_ratio_from_returns

strategies = {
    'Strategy A': [0.01, -0.005, 0.02, ...],
    'Strategy B': [0.015, 0.005, -0.01, ...],
    'Strategy C': [0.008, 0.012, 0.003, ...]
}

for name, returns in strategies.items():
    sharpe = sharpe_ratio_from_returns(returns, periods_per_year=52)
    print(f"{name}: Sharpe = {sharpe:.4f}")
```

## Important Considerations

### 1. **Periods Per Year**
Choose the correct value based on your data frequency:
- **Daily**: 252 (standard trading days)
- **Weekly**: 52
- **Monthly**: 12
- **Hourly (24/7 crypto)**: 8760

### 2. **Risk-Free Rate**
- Often set to 0.0 for simplicity
- US Treasury rate: ~4-5% annually (as of 2024)
- For crypto, 0% is common since it's a 24/7 market

### 3. **Interpretation**
- **Sharpe > 1.0**: Good risk-adjusted returns
- **Sharpe > 2.0**: Very good
- **Sharpe > 3.0**: Excellent
- **Sharpe < 1.0**: Poor risk-adjusted returns
- **Negative Sharpe**: Strategy loses money on average

### 4. **Crypto-Specific Considerations**
- Crypto is more volatile than traditional assets
- Lower Sharpe ratios are more common
- Consider using Sortino ratio (only penalizes downside)
- Sharpe > 0.5 is often considered acceptable for crypto

## Troubleshooting

### Issue: Sharpe ratio returns 0

**Causes:**
1. All returns are 0 (no trading activity)
2. Standard deviation is 0 (constant returns)
3. Insufficient data points

**Solution:**
```python
# Check your returns data
print(f"Number of returns: {len(returns)}")
print(f"Mean return: {np.mean(returns)}")
print(f"Std dev: {np.std(returns)}")
```

### Issue: Sharpe ratio is extremely high/low

**Causes:**
1. Very few data points (overfitting)
2. Incorrect periods_per_year setting
3. Data contains errors

**Solution:**
```python
# Ensure you have enough data
assert len(returns) >= 30, "Need at least 30 data points"

# Check for outliers
print(f"Min return: {min(returns)}")
print(f"Max return: {max(returns)}")
```

## Integration with Your Pipeline

### In `run_hpo.py`

When optimizing with Optuna, you can now use Sharpe ratio as your objective:

```python
# After backtesting in your objective function
from backtesting.sharpe_calculator import sharpe_ratio_from_returns

# Get returns from your backtest
returns = [...] # Extract from backtest results

# Calculate Sharpe as optimization objective
sharpe = sharpe_ratio_from_returns(returns, periods_per_year=52)

# Optuna maximizes, so return negative for minimization
return -sharpe  # Or return sharpe if direction="maximize"
```

### In Backtest Pipeline

The manual Sharpe calculation is now automatic in all backtesting functions:
- Regular backtests
- Parameter optimization
- Walk-forward optimization

## Testing Your Implementation

Run the example script to verify everything works:

```bash
cd backtesting
python example_sharpe_usage.py
```

This will show you:
- How to load and calculate from CSV files
- Portfolio value Sharpe calculations
- Strategy comparisons
- Benchmark comparisons
- Crypto-specific examples

## References

- Original Sharpe Ratio paper: William F. Sharpe (1966)
- For crypto trading, also consider:
  - Sortino Ratio (downside risk only)
  - Calmar Ratio (return/max drawdown)
  - Information Ratio (vs benchmark)

## Quick Reference Card

```python
# Most common use case for your project:
from backtesting.sharpe_calculator import sharpe_ratio_from_returns

# Weekly crypto returns
sharpe = sharpe_ratio_from_returns(
    returns,
    risk_free_rate=0.0,
    periods_per_year=52
)

# Interpretation for crypto:
# > 0.5: Acceptable
# > 1.0: Good
# > 1.5: Very good
# > 2.0: Excellent
```


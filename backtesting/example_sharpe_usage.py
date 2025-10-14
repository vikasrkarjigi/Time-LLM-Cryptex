"""
Example usage of manual Sharpe ratio calculations.

This script demonstrates how to calculate Sharpe ratios from various data sources.
"""

import pandas as pd
import numpy as np
from sharpe_calculator import (
    sharpe_ratio_from_returns,
    sharpe_ratio_from_prices,
    sharpe_ratio_from_dataframe,
    sortino_ratio,
    information_ratio
)


def example_1_from_csv():
    """Calculate Sharpe ratio from a CSV file with predictions."""
    print("=" * 60)
    print("Example 1: Calculate Sharpe from CSV inference file")
    print("=" * 60)
    
    # Load your inference CSV
    df = pd.read_csv("data/inference_returns_w.csv")
    
    # Calculate Sharpe from actual returns
    actual_returns = df['returns'].dropna()
    sharpe_actual = sharpe_ratio_from_returns(
        actual_returns, 
        risk_free_rate=0.0,  # 0% risk-free rate
        periods_per_year=52   # Weekly data
    )
    print(f"Sharpe ratio (actual returns): {sharpe_actual:.4f}")
    
    # Calculate Sharpe from predicted returns (if available)
    if 'returns_predicted_1' in df.columns:
        predicted_returns = df['returns_predicted_1'].dropna()
        sharpe_predicted = sharpe_ratio_from_returns(
            predicted_returns,
            risk_free_rate=0.0,
            periods_per_year=52
        )
        print(f"Sharpe ratio (predicted returns): {sharpe_predicted:.4f}")
    
    print()


def example_2_from_portfolio_values():
    """Calculate Sharpe ratio from portfolio value changes."""
    print("=" * 60)
    print("Example 2: Calculate Sharpe from portfolio values")
    print("=" * 60)
    
    # Simulate portfolio values over time
    initial_value = 10000
    returns = np.random.normal(0.001, 0.02, 100)  # 100 days of returns
    
    portfolio_values = [initial_value]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    sharpe = sharpe_ratio_from_prices(
        portfolio_values,
        risk_free_rate=0.02,  # 2% annual risk-free rate
        periods_per_year=252   # Daily data
    )
    
    print(f"Simulated portfolio Sharpe ratio: {sharpe:.4f}")
    print(f"Initial value: ${initial_value:,.2f}")
    print(f"Final value: ${portfolio_values[-1]:,.2f}")
    print(f"Total return: {((portfolio_values[-1] / initial_value) - 1) * 100:.2f}%")
    print()


def example_3_compare_strategies():
    """Compare Sharpe ratios of different trading strategies."""
    print("=" * 60)
    print("Example 3: Compare multiple trading strategies")
    print("=" * 60)
    
    # Simulate different strategy returns
    np.random.seed(42)
    
    strategies = {
        'Conservative': np.random.normal(0.0005, 0.01, 252),
        'Moderate': np.random.normal(0.001, 0.015, 252),
        'Aggressive': np.random.normal(0.002, 0.03, 252),
    }
    
    results = []
    for name, returns in strategies.items():
        sharpe = sharpe_ratio_from_returns(returns, periods_per_year=252)
        sortino = sortino_ratio(returns, periods_per_year=252)
        mean_return = np.mean(returns) * 252 * 100  # Annualized %
        volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized %
        
        results.append({
            'Strategy': name,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Ann. Return (%)': mean_return,
            'Ann. Volatility (%)': volatility
        })
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False, float_format='%.4f'))
    print()


def example_4_with_benchmark():
    """Calculate Information Ratio against a benchmark."""
    print("=" * 60)
    print("Example 4: Calculate Information Ratio vs. Benchmark")
    print("=" * 60)
    
    # Simulate strategy and benchmark returns
    np.random.seed(42)
    benchmark_returns = np.random.normal(0.0008, 0.015, 252)
    strategy_returns = benchmark_returns + np.random.normal(0.0002, 0.005, 252)
    
    sharpe_strategy = sharpe_ratio_from_returns(strategy_returns, periods_per_year=252)
    sharpe_benchmark = sharpe_ratio_from_returns(benchmark_returns, periods_per_year=252)
    ir = information_ratio(strategy_returns, benchmark_returns)
    
    print(f"Strategy Sharpe ratio: {sharpe_strategy:.4f}")
    print(f"Benchmark Sharpe ratio: {sharpe_benchmark:.4f}")
    print(f"Information Ratio: {ir:.4f}")
    print()
    
    if ir > 0:
        print("✓ Strategy outperforms benchmark on risk-adjusted basis")
    else:
        print("✗ Strategy underperforms benchmark on risk-adjusted basis")
    print()


def example_5_cryptocurrency_returns():
    """Calculate Sharpe ratio for cryptocurrency returns."""
    print("=" * 60)
    print("Example 5: Cryptocurrency Trading Sharpe Ratio")
    print("=" * 60)
    
    # Example: Weekly BTC returns
    btc_weekly_returns = [
        0.0374, -0.0040, 0.0994, -0.0454, -0.0339,
        -0.0217, -0.0022, -0.0721, -0.0159, -0.0500,
        -0.0016, 0.0573, -0.0260, -0.1045, 0.0970,
        0.1165, 0.0091, 0.0266, 0.0761, 0.0251
    ]
    
    sharpe = sharpe_ratio_from_returns(
        btc_weekly_returns,
        risk_free_rate=0.0,
        periods_per_year=52  # Weekly data
    )
    
    sortino = sortino_ratio(
        btc_weekly_returns,
        risk_free_rate=0.0,
        periods_per_year=52
    )
    
    total_return = ((1 + np.sum(btc_weekly_returns)) - 1) * 100
    volatility = np.std(btc_weekly_returns) * np.sqrt(52) * 100
    
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Sortino Ratio: {sortino:.4f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Volatility: {volatility:.2f}%")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MANUAL SHARPE RATIO CALCULATION EXAMPLES")
    print("=" * 60 + "\n")
    
    try:
        example_1_from_csv()
    except FileNotFoundError:
        print("Skipping Example 1 (CSV file not found)\n")
    
    example_2_from_portfolio_values()
    example_3_compare_strategies()
    example_4_with_benchmark()
    example_5_cryptocurrency_returns()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


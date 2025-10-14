"""
Standalone Sharpe Ratio Calculator

This module provides various methods to calculate the Sharpe ratio manually.
Can be used independently of backtrader or other frameworks.
"""

import numpy as np
import pandas as pd

def calculate_sharpe_ratio_manual(strategy_result, risk_free_rate=0.0, periods_per_year=252):
    """
    Manually calculate Sharpe ratio from strategy results.
    
    Args:
        strategy_result: Backtrader strategy result object
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Number of trading periods per year (252 for daily, 52 for weekly, 12 for monthly)
    
    Returns:
        float: Sharpe ratio
    """
    try:
        # Get the returns from the strategy's portfolio values
        # Access the strategy's data to get portfolio values over time
        portfolio_values = []
        
        # If we have access to the strategy object, get its observers
        if hasattr(strategy_result, 'observers'):
            # Try to get broker observer which tracks portfolio value
            for observer in strategy_result.observers:
                if hasattr(observer, 'lines') and hasattr(observer.lines, 'value'):
                    portfolio_values = list(observer.lines.value.array)
                    break
        
        # If we couldn't get portfolio values from observers, use the returns analyzer
        if not portfolio_values or len(portfolio_values) == 0:
            returns_analysis = strategy_result.analyzers.returns.get_analysis()
            if 'rtot' in returns_analysis:
                # If we only have total return, we can't calculate a proper Sharpe ratio
                # Return None to indicate we need to use periodic returns
                return None
        
        # Calculate periodic returns from portfolio values
        if portfolio_values and len(portfolio_values) > 1:
            returns = []
            for i in range(1, len(portfolio_values)):
                if portfolio_values[i-1] != 0:
                    ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                    returns.append(ret)
        else:
            return None
        
        if not returns or len(returns) == 0:
            return 0.0
        
        # Calculate mean and standard deviation of returns
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)  # Sample standard deviation
        
        # Avoid division by zero
        if std_return == 0:
            return 0.0
        
        # Calculate Sharpe ratio
        # Annualized: sqrt(periods_per_year) * (mean_return - risk_free_rate/periods_per_year) / std_return
        risk_free_per_period = risk_free_rate / periods_per_year
        sharpe_ratio = np.sqrt(periods_per_year) * (mean_return - risk_free_per_period) / std_return
        
        return sharpe_ratio
        
    except Exception as e:
        print(f"Error calculating Sharpe ratio manually: {e}")
        return 0.0



def sharpe_ratio_from_returns(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate Sharpe ratio from a list/array of periodic returns.
    
    Formula: Sharpe = sqrt(N) * (mean(returns) - rf) / std(returns)
    where N is the number of periods per year
    
    Args:
        returns: List, array, or pandas Series of periodic returns (e.g., daily, weekly)
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Number of trading periods per year
                         - 252 for daily returns
                         - 52 for weekly returns
                         - 12 for monthly returns
    
    Returns:
        float: Annualized Sharpe ratio
        
    Example:
        >>> returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        >>> sharpe = sharpe_ratio_from_returns(returns, periods_per_year=252)
        >>> print(f"Sharpe ratio: {sharpe:.4f}")
    """
    if returns is None or len(returns) == 0:
        return 0.0
    
    returns_array = np.array(returns)
    
    # Remove NaN values
    returns_array = returns_array[~np.isnan(returns_array)]
    
    if len(returns_array) == 0:
        return 0.0
    
    # Calculate mean and standard deviation
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)  # Sample standard deviation
    
    # Avoid division by zero
    if std_return == 0:
        return 0.0
    
    # Calculate per-period risk-free rate
    risk_free_per_period = risk_free_rate / periods_per_year
    
    # Calculate and annualize Sharpe ratio
    sharpe_ratio = np.sqrt(periods_per_year) * (mean_return - risk_free_per_period) / std_return
    
    return sharpe_ratio


def sharpe_ratio_from_prices(prices, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate Sharpe ratio from a list/array of prices.
    
    Args:
        prices: List, array, or pandas Series of prices
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Number of trading periods per year
    
    Returns:
        float: Annualized Sharpe ratio
        
    Example:
        >>> prices = [100, 101, 100.5, 102, 103.5, 102.5]
        >>> sharpe = sharpe_ratio_from_prices(prices, periods_per_year=252)
        >>> print(f"Sharpe ratio: {sharpe:.4f}")
    """
    if prices is None or len(prices) < 2:
        return 0.0
    
    prices_array = np.array(prices)
    
    # Calculate returns
    returns = np.diff(prices_array) / prices_array[:-1]
    
    return sharpe_ratio_from_returns(returns, risk_free_rate, periods_per_year)


def sharpe_ratio_from_portfolio_values(portfolio_values, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate Sharpe ratio from portfolio values over time.
    
    Args:
        portfolio_values: List, array, or pandas Series of portfolio values over time
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Number of trading periods per year
    
    Returns:
        float: Annualized Sharpe ratio
    """
    return sharpe_ratio_from_prices(portfolio_values, risk_free_rate, periods_per_year)


def sharpe_ratio_from_dataframe(df, return_column='returns', risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate Sharpe ratio from a pandas DataFrame.
    
    Args:
        df: pandas DataFrame containing returns or prices
        return_column: Name of the column containing returns (if using returns)
                      or prices (if using prices)
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Number of trading periods per year
    
    Returns:
        float: Annualized Sharpe ratio
        
    Example:
        >>> df = pd.DataFrame({'returns': [0.01, -0.005, 0.02, 0.015, -0.01]})
        >>> sharpe = sharpe_ratio_from_dataframe(df, 'returns', periods_per_year=252)
        >>> print(f"Sharpe ratio: {sharpe:.4f}")
    """
    if return_column not in df.columns:
        raise ValueError(f"Column '{return_column}' not found in DataFrame")
    
    return sharpe_ratio_from_returns(df[return_column].values, risk_free_rate, periods_per_year)


def rolling_sharpe_ratio(returns, window=30, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate rolling Sharpe ratio over a specified window.
    
    Args:
        returns: List, array, or pandas Series of periodic returns
        window: Rolling window size (number of periods)
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Number of trading periods per year
    
    Returns:
        numpy array or pandas Series: Rolling Sharpe ratios
        
    Example:
        >>> returns = np.random.normal(0.001, 0.02, 100)
        >>> rolling_sharpe = rolling_sharpe_ratio(returns, window=30, periods_per_year=252)
        >>> print(f"Latest Sharpe: {rolling_sharpe[-1]:.4f}")
    """
    if isinstance(returns, (list, np.ndarray)):
        returns = pd.Series(returns)
    
    risk_free_per_period = risk_free_rate / periods_per_year
    
    # Calculate rolling mean and std
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std(ddof=1)
    
    # Calculate rolling Sharpe
    rolling_sharpe = np.sqrt(periods_per_year) * (rolling_mean - risk_free_per_period) / rolling_std
    
    return rolling_sharpe


def information_ratio(returns, benchmark_returns):
    """
    Calculate Information Ratio (similar to Sharpe but relative to a benchmark).
    
    Formula: IR = mean(excess_returns) / std(excess_returns)
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
    
    Returns:
        float: Information ratio
        
    Example:
        >>> portfolio_returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        >>> benchmark_returns = [0.008, -0.003, 0.015, 0.012, -0.008]
        >>> ir = information_ratio(portfolio_returns, benchmark_returns)
        >>> print(f"Information Ratio: {ir:.4f}")
    """
    returns_array = np.array(returns)
    benchmark_array = np.array(benchmark_returns)
    
    # Calculate excess returns
    excess_returns = returns_array - benchmark_array
    
    # Calculate mean and std of excess returns
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    return mean_excess / std_excess


def sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate Sortino ratio (uses downside deviation instead of total volatility).
    
    Formula: Sortino = sqrt(N) * (mean(returns) - rf) / downside_std
    
    Args:
        returns: List, array, or pandas Series of periodic returns
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Number of trading periods per year
    
    Returns:
        float: Annualized Sortino ratio
        
    Example:
        >>> returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        >>> sortino = sortino_ratio(returns, periods_per_year=252)
        >>> print(f"Sortino ratio: {sortino:.4f}")
    """
    if returns is None or len(returns) == 0:
        return 0.0
    
    returns_array = np.array(returns)
    returns_array = returns_array[~np.isnan(returns_array)]
    
    if len(returns_array) == 0:
        return 0.0
    
    mean_return = np.mean(returns_array)
    
    # Calculate downside deviation (only negative returns)
    negative_returns = returns_array[returns_array < 0]
    
    if len(negative_returns) == 0:
        # No negative returns - undefined Sortino, return a large number
        return float('inf')
    
    downside_std = np.std(negative_returns, ddof=1)
    
    if downside_std == 0:
        return 0.0
    
    risk_free_per_period = risk_free_rate / periods_per_year
    sortino = np.sqrt(periods_per_year) * (mean_return - risk_free_per_period) / downside_std
    
    return sortino


# Example usage
if __name__ == "__main__":
    # Example 1: Calculate from returns
    returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.005, 0.03, -0.02, 0.01, 0.008]
    sharpe = sharpe_ratio_from_returns(returns, periods_per_year=252)
    print(f"Sharpe ratio (from returns): {sharpe:.4f}")
    
    # Example 2: Calculate from prices
    prices = [100, 101, 100.5, 102, 103.5, 102.5, 103, 106, 104, 105]
    sharpe_prices = sharpe_ratio_from_prices(prices, periods_per_year=252)
    print(f"Sharpe ratio (from prices): {sharpe_prices:.4f}")
    
    # Example 3: Calculate Sortino ratio
    sortino = sortino_ratio(returns, periods_per_year=252)
    print(f"Sortino ratio: {sortino:.4f}")
    
    # Example 4: Calculate Information Ratio
    benchmark_returns = [0.008, -0.003, 0.015, 0.012, -0.008, 0.004, 0.025, -0.015, 0.009, 0.007]
    ir = information_ratio(returns, benchmark_returns)
    print(f"Information Ratio: {ir:.4f}")


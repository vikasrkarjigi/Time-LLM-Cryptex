"""
Standalone Sharpe Ratio Calculator

This module provides various methods to calculate the Sharpe ratio manually.
Can be used independently of backtrader or other frameworks.
"""
import numpy as np

def sharpe_ratio_from_returns(returns, risk_free_rate=0.0, periods_per_year=365):
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



def sortino_ratio(returns, risk_free_rate=3.0, periods_per_year=365):
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
        return -float('inf')
    
    downside_std = np.std(negative_returns, ddof=1)
    
    if downside_std == 0:
        return 0.0
    
    risk_free_per_period = risk_free_rate / periods_per_year
    sortino = np.sqrt(periods_per_year) * (mean_return - risk_free_per_period) / downside_std
    
    return sortino


def sharpe_from_trade_log(trade_log_df, periods_per_year=365, risk_free_rate=0.0):
    
    """
    Calculate Sharpe ratio from trade log DataFrame.
    
    Args:
        trade_log_df: DataFrame from TradeLog analyzer
        periods_per_year: Number of trading periods per year (365 for daily)
        risk_free_rate: Annual risk-free rate
    
    Returns:
        float: Sharpe ratio calculated from trade returns
    """
    if trade_log_df.empty or 'return' not in trade_log_df.columns:
        return None
    
    # Get valid returns (non-NaN)
    valid_returns = trade_log_df['return'].dropna()
    
    if len(valid_returns) == 0:
        return None
    
    # Calculate Sharpe ratio using the imported function
    return sharpe_ratio_from_returns(valid_returns.values, risk_free_rate, periods_per_year)

def sortino_from_trade_log(trade_log_df, periods_per_year=365, risk_free_rate=0.0):
    """
    Calculate Sortino ratio from trade log DataFrame.
    
    The Sortino ratio is similar to the Sharpe ratio but uses downside deviation
    instead of total volatility, focusing only on negative returns.
    
    Args:
        trade_log_df: DataFrame from TradeLog analyzer
        periods_per_year: Number of trading periods per year (365 for daily)
        risk_free_rate: Annual risk-free rate
    
    Returns:
        float: Sortino ratio calculated from trade returns
    """
    if trade_log_df.empty or 'return' not in trade_log_df.columns:
        return None
    
    # Get valid returns (non-NaN)
    valid_returns = trade_log_df['return'].dropna()
    
    if len(valid_returns) == 0:
        return None
    
    # Calculate Sortino ratio using the imported function
    return sortino_ratio(valid_returns.values, risk_free_rate, periods_per_year)


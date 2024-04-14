import pandas as pd


def common_moving_average(time_series: pd.Series, window_size: int = 5):
    """
    Common moving average for time series smoothing.
    
    Parameters
    ----------
    time_series : pd.Series
        Time series to apply common moving average.
    window_size : int
        Window of moving average.
        
    Returns
    -------
    result : list
        Moving averaged time series.
    """
    moving_average = time_series.rolling(window=window_size).mean()
    return moving_average


def exponential_smoothing(time_series: pd.Series, alpha: float):
    """
    Exponential smoothing for time series.
    
    Parameters
    ----------
    time_series : pd.Series
        Time series to apply common moving average.
    alpha : float
        Multiplier of actual time series i_th value.
        
    Returns
    -------
    result : list
        Exponentially smoothed time series.
    """
    result = [time_series.iloc[0]]
    for i in range(1, len(time_series)):
        result.append(alpha * time_series.iloc[i] + (1 - alpha) * result[i - 1])
    return result


def double_exponential_smoothing(time_series: pd.Series, alpha: float, beta: float):
    """
    Double exponential smoothing for time series.
    
    Parameters
    ----------
    time_series : pd.Series
        Time series to apply common moving average.
    alpha : float
        Multiplier of level.
    beta : float
        Multiplier of trend.
        
    Returns
    -------
    result : list
        Double exponentially smoothed time series.
    """
    result = [time_series.iloc[0]]
    for i in range(1, len(time_series)):
        if i == 1:
            level, trend = time_series.iloc[0], time_series.iloc[1] - time_series.iloc[0]
        if i >= len(time_series):
            value = result[-1]
        else:
            value = time_series.iloc[i]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


if __name__ == '__main__':
    pass

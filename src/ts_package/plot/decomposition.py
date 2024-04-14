import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm
from matplotlib import pyplot as plt


def plot_decomposition(decomposition_result, date):
    """Draw time series decomposition"""
    observed = decomposition_result.observed
    seasonal = decomposition_result.seasonal
    trend = decomposition_result.trend
    residuals = decomposition_result.resid
    
    plt.figure(figsize=(12, 14))

    plt.subplot(4, 1, 1)
    plt.title('Observed')
    plt.plot(date, observed)

    plt.subplot(4, 1, 2)
    plt.title('seasonal')
    plt.plot(date, seasonal)

    plt.subplot(4, 1, 3)
    plt.title('trend')
    plt.plot(date, trend)

    plt.subplot(4, 1, 4)
    plt.title('residuals')
    plt.scatter(date, residuals)

    return


def tsplot(y, lags=None):
    """Draw time series, acf, pacf, qq and probability plots"""
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context('bmh'):    
        plt.figure(figsize=(15, 10))
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        
        # График автокорреляции
        sm.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        # График частичной автокорреляции
        sm.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        # QQ график
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        # График вероятности
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
        plt.tight_layout()
        
    return

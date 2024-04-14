import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_time_series(df: pd.DataFrame, x: str|pd.Series, y: str|pd.Series, title: str = '', xlabel: str = 'Date', ylabel: str = 'Value'):
    """Draw time series"""
    plt.figure(figsize=(12, 4))
    sns.lineplot(df, x=x, y=y)
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    return

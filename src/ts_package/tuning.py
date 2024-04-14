import optuna
import pandas as pd
from catboost import CatBoostRegressor

from ts_package.validation.validation import get_time_series_cv_score


def objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series):
    """
    Objective function for optuna study.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial instance.
    X : pd.DataFrame
        Features data.
    y : pd.Series
        Target variable.

    Returns
    -------
    Target metric value.
    """
    parameters = {
        'loss_function': trial.suggest_categorical('loss_function', ['RMSE', 'MAE', 'MAPE']),
        'depth': trial.suggest_int('depth', 4, 8),
        'iterations': trial.suggest_int('iterations', 200, 1500),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'random_state': 777,
        'verbose': False
    }
    cv_result = get_time_series_cv_score(X, y, parameters, CatBoostRegressor)

    return cv_result['bm_mean']

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from ts_package.validation.metric import BusinessSimulation


def get_time_series_cv_score(
        X: pd.DataFrame, y: pd.Series, model_parameters: dict, model_class: CatBoostRegressor,
        return_calibrated_model: CatBoostRegressor = False
):
    """
    Perform time series cross-validation with specified model_parameters and model_class.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame.
    y : pd.Series
        Target variable.
    model_parameters : dict
        Parameters/hyperparameters to pass into model_class instance.
    model_class : CatBoostRegressor|other
        CatBoostRegressor class actually.
    return_calibrated_model : CatBoostRegressor|other
        Fitted CatBoostRegressor instance by time series cross validation.
    
    Returns
    -------
    result : dict
        Dictionary with mean and std over all time series splitted folds of cross-validation.
    """
    business_metrics = []
    maes = []
    splitter = TimeSeriesSplit(n_splits=7) 
    for train_indexes, val_indexes in splitter.split(X):
        X_train, y_train = X.iloc[train_indexes, :], y.iloc[train_indexes]
        X_val, y_val = X.iloc[val_indexes, :], y.iloc[val_indexes]

        model_instance = model_class(**model_parameters)
        model_instance.fit(X_train, y_train, verbose_eval=500)

        predictions = model_instance.predict(X_val)
        bs = BusinessSimulation()
        business_metric = bs.evaluate_performance(y_val, predictions)
        business_metrics.append(business_metric)
        mae = mean_absolute_error(y_val, predictions)
        maes.append(mae)
    
    result = {
        'bm_mean': np.mean(business_metrics),
        'bm_std': np.std(business_metrics),
        'mae_mean': np.mean(maes),
        'mae_std': np.std(maes)
    }
    if return_calibrated_model:
        return result, model_instance
    return result

import numpy as np
from catboost import (CatBoostRegressor, EFeaturesSelectionAlgorithm,
                      EShapCalcType, Pool)


def get_catboost_feature_importances(model: CatBoostRegressor):
    """Feature importance of CatBoostRegressor"""
    importance_idx = np.argsort(model.feature_importances_)[::-1]
    return dict(
        zip(
            np.array(model.feature_names_)[importance_idx],
            model.feature_importances_[importance_idx]
        )
    )


def select_features_wrapped_catboost(
        features: list, algorithm: EFeaturesSelectionAlgorithm, train_pool: Pool, val_pool: Pool,
        steps: int = 1, n_features: int = 10, 
    ):
    """
    Performs different CatBoost feature selection wrapped algorithms.

    Parameters
    ----------
    features : list
        Features list to select from.
    algorithm : EFeaturesSelectionAlgorithm
        CatBoost feature selection algorithm instance.
    train_pool : Pool
        CatBoost pool for training data.
    val_pool : Pool
        CatBoost pool for validation data.
    steps : int
        Number of performing steps. Bigger is better.
    n_features : int
        Final number of features after selection.
    
    Returns
    -------
    summary : dict
        Summary of the performed feature selection.
    """
    print('Algorithm for feature selection:', algorithm)
    model = CatBoostRegressor(random_seed=777)
    summary = model.select_features(
        train_pool,
        eval_set=val_pool,
        features_for_select=features,
        num_features_to_select=n_features,
        steps=steps,
        algorithm=algorithm,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=True,
        logging_level='Silent',
        plot=False
    )
    return summary

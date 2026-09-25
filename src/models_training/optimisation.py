import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def optimisation(model, param_grid, x_train, x_test, y_train, y_test):

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(x_train, y_train)

    # best model
    best_rf = grid_search.best_estimator_

    # Evaluate the Optimized Model
    y_pred = best_rf.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    return {
        "best_params": grid_search.best_params_,
        "y_pred": y_pred,
        "evaluation": {"rmse": rmse, "mae": mae, "r2": r2}
    }

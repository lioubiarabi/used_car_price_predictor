import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def training(model, x_train, y_train, x_test, y_test):
    # training model
    model.fit(x_train, y_train)

    # make predictions
    y_pred = model.predict(x_test)

    # Evaluate performance
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "y_pred": y_pred,
        "evaluation": {"rmse": rmse, "mae": mae, "r2": r2}
    }

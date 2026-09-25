from sklearn.model_selection import train_test_split
from src.data_loader.cleaned_results import load_cleaned_data
from src.models_training.training import training
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import pandas as pd

df = load_cleaned_data()

# separate features
x = df.drop(columns=["selling_price"])
y = df["selling_price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = [
    {"name": "LinearRegression", "object" : LinearRegression()},
    {"name": "RandomForest", "object": RandomForestRegressor(random_state=42)},
    {"name": "XGBRegressor", "object": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        objective='reg:squarederror'
    )},
    {"name": "SVR", "object": SVR(kernel='rbf')}
]

predictions_df = pd.DataFrame({"y_test": y_test.reset_index(drop=True)})
metrics_data = []
for model in models:
    result = training(model["object"], x_train, y_train, x_test, y_test)

    predictions_df[f"{model["name"]}"] = result['y_pred']
    metrics_data.append({
        "Model": model["name"],
        "RMSE": result['evaluation']['rmse'],
        "MAE": result['evaluation']['mae'],
        "R2": result['evaluation']['r2']
    })

pd.DataFrame(metrics_data).to_csv("../../data/training/model_predictions.csv", index=False)
predictions_df.to_csv("../../data/processed/model_metrics.csv", index=False)

print("Done: training")
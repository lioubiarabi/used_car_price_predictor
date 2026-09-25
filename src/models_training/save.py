import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.data_loader.cleaned_results import load_cleaned_data
from src.models_training.training import training

df = load_cleaned_data()

x = df.drop(columns=["selling_price"])
y = df["selling_price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = XGBRegressor(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=300,
        subsample=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
final_model = training(model, x_train, y_train, x_test, y_test)

joblib.dump(final_model['model'], "../../data/models/xgboost_car_price_model.pkl")

print(f"Success! Champion model saved")

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.svm import SVR
from src.models_training.optimisation import optimisation

from src.data_loader.cleaned_results import load_cleaned_data

df = load_cleaned_data()

X = df.drop(columns=["selling_price"])
y = df["selling_price"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    {
        "name": "LinearRegression",
        "object" : LinearRegression(),
        "params": {'fit_intercept': [True, False], 'positive': [True, False]}
    },
    {
        "name": "RandomForest",
        "object": RandomForestRegressor(random_state=42),
        "params": {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    {
        "name": "XGBRegressor",
        "object": XGBRegressor(random_state=42, objective='reg:squarederror'),
        "params": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    },
    {
        "name": "SVR",
        "object": Pipeline([('scaler', StandardScaler()), ('model', SVR())]),
        "params":{
            'model__kernel': ['rbf', 'linear'],
            'model__C': [0.1, 1, 10, 100],
            'model__gamma': ['scale', 'auto', 0.1]
        }
    }
]

text_content = ""

for model in models:
    results = optimisation(model["object"], model["params"], x_train, x_test, y_train, y_test)

    text_content += f"--- {model['name']} ---\n"
    text_content += f"Best Parameters: {results['best_params']}\n"
    text_content += f"RMSE : {results['evaluation']['rmse']:,.2f}\n"
    text_content += f"MAE  : {results['evaluation']['mae']:,.2f}\n"
    text_content += f"R²   : {results['evaluation']['r2']:.4f}\n"
    text_content += "-" * 40 + "\n\n"

# write results in txt file
with open("../../data/processed/optimization_results.txt", "w", encoding="utf-8") as file:
    file.write(text_content)

print("Done: models optimisation")
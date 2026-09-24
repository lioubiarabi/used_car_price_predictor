from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

from src.data_loader.cleaned_results import load_cleaned_data
df = load_cleaned_data()

x = df[['km_driven']]
y = df['year']

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor())
])

mod = GridSearchCV(estimator=pipe,
                   param_grid={
                       "model__n_neighbors":range(1, 11)
                   },
                   cv=3)
mod.fit(x, y)
pred = mod.predict(x)

print(df[['km_driven', 'year']])
print(pred)

print(mod.best_params_)

import matplotlib.pylab as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from src.data_loader.cleaned_results import load_cleaned_data
df = load_cleaned_data()

x = df[['km_driven']]
y = df['year']

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LinearRegression())
])

pipe.fit(x, y)
pred = pipe.predict(x)
plt.scatter(pred, y)
plt.show()

print(df[['km_driven', 'year']])
print(pred)

mod = LinearRegression()

mod.fit(x, y)
pred = mod.predict(x)
print(pred)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from src.data_loader.cleaned_results import load_cleaned_data
df = load_cleaned_data()

x = df[['km_driven']]
y = df['selling_price']

mod = LinearRegression()

mod.fit(x, y)
pred = mod.predict(x)


print(df[['km_driven', 'selling_price']])
print(pred)

mod2 = KNeighborsRegressor()

mod2.fit(x, y)
pred2 = mod2.predict(x)
print(pred2)
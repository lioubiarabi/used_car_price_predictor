import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from src.data_loader.cleaned_results import load_cleaned_data
df = load_cleaned_data()

x = df[['km_driven']]
y = df['year']

# Isolate the scaler to transform the data so we can graph it
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Create a side-by-side visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: The original raw data
ax1.scatter(x, y, alpha=0.5, color='blue')
ax1.set_title("Before Scaling: Original Data")
ax1.set_xlabel("Kilometers Driven (Raw)")
ax1.set_ylabel("Year")

# Plot 2: The scaled data
ax2.scatter(x_scaled, y, alpha=0.5, color='red')
ax2.set_title("After Scaling: StandardScaler")
ax2.set_xlabel("Kilometers Driven (Standardized)")
ax2.set_ylabel("Year")

plt.tight_layout()
plt.show()

# You can still run your pipeline below this!
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LinearRegression())
])
pipe.fit(x, y)
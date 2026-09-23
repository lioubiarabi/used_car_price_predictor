
from src.data_loader.fetch import load_raw_data



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
df = load_raw_data()

# 2. Set up the plotting area (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 3. Create a boxplot for selling_price
sns.boxplot(y=df['selling_price'], ax=axes[0], color='skyblue')
axes[0].set_title('Outliers in Selling Price')
axes[0].set_ylabel('Selling Price')

# 4. Create a boxplot for km_driven
sns.boxplot(y=df['km_driven'], ax=axes[1], color='lightgreen')
axes[1].set_title('Outliers in Kilometers Driven')
axes[1].set_ylabel('Kilometers Driven')

# 5. Display the plots
plt.tight_layout()
plt.show()
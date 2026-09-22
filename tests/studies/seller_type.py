from src.data_loader.fetch import load_raw_data
import pandas as pd

df = load_raw_data()
df_column = df['seller_type']

counts = df_column.value_counts()
percentages = df_column.value_counts(normalize=True) * 100

summary_df = pd.DataFrame({
    'Count': counts,
    'Percentage': percentages
})
print(summary_df)

price_per_seller_type = df.groupby(['seller_type'])['selling_price'].describe()
print(price_per_seller_type[['mean', 'min', 'max']])

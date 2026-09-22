from src.data_loader.fetch import load_raw_data
import pandas as pd




df = load_raw_data()
df_column = df['owner']

counts = df_column.value_counts()
percentages = df_column.value_counts(normalize=True) * 100

summary_df = pd.DataFrame({
    'Count': counts,
    'Percentage': percentages
})
print(summary_df)


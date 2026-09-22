from src.data_loader.fetch import load_raw_data
from src.data_processing.clean import clean_data



df = load_raw_data()
df_column = df['year']

print(df_column.describe())

print("--- after cleaning ---")

df = clean_data(load_raw_data())
df_column = df['year']

print(df_column.describe())
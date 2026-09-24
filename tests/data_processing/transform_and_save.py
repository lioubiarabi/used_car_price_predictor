from src.data_loader.fetch import load_raw_data
from src.data_processing.clean import clean_data
from src.data_processing.transform import transform_data

df = load_raw_data()
cleaned_df = clean_data(df)
transformed_df = transform_data(cleaned_df)

print(transformed_df)
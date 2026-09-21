from src.data_loader.fetch import load_raw_data

df = load_raw_data()
print(df['owner'].mode()[0])
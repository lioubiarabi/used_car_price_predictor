from src.data_loader.fetch import load_raw_data
import pandas as pd




df = load_raw_data()

split_columns = df['name'].str.split(' ', n=2, expand=True)

# 3. Assign the new columns to the dataframe
df['brand'] = split_columns[0]
df['model'] = split_columns[1]
df['variant'] = split_columns[2]

# 4. Drop the original 'name' column if you no longer need it (optional)
# df = df.drop('name', axis=1)

df['brand'] = df['name'].str.split(' ').str[0]

# Calculate the average year by brand, round it, and sort from newest to oldest
avg_year = df.groupby(['brand', 'model'])['year'].mean()

# Display the results
print(avg_year)

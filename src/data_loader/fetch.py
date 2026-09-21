import pandas as pd

def load_raw_data(file_path = "../../data/raw/car-prices.csv"):
    # fetch data from csv file

    try:
        df = pd.read_csv(file_path)

        from scipy import stats
        import numpy as np

        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(df['selling_price'].dropna()))

        # Keep only rows where Z-score is less than 3
        df = df[(z_scores < 3)]

        return df
    except FileNotFoundError :
        print("Error: failed fetching cars prices file!")
        return None
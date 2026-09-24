import pandas as pd

def load_cleaned_data(file_path = "../../data/processed/cleaned_results.csv"):
    # fetch data from csv file

    try:
        df = pd.read_csv(file_path)

        from scipy import stats
        import numpy as np

        return df
    except FileNotFoundError :
        print("Error: failed fetching cars prices file!")
        return None
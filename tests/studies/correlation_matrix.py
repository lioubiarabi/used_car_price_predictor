import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.data_loader.fetch import load_raw_data


def plot_correlation_matrix(df):
    """Calculates and plots the correlation matrix for numerical features."""
    # 1. Isolate numerical columns
    num_cols = ['year', 'km_driven', 'selling_price', 'fuel', 'owner']

    """ encode owner and fuel column"""
    owner_mapping = {
        'Test Drive Car': 0,
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }
    df['owner'] = df['owner'].map(owner_mapping)

    fuel_mapping = {
        'Diesel': 1,
        'Petrol': 2,
        'CNG': 3,
        'LPG': 4,
        'Electric': 5
    }

    # 2. Apply the mapping to the column
    df['fuel'] = df['fuel'].map(fuel_mapping)

    df_numeric = df[num_cols]

    # 2. Calculate the correlation matrix mathematically
    corr_matrix = df_numeric.corr()
    print("--- Mathematical Correlation Matrix ---")
    print(corr_matrix)

    # 3. Visualize it as a Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,  # Shows the actual numbers inside the squares
        cmap='coolwarm',  # Red for positive correlation, Blue for negative
        fmt=".2f",  # Rounds numbers to 2 decimal places
        linewidths=0.5
    )
    plt.title("Correlation Matrix: Selling Price vs Features")
    plt.tight_layout()
    plt.show()

# If running directly:
# df = pd.read_csv('../data/raw/car-price-6aad030a5ff4f172056750.csv')
# plot_correlation_matrix(df)

plot_correlation_matrix(load_raw_data())
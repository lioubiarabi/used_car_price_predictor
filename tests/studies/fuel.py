from src.data_loader.fetch import load_raw_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = load_raw_data()
df_column = df['fuel']

counts = df_column.value_counts()
percentages = df_column.value_counts(normalize=True) * 100

summary_df = pd.DataFrame({
    'Count': counts,
    'Percentage': percentages
})
print(summary_df)


def plot_fuel_analysis(df):
    """
    Generates visualizations to analyze the 'fuel' column.
    Requires seaborn and matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Count Plot: Added hue='fuel' and legend=False
    sns.countplot(data=df, x='fuel', ax=axes[0], palette='Set2', hue='fuel', legend=False)
    axes[0].set_title('Volume of Cars by Fuel Type')
    axes[0].set_xlabel('Fuel Type')
    axes[0].set_ylabel('Number of Cars')

    # 2. Box Plot: Added hue='fuel' and legend=False
    sns.boxplot(data=df, x='fuel', y='selling_price', ax=axes[1], palette='Set2', hue='fuel', legend=False)
    axes[1].set_title('Impact of Fuel Type on Selling Price')
    axes[1].set_xlabel('Fuel Type')
    axes[1].set_ylabel('Selling Price')

    plt.tight_layout()
    plt.show()

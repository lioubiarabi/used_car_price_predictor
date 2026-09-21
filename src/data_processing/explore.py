import pandas as pd

def print_basic_stats(df):
    # Prints shape, info, and missing value counts

    print("--- Dataset Shape ---")
    print(f"{df.shape[0]} rows, {df.shape[1]} columns\n")

    print("--- Data Types and Non-Null Counts ---")
    df.info()
    print("\n")

    print("--- Missing Values Count ---")
    print(df.isnull().sum(), "\n")

    print("--- Descriptive Statistics ---")
    print(df.describe(include='all'), "\n")



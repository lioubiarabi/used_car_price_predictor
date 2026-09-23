import os
from datetime import datetime
from scipy import stats
import numpy as np

def clean_data(df):
    """split the name into brand model variant"""
    split_columns = df['name'].str.split(' ', n=2, expand=True)
    df['brand'] = split_columns[0]
    df['model'] = split_columns[1]
    df['variant'] = split_columns[2]
    df['brand'] = df['name'].str.split(' ').str[0]

    # replace the outlier values in km_driven with NaN
    z_scores_km = np.abs(stats.zscore(df['km_driven'], nan_policy='omit'))
    df.loc[(z_scores_km >= 3), 'km_driven'] = np.nan

    # remove the outlier values in selling_price
    z_scores_sp = np.abs(stats.zscore(df['selling_price'], nan_policy='omit'))
    df = df[(z_scores_sp < 3)]


    # Calculate and replace the average year by brand and model
    df['year'] = df['year'].fillna(
        df.groupby(['brand', 'model'])['year'].transform('mean').round()
    )

    # Fill missing fuel with the most common fuel type by model and
    df['fuel'] = df['fuel'].fillna(
        df.groupby(['brand', 'model'])['fuel'].transform(lambda x: x.mode()[0] if not x.mode().empty else x)
    )

    # drop the still missing year and fuel values raw
    df = df.dropna(subset=['year', 'fuel'])

    # fill missing km_driven based on year
    df['km_driven'] = df['km_driven'].fillna(
        df.groupby('year')['km_driven'].transform('mean').round()
    )

    # fill the owner based on the year

    conditional_owners = df['year'].apply(
        lambda y: 'Second Owner' if y < 2011 else 'First Owner'
    )
    df['owner'] = df['owner'].fillna(conditional_owners)

    # fill the seller_type based on the price
    df['seller_type'] = df['seller_type'].fillna(df['selling_price'].mode()[0])




    return df



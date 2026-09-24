import pandas as pd

def transform_data(df):
    # int columns
    df = df.astype({
        'year': int,
        'km_driven': int,
        'selling_price': int
    })

    # drop unwanted columns
    df['brand_model'] = df['brand'] + ' ' + df['model']
    df = df.drop(columns=['name', 'brand', 'model','variant'])

    # label encoding

    df['owner'] = df['owner'].map({
        'Test Drive Car': 0,
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    })

    df['seller_type'] = df['seller_type'].map({
        'Individual': 0,
        'Dealer': 1,
        'Trustmark Dealer': 2
    })

    df['transmission'] = df['transmission'].map({
        'Manual': 0,
        'Automatic': 1,
    })

    df['fuel'] = df['fuel'].map({
        'Diesel': 0,
        'Petrol': 1,
        'CNG': 2,
        'LPG': 3,
        'Electric': 4
    })

    df['brand_model'] = df['brand_model'].astype('category').cat.codes
    

    df.to_csv('../../data/processed/cleaned_results.csv', index=False)

    return df

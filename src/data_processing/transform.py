def transform_data(df):
    # int columns
    df = df.astype({
        'year': int,
        'km_driven': int,
        'selling_price': int
    })

    df = df.drop(columns=['brand', 'model', 'variant'])

    df.to_csv('../../data/processed/results.csv', index=False)

    return df
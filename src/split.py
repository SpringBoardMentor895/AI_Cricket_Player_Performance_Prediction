def time_based_split(df, split_ratio=0.8):
    split_index = int(len(df) * split_ratio)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    return train, test

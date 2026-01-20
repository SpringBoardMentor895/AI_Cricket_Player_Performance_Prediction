def add_rolling_features(df):
    df = df.sort_values("match_id")

    df["avg_runs_last_5"] = (
        df.groupby("batter")["runs_scored"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["avg_sr_last_5"] = (
        df.groupby("batter")["strike_rate"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df

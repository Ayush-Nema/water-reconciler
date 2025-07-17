import pandas as pd


def transform_table(df_raw):
    # Columns that represent months (e.g. "2024_10", etc.)
    month_cols = [col for col in df_raw.columns if col.startswith("202")]

    # Melt the wide format into long format
    df_long = df_raw.melt(
        id_vars=["Flat_ID", "is_vacant", "is_faulty"],
        value_vars=month_cols,
        var_name="Month",
        value_name="Consumption"
    )

    # Convert Month from string like "2024_10" to datetime
    df_long["Month"] = pd.to_datetime(df_long["Month"], format="%Y_%m")

    # Sort for time series consistency
    df_long = df_long.sort_values(["Flat_ID", "Month"]).reset_index(drop=True)
    return df_long


if __name__ == '__main__':
    sample_df = pd.read_excel("data/sample_water_data.xlsx", sheet_name="Sheet2")
    op = transform_table(sample_df)
    print(op.head())

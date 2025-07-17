import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# ---- Parameters ----
LAG_MONTHS = 3


# ---- 1. Load & Preprocess ----
def create_lag_features(df, lag_months=3):
    df = df.sort_values(['Flat_ID', 'Month']).copy()
    for lag in range(1, lag_months + 1):
        df[f'Lag_{lag}'] = df.groupby('Flat_ID')['Consumption'].shift(lag)
    return df


def add_time_features(df):
    df['Month_Num'] = df['Month'].dt.month
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month_Num'] / 12)
    return df


def prepare_features(df):
    df = create_lag_features(df, LAG_MONTHS)
    df = add_time_features(df)
    df['Total_Society_Usage_Last_Month'] = df.groupby('Month')['Consumption'].transform('sum').shift(1)
    return df


# ---- 2. Train/Test Split ----
def split_data(df, test_months=6):
    max_month = df['Month'].max()
    cutoff = max_month - pd.DateOffset(months=test_months)
    train = df[df['Month'] <= cutoff]
    test = df[df['Month'] > cutoff]
    return train, test


# ---- 3. Train Model ----
def train_model(train_df, features, target='Consumption'):
    model = LGBMRegressor()
    model.fit(train_df[features], train_df[target])
    return model


# ---- 4. Predict with Vacancy Handling ----
def predict_with_vacancy(model, df, features):
    preds = model.predict(df[features])
    # Set prediction to 0 for vacant flats
    preds[df['Vacant'] == 1] = 0
    return preds


# ---- 5. Normalize to Match Total Usage ----
def normalize_predictions(preds, actual_total):
    scale = actual_total / preds.sum() if preds.sum() > 0 else 0
    return preds * scale


# ---- 6. Full Pipeline ----
def run_pipeline(df):
    df = prepare_features(df)
    df = df.dropna().copy()  # Drop rows with NaN lag features

    train_df, test_df = split_data(df)

    features = [f'Lag_{i}' for i in range(1, LAG_MONTHS + 1)] + ['Month_Sin', 'Month_Cos',
                                                                 'Total_Society_Usage_Last_Month']

    model = train_model(train_df, features)

    test_preds = predict_with_vacancy(model, test_df, features)

    # Normalize to match total society consumption
    actual_total = test_df['Consumption'].sum()
    normalized_preds = normalize_predictions(test_preds, actual_total)

    # Evaluation
    mae = mean_absolute_error(test_df['Consumption'], normalized_preds)
    print(f"MAE after normalization: {mae:.2f} liters")

    test_df['Predicted'] = normalized_preds
    return model, test_df[['Flat_ID', 'Month', 'Consumption', 'Vacant', 'Predicted']]

# ---- Example usage ----
# df = pd.read_csv("data/sample_water_data.xlsx", sheet_name='Sheet2', parse_dates=['Month'])
# model, results = run_pipeline(df)
# print(results.head())

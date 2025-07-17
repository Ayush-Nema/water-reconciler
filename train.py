import numpy as np
import pandas as pd
from rich.traceback import install
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from preprocessor import transform_table

install(show_locals=False)

# Define number of lag months
LAG_MONTHS = 3


# --- Helper Functions ---

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
    return df.dropna().copy()  # Drop rows with NaNs from lag features


def split_data(df, test_months=2):
    max_month = df['Month'].max()
    cutoff = max_month - pd.DateOffset(months=test_months)
    train = df[df['Month'] <= cutoff]
    test = df[df['Month'] > cutoff]
    return train, test


def train_model(train_df, features, target='Consumption'):
    model = XGBRegressor()
    model.fit(train_df[features], train_df[target])
    return model


def predict_with_vacancy(model, df, features):
    preds = model.predict(df[features])
    preds[df['is_vacant'] == 1] = 0  # set predictions to 0 if flat is vacant
    return preds


def normalize_predictions(preds, actual_total):
    scale = actual_total / preds.sum() if preds.sum() > 0 else 0
    return preds * scale


# --- Pipeline ---
def run_model_pipeline(df):
    df = prepare_features(df)

    # Train/Test split (last 2 months for test)
    train_df, test_df = split_data(df)

    # Features to use
    features = [f'Lag_{i}' for i in range(1, LAG_MONTHS + 1)] + [
        'Month_Sin', 'Month_Cos', 'Total_Society_Usage_Last_Month'
    ]

    model = train_model(train_df, features)
    test_preds = predict_with_vacancy(model, test_df, features)

    # Normalize predictions to match total water usage
    actual_total = test_df['Consumption'].sum()
    normalized_preds = normalize_predictions(test_preds, actual_total)

    test_df['Predicted'] = normalized_preds
    mae = mean_absolute_error(test_df['Consumption'], normalized_preds)

    return test_df[['Flat_ID', 'Month', 'Consumption', 'is_vacant', 'Predicted']], mae


# Run pipeline on reshaped data
DATA_PATH = 'data/sample_water_data.xlsx'
SHEET = 'Sheet2'
df_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET)
results_df, mae_score = run_model_pipeline(transform_table(df_raw))
print(results_df, mae_score)

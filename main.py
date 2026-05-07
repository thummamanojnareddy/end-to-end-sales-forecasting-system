import os
import joblib

from src.preprocessing import load_data, handle_missing_dates
from src.feature_engineering import create_features

from src.train_arima import train_arima
from src.train_prophet import train_prophet
from src.train_xgboost import train_xgboost
from src.train_lstm import train_lstm

from src.model_selector import select_best_model
from src.predict import forecast_next_8_weeks


# =========================
# Load and preprocess data
# =========================

df = load_data('data/sales_data.csv')

df = handle_missing_dates(df)

df = create_features(df)


# =========================
# Train / Validation Split
# =========================

train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]

valid = df.iloc[train_size:]


# =========================
# Create models folder
# =========================

os.makedirs('models', exist_ok=True)


# =========================
# Train or Load Models
# =========================

results = {}


# ---------- ARIMA ----------

if os.path.exists('models/arima_mae.pkl'):

    arima_mae = joblib.load(
        'models/arima_mae.pkl'
    )

    print("Loaded saved ARIMA model")

else:

    arima_mae = train_arima(
        train,
        valid
    )

    joblib.dump(
        arima_mae,
        'models/arima_mae.pkl'
    )

    print("ARIMA model trained and saved")

results['ARIMA'] = arima_mae


# ---------- Prophet ----------

if os.path.exists('models/prophet_mae.pkl'):

    prophet_mae = joblib.load(
        'models/prophet_mae.pkl'
    )

    print("Loaded saved Prophet model")

else:

    prophet_mae = train_prophet(
        train,
        valid
    )

    joblib.dump(
        prophet_mae,
        'models/prophet_mae.pkl'
    )

    print("Prophet model trained and saved")

results['Prophet'] = prophet_mae


# ---------- XGBoost ----------

if os.path.exists('models/xgb_mae.pkl'):

    xgb_mae = joblib.load(
        'models/xgb_mae.pkl'
    )

    print("Loaded saved XGBoost model")

else:

    xgb_mae = train_xgboost(
        train,
        valid
    )

    joblib.dump(
        xgb_mae,
        'models/xgb_mae.pkl'
    )

    print("XGBoost model trained and saved")

results['XGBoost'] = xgb_mae


# ---------- LSTM ----------

if os.path.exists('models/lstm_model.keras'):

    print("Loaded saved LSTM model")

else:

    train_lstm(train)

    print("LSTM model trained and saved")


# =========================
# Select Best Model
# =========================

best_model = select_best_model(results)

print("\nBest Model:", best_model)


# =========================
# Predictions
# =========================

sample_features = (
    valid
    .drop(columns=['sales'])
    .head(8)
)

predictions = forecast_next_8_weeks(
    sample_features
)

print("\nNext 8 Week Predictions:\n")

for i, pred in enumerate(
    predictions,
    start=1
):

    print(
        f"Week {i}: {pred:,.2f}"
    )
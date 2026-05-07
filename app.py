from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('models/xgboost.pkl')

@app.get('/')
def home():

    return {
        'message': 'Forecasting API Running'
    }

@app.post('/forecast')
def forecast(state: str):

    sample = pd.DataFrame({
        'lag_1': [100],
        'lag_7': [120],
        'lag_30': [150],
        'rolling_mean_7': [130],
        'rolling_std_7': [15],
        'month': [6],
        'weekofyear': [22],
        'quarter': [2],
        'year': [2026],
        'trend': [100],
        'holiday_flag': [0]
    })

    prediction = model.predict(sample)

    return {
        'state': state,
        'forecast': prediction.tolist()
    }

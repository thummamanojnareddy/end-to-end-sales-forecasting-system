from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import joblib

def train_prophet(train, valid):

    prophet_train = train[['date', 'sales']]
    prophet_train.columns = ['ds', 'y']

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(prophet_train)

    future = model.make_future_dataframe(
        periods=len(valid),
        freq='W'
    )

    forecast = model.predict(future)

    predictions = forecast['yhat'].tail(len(valid))

    mae = mean_absolute_error(valid['sales'], predictions)

    joblib.dump(model, 'models/prophet.pkl')

    return mae

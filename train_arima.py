from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import joblib

def train_arima(train, valid):

    model = SARIMAX(
        train['sales'],
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
    )

    fitted = model.fit(disp=False)

    predictions = fitted.predict(
        start=len(train),
        end=len(train)+len(valid)-1
    )

    mae = mean_absolute_error(valid['sales'], predictions)

    joblib.dump(fitted, 'models/arima.pkl')

    return mae

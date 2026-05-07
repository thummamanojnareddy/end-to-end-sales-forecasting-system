from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib

FEATURES = [
    'lag_1',
    'lag_7',
    'lag_30',
    'rolling_mean_7',
    'rolling_std_7',
    'month',
    'weekofyear',
    'quarter',
    'year',
    'trend',
    'holiday_flag'
]

def train_xgboost(train, valid):

    X_train = train[FEATURES]
    y_train = train['sales']

    X_valid = valid[FEATURES]
    y_valid = valid['sales']

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, predictions)

    joblib.dump(model, 'models/xgboost.pkl')

    return mae

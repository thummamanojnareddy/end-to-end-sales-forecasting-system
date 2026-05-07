import joblib


def forecast_next_8_weeks(features_df):

    # Load saved XGBoost model
    model = joblib.load('models/xgboost.pkl')

    # Remove unsupported columns
    drop_cols = ['date', 'state', 'category']

    for col in drop_cols:

        if col in features_df.columns:

            features_df = features_df.drop(columns=[col])

    # Predict sales
    predictions = model.predict(features_df)

    # Return next 8 weeks forecast
    return predictions[:8]
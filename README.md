# 📈 End-to-End Time Series Forecasting System

Production-ready time series forecasting platform for predicting the next 8 weeks of state-wise sales using ARIMA, Prophet, XGBoost, and LSTM with FastAPI backend and Streamlit dashboard.

---

## 🚀 Project Highlights

✅ Multiple Forecasting Models
✅ Automatic Best Model Selection
✅ Advanced Feature Engineering
✅ Time-Series Validation Without Leakage
✅ FastAPI REST API
✅ Interactive Streamlit Dashboard
✅ Model Persistence & Reusability
✅ Production-Style ML Pipeline

---

## 📊 Models Implemented

| Model          | Type                            |
| -------------- | ------------------------------- |
| ARIMA / SARIMA | Statistical Forecasting         |
| Prophet        | Trend & Seasonality Forecasting |
| XGBoost        | Machine Learning                |
| LSTM           | Deep Learning                   |

---

## 🧠 Feature Engineering

The system automatically creates:

* Lag Features (t-1, t-7, t-30)
* Rolling Mean & Rolling Standard Deviation
* Month Features
* Week Features
* Trend Features
* Seasonal Patterns

---

## 🏆 Best Model Selection

All models are evaluated using validation metrics.

The system automatically selects the best-performing model based on forecasting error.

Evaluation Metrics:

* MAE
* RMSE
* Forecast Accuracy

---

## 🖥️ Frontend Dashboard

The Streamlit dashboard provides:

* State-wise forecasting
* Interactive visualizations
* Historical sales analysis
* Forecast charts
* Prediction tables
* Best model information

---

## 🔌 FastAPI Backend

REST API endpoints are provided using FastAPI.

API Features:

* Forecast generation
* JSON response support
* Swagger UI documentation
* Production-ready backend architecture

---

## 🏗️ Project Architecture

```text
Dataset
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
ARIMA | Prophet | XGBoost | LSTM
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Predictions API
   ↓
Streamlit Dashboard
```

---

## 📂 Project Structure

## Project Overview

This project is a production-style end-to-end time series forecasting system developed for forecasting the next 8 weeks of sales for each state using historical sales data.

The system:

* Trains multiple forecasting models
* Performs feature engineering
* Handles missing values and missing dates
* Automatically selects the best-performing model
* Exposes predictions through API
* Provides an interactive frontend dashboard

---

# Problem Statement

Forecast the next 8 weeks of sales for each state using historical data.

The system should:

* Handle missing dates and missing values
* Capture seasonality and trends
* Compare multiple models
* Automatically select the best model
* Serve predictions through REST API

---

# Models Implemented

The following forecasting models were implemented and compared:

1. ARIMA / SARIMA
2. Facebook Prophet
3. XGBoost
4. LSTM (Deep Learning)

---

# Feature Engineering

The project includes advanced time-series feature engineering:

* Lag Features

  * t-1
  * t-7
  * t-30

* Rolling Statistics

  * Rolling Mean
  * Rolling Standard Deviation

* Date Features

  * Month
  * Week
  * Day of Week

* Time-Series Validation

  * Train-validation split without leakage

---

# Technologies Used

## Backend

* Python
* FastAPI
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Prophet
* TensorFlow / Keras
* Statsmodels

## Frontend

* Streamlit
* Matplotlib

---

# Project Structure

```text
forecasting-system
│
├── api
│   └── app.py
│
├── data
│   └── sales_data.csv
│
├── models
│   ├── xgboost.pkl
│   ├── lstm_model.keras
│   ├── arima_mae.pkl
│   └── prophet_mae.pkl
│
├── notebooks
│
├── src
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_arima.py
│   ├── train_prophet.py
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── predict.py
│   ├── evaluate.py
│   └── model_selector.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Data Preprocessing

The preprocessing pipeline performs:

* Handling mixed date formats
* Cleaning sales values
* Missing date handling
* Missing value interpolation
* Sorting time-series data

---

# Model Selection Strategy

The system evaluates all models using validation metrics.

The model with the lowest forecasting error is automatically selected as the best model.

Evaluation Metrics:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* Forecast comparison

---

# How to Run the Project

## Step 1: Create Virtual Environment

```powershell
python -m venv venv
```

---

## Step 2: Activate Environment

```powershell
venv\Scripts\activate
```

---

## Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## Step 4: Run Model Training

```powershell
python main.py
```

This will:

* Load dataset
* Perform preprocessing
* Create features
* Train models
* Select best model
* Generate predictions

---

# Running FastAPI Backend

## Start API Server

```powershell
uvicorn api.app:app --reload
```

---

## Open Swagger Documentation

Open browser:

```text
http://127.0.0.1:8000/docs
```

---

# Running Streamlit Frontend

## Start Dashboard

```powershell
streamlit run app.py
```

---

# Project Screenshots

Add your frontend screenshots inside a new folder:

```text
forecasting-system/screenshots
```

Example:

```text
screenshots
├── dashboard.png
├── forecast_chart.png
└── prediction_table.png
```

Then include them in README using:

```markdown
## Dashboard Preview

![Dashboard](screenshots/dashboard.png)

## Forecast Visualization

![Forecast Chart](screenshots/forecast_chart.png)

## Prediction Results

![Prediction Table](screenshots/prediction_table.png)
```

---

# Frontend Features

The dashboard includes:

* State selection
* Forecast generation
* Historical sales visualization
* Forecast charts
* Prediction tables
* Best model display

---

# Example Prediction Output

```text
Next 8 Week Predictions:

Week 1: 178,353,170
Week 2: 180,803,980
Week 3: 179,475,280
Week 4: 181,131,780
Week 5: 183,895,420
Week 6: 182,955,330
Week 7: 179,948,980
Week 8: 181,773,980
```

---

# Key Highlights

* End-to-end ML pipeline
* Production-style architecture
* Multiple forecasting algorithms
* Automated model selection
* REST API integration
* Frontend dashboard
* Reusable model persistence
* Time-series feature engineering
* Scalable forecasting workflow

---

# Future Improvements

Possible future enhancements:

* Cloud deployment
* CI/CD integration
* Real-time forecasting
* SHAP explainability
* Confidence intervals
* Docker containerization
* Automated retraining pipeline

---

# Conclusion

This project demonstrates a complete production-ready forecasting workflow combining statistical, machine learning, and deep learning approaches.

The system successfully forecasts future sales while supporting automated model selection, API serving, and interactive visualization through a modern frontend dashboard.

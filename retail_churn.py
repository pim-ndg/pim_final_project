import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
import joblib
from utilities.data_processing import load_cleaned_data


@st.cache_data
def load_data():
    cdf = load_cleaned_data()
    cdf = create_features(cdf)
    return cdf

# Feature engineering
@st.cache_data
def create_features(df):
    # Revenue
    df['TotalSpending'] = df['Quantity'] * df['UnitPrice']
    
    # Recency
    df['DaysSinceLastPurchase'] = (df['InvoiceDate'].max() - df['InvoiceDate']).dt.days
    # Recency: Calculate the number of days since a customer's last purchase. This can help predict if a customer is likely to make another purchase soon
    df['LastPurchaseDate'] = df.groupby('CustomerID')['InvoiceDate'].transform('max')
    
    df['Recency'] = (df['LastPurchaseDate'] - df['InvoiceDate']).dt.days
    
    # Frequency
    df['FrequencyScore'] = df.groupby('CustomerID')['InvoiceNo'].transform('count').astype(int)
    # Frequency: Calculate the number of purchases made by each customer. This can help predict how often a customer is likely to make purchases.
    df['Frequency'] = df.groupby('CustomerID')['InvoiceNo'].transform('count')
    
    # Monetary
    df['MonetaryScore'] = df.groupby('CustomerID')['TotalSpending'].transform('sum').astype(int)
    # Monetary Value: Calculate the total monetary value of each customer's purchases. This can help predict how much a customer is likely to spend.
    df['MonetaryValue'] = df['TotalSpending']
    df['MonetaryValue'] = df.groupby('CustomerID')['MonetaryValue'].cumsum()

    # RFM
    df['RFMScore'] = df.groupby('CustomerID')['InvoiceDate'].rank(method='dense', ascending=False).astype(int)
    
    # Average Order Value: Calculate the average order value for each customer. This can help predict how much a customer is likely to spend per order.
    df['AvgOrderValue'] = df['TotalSpending'] / df['Frequency']

    # Repeat Purchase Ratio: Calculate the ratio of repeat purchases to total purchases for each customer. This can help predict how likely a customer is to make repeat purchases.
    df['RepeatPurchaseRatio'] = df.groupby('CustomerID')['InvoiceNo'].transform('count') / df['Frequency']

    # Churn: Create a binary target variable indicating whether a customer has churned (i.e., not made a purchase in a certain time period). This can help predict which customers are likely to churn.
    # df['Churn'] = (df.groupby('CustomerID')['InvoiceDate'].transform('max') + pd.Timedelta(days=60) < df['InvoiceDate']).astype(int)
    # df['Churn'] = (df['InvoiceDate'] - df['LastPurchaseDate'] > pd.Timedelta(days=60)).astype(int)
    # Dued to the data quite outdated, so we go with the number of bought for calcualte churn instead
    # df['Churn'] = (df.groupby('CustomerID')['InvoiceNo'].transform('count') == 1).astype(int)
    df['Churn'] = (df['Recency'] > 60) | (df['Frequency'] == 1) | (df['MonetaryValue'] < 100) | (df['AvgOrderValue'] < 50) | (df['RepeatPurchaseRatio'] < 0.2)
    df['Churn'] = df['Churn'].astype(int)

    return df

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type='RandomForestRegressor', **kwargs):
    if model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(**kwargs)
    elif model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if isinstance(model, RandomForestRegressor):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mae, mse, r2
    elif isinstance(model, RandomForestClassifier):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return accuracy, precision, recall, f1

def predict(model, X):
    y_pred = model.predict(X)
    return y_pred

def predict_churn(recency, frequency, monetary_value, avg_order_value, repeat_purchase_ratio):
    model = joblib.load("models/churn_model.joblib")
    new_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency], 
        'MonetaryValue': [monetary_value],
        'AvgOrderValue': [avg_order_value],
        'RepeatPurchaseRatio': [repeat_purchase_ratio]
    })
    prediction = model.predict(new_data)[0]
    return prediction

def run():
    st.title("Online Retail Churn Prediction", anchor=False)
    
    cdf = load_data()
    
    st.subheader("Select Customer", anchor=False)
    customer_id = st.selectbox("Choose a customer", cdf['CustomerID'].unique())
    
    st.subheader("Input RFM, Average Order Value, and Repeat Purchase Ratio", anchor=False)
    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, step=1)
    monetary_value = st.number_input("Monetary Value (total spending)", min_value=0.0, step=0.01)
    avg_order_value = st.number_input("Average Order Value", min_value=0.0, step=0.01)
    repeat_purchase_ratio = st.number_input("Repeat Purchase Ratio", min_value=0.0, max_value=1.0, step=0.01)
    
    # Prepare the data
    X = cdf[['Recency', 'Frequency', 'MonetaryValue', 'AvgOrderValue', 'RepeatPurchaseRatio']]
    y = cdf['Churn']

    X_train, X_test, y_train, y_test = split_data(X, y)

    if 'churn_model' not in st.session_state:
        if st.button("Train Churn Model"):
            model = train_model(X_train, y_train, model_type='RandomForestRegressor', n_estimators=100, random_state=42)
            st.session_state.churn_model = model
            joblib.dump(model, "models/churn_model.joblib")
            st.experimental_rerun()
    else:
        model = st.session_state.churn_model
        st.write("Churn model has already been trained.")
    
    # if st.button("Evaluate Churn Model"):
    #     model = joblib.load("models/churn_model.joblib")
    #     mae, mse, r2 = evaluate_model(model, X_test, y_test)
    #     churn_metrics = f"""
    #     |Metrics|Result|
    #     |-------|------|
    #     |MAE|{mae:,.2f}|
    #     |MSE|{mse:,.2f}|
    #     |R-squared|{r2:,.2f}|
    #     """
    #     st.markdown(churn_metrics)
    
    if st.button("Predict Churn"):
        if 'churn_model' in st.session_state:
            prediction = predict_churn(recency, frequency, monetary_value, avg_order_value, repeat_purchase_ratio)
            st.write(f'Predicted churn probability for customer {customer_id}: {prediction:.2f}')
        else:
            st.error("Please train the churn model first before making predictions.")
        

# region [Mock Test Data]
# data = {
#     'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'Recency': [30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
#     'Frequency': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
#     'MonetaryValue': [100, 200, 300, 400, 500, 100, 200, 300, 400, 500],
#     'AvgOrderValue': [50, 100, 150, 200, 250, 50, 100, 150, 200, 250],
#     'RepeatPurchaseRatio': [0.2, 0.4, 0.6, 0.8, 1.0, 0.2, 0.4, 0.6, 0.8, 1.0],
#     'Churn': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
# }
# endregion
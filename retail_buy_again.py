import streamlit as st
import pandas as pd
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
import joblib
from utilities.data_processing import load_cleaned_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
    df['Churn'] = (df['Recency'] > 60) & (df['Frequency'] == 1) & (df['MonetaryValue'] < 100) & (df['AvgOrderValue'] < 50) & (df['RepeatPurchaseRatio'] < 0.2)
    df['Churn'] = df['Churn'].astype(int)

    return df

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, **kwargs):
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    rpt_clf = classification_report(y_test, y_pred)
    rpt_cf = confusion_matrix(y_test, y_pred)
    return accuracy, rpt_clf, rpt_cf

def predict_buy_again(model, recency, frequency, monetary_value, avg_order_value, repeat_purchase_ratio):
    new_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency], 
        'MonetaryValue': [monetary_value],
        'AvgOrderValue': [avg_order_value],
        'RepeatPurchaseRatio': [repeat_purchase_ratio]
    })
    prediction = model.predict_proba(new_data)[0][1]
    return prediction

def main():
    st.title("Online Retail Buy Again Prediction")
    
    cdf = load_data()
    
    
    cdf['BuyAgain'] = 0
    cdf.loc[cdf.groupby('CustomerID')['InvoiceNo'].transform('count') > 1, 'BuyAgain'] = 1

    
    # Create a new target variable 'BuyAgain'
    cdf['BuyAgain'] = ((cdf['RepeatPurchaseRatio'] > 0.7) & (cdf['Frequency'] > 2)).astype(int)
    
    st.subheader("Select Customer")
    customer_id = st.selectbox("Choose a customer", cdf['CustomerID'].unique())
    
    # Filter the data for the selected customer
    customer_data = cdf[cdf['CustomerID'] == customer_id][['Recency', 'Frequency', 'MonetaryValue', 'AvgOrderValue', 'RepeatPurchaseRatio']]
    
    st.subheader("Input RFM, Average Order Value, and Repeat Purchase Ratio")
    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1, value=customer_data['Recency'].values[0])
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, step=1, value=customer_data['Frequency'].values[0])
    monetary_value = st.number_input("Monetary Value (total spending)", min_value=0.0, step=0.01, value=customer_data['MonetaryValue'].values[0])
    avg_order_value = st.number_input("Average Order Value", min_value=0.0, step=0.01, value=customer_data['AvgOrderValue'].values[0])
    repeat_purchase_ratio = st.number_input("Repeat Purchase Ratio", min_value=0.0, max_value=1.0, step=0.01, value=customer_data['RepeatPurchaseRatio'].values[0])
    
    # Prepare the data for buy again prediction
    X = cdf[['Recency', 'Frequency', 'MonetaryValue', 'AvgOrderValue', 'RepeatPurchaseRatio']]
    y = cdf['BuyAgain']
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    if 'buy_again_model' not in st.session_state:     
        if st.button("Train Buy Again Model"):
            model = train_model(X_train, y_train)
            st.session_state.buy_again_model = model
            joblib.dump(model, "models/buy_again_model.joblib")
            st.experimental_rerun()
    else:
        model = st.session_state.buy_again_model
        st.write("Buy again model has already been trained.")
    
    # if st.button("Evaluate Buy Again Model"):
    #     model = joblib.load("models/buy_again_model.joblib")
    #     accuracy, rpt_clf, rpt_cf = evaluate_model(model, X_test, y_test)
    #     buy_again_metrics = f"""
    #     |Metrics|Result|
    #     |-------|------|
    #     |Accuracy|{accuracy:.2f}|
    #     """
    #     st.markdown(buy_again_metrics)
    
    if st.button("Predict Buy Again"):
        if 'buy_again_model' in st.session_state:
            prediction = predict_buy_again(model, recency, frequency, monetary_value, avg_order_value, repeat_purchase_ratio)
            st.write(f'Predicted Buy Again Probability for customer {customer_id}: {prediction:.2f}')
        else:
            st.error("Please train the Buy again model first before making predictions.")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from utilities.data_processing import load_cleaned_data
import joblib

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

def train_customer_segmentation(X):
    # Spending Segment
    kmeans_spending = KMeans(n_clusters=4, random_state=42)
    spending_labels = kmeans_spending.fit_predict(X[['MonetaryValue', 'AvgOrderValue', 'RepeatPurchaseRatio']])
    
    # Loyalty Segment
    kmeans_loyalty = KMeans(n_clusters=4, random_state=42)
    loyalty_labels = kmeans_loyalty.fit_predict(X[['Recency', 'Frequency', 'RepeatPurchaseRatio']])
    
    return spending_labels, loyalty_labels, kmeans_spending, kmeans_loyalty

def evaluate_customer_segmentation(cdf, spending_labels, loyalty_labels):
    # Assign segment names
    cdf['SpendingSegment'] = spending_labels
    cdf['LoyaltySegment'] = loyalty_labels
    
    cdf['SpendingSegmentName'] = cdf['SpendingSegment'].map({0: 'Top Spender', 1: 'Mid Spender', 2: 'Low Spender', 3: 'Never Buy'})
    cdf['LoyaltySegmentName'] = cdf['LoyaltySegment'].map({0: 'Loyal', 1: 'Occasional', 2: 'Discount Seeker', 3: 'Churned'})
    
    return cdf

def predict_customer_segments(cdf, customer_id, kmeans_spending, kmeans_loyalty):
    # Filter the data for the selected customer
    customer_data = cdf[cdf['CustomerID'] == customer_id][['MonetaryValue', 'AvgOrderValue', 'RepeatPurchaseRatio', 'Recency', 'Frequency']]
    
    # Ensure column order matches the trained models
    spending_cols = ['MonetaryValue', 'AvgOrderValue', 'RepeatPurchaseRatio']
    loyalty_cols = ['Recency', 'Frequency', 'RepeatPurchaseRatio']
    customer_data_spending = customer_data[spending_cols]
    customer_data_loyalty = customer_data[loyalty_cols]
    
    # Predict spending and loyalty segments for the selected customer
    spending_segment = kmeans_spending.predict(customer_data_spending)[0]
    loyalty_segment = kmeans_loyalty.predict(customer_data_loyalty)[0]
    
    # Map segment names
    spending_segment_name = {0: 'Top Spender', 1: 'Mid Spender', 2: 'Low Spender', 3: 'Never Buy'}.get(spending_segment)
    loyalty_segment_name = {0: 'Loyal', 1: 'Occasional', 2: 'Discount Seeker', 3: 'Churned'}.get(loyalty_segment)
    
    return spending_segment_name, loyalty_segment_name

def main():
    st.title("Online Retail Customer Segmentation")
    
    cdf = load_data()
    
    # Prepare the data for segmentation
    X = cdf[['MonetaryValue', 'AvgOrderValue', 'RepeatPurchaseRatio', 'Recency', 'Frequency']]
    
    if 'kmeans_spending' not in st.session_state or 'kmeans_loyalty' not in st.session_state:
        if st.button("Train Customer Segmentation"):
            spending_labels, loyalty_labels, kmeans_spending, kmeans_loyalty = train_customer_segmentation(X)
            
            # Save the trained models
            joblib.dump(kmeans_spending, "models/kmeans_spending.joblib")
            joblib.dump(kmeans_loyalty, "models/kmeans_loyalty.joblib")
            
            # Store the trained models in session state
            st.session_state.kmeans_spending = kmeans_spending
            st.session_state.kmeans_loyalty = kmeans_loyalty
            
            cdf = evaluate_customer_segmentation(cdf, spending_labels, loyalty_labels)
            st.write("Customer segmentation training and evaluation completed.")
            st.experimental_rerun()
    else:
        st.write("Customer segmentation models have already been trained.")
    
    st.subheader("Select Customer")
    customer_id = st.selectbox("Choose a customer", cdf['CustomerID'].unique())
    
    if st.button("Predict Customer Segments"):
        if 'kmeans_spending' in st.session_state and 'kmeans_loyalty' in st.session_state:
            # Load the trained models from session state
            kmeans_spending = st.session_state.kmeans_spending
            kmeans_loyalty = st.session_state.kmeans_loyalty
            
            spending_segment_name, loyalty_segment_name = predict_customer_segments(cdf, customer_id, kmeans_spending, kmeans_loyalty)
            st.write(f"Spending Segment: {spending_segment_name}")
            st.write(f"Loyalty Segment: {loyalty_segment_name}")
        else:
            st.error("Please train the customer segmentation models first before making predictions.")
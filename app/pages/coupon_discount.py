import streamlit as st
import pandas as pd
import joblib

def show():
    # Load the model and scaler
    model = joblib.load('models/discount_prediction_model.joblib')
    scaler = joblib.load('models/discount_prediction_scaler.joblib')

    # Load customer features
    customer_features = pd.read_csv('data/processed/customer_features.csv', index_col=0)

    st.title('Coupon and Discount Value Predictions')

    customer_id = st.selectbox('Select a customer ID:', customer_features.index)

    if st.button('Get Discount Prediction'):
        customer_data = customer_features.loc[customer_id, ['PurchaseFrequency', 'TotalSpent', 'CustomerLifetime', 'AverageOrderValue']].values.reshape(1, -1)
        scaled_data = scaler.transform(customer_data)
        predicted_discount = model.predict(scaled_data)[0]
        
        st.write(f"Customer ID: {customer_id}")
        st.write(f"Predicted Discount: ${predicted_discount:.2f}")
        
        st.write("\nCustomer Features:")
        st.write(customer_features.loc[customer_id])

    st.write("\nNote: This model predicts a discount value based on the customer's purchase history and behavior.")

if __name__ == "__main__":
    show()
import streamlit as st
import pandas as pd
import joblib

# Load the preprocessed customer data and segmentation model
customer_data = pd.read_csv('data/processed/customer_data.csv')
segmentation_model = joblib.load('models/segmentation_model.joblib')

def show():
    st.title("Customer Segmentation")
    
    st.write(customer_data[customer_data["segment"]>0].sample(5))

    # Get the unique customer IDs
    unique_customer_ids = customer_data['CustomerID'].unique()

    # Allow the user to select a customer from the list
    customer_id = st.selectbox("Select a Customer", unique_customer_ids)

    if customer_id:
        # Check if any rows match the customer_id
        matching_rows = customer_data[customer_data['CustomerID'] == customer_id]
        
        if not matching_rows.empty:
            # Retrieve the customer's segment
            customer_data_row = customer_data[customer_data['CustomerID'] == customer_id].iloc[0]
            customer_segment = customer_data_row['segment']

            # Display the customer's segment and buy-again probability
            st.write(f"Customer {customer_id} belongs to Segment {customer_segment}")

            segment_probabilities = joblib.load('models/segment_probabilities.joblib')
            buy_again_probability = segment_probabilities[customer_segment]

            st.write(f"Buy-Again Probability: {buy_again_probability*100}%")
        else:
            st.write(f"No data found for Customer {customer_id}")

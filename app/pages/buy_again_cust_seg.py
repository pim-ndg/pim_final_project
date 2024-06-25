import streamlit as st
import joblib
import pandas as pd

# Load the segment-probability mapping
customer_data = pd.read_csv('data/processed/customer_data.csv')
segment_probabilities = joblib.load('models/segment_probabilities.joblib')

def get_buy_again_recommendation(customer_segment):
    # Get the buy-again probability for the customer's segment
    buy_again_probability = segment_probabilities[customer_segment]
    
    # Implement the buy-again recommendation logic
    if buy_again_probability >= 0.7:
        recommendation = "High probability to buy again. Recommend targeted promotions."
    elif buy_again_probability >= 0.5:
        recommendation = "Medium probability to buy again. Recommend general promotions."
    else:
        recommendation = "Low probability to buy again. Recommend customer retention strategies."
    
    return recommendation

def show():
    st.title("Buy-Again Recommendation")
    
    st.write(customer_data[customer_data["segment"]>0].sample(5))

    # Get the unique customer IDs
    unique_customer_ids = customer_data['CustomerID'].unique()

    # Allow the user to select a customer from the list
    customer_id = st.selectbox("Select a Customer", unique_customer_ids)

    if customer_id:
        # Retrieve the customer's segment
        customer_segment = customer_data[customer_data['CustomerID'] == customer_id]['segment'].values[0]

        # Get the buy-again recommendation for the customer
        recommendation = get_buy_again_recommendation(customer_segment)
        st.write(f"Recommendation for Customer {customer_id}: {recommendation}")
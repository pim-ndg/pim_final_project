import streamlit as st
import pandas as pd
from models.buy_again_model import BuyAgainModel
from utils.data_loader import load_data

@st.cache_resource
def load_model():
    model = BuyAgainModel()
    model.load()
    return model

def show():
    st.title("Buy Again Recommendations")

    # Load data
    data = load_data()

    # Display data info
    st.write("Data Info:")
    st.write(data.info())
    st.write("Data Sample:")
    st.write(data.head())

    # Load model (this will only happen once)
    model = load_model()
    st.success("Model, scaler, and product columns loaded successfully.")

    # Make predictions
    try:
        customer_ids = data['CustomerID'].unique()
        customer_id = st.selectbox("Select a customer", customer_ids)
        
        if st.button("Get Recommendations"):
            recommendations = model.get_recommendations(customer_id, data)
            
            if not recommendations.empty:
                st.write("Top 5 Product Recommendations:")
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    st.write(f"{i}. Product ID: {row['Product ID']}, Probability: {row['Normalized Score']:.2f}")
            else:
                st.write("No recommendations available for this customer.")
                if st.button("Train model for this customer"):
                    # Filter data for the selected customer
                    customer_data = data[data['CustomerID'] == customer_id]
                    if not customer_data.empty:
                        with st.spinner("Training model for the selected customer..."):
                            model.train(customer_data)
                        st.success("Model trained for the selected customer.")
                        
                        # Get recommendations after training
                        recommendations = model.get_recommendations(customer_id, data)
                        if not recommendations.empty:
                            st.write("Top 5 Product Recommendations after training:")
                            for i, (product, score) in enumerate(recommendations.items(), 1):
                                st.write(f"{i}. Product ID: {product}, Score: {score:.2f}")
                        else:
                            st.write("Still no recommendations available after training.")
                    else:
                        st.error("No data available for the selected customer.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check the data format and ensure it matches the expected structure.")

if __name__ == "__main__":
    show()
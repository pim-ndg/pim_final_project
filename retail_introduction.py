import streamlit as st

intro = """
## Welcome to our Online Retail ML webapp
We are developing an ML web app with the following features:
1. Training: Users can train Machine Learning models using the Online Retail data by selecting features such as Recency, Frequency, Monetary Value, and Repeat Purchase Ratio.
2. Evaluation: Users can evaluate the performance of the trained models using test data, looking at metrics like Accuracy, MAE, MSE, or Confusion Matrix.
3. Prediction: Users can input a CustomerID and view the model's predictions, such as Next Purchase Date, Buy Again Probability, or Product Recommendations.

We are using Streamlit as the web app framework to allow users to easily interact with the input parameters.

We are using the UCI Online Retail II dataset which contains online retail transaction data from a UK-based retail company, with the following characteristics:
The data covers the period from 1st December 2009 to 9th December 2011.
It includes information about the products (StockCode, Description), the quantity sold (Quantity), the unit price (UnitPrice), and the transaction date (InvoiceDate).
It also includes customer information (CustomerID), which can be used to analyze customer buying behavior.
The data can be used for tasks like Customer Segmentation, Next Purchase Prediction, and Product Recommendation.

With this data, we can develop an ML web app that can help businesses analyze and predict customer behavior.

### UCI Online Retail Variables

| Column Name | Ideal Data Type | Description |
|-------------|-------------|-----------|
|InvoiceNo (Invoice number)| Nominal. |A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation. |
|StockCode (Product (item) code)| Nominal. |A 5-digit integral number uniquely assigned to each distinct product. |
|Description (Product (item) name)| Nominal. ||
|Quantity | Numeric.	|The quantities of each product (item) per transaction.|
|InvoiceDate (Invice date and time)| Numeric. |The day and time when a transaction was generated. |
|UnitPrice (Unit price)| Numeric. |Product price per unit in sterling (£). |
|CustomerID (Customer number) | Nominal. |A 5-digit integral number uniquely assigned to each customer. |
|Country (Country name)| Nominal.| The name of the country where a customer resides.|

Reference: [UCI - Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
"""

def run():
    st.markdown(intro)
    st.write("### Select the sidebar menu to start working on our app.")

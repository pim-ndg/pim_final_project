import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

file_path = os.path.join('data', 'online_retail_II.csv')

@st.cache_data(ttl=3600, show_spinner="Load data...")
def load_data(file_path=file_path):
    ext = file_path.split(".")[1]
    if ext == "csv":
        read_func = pd.read_csv
    elif ext == "xlsx":
        read_func = pd.read_excel
    else:
        return None
    
    df = read_func(
        file_path, skiprows=1,
        names=['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 
               'UnitPrice', 'CustomerID', 'Country'],
            parse_dates=['InvoiceDate']
    )
        
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df = df.dropna()
    df = df.drop_duplicates()
    # df = df.drop_duplicates(subset=['InvoiceNo', 'StockCode', 'CustomerID'])
    
    return df

@st.cache_data
def load_cleaned_data():
    df = pd.read_csv("data/online_retail_no_outliers.csv", parse_dates=["InvoiceDate"])
    df = df.drop("Unnamed: 0", axis=1)
    return df

def prep_features_for_churn_model(df):
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Create a new DataFrame with the latest invoice date for each customer
    # latest_invoice = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    # Create a new DataFrame with the latest invoice date for each customer
    latest_invoice = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    latest_invoice = latest_invoice.rename(columns={'InvoiceDate': 'latest_invoice_date'})


    # Merge the latest invoice date with the original DataFrame
    # df = pd.merge(df, latest_invoice, on=['CustomerID', 'InvoiceDate'], how='left')
    df = pd.merge(df, latest_invoice, on='CustomerID', how='left')

    # Calculate recency as the number of days since the latest invoice date
    # df['recency'] = (df['InvoiceDate_x'] - df['InvoiceDate_y']).dt.days
    # Calculate recency as the number of days since the latest invoice date
    df['recency'] = (df['InvoiceDate'].max() - df['latest_invoice_date']).dt.days

    # Calculate frequency as the number of purchases for each customer
    df['frequency'] = df.groupby('CustomerID')['InvoiceNo'].transform('count')

    # Calculate monetary as the total amount spent by each customer
    df['monetary'] = df['Quantity'] * df['UnitPrice']
    df['monetary'] = df.groupby('CustomerID')['monetary'].transform('sum')

    # Normalize the RFM scores
    df['r_score'] = pd.qcut(df['recency'], 5, labels=range(5, 0, -1))
    df['f_score'] = pd.qcut(df['frequency'], 5, labels=range(1, 6))
    df['m_score'] = pd.qcut(df['monetary'], 5, labels=range(1, 6))
    
    # Calculate the overall RFM score
    # Calculate the overall RFM score
    df['RFM_Score'] = (df['r_score'].astype(int) + df['f_score'].astype(int) + df['m_score'].astype(int)) / 3

    st.write(df['RFM_Score'].unique())
    
    # Define the weights for each factor
    recency_weight = 0.4
    frequency_weight = 0.3
    monetary_weight = 0.3
    
    # Calculate the normalized recency, frequency, and monetary values
    df['recency_norm'] = (df['recency'] - df['recency'].min()) / (df['recency'].max() - df['recency'].min())
    df['frequency_norm'] = (df['frequency'] - df['frequency'].min()) / (df['frequency'].max() - df['frequency'].min())
    df['monetary_norm'] = (df['monetary'] - df['monetary'].min()) / (df['monetary'].max() - df['monetary'].min())

    
    # Calculate the churn score
    # df['churn_score'] = 0
    # df.loc[(df['recency'] > 90) & (df['frequency'] < 3) & (df['monetary'] < 5000), 'churn_score'] = 1
    # Calculate the churn score
    df['churn_score'] = (recency_weight * df['recency_norm']) + (frequency_weight * df['frequency_norm']) + (monetary_weight * df['monetary_norm'])

    # Sort the data by churn score
    df = df.sort_values('churn_score', ascending=False)
    st.write(df['churn_score'].min(),df['churn_score'].max())
    
    # Define churn based on the RFM score
    churn_threshold = 0.4
    # Find the churn threshold that results in a 50:50 split
    # total_customers = len(df)
    # churned_customers = int(total_customers / 3)
    # churn_threshold = df.iloc[churned_customers]['churn_score']
    # Create the churn column (assuming a threshold of 365 days for churn)
    # df['churn'] = (df['recency'] > 365).astype(int)
    # df['churn'] = np.where(df['RFM_Score'] < churn_threshold, 1, 0)
    # Determine the churn status
    df['churn'] = (df['churn_score'] >= churn_threshold).astype(int)
    st.write(df['churn'].unique())
    
    # Display the results
    st.write(df[['recency', 'frequency', 'monetary', 'churn_score', 'churn']])
    
    # Create the Streamlit app
    st.title('RFM Score vs. Churn Probability')

    import matplotlib.pyplot as plt
    # Plot the RFM scores against churn probability
    fig, ax = plt.subplots()
    ax.scatter(df['RFM_Score'], df['churn'])
    ax.set_xlabel('RFM Score')
    ax.set_ylabel('Churn Probability')
    ax.set_title('RFM Score vs. Churn Probability')

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    return df

def load_performance_metrics():
    # Load the model and test data
    model = joblib.load('models/customer_product_preference_model.joblib')
    X_test = pd.read_csv('data/x_test_customer_product_preference_model.csv')
    y_test = pd.read_csv('data/y_test_customer_product_preference_model.csv')['PreferredCategory']

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    }

def load_example_data(features):
    # Load the actual dataset
    df = pd.read_csv('data/customer_purchase_data.csv')  # Replace with your actual data file

    # Select only the required features
    df = df[features]

    # Return the first 100 rows (or fewer if the dataset is smaller)
    return df.head(100)

def get_feature_explanations():
    return """
    - **UniqueProductCount**: Number of unique products purchased by the customer
    - **TotalQuantity**: Total number of items purchased by the customer
    - **TotalAmountSpent**: Total amount spent by the customer
    - **Card_Quantity**: Number of card items purchased
    - **Christmas_Quantity**: Number of Christmas items purchased
    - **Garden_Quantity**: Number of garden items purchased
    - **Gift_Quantity**: Number of gift items purchased
    - **Kitchen_Quantity**: Number of kitchen items purchased
    - **Other_Quantity**: Number of other items purchased
    - **Toy_Quantity**: Number of toy items purchased
    - **Card_TotalAmount**: Total amount spent on card items
    - **Christmas_TotalAmount**: Total amount spent on Christmas items
    - **Garden_TotalAmount**: Total amount spent on garden items
    - **Gift_TotalAmount**: Total amount spent on gift items
    - **Kitchen_TotalAmount**: Total amount spent on kitchen items
    - **Other_TotalAmount**: Total amount spent on other items
    - **Toy_TotalAmount**: Total amount spent on toy items
    """

def prepare_customer_data(df, customer_id):
    # Filter data for the specific customer
    customer_data = df[df['CustomerID'] == customer_id]

    # Prepare features for the customer
    customer_features = customer_data.groupby('CustomerID').agg({
        'StockCode': lambda x: len(set(x)),  # Number of unique products purchased
        'Quantity': 'sum',
        'TotalAmount': 'sum'
    })
    customer_features.columns = ['UniqueProductCount', 'TotalQuantity', 'TotalAmountSpent']

    # Calculate quantity and amount spent per category for the customer
    category_pivot = customer_data.pivot_table(
        values=['Quantity', 'TotalAmount'], 
        index='CustomerID', 
        columns='ProductCategory', 
        aggfunc='sum', 
        fill_value=0
    )
    
    category_pivot.columns = [f'{col[1]}_{col[0]}' for col in category_pivot.columns]

    # Merge features
    customer_features = customer_features.join(category_pivot)

    return customer_features

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
def preprocess_data_for_anomaly(df):
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Convert 'InvoiceDate' to datetime and extract numeric features
    df_processed['InvoiceDate'] = pd.to_datetime(df_processed['InvoiceDate'])
    df_processed['InvoiceYear'] = df_processed['InvoiceDate'].dt.year
    df_processed['InvoiceMonth'] = df_processed['InvoiceDate'].dt.month
    df_processed['InvoiceDay'] = df_processed['InvoiceDate'].dt.day
    
    # Calculate additional features
    df_processed['AverageOrderValue'] = df_processed['Quantity'] * df_processed['UnitPrice']
    df_processed['AverageOrderValue'] = df_processed.groupby(['CustomerID'])['AverageOrderValue'].transform('mean')
    df_processed['TotalQuantity'] = df_processed.groupby(['CustomerID'])['Quantity'].transform('sum')
    df_processed['AverageUnitPrice'] = df_processed.groupby(['CustomerID'])['UnitPrice'].transform('mean')
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['InvoiceNo', 'StockCode', 'CustomerID', 'Country']
    for col in categorical_cols:
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Select features for anomaly detection
    features = ['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'CustomerID', 'InvoiceYear', 'InvoiceMonth', 'InvoiceDay', 'AverageOrderValue', 'TotalQuantity', 'AverageUnitPrice']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df_processed[features] = imputer.fit_transform(df_processed[features])
    
    # Normalize the features
    scaler = StandardScaler()
    df_processed[features] = scaler.fit_transform(df_processed[features])
    
    df_processed["anomaly"] = None
    return df_processed, features, scaler
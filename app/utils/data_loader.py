import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    xlsx_path = "data/raw_data/Online Retail.xlsx" #v1
    data_path = "data/raw_data/online_retail_II.csv" #v2
    
    # df = pd.read_excel(xlsx_path)
    df = pd.read_csv(data_path, skiprows=1, 
                     names=['InvoiceNo', 'StockCode', 'Description', 'Quantity', 
                            'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'],
                     parse_dates=["InvoiceDate"],
                     dtype={"InvoiceNo":"str", "StockCode":"str"},
                     )
    
    df.dropna(inplace=True)

    df['StockCode'] = df['StockCode'].astype(str)
    
    return df

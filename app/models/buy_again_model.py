import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib

class BuyAgainModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.product_columns = None
        self.model_path = 'models/buy_again_model.joblib'
        self.scaler_path = 'models/buy_again_scaler.joblib'
        self.columns_path = 'models/product_columns.joblib'

    def load(self):
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.product_columns = joblib.load(self.columns_path)

    def train(self, data):
        customer_product_matrix = self._prepare_data(data)
        
        if self.product_columns is None:
            self.product_columns = customer_product_matrix.columns
        else:
            # Ensure all columns from the original model are present
            for col in self.product_columns:
                if col not in customer_product_matrix.columns:
                    customer_product_matrix[col] = 0
            customer_product_matrix = customer_product_matrix[self.product_columns]
        
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        X = self.scaler.fit_transform(customer_product_matrix.values)
        
        if self.model is None:
            self.model = NearestNeighbors(n_neighbors=5, metric='cosine')
        
        self.model.fit(X)

    def get_recommendations(self, customer_id, data, n_recommendations=5):
        customer_product_matrix = self._prepare_data(data)
        
        # Ensure the customer_product_matrix has the same columns as the training data
        for col in self.product_columns:
            if col not in customer_product_matrix.columns:
                customer_product_matrix[col] = 0
        customer_product_matrix = customer_product_matrix[self.product_columns]
        
        if customer_id not in customer_product_matrix.index:
            return pd.Series()
        
        customer_vector = customer_product_matrix.loc[customer_id].values.reshape(1, -1)
        customer_vector_scaled = self.scaler.transform(customer_vector)
        
        distances, indices = self.model.kneighbors(customer_vector_scaled)
        similar_customers = customer_product_matrix.index[indices[0][1:]]
        
        recommended_products = customer_product_matrix.loc[similar_customers].sum()
        recommended_products = recommended_products[recommended_products > 0].sort_values(ascending=False)
        
        # Get the minimum and maximum scores
        min_score = recommended_products.min()
        max_score = recommended_products.max()
        
        # Convert the scores to probabilities between 0 and 1
        recommended_products_normalized = (recommended_products - min_score) / (max_score - min_score)
        
        # Create a new Series with both raw and normalized scores
        recommended_products_df = pd.DataFrame({
            'Product ID': recommended_products.index,
            'Raw Score': recommended_products.values,
            'Normalized Score': recommended_products_normalized.values
        })
        
        return recommended_products_df.head(n_recommendations)

    def _prepare_data(self, data):
        # Ensure the data has the required columns
        required_columns = ['CustomerID', 'StockCode', 'Quantity']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain the following columns: {required_columns}")
        
        # Convert 'StockCode' to string if it's not already
        data['StockCode'] = data['StockCode'].astype(str)
        
        # Create the customer-product matrix
        customer_product_matrix = data.pivot_table(
            values='Quantity', 
            index='CustomerID', 
            columns='StockCode', 
            aggfunc='sum', 
            fill_value=0
        )
        
        return customer_product_matrix
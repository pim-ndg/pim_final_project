from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import (setup, compare_models, pull, save_model, 
                                load_model, tune_model, create_model,
                                evaluate_model)
# import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import os 
from utilities.data_processing import load_data, preprocess_data_for_anomaly, load_cleaned_data
import joblib
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tempfile
from sklearn.model_selection import cross_validate
import line_profiler




def evaluate_customer(test_sets, model):
    X_test, y_test = test_sets
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred == 1).astype(int)
    
    # Convert y_test to binary targets
    y_test_binary = (y_test == 1).astype(int)
    
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    precision = precision_score(y_test_binary, y_pred_binary)
    recall = recall_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# if os.path.exists('data/online_retail_II.csv'): 
#     df = pd.read_csv('data/online_retail_II.csv', index_col=None)
# df = load_data()
df = load_cleaned_data()
# st.write(df.sample(5))
if "data" not in st.session_state:
    st.session_state["data"] = df

with st.sidebar: 
    st.image("./assets/Logo_PIM.png")
    st.title("Online Retail")
    choice = st.radio("Navigation", [
        "Introduction",
        "Auto Profiling", #"Auto Modeling", #"Download","Upload", 
        # "Super MVP", 
        "Churn", "CLV", 
        # "Next Purchase", 
        "Buy Again", "Customer Segment",
        # "CLV", "Product Demand", "Buy Again","CLV2"
        ])
    st.info(f"The capstone project for non-degree the *Data Science and Big Data Technology for Business Analytics* class.")

if choice == "Super MVP":
    import super_mvp
    super_mvp.main()
if choice == "Churn":
    import retail_churn
    retail_churn.run()
if choice == "CLV":
    import retail_clv as retail_clv
    retail_clv.run()
if choice == "Next Purchase":
    import retail_next_purchase as np
    np.main()
if choice == "Buy Again":
    import retail_buy_again as ba
    ba.main()
if choice == "Customer Segment":
    import retail_customer_segment as cs
    cs.main()
if choice == "Introduction":
    st.write("Online Retail Recommendation System")
    import retail_introduction as intro
    intro.run()

if choice == "Upload":
    st.title("Upload Your Dataset")
    max_file_size = 32 * 1024 * 1024  # 32 MB
    file = st.file_uploader("Upload Your Dataset",type=["csv", "xlsx"])
    if file:
        if len(file.getvalue()) > max_file_size:
            st.error(f"File size exceeds the limit of {max_file_size / (1024 * 1024):.2f} MB.")
            st.write("We will use default file instead.")
            df = load_data()
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file_path = os.path.join(tmp_dir, file.name)
                with open(tmp_file_path, 'wb') as f:
                    f.write(file.getvalue())

                df = load_data(tmp_file_path)
                df.to_csv(f'dataset_{file.name.split(".")[0]}.csv', index=None)
        st.dataframe(df)

if choice == "Auto Profiling":
    import retail_profiling
    with st.spinner("Auto profiling in process, please wait..."):
        profiler = line_profiler.LineProfiler()
        profiler.add_function(retail_profiling.run())
        profiler.run('retail_profiling')
        profiler.print_stats()
    
if choice == "Auto Modeling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    
    
    if st.button('Run Modeling'):
        # Get the list of categorical columns
        categorical_cols = df.select_dtypes(include='object').columns
        # Determine the chunk size based on available memory
        
        with st.spinner("Categorical Encoding..."):
            
            if 'encoded_df' in st.session_state:
                encoded_df = st.session_state['encoded_df']
            else:
                chunk_size = 1000
                encoded_df = pd.DataFrame()
                
                # Iterate over the data in chunks
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    
                    # Encode the categorical columns in the chunk
                    for col in categorical_cols:
                        encoder = OneHotEncoder()
                        encoded = encoder.fit_transform(chunk[[col]])
                        encoded_cols = encoder.get_feature_names_out([col])
                        encoded_chunk = pd.DataFrame(encoded.toarray(), columns=encoded_cols)
                        encoded_df = pd.concat([encoded_df, encoded_chunk], axis=1)
                        
                    # Clear the chunk from memory
                    del chunk
                st.session_state['encoded_df'] = encoded_df
        
                df = pd.concat([df, encoded_df], axis=1)
                df = df.drop(categorical_cols, axis=1)
        
        with st.spinner("Running data modeling, please wait..."):
            if 'setup_df' in st.session_state:
                setup_df = st.session_state['setup_df']
            else:
                setup(df, target=chosen_target, verbose=False, remove_outliers=True, feature_interaction=True, feature_selection=True)
                setup_df = pull()
                st.session_state['setup_df'] = setup_df
            st.dataframe(setup_df)
            
            # Tune the best model
            tuned_model = tune_model(chosen_target, optimize='R2', fold=5, n_iter=20)
            
            # Compare models
            # compare_df = compare_models(include=['lightgbm', 'catboost', 'xgboost'], fold=5, sort='R2')
            if 'compare_df' in st.session_state:
                compare_df = st.session_state['compare_df']
            else:
                compare_df = compare_models(include=['lightgbm', 'catboost', 'xgboost'], fold=5, sort='R2')
                st.session_state['compare_df'] = compare_df
            st.dataframe(compare_df)
            
            # Select the best model
            best_model = create_model(compare_df.index[0])
            
            # Evaluate the best model using holdout data
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            evaluate_model(best_model, train=train, test=test)
            
            # Cross-validate the best model
            # cv_results = cross_validate(best_model, df, chosen_target, cv=5, scoring=['r2', 'neg_mean_squared_error'])
            if 'cv_results' in st.session_state:
                cv_results = st.session_state['cv_results']
            else:
                cv_results = cross_validate(best_model, df, chosen_target, cv=5, scoring=['r2', 'neg_mean_squared_error'])
                st.session_state['cv_results'] = cv_results
            st.write(f"Average R2: {cv_results['test_r2'].mean():.2f}")
            st.write(f"Average MSE: {-cv_results['test_neg_mean_squared_error'].mean():.2f}")
            
            # Save the best model
            save_model(best_model, 'best_model')
            joblib.dump(best_model, "./models/best_model.joblib")
        st.success("Data modeling completed!")

if choice == "CLV_":
    import retail_clv
    retail_clv.run()
    
if choice == "Product Demand":
    import retail_product_demand
    retail_product_demand.run()
    

if choice == "Anomaly":
    import retail_anomaly
    retail_anomaly.run()
    
if choice == "Download": 
    with open('./models/best_model.joblib', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.joblib")

if choice == "Buy Again_":
    import retail_buyagain
    retail_buyagain.run()

if choice == "CLV2":
    import retail_customer_lifetime_value as rclv
    rclv.run()
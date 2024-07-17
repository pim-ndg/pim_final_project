import streamlit as st
from ydata_profiling import ProfileReport
from utilities.data_processing import load_data
from streamlit_pandas_profiling import st_profile_report

@st.cache_resource(show_spinner="Getting profile report, please wait...")
def get_profile_report(df):
    profile_report = ProfileReport(df, title="UCI OnlineRetail")
    return profile_report 

def run():
    st.title("Data Profiling - Online Retail")
    df = load_data()
    if st.button("Process data profiling"):
        if "profile_report" not in st.session_state:
            profile_report = get_profile_report(df)
            st.session_state["profile_report"] = profile_report
        else:
            profile_report = st.session_state["profile_report"]
        
        st_profile_report(profile_report)
import streamlit as st
from components.sidebar import show as show_sidebar

def main():
    st.set_page_config(
        page_title="PIM - ML Recommendation System", 
        page_icon=":taco:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    show_sidebar()

if __name__ == "__main__":
    main()
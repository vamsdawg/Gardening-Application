import streamlit as st
import lawn_care_app
import plant_care_app

# Streamlit "Smart Garden Assistant" - Main Entry Point
# Usage: pip install -r requirements.txt
# Run: streamlit run app.py

st.set_page_config(page_title="Smart Garden Assistant", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("Navigation")

def reset_page_state():
    """Reset analysis state when switching pages"""
    st.session_state.lawn_analyzed = False
    st.session_state.plant_analyzed = False
    if 'lawn_image' in st.session_state:
        del st.session_state.lawn_image
    if 'plant_image' in st.session_state:
        del st.session_state.plant_image

page = st.sidebar.radio("Go to", ["Lawn Care", "Plant Care"], on_change=reset_page_state)

if page == "Lawn Care":
    lawn_care_app.run()
elif page == "Plant Care":
    plant_care_app.run()

import streamlit as st

st.set_page_config(page_title='바쁜사람들', layout="wide", initial_sidebar_state="collapsed")

col1, col2, col3 = st.columns(3)

with col1:
   st.header("식물병원이란?")
   st.video('https://youtu.be/n_QOv-nY_zM')



import streamlit as st

st.set_page_config(page_title='바쁜사람들', layout="wide", initial_sidebar_state="collapsed")

col1, col2, col3 = st.columns([2,1,1])

with col1:
   st.header("식물병원이란?")
   st.video('https://youtu.be/n_QOv-nY_zM')


import os

path = os.getcwd()

path

file_list = os.listdir(path)

file_list

list_remove = [
"NanumBarunGothic (2).ttf",
"NanumBarunGothic (9).ttf",
"NanumBarunGothic (3).ttf",
"NanumBarunGothic (13).ttf",
"NanumBarunGothic (10).ttf",
"NanumBarunGothic (4).ttf",
"NanumBarunGothic (15).ttf",
"NanumBarunGothic (12).ttf",
"NanumBarunGothic (11).ttf",
"NanumBarunGothic.ttf",
"NanumBarunGothic (1).ttf",
"NanumBarunGothic (8).ttf",
"NanumBarunGothic (7).ttf",
"NanumBarunGothic (6).ttf",
"NanumBarunGothic (14).ttf"
]

for i in list_remove:
   os.remove(i)
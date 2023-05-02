import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import koreanize_matplotlib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.title('한눈에 보는 데이터 프레임')
agree = st.checkbox('밴드')
agree2 = st.checkbox('식물갤러리')

options = st.multiselect(
    '단어를 선택하세요',
    ['식물', '몬스테라', '영양제',])

st.write('You selected:', options)



    


  

col1, col2 = st.columns(2)
with col1:
    option_1 = st.selectbox('Option 1', ['A', 'B', 'C'],key='option1')
with col2:
    option_2 = st.selectbox('Option 2', ['X', 'Y', 'Z'],key='option2')

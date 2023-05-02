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



    
title = st.text_input('Movie title', 'Life of Brian')
st.write('The current movie title is', title)

    
option = st.selectbox(
        "How would you like to be contacted?",
        ("Email", "Home phone", "Mobile phone")

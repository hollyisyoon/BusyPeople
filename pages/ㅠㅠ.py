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
    option_1 = st.selectbox('Option 1', ['자사', '경쟁사'],key='option1')
with col2:
        # 시작일과 종료일을 지정합니다.
    start_date = dt.date(2023, 1, 1)
    end_date = dt.date.today()

    # 시작일과 종료일 사이의 모든 날짜를 생성합니다.
    dates = pd.date_range(start_date, end_date)

    # 날짜를 원하는 형식으로 문자열로 변환합니다.
    date_strings = [date.strftime('%Y-%m-%d') for date in dates]

    # selectbox에 날짜 목록을 전달합니다.
    selected_date = st.selectbox('날짜 선택', date_strings,key='option2')

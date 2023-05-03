import streamlit as st
import plotly.express as px

import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

from wordcloud import WordCloud

########################################################################################################################

df_리뷰_감성분석결과 = pd.read_csv('/app/busypeople-stramlit/data/리뷰_감성분석결과.csv')
df_리뷰_감성분석결과['time'] = pd.to_datetime(df_리뷰_감성분석결과['time'])

df_파이차트 = pd.DataFrame(df_리뷰_감성분석결과['감성결과'].value_counts())
values = '감성결과'
names = df_파이차트.index

px.pie(df_파이차트, values='감성결과', names=names)

########################################################################################################################
# 레이아웃

col1, col2, col3 = st.columns([2,1,1])

with col1:
   st.plotly_chart(파이차트)


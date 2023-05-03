import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from collections import Counter

from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from wordcloud import WordCloud
########################################################################################################################
def get_count_top_words(df, start_date=None, last_date=None, num_words=10, name=None, 감성결과 = None, item = None, source = None ):
    if name is not None:
        df = df[df['name'] == name]
    if 감성결과 is not None:
        df = df[df['감성결과'] == 감성결과]
    if item is not None:
        df = df[df['item'] == item]
    if source is not None:
        df = df[df['source'] == source]
    if start_date is None:
        start_date = df['time'].min().strftime('%Y-%m-%d')
    if last_date is None:
        last_date = df['time'].max().strftime('%Y-%m-%d')
    df = df[(df['time'] >= start_date) & (df['time'] <= last_date)]
    count_vectorizer = CountVectorizer()
    count = count_vectorizer.fit_transform(df['kha_nng_은어전처리_sentence'].values)
    count_df = pd.DataFrame(count.todense(), columns=count_vectorizer.get_feature_names_out())
    count_top_words = count_df.sum().sort_values(ascending=False).head(num_words).to_dict()
    return count_top_words
########################################################################################################################
# 데이터 로드
df_리뷰_감성분석결과 = pd.read_csv('/app/busypeople-stramlit/data/리뷰_감성분석결과.csv')
df_리뷰_감성분석결과['time'] = pd.to_datetime(df_리뷰_감성분석결과['time'])

words = get_count_top_words(df_리뷰_감성분석결과)
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
# 레이아웃
col1, col2, col3 = st.columns([1,1,1])
########################################################################################################################
with col1:
    # 파이차트
    df_파이차트 = pd.DataFrame(df_리뷰_감성분석결과['감성결과'].value_counts())
    pie_chart = go.Figure(data=[go.Pie(labels=list(df_파이차트.index), values=df_파이차트['count'])])
    st.plotly_chart(pie_chart, use_container_width=True)
########################################################################################################################
with col2:
    # 워드클라우드
    cand_mask = np.array(Image.open('/app/busypeople-stramlit/data/circle.png'))
    워드클라우드 = WordCloud(
        background_color="white", 
        max_words=1000,
        font_path = "/app/busypeople-stramlit/font/NanumBarunGothic.ttf", 
        contour_width=3, 
        colormap='Spectral', 
        contour_color='white',
        mask=cand_mask).generate_from_frequencies(words)
    fig, ax = plt.subplots()
    ax.imshow(워드클라우드, interpolation='bilinear')
    st.pyplot(fig, use_container_width=True)
########################################################################################################################

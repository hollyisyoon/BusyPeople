import pandas as pd
import koreanize_matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import to_rgba
import plotly.graph_objects as go
import plotly.express as px
import ast
import time

import streamlit as st
from streamlit_extras.let_it_rain import rain

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore", message="PyplotGlobalUseWarning")
import networkx as nx
from gensim.models import Word2Vec
import time

df = pd.read_csv('/app/busypeople-stramlit/data/plant_gallery.csv', encoding='utf8')
df['time'] = pd.to_datetime(df['time'])
df['name'] = df['name'].astype(str)

def time_series(df, start_date, end_date, media, search_word):
    df = df[df['name'] == media]
    df = df[(df['time'] >= start_date) & (df['time'] <= last_date)]
    # countvectorizer
    count_vectorizer = CountVectorizer()
    count = count_vectorizer.fit_transform(df['title+content'].values)
    count_df = pd.DataFrame(count.todense(), columns=count_vectorizer.get_feature_names_out())

    count_df['date'] = pd.to_datetime(df['time']).date()
    count_df = count_df[['date', search_word]]
    top_words = count_df.iloc[:, 1:].sum().sort_values(ascending=False).head(100).index
    top_words = [search_word]
    time_top_words = count_df.groupby('date')[top_words].sum()
    return time_top_words

search_word = st.text_input('어떤 키워드의 트렌드를 볼까요?', value='제라늄')

col1, col2, col3 = st.beta_columns(3)
with col1:
    start_date = st.date_input("시작 날짜",
                           value=datetime.today() - timedelta(days=45),
                           min_value=datetime(2022, 4, 27),
                           max_value=datetime(2023, 4, 26))
with col2:
    end_date = st.date_input("끝 날짜", 
                         value=datetime.today() - timedelta(days=30),    
                         min_value=datetime(2022, 4, 27),
                         max_value=datetime(2023, 4, 26))
with col3:
    keyword_no = st.number_input("📌 키워드", value=50, min_value=1, step=1)

media = st.selectbox('매체',('식물갤러리', '네이버카페'))

start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)
time_series(df, start_date, end_date, media, search_word)
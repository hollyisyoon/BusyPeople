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
import os
import numpy as np

rain(emoji="🦝",
    font_size=54,
    falling_speed=10,
    animation_length="infinite")

#데이터 불러오기
df = pd.read_csv('/app/busypeople-stramlit/data/plant_gallery.csv', encoding='utf8')
df['time'] = pd.to_datetime(df['time'])
def to_list(text):
    return ast.literal_eval(text)

def get_tfidf_top_words(df, start_date, last_date, num_words, media):
    df = df[df['name'] == media]
    df = df[(df['time'] >= start_date) & (df['time'] <= last_date)]
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf = tfidf_vectorizer.fit_transform(df['title+content'].values)
    tfidf_df = pd.DataFrame(tfidf.todense(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_top_words = tfidf_df.sum().sort_values(ascending=False).head(num_words).to_dict()
    tfidf_top_words = dict(tfidf_top_words)
    return tfidf_top_words

def get_count_top_words(df, start_date, last_date, num_words, media):
    df = df[df['name'] == media]
    df = df[(df['time'] >= start_date) & (df['time'] <= last_date)]
    count_vectorizer = CountVectorizer(stop_words=stopwords)
    count = count_vectorizer.fit_transform(df['title+content'].values)
    count_df = pd.DataFrame(count.todense(), columns=count_vectorizer.get_feature_names_out())
    count_top_words = count_df.sum().sort_values(ascending=False).head(num_words).to_dict()
    return count_top_words

def keyword_timeseries(df, start_date, last_date, media, keyword):
    df['title+content'] = df['title+content'].astype(str)
    df = df[(df['title+content'].str.contains(keyword)) & (df['name'] == media)]
    mask = (df['time'] >= start_date) & (df['time'] <= last_date)
    df = df.loc[mask]
    df_daily_views = df.groupby(df['time'].dt.date)['view'].sum().reset_index()
    return df_daily_views

def get_words(df, col, keyword):
    df[col] = df[col].map(to_list)
    text_list=[]
    for sublist in df[col]:
        text_list.append(sublist)
    model = Word2Vec(text_list, vector_size=100, window=5, min_count=1, workers=3, epochs=30)
    try:
        similar_words = model.wv.most_similar(keyword, topn=10)
        results = [(keyword, word, score) for word, score in similar_words]
        return results
    except:
        return None

def show_modal(df):
    st.table(df) 

#### 대시보드 시작 #####
st.title('외부 트렌드 모니터링 대시보드')

#### 인풋 필터 #####
col1, col2, col3 = st.beta_columns(3)
with col1:
    start_date = st.date_input("시작 날짜",
                           value=datetime.today() - timedelta(days=45),
                           min_value=datetime(2022, 7, 27),
                           max_value=datetime(2023, 4, 26))
with col2:
    end_date = st.date_input("끝 날짜", 
                         value=datetime.today() - timedelta(days=30),    
                         min_value=datetime(2022, 7, 27),
                         max_value=datetime(2023, 4, 26))
with col3:
    keyword_no = st.number_input("📌 키워드", value=50, min_value=1, step=1)

col1, col2, col3 = st.beta_columns(3)    
with col1:
    type = st.selectbox('기준',('상대 빈도(TF-IDF)','단순 빈도(Countvertize)'))
with col2:
    media = st.selectbox('매체',('식물갤러리', '네이버카페'))
with col3:
    input_str = st.text_input('제거할 키워드')
    stopwords = [x.strip() for x in input_str.split(',')]

# 타입 옵션
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)
if type == '단순 빈도(Countvertize)' :
    words = get_count_top_words(df, start_date, end_date, keyword_no, media)
else :
    words = get_tfidf_top_words(df, start_date, end_date, keyword_no, media)

#워드클라우드
wc = WordCloud(background_color="white", colormap='Spectral', contour_color='steelblue', font_path="/app/busypeople-stramlit/font/NanumBarunGothic.ttf")
wc.generate_from_frequencies(words)

###########동적 워드 클라우드####################
# 컬러 팔레트 생성
word_list=[]
freq_list=[]
fontsize_list=[]
position_list=[]
orientation_list=[]
color_list=[]

for (word, freq), fontsize, position, orientation, color in wc.layout_:
    word_list.append(word)
    freq_list.append(freq)
    fontsize_list.append(fontsize)
    position_list.append(position)
    orientation_list.append(orientation)
    color_list.append(color)

# get the positions
x=[]
y=[]
for i in position_list:
    x.append(i[0])
    y.append(i[1])

# WordCloud 시각화를 위한 Scatter Plot 생성
fig = go.Figure(go.Scatter(
    x=x, y=y, mode="text",
    text=word_list,
    textfont=dict(size=fontsize_list, color=color_list),
))
fig.update_layout(title="WordCloud", xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), hovermode='closest')
st.plotly_chart(fig, use_container_width=True)

###### 바그래프 #####
words_count = Counter(words)
words_df = pd.DataFrame([words_count]).T
st.bar_chart(words_df)


#######세부 키워드########
### 시계열 그래프 ###
st.title('관련 키워드 알아보세요!')

search_word = st.text_input('🔮 무슨 키워드가 궁금하신가요', value='제라늄')
df_daily_views = keyword_timeseries(df, start_date, end_date, media, search_word)
fig = px.line(df_daily_views, x='time', y='view')
st.plotly_chart(fig, use_container_width=True)

#### 연관검색어 #####
if st.button('분석을 시작하기'):
    with st.spinner('분석 중입니다...'):
        # Define the data
        data = get_words(df,'title+content', search_word)
        if data is None:
            st.warning('다른 키워드를 입력해주세요. 추천 키워드 : 제라늄🌸')
        else:
            df_data = pd.DataFrame(data, columns=["키워드", "연관 키워드", "유사도"])

        # Create the network graph
        G = nx.DiGraph()
        for row in data:
            G.add_edge(row[0], row[1], weight=row[2])

        pos = nx.spring_layout(G)

        labels = {}
        for edge in G.edges(data=True):
            labels[(edge[0], edge[1])] = f"{edge[2]['weight']:.2f}"

        edge_widths = [data[i][2] for i in range(len(data))]

        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_labels(G, pos, font_size=12, font_family='NanumGothic', font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=12, font_family='NanumBarunGothic')

        st.success(f"<{search_word}>에 대한 연관어 분석 결과입니다😀")
        plt.axis('off')
        st.pyplot()

        expander = st.expander('분석 결과 데이터 보기')
        with expander:
            show_modal(df_data)


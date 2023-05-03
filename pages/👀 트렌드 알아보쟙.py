import pandas as pd
import koreanize_matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import to_rgba
import plotly.graph_objects as go
import ast
import time

import streamlit as st
from streamlit_extras.let_it_rain import rain

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud
from datetime import datetime, timedelta


rain(emoji="🦝",
    font_size=54,
    falling_speed=10,
    animation_length="infinite")

#데이터 불러오기
df = pd.read_csv('/app/busypeople-stramlit/data/plant_gallery.csv', encoding='utf8')
df['time'] = pd.to_datetime(df['time'])
df['name'] = df['name'].astype(str)

stopwords=['식물']

def get_tfidf_top_words(df, start_date, last_date, num_words, media):
    df = df[df['name'] == media]
    start_date = pd.to_datetime(start_date)
    last_date = pd.to_datetime(last_date)
    df = df[(df['time'] >= start_date) & (df['time'] <= last_date)]
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf = tfidf_vectorizer.fit_transform(df['title+content'].values)
    tfidf_df = pd.DataFrame(tfidf.todense(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_top_words = tfidf_df.sum().sort_values(ascending=False).head(num_words).to_dict()
    tfidf_top_words = dict(tfidf_top_words)
    return tfidf_top_words

def get_count_top_words(df, start_date, last_date, num_words, media):
    df = df[df['name'] == media]
    start_date = pd.to_datetime(start_date)
    last_date = pd.to_datetime(last_date)
    df = df[(df['time'] >= start_date) & (df['time'] <= last_date)]
    count_vectorizer = CountVectorizer(stop_words=stopwords)
    count = count_vectorizer.fit_transform(df['title+content'].values)
    count_df = pd.DataFrame(count.todense(), columns=count_vectorizer.get_feature_names_out())
    count_top_words = count_df.sum().sort_values(ascending=False).head(num_words).to_dict()
    return count_top_words

st.title('외부 트렌드 모니터링 대시보드')
#인풋
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

col1, col2, col3 = st.beta_columns(3)    
with col1:
    type = st.selectbox('기준',('상대 빈도(TF-IDF)','단순 빈도(Countvertize)'))
with col2:
    media = st.selectbox('매체',('식물갤러리', '네이버카페'))
with col3:
    input_str = st.text_input('제거할 키워드')
    # stopwords = [x.strip() for x in input_str.split(',')]


# 타입 옵션
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)
if type == '단순 빈도(Countvertize)' :
    words = get_count_top_words(df, start_date, end_date, keyword_no, media)
else :
    words = get_tfidf_top_words(df, start_date, end_date, keyword_no, media)

#워드클라우드
wc = WordCloud(background_color="white", colormap='Spectral', contour_color='steelblue')
wc.generate_from_frequencies(words)


############동적 워드 클라우드####################
# # Wordcloud를 위한 데이터 프레임 생성
# words_dict = dict(wc.words_)
# df = pd.DataFrame({
#     'text': list(words_dict.keys()),
#     'size': list(words_dict.values()),
#     'color': np.random.choice(palette, len(words_dict))
# })

# # 컬러 팔레트 생성
# palette = np.random.choice(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
#                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'], 50, replace=True)

# # WordCloud 시각화를 위한 Scatter Plot 생성
# fig = go.Figure(go.Scatter(
#     x=[0], y=[0], mode="text",
#     text=df['text'],
#     textfont=dict(size=df['size'], color=df['color']),
# ))
# fig.update_layout(title="WordCloud", xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), hovermode='closest')
# st.plotly_chart(fig)


##########정적 워드 클라우드##########
fig1, ax1 = plt.subplots()
ax1.imshow(wc, interpolation='bilinear')
ax1.axis("off")
st.pyplot(fig1)


# 바그래프
words_count = Counter(words)
words_df = pd.DataFrame([words_count]).T
st.bar_chart(words_df)

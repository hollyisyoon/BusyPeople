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

def get_tfidf_top_words(df, start_date, last_date, num_words, media):
    df = df[df['name'] == media]
    start_date = pd.to_datetime(start_date)
    last_date = pd.to_datetime(last_date)
    df = df[(df['time'] >= start_date) & (df['time'] <= last_date)]
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf = tfidf_vectorizer.fit_transform(df['title+content'].values)
    tfidf_df = pd.DataFrame(tfidf.todense(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_top_words = tfidf_df.sum().sort_values(ascending=False).head(num_words).to_dict()
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
    type = st.selectbox('기준',('단순 빈도(Countvertize)', '상대 빈도(TF-IDF)'))
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

# 색 구분
def create_colorscale(color_list):
    """
    구간별 색 구분
    """
    n = len(color_list)
    scale = []
    for i in range(n):
        if i == 0:
            scale.append([0, color_list[i]])
        elif i == n - 1:
            scale.append([1, color_list[i]])
        else:
            scale.append([(i / (n - 1)), color_list[i]])
    return scale

wc = WordCloud(background_color="white", colormap='Spectral', font_path='/app/busypeople-stramlit/font/NanumBarunGothic.ttf')
wc.generate_from_frequencies(words)

colors = wc.to_array()
colors = colors / 255.0
colors = colors.reshape(-1, 4)
colors = np.apply_along_axis(lambda x: to_rgba(x), 1, colors)
num_intervals = 5
cscale = []
for i in range(num_intervals):
    start = i / num_intervals
    end = (i + 1) / num_intervals
    interval_color = np.mean(colors[(colors[:, 2] >= start) & (colors[:, 2] < end)], axis=0)
    cscale.append([i / (num_intervals - 1), 'rgb' + str(tuple(interval_color[:3] * 255))])

fig = go.Figure(go.Image(z=colors))
fig.update_layout(
    width=700,
    height=700,
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    margin=dict(l=0, r=0, t=0, b=0),
    coloraxis=dict(
        showscale=False,
        colorscale=create_colorscale(cscale),
        colorbar=dict(tickvals=np.linspace(0, 1, num_intervals), ticktext=[f'{i + 1}구간' for i in range(num_intervals)])
    )
)
st.plotly_chart(fig)


words_count = Counter(words)
words_df = pd.DataFrame([words_count]).T
st.bar_chart(words_df)

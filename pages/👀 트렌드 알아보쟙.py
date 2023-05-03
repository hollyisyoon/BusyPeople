import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import ast
import time

import streamlit as st
from datetime import datetime, timedelta
from streamlit_extras.let_it_rain import rain
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud
from datetime import datetime, timedelta
import koreanize_matplotlib

# plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font', family = 'NanumBarunGothic')

rain(emoji="🦝",
    font_size=54,
    falling_speed=10,
    animation_length="infinite")

#데이터 전처리
# def to_list(text):
#     try:
#         return ast.literal_eval(text)
#     except (ValueError, SyntaxError):
#         return []

#데이터 불러오기
df = pd.read_csv('https://raw.githubusercontent.com/seoinhyeok96/BusyPeople/main/data/plant_gallery.csv', encoding='utf8')
# df['title+content'] = df['title+content'].map(to_list)
df['time'] = pd.to_datetime(df['time'])
df['name'] = df['name'].astype(str)

def get_tfidf_top_words(df, start_date, last_date, num_words, media):
    df = df[df['name'] == media]
    start_date = pd.to_datetime(start_date)
    last_date = pd.to_datetime(last_date)
    df = df[(df['time'] >= start_date) & (df['time'] <= last_date)]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(df['title+content'].values)
    tfidf_df = pd.DataFrame(tfidf.todense(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_top_words = tfidf_df.sum().sort_values(ascending=False).head(num_words).to_dict()
    return tfidf_top_words

def get_count_top_words(df, start_date, last_date, num_words, media):
    df = df[df['name'] == media]
    start_date = pd.to_datetime(start_date)
    last_date = pd.to_datetime(last_date)
    df = df[(df['time'] >= start_date) & (df['time'] <= last_date)]
    count_vectorizer = CountVectorizer()
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
    pre_input = ['식물', '화분', '사진', '오늘']
    exceptwords = st.multiselect('제외할 키워드', pre_input, allow_input=True)
    # if stopwords:
    #     stopwords = [x.strip() for x in ','.join(stopwords).split(',')]


    # input_str = st.text_input('Enter hashtags separated by commas')
    # stopwords = [x.strip() for x in input_str.split(',')]

# Get top words
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)
if type == '단순 빈도(Countvertize)' :
    words = get_count_top_words(df, start_date, end_date, keyword_no, media)
else :
    words = get_tfidf_top_words(df, start_date, end_date, keyword_no, media)

# Create word cloud
wc = WordCloud(background_color="white", 
            #    max_font_size=1000, 
               colormap='Spectral', 
               contour_color='steelblue',
               font_path='/app/busypeople-stramlit/font/NanumBarunGothic.ttf')
wc.generate_from_frequencies(words)
fig1, ax1 = plt.subplots()
ax1.imshow(wc, interpolation='bilinear')
ax1.axis("off")
st.pyplot(fig1)

# Create bar graph
words_count = Counter(words)
words_df = pd.DataFrame([words_count]).T
# words_df = words_df.sort_values('count', ascending=False, inplace=True)

# fig2, ax2 = plt.subplots(figsize=(10, 4))
# words_df.plot(kind='bar', ax=ax2)
# ax2.set_title('Top Words')
# ax2.set_xlabel('Words')
# ax2.set_ylabel('Count')
# ax2.tick_params(axis='x', labelrotation=45, labelsize=8)
# label_size = st.slider('X-Axis Label Size', 1, 20, 8)
# ax2.tick_params(axis='x', labelrotation=45, labelsize=label_size)
st.bar_chart(words_df)
         
     

import pandas as pd
import matplotlib.pyplot as plt
import ast
import time

import streamlit as st
from datetime import datetime, timedelta
from streamlit_extras.let_it_rain import rain
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud
from datetime import datetime, timedelta

rain(emoji="🦝",
    font_size=54,
    falling_speed=10,
    animation_length="infinite")

#데이터 전처리
def to_list(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

df = pd.read_csv('https://raw.githubusercontent.com/seoinhyeok96/BusyPeople/main/data/plant_gallery.csv')
df['title+content'] = df['title+content'].map(to_list)
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

plt.rc('font', family='NanumBarunGothic')
st.title('외부 트렌드 모니터링 대시보드')
#인풋
col1, col2, col3 = st.beta_columns(3)
with col1:
    start_date = st.date_input("👉🏻 시작 날짜",
                           value=datetime.today() - timedelta(days=45),
                           min_value=datetime(2022, 4, 27),
                           max_value=datetime(2023, 4, 26))
with col2:
    end_date = st.date_input("끝 날짜 👈🏻", 
                         value=datetime.today() - timedelta(days=30),    
                         min_value=datetime(2022, 4, 27),
                         max_value=datetime(2023, 4, 26))
with col3:
    keyword_no = st.number_input("📌 키워드", value=50, min_value=1, step=1)

col1, col2, col3 = st.beta_columns(3)    
with col1:
    st.write("🗓 ", start_date, "~", end_date)    
with col2:
    st.write(keyword_no, '개의 키워드 선택')   
with col3:
    st.write('식물갤러리')   

# Get top words
# start_date = pd.to_datetime(start_date)
# end_date = pd.to_datetime(end_date)
df = df[(df['name'] == '식물갤러리') & (df['time'] >= start_date) & (df['time'] <= end_date)]
words = get_tfidf_top_words(df, start_date, end_date, keyword_no, '식물갤러리')

# Create word cloud
wc = WordCloud(background_color="white", 
               max_font_size=1000, 
               colormap='Spectral', 
               contour_color='steelblue',
               font_path='NanumBarunGothic.ttf')
wc.generate_from_frequencies(words)
fig1, ax1 = plt.subplots()
ax1.imshow(wc, interpolation='bilinear')
ax1.axis("off")
st.pyplot(fig1)

# Create bar graph
words_count = Counter(words)
words_df = pd.DataFrame.from_dict(words_count, orient='index', columns=['count'])
words_df.sort_values('count', ascending=False, inplace=True)
fig2, ax2 = plt.subplots(figsize=(10, 4))
words_df.plot(kind='bar', ax=ax2)
ax2.set_title('Top Words')
ax2.set_xlabel('Words')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', labelrotation=45, labelsize=8)
label_size = st.slider('X-Axis Label Size', 1, 20, 8)
ax2.tick_params(axis='x', labelrotation=45, labelsize=label_size)
st.pyplot(fig2)
         
     

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

rain(emoji="🦝",
    font_size=54,
    falling_speed=10,
    animation_length="infinite")

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
    media = st.multiselect('모니터링할 곳은~?',['식물갤러리'], default='식물갤러리')
     

#데이터 전처리
def to_list(text):
    return ast.literal_eval(text)
df = pd.read_csv('https://raw.githubusercontent.com/seoinhyeok96/BusyPeople/main/data/plant_gallery.csv')
df['title+content'] = df['title+content'].map(to_list)
df['time'] = pd.to_datetime(df['time'])

#워드 클라우드
def plot_wordcloud(words):
    wc = WordCloud(background_color="white", 
                   max_words=1000, 
                   contour_width=3, 
                   colormap='Spectral', 
                   contour_color='steelblue')
    wc.generate_from_frequencies(words)
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")

def plot_bar(words):
    words_count = Counter(words)
    words_df = pd.DataFrame.from_dict(words_count, orient='index', columns=['count'])
    words_df.sort_values('count', ascending=False, inplace=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    words_df.plot(kind='bar', ax=ax)
    ax.set_title('Top Words')
    ax.set_xlabel('Words')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', labelrotation=45, labelsize=8) 
    label_size = st.slider('X-Axis Label Size', 1, 20, 8)
    ax.tick_params(axis='x', labelrotation=45, labelsize=label_size)
    st.pyplot(fig)    
    
def get_tfidf_top_words(df, start_date, last_date, num_words, name):
    df = df[df['name'] == name]
    start_date = pd.to_datetime(start_date)
    last_date = pd.to_datetime(last_date)
    df = df[(df['time'] >= start_date) & (df['time'] <= last_date)]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(df['title+content'].values)
    tfidf_df = pd.DataFrame(tfidf.todense(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_top_words = tfidf_df.sum().sort_values(ascending=False).head(num_words).to_dict()
    plt.figure(figsize=(12, 6))
    plot_wordcloud(tfidf_top_words)
    plot_bar(tfidf_top_words)
        
def main():
    get_tfidf_top_words(df, start_date, end_date, keyword_no, media[0])
    
if __name__ == '__main__':
    main()    

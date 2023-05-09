import koreanize_matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import to_rgba
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyvis.network import Network
import networkx as nx
import gensim
from gensim.models import Word2Vec
from PIL import Image

import pandas as pd
import ast
import time
from datetime import datetime, timedelta
import itertools
from markdownlit import mdlit

#스트림잇
import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_tags import st_tags
import warnings
warnings.filterwarnings("ignore", message="PyplotGlobalUseWarning")

#계산
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud


# CSS 스타일 정의
css_code = """
<style>
    .custom-sidebar {
        padding: 20px;
        background-color: #f2f2f2;
        font-size: 18px;
        color: #333;
    }
    
    .custom-sidebar a {
        color: #333;
        text-decoration: none;
    }
</style>
"""

STYLE = """
.callout {
    padding: 1em;
    border-radius: 0.5em;
    background-color: #F8F8F8;
    border-left: 4px solid #195ef7;
    margin-bottom: 1em;
    color: black;
}

h3 {
    font-size: 1.5em;
    font-weight: medium;
    padding: 0.3em;
    margin: 0.3em 0em;
    background-color: #f5f5f5;
    border-radius: 10px;
}
"""

##############메인 콘텐츠
st.title("트렌드 키워드 분석")

#########Section1 - wordcloud############
st.markdown("<h2 id='section1'>🔎 키워드 발굴</h2>", unsafe_allow_html=True)

##데이터##
def to_list(text):
    return ast.literal_eval(text)

df = pd.read_csv('/app/busypeople-streamlit/data/df_트렌드_github_.csv')
df['날짜'] = pd.to_datetime(df['날짜'])
def extract_df(df, media, start_date, end_date, effect_size):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    standard_df = df[(df['매체'] == media) & (df['날짜'] >= start_date) & (df['날짜'] <= end_date) & (df['영향도'] >= effect_size)]
    range_days = (end_date - start_date) + timedelta(days = 1)
    new_day = start_date - range_days
    new_day = pd.Timestamp(new_day)
    new_df = df[(df['매체'] == media) & (df['날짜'] >= new_day) & (df['날짜'] < start_date) & (df['영향도'] >= effect_size)]    
    return standard_df, new_df

##인풋 필터 - 전체용##
col1, col2, col3 = st.beta_columns(3)
min_date = datetime(2022, 6, 1)
max_date = datetime(2023, 4, 26)
with col1:
    start_date = st.date_input("시작 날짜",
                               value=datetime(2022,6,1),
                               min_value=min_date,
                               max_value=max_date - timedelta(days=7))
    # 끝 날짜를 선택할 때 최소 날짜는 시작 날짜이며, 최대 날짜는 90일 이전까지로 제한
    end_date = st.date_input("끝 날짜",
                             value=datetime(2022,6,15),
                             min_value=start_date + timedelta(days=7),
                             max_value=start_date + timedelta(days=60))
with col2:
    media = st.selectbox('매체',('식물갤러리', '식물병원', '네이버카페', '네이버블로그', '네이버포스트'), help="확인하고 싶은 외부 데이터의 매체를 선택할 수 있습니다.")
with col3:
    temp_effect_size = st.slider('영향도 범위', 0, 100, 83, help="각 매체별 콘텐츠의 영향도(조회수)를 점수화한 값입니다. 0에 가까울 수록 영향도가 높습니다.")    
    effect_size = (100-int(temp_effect_size)*2)/100

standard_df, new_df = extract_df(df, media, start_date, end_date, effect_size)

##인풋 필터 - 워드클라우드용##
expander = st.expander('세부필터')
with expander:
    col1, col2= st.beta_columns(2)    
    with col1:
        type = st.selectbox('기준',('빈도(Count)','상대 빈도(TF-IDF)'), 
                            help="""Count: 단어의 빈도 순으로 크기를 설정합니다. TF-IDF: 전체 리뷰 내 빈도와 개별 리뷰 내 빈도를 모두 고려해 크기를 설정합니다.""")
    with col2:
        keyword_no = st.number_input("키워드 수", value=100, min_value=1, step=50,
                                     help="워드 클라우드를 통해 보고 싶은 단어의 갯수를 제어할 수 있습니다.")   
    stopwords = st_tags(
        label = '제거할 키워드',
        text = '직접 입력해보세요',
        value = ['식물', '화분'],
        suggestions = ['식물', '화분'],
        key = '1')

##워드 클라우드##
def get_tfidf_top_words(df, keyword_no):
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf = tfidf_vectorizer.fit_transform(df['제목+내용(nng)'].values)
    tfidf_df = pd.DataFrame(tfidf.todense(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_top_words = tfidf_df.sum().sort_values(ascending=False).head(keyword_no).to_dict()
    tfidf_top_words = dict(tfidf_top_words)
    return tfidf_top_words

def get_count_top_words(df, keyword_no):
    count_vectorizer = CountVectorizer(stop_words=stopwords)
    count = count_vectorizer.fit_transform(df['제목+내용(nng)'].values)
    count_df = pd.DataFrame(count.todense(), columns=count_vectorizer.get_feature_names_out())
    count_top_words = count_df.sum().sort_values(ascending=False).head(keyword_no).to_dict()
    return count_top_words

def generate_treemap(words):
    df = pd.DataFrame(list(words.items()), columns=['Word', 'Frequency'])
    fig = px.treemap(df, path=['Word'], values='Frequency',
                     color='Frequency', color_continuous_scale='Spectral')
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

try :
    if type == '빈도(Count)' :
        words = get_count_top_words(standard_df, keyword_no)
    else :
        words = get_tfidf_top_words(standard_df, keyword_no)

    #워드클라우드
    wc = WordCloud(background_color="white", colormap='Spectral', contour_color='steelblue', font_path="/app/busypeople-streamlit/font/Pretendard-Bold.otf")
    wc.generate_from_frequencies(words)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    
except :
    st.warning('영향도 범위를 조정해주세요! 데이터가 부족합니다')    

#########Section2 - 키워드 큐레이팅############
st.markdown("---")
st.markdown("<h2 id='section2'>💎 키워드 랭킹</h2>", unsafe_allow_html=True)

def get_top_keyword(df):
    content_list = []
    for values in df['제목+내용(nng)']:
        content_list.extend(eval(values))

    result_dict = {}
    for word in content_list:
        word_df = df[df['제목+내용(nng)'].str.contains(word)]
        if len(word_df) > 0:
            avg_views = word_df['영향도'].mean()
            urls = word_df['URL'].tolist()
            result_dict[word] = {'평균 영향도': float(avg_views), 'URL': urls}

    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1]['평균 영향도'], reverse=True))

    keywords = []
    avg_views = []
    urls = []

    for key, value in result_dict.items():
        keywords.append(key)
        avg_views.append(value['평균 영향도'])
        urls.append('\n'.join(value['URL']))

    result_df = pd.DataFrame({
        '평균 영향도': avg_views,
        '키워드': keywords,
        'URL': urls
    })

    return result_df

def new_keyword(standard_df, new_df):
    df['제목+내용(nng)'] = df['제목+내용(nng)'].map(to_list)
    content_list_1 = []
    content_list_1.extend(list(itertools.chain.from_iterable([eval(i) for i in standard_df['제목+내용(nng)']])))
    content_list_2 = []
    content_list_2.extend(list(itertools.chain.from_iterable([eval(i) for i in new_df['제목+내용(nng)']])))

    new_keywords = set(content_list_1) - set(content_list_2)   
    result_dict = {}
    # 이번달에만 있는 
    for word in new_keywords:
        word_df = standard_df[standard_df['제목+내용(nng)'].str.contains(word)]
        if len(word_df) > 0:
            avg_views = word_df['영향도'].mean()
            urls = word_df['URL'].tolist()
            result_dict[word] = {'평균 영향도': float(avg_views), 'URL': urls}
            
    # 조회수 높은순으로 정렬        
    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1]['평균 영향도'], reverse=True))    

    # 결과 딕셔너리를 데이터프레임으로 변환
    keywords = []
    avg_views = []
    urls = []
    
    for key, value in result_dict.items():
        keywords.append(key)
        avg_views.append(value['평균 영향도'])
        urls.append('\n'.join(value['URL']))
    
    result_df = pd.DataFrame({
        '평균 영향도': avg_views,
        '키워드': keywords,
        'URL': urls
    })
    return result_df

def rising_keyword(standard_df, new_df):
    # 데이터 합치기 
    df = pd.concat([standard_df, new_df])

    # 날짜 구하기
    이번주마지막날 = df['날짜'].max()
    이번주첫날 = (df['날짜'].max() - timedelta(days=7))
    지난주첫날 = 이번주첫날 - timedelta(days=7)
    
    이번주_df = df[(df['날짜'] > 이번주첫날) & (df['날짜'] <= 이번주마지막날)]
    지난주_df = df[(df['날짜'] > 지난주첫날) & (df['날짜'] <= 이번주첫날)]
        
    # 중복값 제거한 새로운 열 추가
    이번주_df = 이번주_df.copy()
    이번주_df['unique_content'] = 이번주_df['제목+내용(nng)'].apply(lambda x: ast.literal_eval(x))
    이번주_df['unique_content'] = 이번주_df['unique_content'].apply(lambda x: list(set(x)))

    지난주_df = 지난주_df.copy()
    지난주_df['unique_content'] = 지난주_df['제목+내용(nng)'].apply(lambda x: ast.literal_eval(x))
    지난주_df['unique_content'] = 지난주_df['unique_content'].apply(lambda x: list(set(x)))

    this_week_words = list(이번주_df['unique_content'].explode())
    last_week_words = list(지난주_df['unique_content'].explode())

    this_week_word_counts = Counter(this_week_words)
    last_week_word_counts = Counter(last_week_words)

    # 이번주와 지난주에 모두 언급된 단어를 모은 집합
    common_words = set(this_week_word_counts.keys()) & set(last_week_word_counts.keys())
    result = {}
    for word in common_words:
        # 해당 단어가 언급된 모든 URL을 리스트로 모음
        url_list = list(이번주_df.loc[이번주_df['unique_content'].apply(lambda x: word in x)]['URL'])
        # 영향도가 가장 높은 URL을 찾아서 출력
        url = max(url_list, key=lambda x: 이번주_df.loc[이번주_df['URL'] == x, '영향도'].iloc[0])
        increase_rate = (this_week_word_counts[word] - last_week_word_counts[word]) / this_week_word_counts[word]
        result[word] = {'상승률': increase_rate, 'URL': url}

    keywords = []
    ups = []
    urls = []

    for word, data in sorted(result.items(), key=lambda x: x[1]['상승률'], reverse=True):
        if data['상승률']>0:
            keywords.append(word)
            ups.append(data['상승률'])
            urls.append(data['URL'])

    result_df = pd.DataFrame({
        '상승률': ups,
        '키워드': keywords,
        'URL': urls
    })

    if len(result_df.index) >= 1 :
        return result_df

##키워드##
st.markdown(f"<style>{STYLE}</style>", unsafe_allow_html=True)
st.markdown(f"""<h3>📌 영향도 높은 키워드</h3>""", unsafe_allow_html=True)
try:
    top_keyword = get_top_keyword(standard_df)
    grouped_top_keyword = top_keyword.groupby('URL').agg({'키워드': list, '평균 영향도': 'first'}).reset_index()
    grouped_top_keyword = grouped_top_keyword[['평균 영향도', '키워드', 'URL']].sort_values(by='평균 영향도', ascending=False).reset_index(drop=True)
    grouped_top_keyword['평균 영향도'] = round(grouped_top_keyword['평균 영향도']*100,2)
    st.dataframe(grouped_top_keyword.head(20))
except:
    st.warning('해당 기간내 영향도 높은 키워드가 없습니다')

st.markdown(f"<style>{STYLE}</style>", unsafe_allow_html=True)
st.markdown(f"""<h3>⭐️ 신규 키워드</h3>""", unsafe_allow_html=True)
try:
    new_keyword = new_keyword(standard_df, new_df)
    grouped_new_keyword = new_keyword.groupby('URL').agg({'키워드': list, '평균 영향도': 'first'}).reset_index()
    grouped_new_keyword = grouped_new_keyword[['평균 영향도', '키워드', 'URL']].sort_values(by='평균 영향도', ascending=False).reset_index(drop=True)
    grouped_new_keyword['평균 영향도'] = round(grouped_new_keyword['평균 영향도'] * 100, 2)
    st.dataframe(grouped_new_keyword)
except:
    st.warning('해당 기간내 신규 키워드가 없습니다')

st.markdown(f"<style>{STYLE}</style>", unsafe_allow_html=True)
st.markdown(f"""<h3>📈 급상승 키워드</h3>""", unsafe_allow_html=True)
try:
    rising_keyword = rising_keyword(standard_df, new_df)
    grouped_rising_keyword = rising_keyword.groupby('URL').agg({'키워드': list, '상승률': 'first'}).reset_index()
    grouped_rising_keyword = grouped_rising_keyword[['상승률', '키워드', 'URL']].sort_values(by='상승률', ascending=False).reset_index(drop=True)
    grouped_rising_keyword['상승률'] = grouped_rising_keyword['상승률'].apply(lambda x: f'{round(x * 100, 2)}%')
    st.dataframe(grouped_rising_keyword)
except:
    st.warning('해당 기간내 급상승 키워드가 없습니다')





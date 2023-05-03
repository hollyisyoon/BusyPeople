import streamlit as st
import pandas as pd
import ast
import gensim
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

stopwords = [ '합니다', '하는', '할', '하고', '한다', 
             '그리고', '입니다', '그', '등', '이런', '및','제', '더','언늘','결국','생각','식물키',
             '감사','ㅋㅋ','진짜','완전','요ㅎ','사용','정도','엄마','아이','원래','식물']

def to_list(text):
    return ast.literal_eval(text)

def lda_modeling(tokens, num_topics=4, passes=10):
    # word-document matrix
    dictionary = gensim.corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]

    # Train the LDA model
    model = gensim.models.ldamodel.LdaModel(corpus,
                                            num_topics=num_topics,
                                            id2word=dictionary, # 단어매트릭스
                                            passes=passes, # 학습반복횟수
                                            random_state=100) 
    return model, corpus, dictionary

def print_topic_model(topics, rating):
    topic_values = []
    for topic in topics:
        topic_value = topic[1]
        topic_values.append(topic_value)
    topic_model = pd.DataFrame({"topic_num": list(range(1, len(topics) + 1)), "word_prop": topic_values})
    display(topic_model)

# pyLDAvis 오류(일단 보류 후 2차 개선)
# def lda_visualize(model, corpus, dictionary, rating):
#     pyLDAvis.enable_notebook()
#     result_visualized = gensim_models.prepare(model, corpus, dictionary, sort_topics=False)
#     pyLDAvis.save_html(result_visualized, 'lda_result.html')
#     pyLDAvis.display(result_visualized)


# 시각화1. 각 주제에서 상위 N개 키워드의 워드 클라우드
def topic_wordcloud(model):
    cloud = WordCloud(background_color='white',
                      font_path = "NanumGothic",
                      width=500,
                      height=500,
                      max_words=10,
                      colormap='tab10',
                      prefer_horizontal=1.0)
    
    topics = model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(6,6), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()


def get_topic_model(data, num_topics=4, passes=10, num_words=10):
    df = pd.read_csv(data)

    # 불용어 리스트
    stopwords = stopwords

    # 문장 리스트 생성
    reviews = []
    for i in range(len(df)):
        reviews.append(to_list(df['kha_nng_은어전처리_sentence'][i]))

    # 텍스트 데이터 전처리
    # 불용어 제거, 단어 인코딩 및 빈도수 계산
    tokens = []
    for review in reviews:
        token_review = [w for w in review if len(w) > 1 and w not in stopwords]
        tokens.append(token_review)

    # # LDA 모델링
    model, corpus, dictionary = lda_modeling(tokens, num_topics=num_topics, passes=passes)

    rating = 'pos' 
    topics = model.print_topics(num_words=num_words)
    print_topic_model(topics, rating)

    # 시각화 결과(pyLDAvis 오류(일단 보류 후 2차 개선)) 
    # lda_visualize(model, corpus, dictionary, rating)

    # 토픽별 워드클라우드 시각화
    topic_wordcloud(model)


############streamlit 구현 ##################
st.title('리뷰_토픽모델링')

# 데이터셋 선택 상자 만들기
dataset = st.selectbox('데이터셋 선택', ['자사(긍정리뷰)', '자사(부정리뷰)', '경쟁사(긍정리뷰)', '경쟁사(부정리뷰)'])

# 선택한 데이터셋에 따라 함수 호출
if dataset == '자사(긍정리뷰)':
    get_topic_model('/app/busypeople-stramlit/data/자사긍정리뷰.csv')
elif dataset == '자사(부정리뷰)':
    get_topic_model('/app/busypeople-stramlit/data/자사부정리뷰.csv')
elif dataset == '경쟁사(긍정리뷰)':
    get_topic_model('/app/busypeople-stramlit/data/경쟁사긍정리뷰.csv')
else:
    get_topic_model('/app/busypeople-stramlit/data/경쟁사부정리뷰.csv')


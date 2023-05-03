import warnings
warnings.filterwarnings("ignore", message="PyplotGlobalUseWarning")
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast
from gensim.models import Word2Vec
import time

#데이터 전처리
df = pd.read_csv('https://raw.githubusercontent.com/seoinhyeok96/BusyPeople/main/data/%ED%8A%B8%EB%A0%8C%EB%93%9C_%EC%A0%9C%EB%AA%A9%2B%EB%82%B4%EC%9A%A9.csv')
def to_list(text):
    return ast.literal_eval(text)
df['title+content'] = df['title+content'].map(to_list)

def get_words(df, col, keyword):
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

def main():
    # 폰트 설정
    plt.rc('font', family='NanumBarunGothic')
    st.title('Networkx 그려보자')
    message = st.text_area('🔮 무슨 키워드가 궁금하신가요 (1개만 입력해주떼용)')
    keyword = message
    if st.button('분석하기'):
        with st.spinner('분석 중입니다...'):
            # Define the data
            data = get_words(df, 'title+content', keyword)
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
    
            st.success(f"<{keyword}>에 대한 연관어 분석 결과입니다😀")
            plt.axis('off')
            st.pyplot()

            expander = st.expander('분석 결과 데이터 보기')
            with expander:
                show_modal(df_data)
            
if __name__ == '__main__':
    main()

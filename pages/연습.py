import streamlit as st

pre_input = ['식물', '화분', '사진', '오늘']
options = pre_input + ["기타"]
stopwords = st.multiselect("옵션을 선택하세요", options, default=["오늘"], allow_input=True)
st.write(stopwords)
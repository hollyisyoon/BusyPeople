import streamlit as st

pre_input = ['식물', '화분', '사진', '오늘']
pre_total = pre_input + ["기타"]
options = st.multiselect(
    'What are your favorite colors',
    pre_total,
    ['Yellow', 'Red'])

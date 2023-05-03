import streamlit as st

options = st.multiselect(
    'What are your favorite colors',
    ['식물', '화분', '사진', '오늘'],
    ['식물', '화분'], allow_input=True)

# import streamlit as st
# options = st.multiselect(
#     'What are your favorite colors',
#     ['Green', 'Yellow', 'Red', 'Blue'],
#     ['Yellow', 'Red'])
# st.write('You selected:', options)

import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

from wordcloud import WordCloud

import streamlit as st

df = pd.read_csv('https://github.com/seoinhyeok96/BusyPeople-stramlit/blob/main/data/%EC%96%B4%EA%B0%84.csv')

df.columns = ['index','count']

dict_0 = dict(zip(df['index'], df['count']))

# Create and generate a word cloud image:
wc = WordCloud(
    max_font_size=200,
    background_color='white',
    # relative_scaling=0.5,
    width=800,
    height=400,
    font_path='/app/busypeople-stramlit/font/NanumBarunGothic.ttf'
    )
cloud = wc.generate_from_frequencies(dict(dict_0))

fig, ax = plt.subplots(
    figsize=(12,12)
)

#주석

# Display the generated image:
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
st.pyplot(fig)

number = st.slider('Insert a number', min_value=10, step=1)
bar_df = df.sort_values(by=['count'],ascending=False).reset_index(drop=True).iloc[:number]
st.bar_chart(bar_df)


from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objs as go

word_list=[]
freq_list=[]
fontsize_list=[]
position_list=[]
orientation_list=[]
color_list=[]

for (word, freq), fontsize, position, orientation, color in wc.layout_:
    word_list.append(word)
    freq_list.append(freq)
    fontsize_list.append(fontsize)
    position_list.append(position)
    orientation_list.append(orientation)
    color_list.append(color)
    
# get the positions
x=[]
y=[]
for i in position_list:
    x.append(i[0])
    y.append(i[1])
        
# get the relative occurence frequencies
new_freq_list = []
for i in freq_list:
    new_freq_list.append(i*100)

trace = go.Scatter(x=x, 
                    y=y, 
                    textfont = dict(size=new_freq_list,
                                    color=color_list),
                    hoverinfo='text',
                    hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                    mode='text',  
                    text=word_list
                    )

layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                    'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})

fig = go.Figure(data=[trace], layout=layout)

fig

import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='markers', hoverinfo='x+y+z',
                                xaxis='x1', yaxis='y1'))

fig.add_trace(go.Scatter(x=[2, 3, 4], y=[5, 6, 7], mode='markers', hoverinfo='x+y+z',
                          xaxis='x2', yaxis='y2'))

fig.update_layout(xaxis=dict(domain=[0, 0.45]), yaxis=dict(domain=[0, 1]),
                  xaxis2=dict(domain=[0.55, 1]), yaxis2=dict(domain=[0, 1]))

fig

url = "https://public.tableau.com/views/KBOOPS/1?:language=ko-KR&:showVizHome=no&:embed=true"
html = '<iframe src=' + url + ' width=1000 height=800></iframe>'
st.markdown(html, unsafe_allow_html=True)

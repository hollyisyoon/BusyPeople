import wget

import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

from wordcloud import WordCloud

import streamlit as st

font_url = 'https://raw.githubusercontent.com/seoinhyeok96/BusyPeople/main/font/NanumBarunGothic.ttf'

wget.download(font_url)

df = pd.read_csv('https://raw.githubusercontent.com/seoinhyeok96/BusyPeople/main/data/%EC%96%B4%EA%B0%84.csv')

df.columns = ['index','count']

dict_0 = dict(zip(df['index'], df['count']))

# Create and generate a word cloud image:
wc = WordCloud(
    max_font_size=200,
    background_color='white',
    # relative_scaling=0.5,
    width=800,
    height=400,
    font_path='NanumBarunGothic.ttf'
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

html = '''<div class='tableauPlaceholder' id='viz1683037741970' style='position: relative'><noscript><a href='#'><img alt='대시보드 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;KB&#47;KBOOPS&#47;1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='KBOOPS&#47;1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;KB&#47;KBOOPS&#47;1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='ko-KR' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1683037741970');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1566px';vizElement.style.height='895px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1566px';vizElement.style.height='895px';} else { vizElement.style.width='100%';vizElement.style.height='2627px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>'''
st.markdown(html, unsafe_allow_html=True)
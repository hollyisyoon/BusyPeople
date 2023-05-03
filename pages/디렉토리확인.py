import os
import psutil

path = os.getcwd()

path

file_list = os.listdir(path)

file_list

메모리 = psutil.virtual_memory()

메모리

사용가능한 = 메모리.available / 1024**3

st.write(f'{round(사용가능한, 1)} GB')
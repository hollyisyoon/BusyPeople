import os
import psutil

path = os.getcwd()

path

file_list = os.listdir(path)

file_list

# 메모리 = psutil.virtual_memory()
# 메모리

# 사용가능한 = 메모리.used / 1024**3
# 사용가능한

제거할파일리스트 = [
"NanumBarunGothic (2).ttf",
"NanumBarunGothic (3).ttf",
"NanumBarunGothic (4).ttf",
"NanumBarunGothic (5).ttf",
"NanumBarunGothic.ttf",
"NanumBarunGothic (1).ttf",
"NanumBarunGothic (8).ttf",
"NanumBarunGothic (7).ttf",
"NanumBarunGothic (6).ttf"
]

for i in 제거할파일리스트:
    os.remove(i)
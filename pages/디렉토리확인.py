import os
import psutil

path = os.getcwd()

path

file_list = os.listdir(path)

file_list

메모리 = psutil.virtual_memory()

메모리

사용가능한 = 메모리.used / 1024**3

사용가능한
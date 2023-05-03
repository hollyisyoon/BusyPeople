import os

path = os.getcwd()

path

file_list = os.listdir(path)

file_list

파일용량 = os.system(f"du -ks -h *")

파일용량
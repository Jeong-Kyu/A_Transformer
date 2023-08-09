# Aihub dataset
import os
import pandas as pd

file_loc = 'C:/Users/jare9/OneDrive/바탕 화면/testdata/한국어-영어 번역(병렬) 말뭉치'
file_name = os.listdir(file_loc)
# 파일명
print(file_name)
# Daraframe형식으로 엑셀 파일 읽기
df = pd.read_excel(file_loc+'/'+file_name[0])

# 데이터 프레임 출력
# print(df.loc[0][0])
print(df.loc[0][1])
 

# import nltk
# nltk.download()

from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

print('단어 토큰화1 :',word_tokenize(df.loc[0][1]))
print('단어 토큰화1 :',word_tokenize(df.loc[0][2]))

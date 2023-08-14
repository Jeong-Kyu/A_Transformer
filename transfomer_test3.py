# Aihub dataset
import os
import pandas as pd
from torch.utils.data import random_split

from nltk.tokenize import word_tokenize


file_loc = 'C:/Users/jare9/OneDrive/바탕 화면/testdata/한국어-영어 번역(병렬) 말뭉치'
file_name = os.listdir(file_loc)

# Daraframe형식으로 엑셀 파일 읽기
df = pd.read_excel(file_loc+'/'+file_name[0])

all_df = []

for t in range(3):
    sub_df ={}
    sub_df['hg'] = word_tokenize(df.iloc[t][1])
    sub_df['eg'] = word_tokenize(df.iloc[t][2])
    all_df.append(sub_df)

train_data,test_data,valid_data = random_split(all_df,[0.7,0.2,0.1])

import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()
        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.fc_q == nn.Linear(hidden_dim, hidden_dim)
        self.fc_k == nn.Linear(hidden_dim, hidden_dim)
        self.fc_v == nn.Linear(hidden_dim, hidden_dim)

        self.fc_o == nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt()(torch.FloatTensor([self.head_dim])).to(device)


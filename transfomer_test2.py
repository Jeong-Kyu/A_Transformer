#BLEU Score
#pip install torchtext==0.6.0

import spacy

spacy_en = spacy.load('en_core_web_sm') #영어 토큰화
spacy_de = spacy.load('de_core_news_sm') #독일어 토큰화
spacy_ko = spacy.load('ko_core_news_sm') #독일어 토큰화

# 토큰 기능 써보기
# tokenized = spacy_en.tokenizer('I am a graduate student.')

# for i, token in enumerate(tokenized):
#     print(f'인덱스 {i}:{token.text}')

def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]

def tokenize_ko(text):
    return [token.text for token in spacy_ko.tokenizer(text)]

from torchtext.data import Field, BucketIterator

SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

from torchtext.datasets import Multi30k

train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=('.de','.en'),fields=(SRC, TRG))
print(f'학습 : {train_dataset.examples}, 평가 : {valid_dataset.examples}, 테스트 : {test_dataset.examples}')

print(vars(train_dataset.examples[30])['src'])
print(vars(train_dataset.examples[30])['trg'])

SRC.build_vocab(train_dataset, min_freq=2)
TRG.build_vocab(train_dataset, min_freq=2)
print(f'len(SRC): {len(SRC.vocab)}')
print(f'len(TRG): {len(TRG.vocab)}')

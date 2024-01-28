#모듈 설치
!pip install mxnet
!pip install gluonnlp pandas tqdm
!pip install sentencepiece
!pip install transformers==4.8.2
!pip install torch
!pip install gluonnlp==0.10.0

#KoBERT & 토크나이저 설치
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'

#모듈 참조 import
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from tqdm import tqdm, tqdm_notebook

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

#구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')

#GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#모델, 토크나이저, 단어화? 불러오기
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

#하이퍼 파라미터 설정
max_len = 64            #한번에 처리할 최대 입력 텍스트 시퀀스 길이
batch_size = 64         #한번에 학습하는 데이터 수
warmup_ratio = 0.1      #학습 초기에 학습률을 점진적으로 증가시키는 비율
num_epochs = 5          #전체 데이터셋에 대한 훈련 횟수
max_grad_norm = 1       #이 값을 초과하는 그라디언트는 비율에 따라 줄어듬, 그라디언트 = 폭발 문제 방지??
log_interval = 200      #학습 중 로그를 출력하는 간격
learning_rate =  5e-5   #학습률, 최적화 과정의 매 스탭마다 가중치를 얼마나 조정할지 설정, 너무 낮으면 학습이 느려짐

#데이터셋 정의
class BERTDataset(Dataset):
    #sent_idx = 문장 데이터 위치 인덱스, label_idx = 라벨 위치 인덱스, bert_tokenizer = BERT모델의 토크나이저 인스턴스
    #vocab = BERT 모델 어휘집, max_len = 문장 최대 길이(길면 자르고 짧으면 패딩, pad = 패딩 여부, pair = 두 개의 입력을 사용하는 경우 ex)질문-응답 쌍
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
        #BERTSentenceTransform = 텍스트를 모델이 이해할 수 있는 형태로 변환(토크나이징, 시퀀스 길이 맞추기, 토큰 타입 ID 생성 등)
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]
    #데이터셋의 특정 인덱스에 해당하는 데이터 반환
    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
      
    #데이터셋의 전체 길이(데이터 포인트 수) 반환
    def __len__(self):
        return (len(self.labels))

#데이터셋 다운로드
!wget https://www.dropbox.com/s/374ftkec978br3d/ratings_train.txt?dl=1
!wget https://www.dropbox.com/s/977gbwh542gdy94/ratings_test.txt?dl=1

#TSV = 탭으로 구분된 파일, field_indices = [1,2] 사용할 열 인덱스 지정 여기서는 2, 3번째, num_discard_samples=1파일의 첫 번째 줄 무시
dataset_train = nlp.data.TSVDataset("ratings_train.txt?dl=1", field_indices=[1,2], num_discard_samples=1)
dataset_test = nlp.data.TSVDataset("ratings_test.txt?dl=1", field_indices=[1,2], num_discard_samples=1)

#데이터셋 토크나이저
tok = tokenizer.tokenize

data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, vocab, max_len, True, False)

#데이터로드
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

#모델 정의
class BERTClassifier(nn.Module):  #nn.Module 클래스를 상속받아 정의
    #bert = 사전 학습된 모델, num_classes = 분류할 클래스 개수, dr_rate = 드롭아웃(과적합 방지) 비율
    def __init__(self, bert, hidden_size = 768, num_classes=2, dr_rate=None, params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        #classifier = 끝에 존재하는 완전 연결 레이어, BERT 출력을 최종 분류 클래스 수에 맞게 변환
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    #유효한 길이인 'valid_length'를 사용해 어텐션 마스크를 생성해 모델이 패딩된 부분을 무시하도록 도와줌
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    #순전파
    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        #pooler = 모델 마지막의  hidden state에서 나온 값으로, 분류 작업을 위한 특징 벡터로 사용
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

#모델 선언
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

#최적화할 파라미터 설정 
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

#최적화 알고리즘 및 손실 함수(역전파 알고리즘) 정의
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

#전체 학습 과정에서의 총 스탭 수 계산 = 배치사이즈 / 1에포크 * 에포크 수
t_total = len(train_dataloader) * num_epochs

#스케쥴러, 초기에 워밍업으로 학습률 증가 후 코사인 함수에 따라 학습률 점진적 감소
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#학습 검증 계산
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

#트레이닝 및 출력
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
      
        #역전파 및 가중치 갱신
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
      
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

    #모델 평가
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))

# 라이브러리 import
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import pandas as pd
import numpy as np


# 구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')
'''
if torch.cuda.is_available():
    model = model.cuda()
'''

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 데이터셋 클래스 정의  / 질문, 답변, 라벨링이 탭으로 구분된 데이터에 대한 데이터셋 클래스
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 데이터 로드
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                q, a, label = line.strip().split('\t')
                self.data.append((q, a, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, a, label = self.data[idx]

        # 질문-답변쌍 토크나이
        encoded_q = self.tokenizer.encode(self.tokenizer.bos_token + q + self.tokenizer.eos_token, 
                                          add_special_tokens=False)
        encoded_a = self.tokenizer.encode(a + self.tokenizer.eos_token, 
                                          add_special_tokens=False)
        
        # 토큰 길이가 최대 길이를 초과하지 않도록 조정
        encoded_pair = encoded_q + encoded_a[:max(self.max_len - len(encoded_q), 0)]

        # 패딩 처리
        padding_length = self.max_len - len(encoded_pair)
        encoded_pair += [self.tokenizer.pad_token_id] * padding_length

        # PyTorch 모델에 입력하기 위해선 tensor 형태로 변환해야 함
        input_ids = torch.tensor(encoded_pair, dtype=torch.long)

        return input_ids, torch.tensor(label, dtype=torch.long)


import torch
from torch import nn

# 모델 정의
class KoGPT2ChatModel(nn.Module):
    def __init__(self, kogpt2_model):
        super(KoGPT2ChatModel, self).__init__()
        self.kogpt2 = kogpt2_model

    def forward(self, input_ids):
        # KoGPT2 모델의 출력을 가져옴
        output = self.kogpt2(input_ids=input_ids, return_dict=True)
        
        # 로그 소프트맥스를 통해 예측 확률을 반환
        logits = output.logits
        return logits

# 모델 인스턴스 생성
model = KoGPT2ChatModel(model)

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss


# 데이터셋 로드
train_dataset = ChatDataset(filepath='train_dataset.txt', tokenizer=tokenizer, max_len=512)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 모델 학습
train(5, model, train_dataloader, optimizer, loss_fn)

# 답변 생성 클래스
def generate_response(question, model, tokenizer, max_len=512):
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():
        input_ids = tokenizer.encode(tokenizer.bos_token + question + tokenizer.eos_token, return_tensors='pt')
      
        output_ids = model.generate(input_ids, max_length=max_len, num_return_sequences=1)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 테스트
question = "오늘 기분은 어때?"
response = generate_response(question, model, tokenizer)
print(response)

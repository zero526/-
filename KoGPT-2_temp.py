# 필요한 라이브러리 설치
!pip install torch
!pip install transformers==4.8.2
!pip install pandas

# 라이브러리 import
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import pandas as pd
import numpy as np

#구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/gdrive')

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# KoGPT2 토크나이저 및 모델 로드
model_name = "skt/kogpt2-base-v2"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 특수 토큰 설정
special_tokens = {
    "bos_token": "</s>", 
    "eos_token": "</s>", 
    "unk_token": "<unk>",
    "pad_token": "<pad>", 
    "mask_token": "<mask>"
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# GPU 설정 (CUDA가 사용 가능한 경우)
if torch.cuda.is_available():
    model = model.cuda()

#데이터셋 클래스 정의
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

        # 질문-답변쌍 토크나이징
        encoded_q = self.tokenizer.encode(self.tokenizer.bos_token + q + self.tokenizer.eos_token,
                                          add_special_tokens=False)
        encoded_a = self.tokenizer.encode(a + self.tokenizer.eos_token,
                                          add_special_tokens=False)

        # 토큰 길이가 최대 길이를 초과하지 않도록 조정
        encoded_pair = encoded_q + encoded_a[:max(self.max_len - len(encoded_q), 0)]

        # 패딩 처리
        padding_length = self.max_len - len(encoded_pair)
        encoded_pair += [self.tokenizer.pad_token_id] * padding_length

        # 주의: PyTorch 모델에 입력하기 위해선 tensor 형태로 변환해야 함
        input_ids = torch.tensor(encoded_pair, dtype=torch.long)

        # attention_mask 생성
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in encoded_pair]
        
        # 리스트를 PyTorch 텐서로 변환
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

        #return input_ids, torch.tensor(label, dtype=torch.long)


import torch
from torch import nn

class KoGPT2ChatModel(nn.Module):
    def __init__(self, kogpt2_model):
        super(KoGPT2ChatModel, self).__init__()
        self.kogpt2 = kogpt2_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.kogpt2(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits if isinstance(outputs, dict) else outputs[0]
        
        return logits

    def generate(self, input_ids, **kwargs):
        return self.kogpt2.generate(input_ids, **kwargs)

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# 모델 인스턴스 생성
model = KoGPT2ChatModel(model)

# 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr=5e-5)

# 손실 함수 설정
# KoGPT2는 LMHeadModel이므로, CrossEntropyLoss를 사용합니다.
# 이 때, 라벨이 없는 부분은 손실 계산에서 제외하기 위해 ignore_index 파라미터를 설정합니다.
loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# GPU 사용 설정
if torch.cuda.is_available():
    model = model.cuda()
    loss_fn = loss_fn.cuda()

def train(epoch, model, dataloader, optimizer, loss_fn):
    model.train()
    for _ in range(epoch):
        for input_ids, attention_mask, labels in dataloader:
            # GPU 설정
            if torch.cuda.is_available():
                input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()

            optimizer.zero_grad()

            # 모델의 출력 계산
            outputs = model(input_ids, attention_mask=attention_mask)

            # 여기서 outputs는 [batch_size, sequence_length, num_classes] 형태를 가질 것으로 예상
            # 첫 번째 토큰의 예측만 사용하여 손실을 계산하기 위해 조정
            outputs = outputs[:, 0, :]

            # 손실 계산 및 역전파
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

# 데이터셋 로드
train_dataset = ChatDataset(filepath='/content/gdrive/My Drive/data/test_data.txt', tokenizer=tokenizer, max_len=512)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 모델 학습
train(5, model, train_dataloader, optimizer, loss_fn)


def generate_response(question, model, tokenizer, max_len=512):
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():
        # 현재 모델이 사용하는 디바이스 확인
        device = next(model.parameters()).device
        
        # 입력 텐서를 모델과 동일한 디바이스로 이동
        input_ids = tokenizer.encode(tokenizer.bos_token + question + tokenizer.eos_token, return_tensors='pt').to(device)
        
        # 답변 생성
        output_ids = model.generate(input_ids, max_length=max_len, num_return_sequences=1)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 테스트
question = "기분이 안 좋아"
response = generate_response(question, model, tokenizer)
print(response)

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

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# KoGPT2 토크나이저 및 모델 로드
model_name = "skt/kogpt2-base-v2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
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
'''
class QADataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                q, a = line.strip().split('\t')  # 질문과 답변 쌍 읽기
                self.data.append((q, a))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, a = self.data[idx]
        encoded_pair = self.tokenizer.encode_plus(q, a, max_length=self.max_len, padding='max_length', truncation=True, add_special_tokens=True, return_tensors="pt")
        input_ids = encoded_pair['input_ids'].squeeze(0)
        attention_mask = encoded_pair['attention_mask'].squeeze(0)
        return input_ids, attention_mask
'''
class QADataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                q, a = line.strip().split('\t')  # 질문과 답변 쌍 읽기
                self.data.append((q, a))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, a = self.data[idx]
        # 질문과 답변에 BOS와 EOS 토큰 명시적 추가
        q_encoded = tokenizer.encode(tokenizer.bos_token + q + tokenizer.eos_token, max_length=self.max_len//2, truncation=True, return_tensors="pt")
        a_encoded = tokenizer.encode(tokenizer.bos_token + a + tokenizer.eos_token, max_length=self.max_len//2, truncation=True, return_tensors="pt")

        # 질문과 답변 인코딩 결합
        input_ids = torch.cat([q_encoded, a_encoded[:, 1:]], dim=1)  # <eos> 중복 제거
        input_ids = input_ids.squeeze(0)

        # attention_mask 생성
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # 모든 토큰에 대해 attention 적용
        attention_mask[input_ids == tokenizer.pad_token_id] = 0  # pad 토큰에 대해서는 attention 비적용

        return input_ids, attention_mask



''' 데이터셋 BOS와 EOS 토큰 ID 확인 테스트
# 첫 번째 데이터 샘플을 가져와 인코딩된 input_ids 확인
sample_input_ids, sample_attention_mask = QADataset(filepath='your_dataset_path.txt', tokenizer=tokenizer, max_len=128)[0]

# BOS, EOS 토큰 ID 확인
print("BOS Token ID:", tokenizer.bos_token_id, "EOS Token ID:", tokenizer.eos_token_id)
print("Encoded Input IDs:", sample_input_ids)

# 인코딩된 텍스트 확인
decoded_text = tokenizer.decode(sample_input_ids)
print("Decoded Text:", decoded_text)
'''

# 모델 정의
import torch
from torch import nn

class KoGPT2ChatModel(nn.Module):
    def __init__(self, kogpt2_model):
        super(KoGPT2ChatModel, self).__init__()
        self.kogpt2 = kogpt2_model

    def forward(self, input_ids, attention_mask=None):
        # KoGPT2 모델의 출력을 가져옴
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

from torch.nn.utils.rnn import pad_sequence

def train(epoch, model, dataloader, optimizer, loss_fn):
    model.train()
    for _ in range(epoch):
        for input_ids, attention_mask in dataloader:  # labels 변수 제거
            if torch.cuda.is_available():
                input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()

            optimizer.zero_grad()

            # 모델의 출력 계산
            outputs = model(input_ids, attention_mask=attention_mask)

            # 손실 함수 계산을 위해 입력 시퀀스를 타겟으로 사용
            # 입력의 첫 번째 토큰을 제외한 나머지를 타겟으로 사용합니다.
            # 여기서는 outputs[:, :-1]를 사용하여 모델이 출력한 로짓,
            # input_ids[:, 1:]를 사용하여 실제 타겟 토큰입니다.
            loss = loss_fn(outputs[:, :-1].reshape(-1, outputs.size(-1)), input_ids[:, 1:].reshape(-1))

            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

# tensor 크기 맞춤
def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]

    # 입력 ID와 어텐션 마스크를 패딩
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return input_ids_padded, attention_masks_padded

# 데이터셋 로드 batch_size = 너무크면 메모리 오버플로우, 작으면 학습시간 길어짐
# DataLoader에 collate_fn 전달
train_dataset = QADataset(filepath='/content/gdrive/My Drive/data/trans_csv.txt', tokenizer=tokenizer, max_len=128)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
#train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 모델 학습
epoch = 10
train(epoch, model, train_dataloader, optimizer, loss_fn)

def generate_response(question, model, tokenizer, max_len=128):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        input_ids = tokenizer.encode(tokenizer.bos_token + question + tokenizer.eos_token, return_tensors='pt').to(device)
        #output_ids = model.generate(input_ids, max_length=max_len, num_return_sequences=1)
        output_ids = model.generate(input_ids, min_length=10, max_length=50, eos_token_id=tokenizer.eos_token_id, early_stopping=True, no_repeat_ngram_size=2, length_penalty=2.0, temperature=0.7, top_k=50, top_p=0.95)

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 예시 사용
question = "요즘 의욕이 없어."
response = generate_response(question, model, tokenizer)
print(response)

'''
# pt 형태로 모델 저장
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

save_model(model, '/content/gdrive/My Drive/data/KoGPT2_003.pt')

# pkl 형태로 모델 저장
import torch
import pickle

def save_model_as_pkl(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

save_model_as_pkl(model, '/content/gdrive/My Drive/data/KoGPT2_003.pkl')
'''

# 라이브러리 설치
!pip install torch==2.2.0
!pip install transformers==4.37.2

# 라이브러리 import
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

#구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/gdrive')

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# KoGPT2 모델 및 토크나이저 로드
model_name = "skt/kogpt2-base-v2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

pad_tokens_dict = {'pad_token': '&amp;amp;lt;pad&amp;amp;gt;'}
num_added_toks = tokenizer.add_special_tokens(pad_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# GPU 설정
if torch.cuda.is_available():
    model = model.cuda()
    
#데이터셋 클래스 정의
class QADataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=512):
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
        
        # 질문과 답변에 BOS, EOS 토큰 명시적 추가
        q_encoded = tokenizer.encode(tokenizer.bos_token + q + tokenizer.eos_token, max_length=self.max_len//2, truncation=True, return_tensors="pt")
        a_encoded = tokenizer.encode(tokenizer.bos_token + a + tokenizer.eos_token, max_length=self.max_len//2, truncation=True, return_tensors="pt")
        
        # 질문과 답변 인코딩 결합
        input_ids = torch.cat([q_encoded, a_encoded[:, 1:]], dim=1)  # &amp;amp;lt;bos&amp;amp;gt;, &amp;amp;lt;eos&amp;amp;gt; 중복 제거
        input_ids = input_ids.squeeze(0)
        
        # attention_mask 생성
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # 모든 토큰 attention 적용
        attention_mask[input_ids == tokenizer.pad_token_id] = 0  # pad 토큰 attention 비적용
        
        return input_ids, attention_mask
        
# 모델 정의
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
loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# GPU 사용 설정
if torch.cuda.is_available():
    model = model.cuda()
    loss_fn = loss_fn.cuda()
    
from torch.nn.utils.rnn import pad_sequence

def train(epoch, model, dataloader, optimizer, loss_fn):
    model.train()
    for _ in range(epoch):
        for input_ids, attention_mask in dataloader:
            if torch.cuda.is_available():
                input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
                
            #옵티마이저 초기화
            optimizer.zero_grad()
            
            # 모델의 출력 계산
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs[:, :-1].reshape(-1, outputs.size(-1)), input_ids[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            print(f"Loss: {loss.item()}")
        print("★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆")
        
def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    
    # 입력 ID와 어텐션 마스크를 패딩
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return input_ids_padded, attention_masks_padded
    
# 데이터셋 로드 batch_size = 너무크면 메모리 오버플로우, 작으면 학습시간 길어짐

#싱글턴 데이터셋
#train_dataset = QADataset(filepath='/content/gdrive/My Drive/data/wellnessQA_01.txt', tokenizer=tokenizer, max_len=512)
#train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

#멀티턴 데이터셋
train_dataset = QADataset(filepath='/content/gdrive/My Drive/data/최종 데이터.txt', tokenizer=tokenizer, max_len=512)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# 모델 학습
epoch = 2
train(epoch, model, train_dataloader, optimizer, loss_fn)
def generate_response(question, model, tokenizer):
    model.eval()
	with torch.no_grad():
        device = next(model.parameters()).device
        input_ids = tokenizer.encode(tokenizer.bos_token + question + tokenizer.eos_token, return_tensors='pt').to(device)
        output_ids = model.generate(input_ids, min_length=10, max_length=512, eos_token_id=tokenizer.eos_token_id, early_stopping=True, no_repeat_ngram_size=2, length_penalty=2.0, temperature=0.7, top_k=50, top_p=0.95)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
# 예시 사용
question = "요즘 너무 우울해"

response = generate_response(question, model, tokenizer)

# response에서 question부분 중복 제거
if response.startswith(question):
    # question 길이만큼 response 앞부분 제거
    response = response[len(question):].strip()
else:
    response = response
    
print(question)
print(response)

# pt 형태로 모델 저장
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
save_model(model, '/content/gdrive/My Drive/data/KoGPT2_multi_2.pt')

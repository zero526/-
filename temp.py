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

#베이스 모델 불러오기
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

#하이퍼 파라미터 설정
max_len = 256
batch_size = 4
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

#데이터셋 정의
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]
    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

#데이터셋 지정
answer_dataset = nlp.data.TSVDataset("/content/drive/data/wellness_dialog_answer.txt", field_indices=[0,1])
category_dataset = nlp.data.TSVDataset("/content/drive/data/wellness_dialog_category.txt", field_indices=[0,1])
text_classification_all_dataset = nlp.data.TSVDataset("/content/drive/data/wellness_dialog_for_text_classification_all.txt", field_indices=[0,1])

#데이터셋 토크나이저
tok = tokenizer.tokenize

answer_train = BERTDataset(answer_dataset, 0, 1, tok, vocab, max_len, True, False)
category_train = BERTDataset(category_dataset, 0, 1, tok, vocab, max_len, True, False)
all_train = BERTDataset(text_classification_all_dataset, 0, 1, tok, vocab, max_len, True, False)

#데이터로드
answer_dataloader = torch.utils.data.DataLoader(answer_train, batch_size=batch_size, num_workers=4)
category_dataloader = torch.utils.data.DataLoader(category_train, batch_size=batch_size, num_workers=4)
all_dataloader = torch.utils.data.DataLoader(all_train, batch_size=batch_size, num_workers=4)

#모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size = 768, num_classes = 359, dr_rate = None, params = None): #num_classes = 분류할 클래스 개수
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

'''
import torch
import torch.nn as nn
from kobert_transformers import get_kobert_model
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel

from model.configuration import get_kobert_config

class KoBERTforSequenceClassfication(BertPreTrainedModel):
    def __init__(self,
                 num_labels=359,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 ):
        super().__init__(get_kobert_config())

        self.num_labels = num_labels
        self.kobert = get_kobert_model()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.kobert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
'''

#모델 선언
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

#최적화할 파라미터 설정 
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

#최적화할 파라미터 설정 
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# + 트레이닝 & 테스트

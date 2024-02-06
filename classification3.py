'''
학습을 위한 데이터 전처리 코드 中 1

'''

import csv
import pandas as pd

# Google Drive 마운트 (코랩 환경에서 필요한 경우)
from google.colab import drive
drive.mount('/content/drive')

# 파일 경로 설정 (실제 경로로 변경 필요)
csv_file_path = '/content/drive/My Drive/data/ChatbotData(0일상다반사1이별부정2사랑긍정).csv'
output_file_path = '/content/drive/My Drive/data/trans_csv.txt'

# CSV 파일 읽기
df = pd.read_csv(csv_file_path, encoding='cp949')  # 'cp949' 인코딩을 사용하여 파일 읽기

# 'Q'와 'A' 컬럼만 선택하고, 각 행을 탭으로 구분된 문자열로 변환
transformed_lines = df['Q'] + '\t' + df['A']

# 변환된 데이터를 텍스트 파일로 저장
with open(output_file_path, 'w', encoding='utf-8') as f:
    for line in transformed_lines:
        f.write(line + '\n')

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# 파일 경로 설정 (실제 경로로 변경 필요)
input_file_path = '/content/drive/My Drive/data/wellnessQA.txt'
output_file_path = '/content/drive/My Drive/data/wellnessQA_01.txt'

# CSV 파일 읽기
# 마지막 컬럼은 무시하고 싶으므로 usecols를 사용하여 필요한 컬럼만 읽어옵니다.
df = pd.read_csv(input_file_path, usecols=[0, 1], encoding='utf-8')

# 'Q'와 'A' 컬럼만 선택하고, 각 행을 탭으로 구분된 문자열로 변환
transformed_data = df.iloc[:, 0] + '\t' + df.iloc[:, 1]

# 변환된 데이터를 텍스트 파일로 저장
with open(output_file_path, 'w', encoding='utf-8') as f:
    for line in transformed_data:
        f.write(line + '\n')

'''
학습을 위한 데이터 전처리 코드 中 1

'''


from google.colab import drive
drive.mount('/content/drive')

# 파일 경로 및 이름
file_path = '/content/drive/My Drive/data/test_data2.txt'
output_file_path = '/content/drive/My Drive/data/trans_test_data2.txt'

# 파일 읽기 및 변환
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 변환된 데이터를 저장할 리스트
transformed_lines = []

# 각 줄을 순회하며 변환
i = 0
while i < len(lines):
    # 현재 줄과 다음 줄을 읽어서 번호를 제거하고 탭으로 구분된 문자열 생성
    line = lines[i].strip()
    if i+1 < len(lines):  # 다음 줄이 존재하는지 확인
        next_line = lines[i+1].strip()
        # '1'로 시작하는 현재 줄과 '2'로 시작하는 다음 줄을 처리
        if line.startswith('1\t') and next_line.startswith('2\t'):
            transformed_line = f"{line[2:]}\t{next_line[2:]}"
            transformed_lines.append(transformed_line)
            i += 2  # 다음 대화 쌍으로 넘어감
            continue
    i += 1

# 변환된 데이터를 출력 파일에 저장
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in transformed_lines:
        output_file.write(line + '\n')

print('변환된 파일이 저장되었습니다.')

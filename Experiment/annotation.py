import pandas as pd
import os

# 사진 파일이 있는 최상위 폴더 경로
train_folder = "C:\\Users\\xianm\\Downloads\\HB\\유형별 두피 이미지\\Training" 
test_folder = "C:\\Users\\xianm\\Downloads\\HB\\유형별 두피 이미지\\Validation" 

# 라벨을 번호로 매핑
label_mapping = {
    "모낭사이홍반": 0,
    "모낭홍반농포": 1,  # 모낭사이홍반과 모낭홍반농포는 같은 라벨 0을 사용
    "미세각질": 2,
    "비듬": 3,
    "피지과다": 4,
    "탈모": 5
}

# 레벨 키워드 및 해당 값 매핑
level_keywords = {"0.양호": 0, "1.경증": 1, "2.중등도": 2,"3.중증": 3}

# 결과를 저장할 DataFrame 초기화
df = pd.DataFrame(columns=["ID", "LABEL", "LEVEL", "PATH"])

# 사진 폴더 순회
for subdir, dirs, files in os.walk(train_folder):
    folder_name = os.path.basename(subdir)
    # 폴더 이름에서 LABEL과 LEVEL 추출
    label_num = None  # 라벨 번호
    level = None
    for label, num in label_mapping.items():
        if label in folder_name:
            label_num = num  # 라벨 이름에 해당하는 번호 할당
            break
    for level_keyword, level_val in level_keywords.items():
        if level_keyword in folder_name:
            level = level_val
            break

    # 해당 폴더의 모든 이미지 파일에 대해 정보 수집
    if label_num is not None and level is not None:
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                df.loc[len(df)] = {
                    "ID": file,
                    "LABEL": label_num,
                    "LEVEL": level,
                    "PATH": os.path.join(subdir, file)
                }
# 결과 CSV 파일로 저장
output_csv = "Train_annotations.csv" 
print("done") 
df.to_csv(output_csv, index=False)

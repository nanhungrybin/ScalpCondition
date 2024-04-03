import pandas as pd
import os

# 사진 파일이 있는 최상위 폴더 경로
train_folder = "/data/hbsuh/ScalpCondition/Training" 
test_folder = "/data/hbsuh/ScalpCondition/Validation" 

# # 라벨을 번호로 매핑
# label_mapping = {
#     "모낭사이홍반": 0,
#     "모낭홍반농포": 1,  # 모낭사이홍반과 모낭홍반농포는 같은 라벨 0을 사용
#     "미세각질": 2,
#     "비듬": 3,
#     "피지과다": 4,
#     "탈모": 5
# }

# # 레벨 키워드 및 해당 값 매핑
# level_keywords = {"0.양호": 0, "1.경증": 1, "2.중등도": 2,"3.중증": 3}

# 결과를 저장할 DataFrame 초기화
df = pd.DataFrame(columns=["ID", "LABEL", "LEVEL", "MULTI","PATH"])


# label과 level에 대한 가능한 값들
possible_labels = [1, 2, 3, 4, 5, 6]
possible_levels = [1, 2, 3, 4]

# 모든 조합에 대한 multi-label 설정
multi_labels = {}
label_count = 0

for label_num in possible_labels:
    for level in possible_levels:
        multi_labels[(label_num, level)] = chr(ord('A') + label_count)
        label_count += 1

for subdir, dirs, files in os.walk(test_folder): ######## change folder
    for folder_name in dirs:

        # 폴더 이름에서 LABEL과 LEVEL 추출
        label_num = int(folder_name.split("_")[0])
        level = int(folder_name.split("_")[1])

        # label과 level에 따른 multi 레이블 값 설정
        if (label_num, level) in multi_labels:
            multi_label = multi_labels[(label_num, level)]

        # 해당 폴더의 모든 이미지 파일에 대해 정보 수집
        folder_path = os.path.join(subdir, folder_name)
        for file in os.listdir(folder_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                df.loc[len(df)] = {
                    "ID": file,
                    "LABEL": label_num,
                    "LEVEL": level,
                    "MULTI": multi_labels,
                    "PATH": os.path.join(folder_path, file)
                }

# 결과 CSV 파일로 저장
output_csv = "/home/goldlab/Project/Experiment/Test_annotations.csv" 
print("done") 
df.to_csv(output_csv, index=False)

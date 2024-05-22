import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

# class MultiDataset(Dataset):
#     def __init__(self, df, transform=None):
#         self.image_id = df['ID'].values
#         self.df = df
#         self.labels = df.iloc[:, 3:].values
#         self.transform = transform

#         # # 데이터 타입을 float32로 변환합니다 (필요하다면)
#         # if self.labels.dtype == object:  # np.object 대신 object를 사용
#         #     self.labels = np.array(self.labels.tolist(), dtype=float)  # 리스트를 통해 재구성

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):

#         image_path = self.df.iloc[idx, self.df.columns.get_loc("PATH")]
#         image = cv2.imread(image_path)
#         label = self.labels[idx]
 
#         if image is None:
#             print(f"The image at path {image_path} was not found.")
#             return None
            
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # # dtype이 object일 경우 float32로 변환
#         # if isinstance(self.labels[idx], np.ndarray) and self.labels[idx].dtype == object:
#         #     label = torch.tensor(self.labels[idx].tolist(), dtype=torch.float32)
#         # else:
#         #     label = torch.tensor(self.labels[idx], dtype=torch.float32)

#         label = torch.tensor(label, dtype=torch.float32)

#         # 라벨에 따른 적절한 변환
        
#         transform = self.transform[label]  # label에 해당하는 적절한 증강 전략 사용

#         if transform:
#             augmented = transform(image=image)
#             image = augmented['image']

#         # if self.transform:
#         #     augmented = self.transform(image=image)
#         #     image = augmented['image']

#         return image, label




label_names = ["Perifollicular erythema", "Follicular erythema with pustules", "Fine scaling", "Dandruff", "Excessive sebum", "Hair loss", "Healthy"]

# min class 기준으로 균등 증강
class MultiDataset(Dataset):
    def __init__(self, df, transform=None, augmentation_strategies=None):
        self.image_id = df['ID'].values
        self.df = df.copy() # 원본 데이터에 영향을 주지 않도록 복사본을 사용
        self.labels = df.iloc[:, 3:].values
        self.transform = transform
        self.augmentation_strategies = augmentation_strategies  # 증강 전략 추가

        # 필요한 증강량 계산
        target_samples = 13000
        augmentation_needed = {
            'Follicular erythema with pustules': max(0, target_samples - df['Follicular erythema with pustules'].sum()),
            'Healthy': max(0, target_samples - df['Healthy'].sum())
        }

        # 데이터 증강 로직 (간단한 복제로 예시)
        for label, extra_needed in augmentation_needed.items():
            # 필요한 수량만큼 샘플 선택하여 복제
            samples_to_augment = df[df[label] == 1].sample(n=extra_needed, replace=True)
            df = pd.concat([df, samples_to_augment])

        # 증강 후 각 라벨별 데이터 개수 다시 계산
        self.updated_class_counts = df.iloc[:, 3:].sum()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, self.df.columns.get_loc("PATH")]
        image = cv2.imread(image_path)
        if image is None:
            print(f"The image at path {image_path} was not found.")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # 활성화된 라벨에 해당하는 변환 선택
        # 라벨별 가중치를 고려하거나 활성화된 라벨을 기반으로 증강 전략을 선택
        active_labels = (label == 1).nonzero(as_tuple=True)[0]

        if len(active_labels) > 0:
            selected_label = label_names[active_labels[0].item()] # 활성화된 라벨
            if self.augmentation_strategies and selected_label in self.augmentation_strategies:  # 증강 전략이 None이 아니면 사용
                selected_transform = self.augmentation_strategies[selected_label]
            else:
                selected_transform = self.transform  # 테스트 일때

        if selected_transform:
            augmented = selected_transform(image=image)
            image = augmented['image'] if 'image' in augmented else augmented

        return image, label
    

# # max class 기준으로 균등 증강
# class MultiDataset(Dataset):
#     def __init__(self, df, transform=None, augmentation_strategies=None):
#         self.image_id = df['ID'].values
#         self.df = df.copy() # 원본 데이터에 영향을 주지 않도록 복사본을 사용
#         self.labels = df.iloc[:, 3:].values
#         self.transform = transform
#         self.augmentation_strategies = augmentation_strategies  # 증강 전략 추가

#         # 클래스별 샘플 수 계산
#         class_counts = df.iloc[:, 3:].sum()

#         # 타겟 샘플 수 설정 (가장 많은 클래스의 샘플 수를 기준으로)
#         target_samples = class_counts.max()

#         # 각 클래스별 필요한 증강량 계산
#         augmentation_needed = {label: max(0, target_samples - count) for label, count in class_counts.items()}

#         # 데이터 증강 로직 (간단한 복제로 예시)
#         for label, extra_needed in augmentation_needed.items():
#             # 필요한 수량만큼 샘플 선택하여 복제
#             samples_to_augment = df[df[label] == 1].sample(n=extra_needed, replace=True)
#             df = pd.concat([df, samples_to_augment])

#         # 증강 후 각 라벨별 데이터 개수 다시 계산
#         self.updated_class_counts = df.iloc[:, 3:].sum()

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         image_path = self.df.iloc[idx, self.df.columns.get_loc("PATH")]
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"The image at path {image_path} was not found.")
#             return None

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)

#         # 활성화된 라벨에 해당하는 변환 선택
#         # 라벨별 가중치를 고려하거나 활성화된 라벨을 기반으로 증강 전략을 선택
#         active_labels = (label == 1).nonzero(as_tuple=True)[0]

#         if len(active_labels) > 0:
#             selected_label = label_names[active_labels[0].item()] # 활성화된 라벨
#             if self.augmentation_strategies and selected_label in self.augmentation_strategies:  # 증강 전략이 None이 아니면 사용
#                 selected_transform = self.augmentation_strategies[selected_label]
#             else:
#                 selected_transform = self.transform  # 테스트 일때

#         if selected_transform:
#             augmented = selected_transform(image=image)
#             image = augmented['image'] if 'image' in augmented else augmented

#         return image, label
    
# 랜덤하게 증강
# import random

# import random

# class MultiDataset(Dataset):
#     def __init__(self, df, transform=None, augmentation_strategies=None):
#         self.image_id = df['ID'].values
#         self.df = df.copy() # 원본 데이터에 영향을 주지 않도록 복사본을 사용
#         self.labels = df.iloc[:, 3:].values
#         self.transform = transform
#         self.augmentation_strategies = augmentation_strategies  # 증강 전략 추가

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         image_path = self.df.iloc[idx, self.df.columns.get_loc("PATH")]
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"The image at path {image_path} was not found.")
#             return None

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)

#         # 활성화된 라벨에 해당하는 변환 선택
#         # 라벨별 가중치를 고려하거나 활성화된 라벨을 기반으로 증강 전략을 선택
#         if self.transform:
#             augmented = self.transform(image=image)
#             image = augmented['image'] if 'image' in augmented else augmented

#         return image, label


    




"""삭제할 데이터는 None으로 설정되어 있으니, None이 아닌 데이터만 모아서 배치로 만듦"""
# 데이터 로더에서 None 반환 시 건너뛰기
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)
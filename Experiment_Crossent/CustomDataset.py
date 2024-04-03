from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform = None, train = True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train
        
        if self.train:
            self.data = pd.read_csv("/home/goldlab/Project/Experiment/Train_annotations.csv")
        else:
            self.data = pd.read_csv("/home/goldlab/Project/Experiment/Test_annotations.csv")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        img_path = self.data.iloc[idx].PATH
        image = Image.open(img_path).convert("RGB")
        
        label = self.data.iloc[idx].LABEL
               
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CustomDataset_LEVEL(Dataset):
    def __init__(self, csv_file, transform = None, train = True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train
        
        if self.train:
            self.data = pd.read_csv("/home/goldlab/Project/Experiment/Train_annotations.csv")
        else:
            self.data = pd.read_csv("/home/goldlab/Project/Experiment/Test_annotations.csv")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        img_path = self.data.iloc[idx].PATH
        image = Image.open(img_path).convert("RGB")
        
        level = self.data.iloc[idx].LEVEL
        
        if self.transform:
            image = self.transform(image)
        
        return image, level
    


class TripletCustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, train=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train

        # 클래스 별로 인덱스를 그룹핑합니다.
        self.labels = self.data['LABEL'].values
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in np.unique(self.labels)}

        if self.train:
            self.data = pd.read_csv("/home/goldlab/Project/Experiment/Train_annotations.csv")
        else:
            self.data = pd.read_csv("/home/goldlab/Project/Experiment/Test_annotations.csv")
            # 테스트 모드에서는 트리플렛 대신 단일 이미지와 레이블을 반환
            self.test_labels = self.data['LABEL'].values
            self.test_data = self.data['PATH'].values
            self.labels_set = set(self.labels)
            self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}
            
    def __getitem__(self, index):
        if self.train:
            img1, label1 = self._retrieve_image(index)

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            img2, _ = self._retrieve_image(positive_index)

            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img3, _ = self._retrieve_image(negative_index)
        else:
            img1, label1 = self.test_data[index], self.test_labels[index]
            img2 = img3 = None

        return (img1, img2, img3), label1 if self.train else label1

    def __len__(self):
        return len(self.data)
    
    def _retrieve_image(self, index):
        img_path = self.data.iloc[index]['PATH']
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[index]['LABEL']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label  

# ToTensor를 리스트로 감싸주기
transform = transforms.Compose([transforms.ToTensor()])

# 이미지 크기를 조정하는 변환 추가
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
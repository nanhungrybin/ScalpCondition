from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform = None, train = True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train
        
        if self.train:
            self.data = pd.read_csv("Train_annotations.csv")
        else:
            self.data = pd.read_csv("Test_annotations.csv")
            
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
            self.data = pd.read_csv("Train_annotations.csv")
        else:
            self.data = pd.read_csv("Test_annotations.csv")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        img_path = self.data.iloc[idx].PATH
        image = Image.open(img_path).convert("RGB")
        
        level = self.data.iloc[idx].LEVEL
        
        if self.transform:
            image = self.transform(image)
        
        return image, level
    

# ToTensor를 리스트로 감싸주기
transform = transforms.Compose([transforms.ToTensor()])

# 이미지 크기를 조정하는 변환 추가
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
import os
import cv2
import time
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.cuda import amp

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold

#from tqdm.notebook import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from sklearn.manifold import TSNE
# pip install timm
# pip install -q timm pytorch-metric-learning

from train import *
# from test import evaluate_model
from Test import *
from model import *
from loss import *
from Customdataset import *

from torch.utils.data.distributed import DistributedSampler


class CFG:
    seed = 42
    model_name = 'tf_efficientnet_b4.ns_jft_in1k' #'tf_efficientnet_b4_ns'
    img_size = 512
    # model_name = 'vit_base_patch16_384'  # Vision Transformer 모델 사용
    # img_size = 384  # ViT 모델에 맞는 이미지 크기
    scheduler = 'CosineAnnealingLR'
    T_max = 10
    lr = 1e-5
    min_lr = 1e-6
    batch_size = 16
    weight_decay = 1e-6
    num_epochs = 10
    num_classes = 4
    embedding_size = 512
    num_folds = 5 # valid set 도 만들기 #############
    n_accumulate = 4
    temperature = 0.1
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 사용할 GPU ID 리스트 설정
    gpu_ids = [0, 1]
    # 기본 장치를 첫 번째 GPU로 설정
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() else "cpu"
    print(device)
    
    
def set_seed(seed = 42):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)   




def initialize_model():
    #num_classes = 4 
    model = CustomModel(CFG.model_name, pretrained=True, num_classes=CFG.num_classes, embedding_size=CFG.embedding_size)
    # model.to(CFG.device)
    model = nn.DataParallel(model, device_ids=CFG.gpu_ids).to(CFG.device) 
    return model


fold_losses = []

def main(model, criterion, optimizer, scheduler, device, num_epochs):

    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=42)

    test_df = df_test
    train_df = df_train
    # Test data custom
    test_data = CustomDataset(TEST_DIR, test_df, transforms=None)
    test_loader = DataLoader(test_data, batch_size=CFG.batch_size, shuffle=False)

    # Train, valid custom. # train, test split은 매개변수가 중요함
    for fold, (train_index, valid_index) in enumerate(skf.split(train_df, train_df['LABEL'])):
        print(f'Fold {fold + 1}')

        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(valid_index)

        train_data = CustomDataset(TRAIN_DIR, train_df, transforms=data_transforms["train"])
        valid_data = CustomDataset(TRAIN_DIR, train_df, transforms=data_transforms["valid"])

        # 데이터로더 초기화
        train_loader = DataLoader(train_data, batch_size=CFG.batch_size, sampler=train_sampler)
        valid_loader = DataLoader(train_data, batch_size=CFG.batch_size, sampler=valid_sampler)

        dataset_sizes = {
        'train' : len(train_data),
        'valid' : len(valid_data)
         }

        dataloaders = {
        'train' : train_loader,
        'valid' : valid_loader
         }

        # 모델 초기화
        model = initialize_model()

        # Train
        model, history = train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device)
        # Evaluate on test set
        print(f"{fold} 폴드 성능평가")
        test_loss, y_true, y_pred, embeddings = evaluate_model(model, criterion, test_loader, device, extract_embeddings=True)
        print("Test Loss: {:.4f}".format(test_loss))
        fold_losses.append(test_loss)

    mean_losses = np.mean(fold_losses)

    return model, history, mean_losses, embeddings, y_true
   
   
def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=123)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(len(np.unique(labels))):
        plt.scatter(reduced_embeddings[labels == i, 0], reduced_embeddings[labels == i, 1], label=i)
    plt.legend()
    plt.show()
    
    
    
    

if __name__ == '__main__':
    
    TRAIN_DIR = '/data/hbsuh/HairLoss/Training'
    TEST_DIR = '/data/hbsuh/HairLoss/Validation'

    df_train = pd.read_csv('/home/goldlab/Project/Experiment3/Train_annotations.csv')
    df_test = pd.read_csv('/home/goldlab/Project/Experiment3/Test_annotations.csv')

    # 라벨링 인코딩 해야할때
    # le = LabelEncoder()
    # df_train.label_group = le.fit_transform(df_train.label_group)
    
    set_seed(CFG.seed)
    
    data_transforms = {
    "train": A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ToTensorV2()], p=1.),

    "valid": A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ToTensorV2()], p=1.)
        }
    
    # 모델정의해주기
    model = initialize_model()
    
    criterion = SupervisedContrastiveLoss(temperature=CFG.temperature).to(CFG.device) # Custom Implementation
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr) 
    
    fold_losses = []  
    
    model, history, mean_losses, embeddings, y_true = main(model, criterion, optimizer, scheduler, device=CFG.device, num_epochs=CFG.num_epochs)
    
    
    # 학습 시각화
    plt.style.use('fivethirtyeight')
    plt.rcParams["font.size"] = "20"
    fig = plt.figure(figsize=(22,8))
    epochs = list(range(CFG.num_epochs))
    plt.plot(epochs, history['train loss'], label='train loss')
    plt.plot(epochs, history['valid loss'], label='valid loss')
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend()
    plt.title('Loss Curve')
    
    
    
    # 임베딩 시각화
    visualize_embeddings(embeddings, y_true)
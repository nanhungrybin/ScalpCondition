import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.init
from train import *
from test_func import *

from Multi_test import *
from Multi_train import *
from MultiDataset import *

from CustomDataset import *
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import Model
import wandb
from Lossfunc import *
from torch.utils.data.distributed import DistributedSampler  # DDP 사용할때 쓰기
from MultiDataset import *
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# 사용할 GPU ID 리스트 설정
gpu_ids = [0, 1]
# 기본 장치를 첫 번째 GPU로 설정
torch.cuda.set_device(gpu_ids[0])
device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() else "cpu"
print(device)

# 랜덤시드 고정
torch.manual_seed(777)

if device =="cuda":
    torch.cuda.manual_seed_all(777)

# # 랜덤 시드 설정
# torch.manual_seed(777)

# # 사용할 GPU 선택
# device = [torch.device("cuda:0"), torch.device("cuda:2")]
# print("Selected GPUs:", device)



######### parameter setting #########


CFG = {
    'num_epochs':20,
    'learning_rate':3e-4,
    'batch_size':32,
    'img_size': 224 #448 #
}

data_transforms = {
    "train": A.Compose([
        A.Resize(CFG["img_size"], CFG["img_size"]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2),
            contrast_limit=(-0.2, 0.2),
            p=0.5
        ),
        A.GaussianBlur(p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.),

    "valid": A.Compose([
        A.Resize(CFG["img_size"], CFG["img_size"]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
}



####################################### 데이터 업로드
### 심각도 분류하고 싶을시에는 CustomDataset_LEVEL
train_df = pd.read_csv('/home/goldlab/Project/Experiment/MULTI_Train_annotations.csv')
test_df = pd.read_csv('/home/goldlab/Project/Experiment/MULTI_Test_annotations.csv')


# 데이터프레임 합치기
all_data_df = pd.concat([train_df, test_df], ignore_index=True)

# 학습, 검증, 테스트로 데이터 분할 (예: 80% 학습, 10% 검증, 10% 테스트)
train_df, temp_df = train_test_split(all_data_df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

############ 조건에 따른 증강 ###############
# # 라벨 이름
label_names = ["Perifollicular erythema", "Follicular erythema with pustules", "Fine scaling", "Dandruff", "Excessive sebum", "Hair loss", "Healthy"]

# 각 라벨의 빈도 계산
class_counts = train_df[label_names].sum()
num_samples = len(train_df)

# # 데이터 증강 전략
threshold = num_samples * 0.1  # 예: 전체 샘플 수의 10% 미만인 클래스를 대상으로 증강
low_frequency_labels = class_counts[class_counts < threshold]

# # 결과 출력
print("Low frequency labels (less than 10% of total samples):")
print(low_frequency_labels)



# 데이터 증강 전략 설정
augmentation_strategies = {
    label: A.Compose([
        A.Resize(CFG["img_size"], CFG["img_size"]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2),
            contrast_limit=(-0.2, 0.2),
            p=0.5
        ),
        A.GaussianBlur(p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
         ToTensorV2()], p=1.) 
    if class_counts[label] < threshold else data_transforms["valid"]
    for label in class_counts.index
}
print("기존 학습 데이터 개수: ",len(train_df))
# ###########################################################
# 데이터 증강 로직 (간단한 복제로 예시)
# for label, extra_needed in augmentation_needed.items():
#     # 필요한 수량만큼 샘플 선택하여 복제
#     samples_to_augment = train_df[train_df[label] == 1].sample(n=extra_needed, replace=True)
#     train_df = pd.concat([train_df, samples_to_augment])

# # 증강 후 각 라벨별 데이터 개수 다시 계산
# updated_class_counts = train_df[label_names].sum()

# # 결과 출력
# print("Updated number of samples per label after augmentation:")
# print(updated_class_counts)
# print("증강 후 학습 데이터 개수: ",len(train_df))
#########################################################
# 마이너 증강 후  

# 데이터셋 객체 생성
#train_dataset = MultiDataset(train_df, transform=data_transforms["train"], augmentation_strategies=augmentation_strategies)
train_dataset = MultiDataset(train_df, transform=data_transforms["train"], augmentation_strategies=None)
valid_dataset = MultiDataset(valid_df, transform=data_transforms["valid"], augmentation_strategies=None)
test_dataset = MultiDataset(test_df, transform=data_transforms["valid"], augmentation_strategies=None)

# # 데이터셋 크기 출력
# print(f"Training set size: {len(train_dataset)}")
# print(f"Validation set size: {len(valid_dataset)}")
# print(f"Test set size: {len(test_dataset)}")


# 증강 후 각 라벨별 샘플 개수 출력
# print("Updated number of samples per label after augmentation:")
# for label, count in zip(label_names, train_dataset.updated_class_counts):
#     print(f"{label}: {count}")


   
################################### 데이터 로더 생성 ###############################
dataloader_train = DataLoader(train_dataset, batch_size = CFG["batch_size"], shuffle=True, collate_fn=collate_fn)
dataloader_valid = DataLoader(valid_dataset, batch_size = CFG["batch_size"], shuffle=True, collate_fn=collate_fn)

dataloader_test = DataLoader(test_dataset, batch_size = CFG["batch_size"], shuffle=False, collate_fn=collate_fn)
###################################################################################

total_samples = len(dataloader_train.dataset)
total_batch = total_samples // CFG["batch_size"]

def wandB_main():

    ######## model load ############

    # 1. RESNET50

    # model = Model.RESNET50(num_classes=7)
    # model = nn.DataParallel(model, device_ids=gpu_ids).to(device)
    
    # 2. pretrained resnet
    # model =  Model.load_pretrained_RESNET(num_classes=7)
    # model = nn.DataParallel(model, device_ids=gpu_ids).to(device)
   
   
    model =  Model.basic_pretrained_vit(num_classes=7)
    model = nn.DataParallel(model, device_ids=gpu_ids).to(device)
    
    # # # 3. eva02 448 x 448
    # model = Model.load_pretrained_eva02(num_classes=7)
    # model = nn.DataParallel(model, device_ids=gpu_ids).to(device)

    # model = Model.load_pretrained_convnext(num_classes=7)
    # model = nn.DataParallel(model, device_ids=gpu_ids).to(device)

    # # 4. swim 224 x 224
    # model = Model.load_pretrained_swim(num_classes=7)
    # model = nn.DataParallel(model, device_ids=gpu_ids).to(device)

    # # 4. 다중 패치 vit
    # model_names = ['vit_base_patch16_224', 'vit_base_patch32_224']  # Different patch sizes
    # model = Model.MultiPatchViT(model_names, num_classes=7)
    # model = nn.DataParallel(model, device_ids=gpu_ids).to(device)
    


    wandb.init()
    config = wandb.config
    

    # 손실 함수 선택
    if config.loss_function == "BinaryCrossEntropyLoss": 
        loss_function = nn.BCEWithLogitsLoss()
    elif config.loss_function == "ScaledBCEWithLogitsLoss":
        loss_function = ScaledBCEWithLogitsLoss()
    elif config.loss_function == "FocalDynamicScaledBCEWithLogitsLoss":
        loss_function = FocalDynamicScaledBCEWithLogitsLoss(initial_scale=1.0, scale_factor=0.05, max_scale=5.0)

  

    # 옵티마이져
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=config.learning_rate)



    training_params = {
    'num_epochs':config.num_epochs,
    'learning_rate':config.learning_rate,
    "optimizer" : config.optimizer,  # optimizer를 문자열이 아닌 torch.optim 객체로 설정 config.optimizer,
    "loss_function" :config.loss_function, #config.loss_function,
    "device": device,
    "total_batch" : total_batch 
    }
    
    ################# 학습과정에서 사용할 변수
    
    num_epochs = training_params["num_epochs"]
    wandb.watch(model, log='all')
    # hist_acc = []
    # hist_loss = []
    for epoch in range(num_epochs):
        #save_path = "/home/goldlab/Project/Experiment/weights/2_resnet50.pth"
        #save_path = "/home/goldlab/Project/Experiment/weights/random_selfpaced_MultiPatch_vit.pth"
        #save_path = "/home/goldlab/Project/Experiment/weights/maxAug_selfpaced_pretrainedresnet.pth"
        #save_path = "/home/goldlab/Project/Experiment/weights/minorAug_selfpaced_MultiPatch_vit.pth"
        save_path = "/home/goldlab/Project/Experiment/weights/originBCE_vit32.pth"
        # model, train_loss, accuracy = train(model, dataloader_train, epoch, num_epochs, optimizer, loss_function, total_batch, device, save_path)
        # wandb.log({"train_acc":accuracy, "train_loss":train_loss, "epoch": epoch})

        # Updated train function now returns two types of accuracies along with the loss
        model, train_loss, train_hamming_acc, train_exact_match_acc,f1_samples, f1_weighted, jaccard = Multi_train(model, dataloader_train, epoch, num_epochs, optimizer, loss_function, device, save_path)
    
        # Log all relevant metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "train_hamming_acc": train_hamming_acc,
            "train_exact_match_acc": train_exact_match_acc,
            "train_f1_samples":f1_samples, 
            "train_f1_weighted":f1_weighted, 
            "train_jaccard":jaccard,
            "epoch": epoch
        })
        
        _, _, test_hamming_acc, _, val_loss,test_f1, f1_weighted, f1_per_class, jaccard, precision, recall = Multi_test(model, dataloader_valid, device, loss_function)
        print( f"valid_hammimg_acc: {test_hamming_acc},valid_loss: {val_loss},valid_jaccard: {jaccard}, valid_f1_samples: {test_f1}, valid_f1_weighted: {f1_weighted},f1_per_class : {f1_per_class}, valid_precision:{precision}, valid_recall:{recall} ")
        
        # Log test metrics to wandb
        wandb.log({
            "valid_hamming_acc": test_hamming_acc,
            "valid_loss": val_loss,
            "valid_f1_samples": test_f1,
            "valid_f1_weighted":f1_weighted, 
            "valid_f1_per_class":f1_per_class,
            "valid_jaccard":jaccard,
            "valid_precision":precision,
            "valid_recall":recall
        })
  
        
    ### 최종테스트 ######################################################
    # all_preds, all_labels, val_accuracy,f1 = test_func(model, dataloader_test, device)  
    # wandb.log({"test_acc":val_accuracy, "test_f1":f1})

    # Updated test_func now returns two types of accuracies and the F1 score
    all_preds, all_labels, test_hamming_acc, test_exact_match_acc, _,test_f1, f1_weighted, f1_per_class, jaccard,precision, recall = Multi_test(model, dataloader_test, device, loss_function)
    print( f"test_jaccard: {jaccard}, test_f1_samples: {test_f1}, test_f1_weighted: {f1_weighted},f1_per_class: {f1_per_class}, test_precision:{precision},test_recall:{recall}" )
    # Log test metrics to wandb
    wandb.log({
        "test_hamming_acc": test_hamming_acc,
        "test_exact_match_acc": test_exact_match_acc,
        "test_f1_samples": test_f1,
        "test_f1_weighted":f1_weighted, 
        "f1_per_class": f1_per_class,
        "test_jaccard":jaccard,
        "test_precision":precision,
        "test_recall":recall
    })


    return all_preds, all_labels

if __name__ == "__main__": # 객체를 불러 오는 것은 main함수에
    #anomaly detect
    torch.autograd.set_detect_anomaly(True)
    
    # output layer를 현재 data에 맞게 수정 
    num_classes = 7    


    ################# hyperparameter tunning with Wandb ###############

    sweep_config = {
        "name": "hyperparameter_tuning",
        "method": "random",
        "metric": {"goal": "maximize", "name": "accuracy"},

        "parameters": {
            "learning_rate": {'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3},
            "num_epochs": {"values": [25]}, 
            "batch_size": {"values": [128]},#,128]},
            "loss_function":{'values': ["BinaryCrossEntropyLoss"]},#"BinaryCrossEntropyLoss"]},
                                        #"FocalDynamicScaledBCEWithLogitsLoss"
                                        #"ListwiseRankingLoss" ScaledBCEWithLogitsLoss
            "optimizer": {"values": ["AdamW"]}  , #"Adam", 
            "dropout": {"values": [0.1, 0.2, 0.3]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="두피 질환 유형_0416_multi")

    ############ Sweep 작업 실행 ########################
    wandb.agent(sweep_id, wandB_main, count=1)

    ############ WandB main 함수 실행 결과 저장 ############
    all_preds, all_labels = wandB_main()
    
    # 멀티 라벨 혼동 행렬 계산
    mcm = multilabel_confusion_matrix(all_labels, all_preds)

    # 각 클래스에 대한 혼동 행렬을 시각화
    fig, axes = plt.subplots(nrows=int(np.ceil(num_classes / 3)), ncols=3, figsize=(15, num_classes * 2))
    axes = axes.flatten()

    for i in range(num_classes):
        sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'Class {i} Confusion Matrix')
        axes[i].set_xlabel('Predicted Labels')
        axes[i].set_ylabel('True Labels')
        axes[i].set_xticklabels(['Negative', 'Positive'])
        axes[i].set_yticklabels(['Negative', 'Positive'])

    # 남은 subplot을 숨김
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('/home/goldlab/Project/Experiment/originBCE_CONFUSION_vit32.png')
    plt.show()

    # Calculate confusion matrices for each class in a multi-label setting
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    cm_sum = np.sum(mcm, axis=0)  # Sum over all confusion matrices for a global view

    # Print correct predictions per class
    for i in range(num_classes):
        correct_samples = mcm[i, 1, 1]  # True Positives
        print(f"Class {i}: {correct_samples} samples correctly predicted.")

    # Calculate and visualize class-wise probabilities
    class_accuracy = [mcm[i, 1, 1] / (mcm[i, 1, 1] + mcm[i, 1, 0]) for i in range(num_classes)]  # TP / (TP + FN)
    annotated_matrix = np.array([[f"{100 * acc:.2f}%" for acc in class_accuracy]])

    # Visualize the confusion matrix as probability percentages
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.array([class_accuracy]), annot=annotated_matrix, fmt='', cmap='Blues', xticklabels=range(num_classes), yticklabels=["Accuracy"])
    plt.xlabel('Class')
    plt.title('Class Accuracy (%)')
    plt.savefig('/home/goldlab/Project/Experiment/originBCE_vit32.png')
    plt.show()


    # # confusion matrix 계산
    # cm = confusion_matrix(all_labels, all_preds) # all_labels, all_preds, 
    
    # def annot_format(matrix):
    #     return np.array([[f'{item:.2f}%' for item in row] for row in matrix])


    # # 각 클래스별로 맞춘 개수 출력
    # for i in range(cm.shape[0]):
    #     correct_samples = cm[i, i]
    #     print(f"Class {i}: {correct_samples} samples correctly predicted.")

    # # 각 클래스별로 맞힌 확률 계산
    # cm_probability = cm / cm.sum(axis=1, keepdims=True) * 100  # multiply here to avoid repeated calc

    # # 각 클래스별 정확도 계산
    # class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # # 각 클래스별 정확도 출력
    # for i, acc in enumerate(class_accuracy):
    #     print(f"Class {i} Accuracy: {acc:.4f}")

    # # confusion matrix 시각화
    # plt.figure(figsize=(8, 6))
    # annotated_matrix = annot_format(cm_probability)
    # sns.heatmap(cm_probability, annot=annotated_matrix, fmt='', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))

    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix (Probability %)')
    # plt.show()


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.init
from train import *
from test_func import *
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


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# 사용할 GPU ID 리스트 설정
gpu_ids = [0, 1, 2,3]
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
}


####################################### 데이터 업로드
### 심각도 분류하고 싶을시에는 CustomDataset_LEVEL

train_dataset = CustomDataset(csv_file='/home/goldlab/Project/Experiment/Train_annotations.csv', transform=transform, train = True )
test_dataset = CustomDataset(csv_file='/home/goldlab/Project/Experiment/Test_annotations.csv', transform=transform, train = False )
    
################################### 데이터 로더 생성 ###############################

# DistributedSampler 사용하여 데이터 로더 생성
# train_sampler = DistributedSampler(train_dataset)
# test_sampler = DistributedSampler(test_dataset)

# dataloader_train = DataLoader(train_dataset, batch_size = CFG["batch_size"], shuffle=False, sampler=train_sampler)
# dataloader_test = DataLoader(test_dataset, batch_size = CFG["batch_size"], shuffle=False, sampler=test_sampler)

dataloader_train = DataLoader(train_dataset, batch_size = CFG["batch_size"], shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size = CFG["batch_size"], shuffle=False)
###################################################################################

total_samples = len(dataloader_train.dataset)
total_batch = total_samples // CFG["batch_size"]

def wandB_main():

    ######## model load ############

    # MODEL = Model.RESNET50(num_classes)
    # model = MODEL.to(device)  
    MODEL = Model.RESNET50(num_classes)
    model = nn.DataParallel(MODEL, device_ids=gpu_ids).to(device) 

  

    wandb.init()
    config = wandb.config
    
    if config.loss_function == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
    # 손실 함수 선택
    # if config.loss_function == "CrossEntropyLoss":
    #     loss_function = nn.CrossEntropyLoss()
    # elif config.loss_function == "ListwiseRankingLoss":
    #     loss_function = ListwiseRankingLoss()

    # elif config.loss_function == "TripletLoss":
    #     loss_function = TripletLoss(margin=1.0)
    # elif config.loss_function == "MultiLabelSoftMarginLoss":
    #     loss_function = nn.MultiLabelSoftMarginLoss()

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
        save_path = "/home/goldlab/Project/Experiment/weights/resnet50.pth"
        
        model, train_loss, accuracy = train(model, dataloader_train, epoch, num_epochs, optimizer, loss_function, total_batch, device, save_path)
        wandb.log({"train_acc":accuracy, "train_loss":train_loss, "epoch": epoch})
        # hist_acc.append(accuracy)
        # hist_loss.append(train_loss)
    ### 최종테스트 ######################################################
    all_preds, all_labels, val_accuracy,f1 = test_func(model, dataloader_test, device)  
    wandb.log({"test_acc":val_accuracy, "test_f1":f1})

    return all_preds, all_labels

if __name__ == "__main__": # 객체를 불러 오는 것은 main함수에

    
    # output layer를 현재 data에 맞게 수정 
    num_classes = 6    


    ################# hyperparameter tunning with Wandb ###############

    sweep_config = {
        "name": "hyperparameter_tuning",
        "method": "random",
        "metric": {"goal": "maximize", "name": "accuracy"},

        "parameters": {
            "learning_rate": {'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3},
            "num_epochs": {"values": [10, 20, 30]}, 
            "batch_size": {"values": [16, 32, 64]},
            "loss_function":{'values': ["CrossEntropyLoss" ]},
                                        #"ListwiseRankingLoss"
            "optimizer": {"values": ["Adam", "AdamW", "RAdam"]}  ,
            "dropout": {"values": [0.1, 0.2, 0.3]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="두피 질환 유형_절차(기본)")

    ############ Sweep 작업 실행 ########################
    wandb.agent(sweep_id, wandB_main, count=2)

    ############ WandB main 함수 실행 결과 저장 ############
    all_preds, all_labels = wandB_main()


    # confusion matrix 계산
    cm = confusion_matrix(all_labels, all_preds) # all_labels, all_preds, 


    # 각 클래스별로 맞춘 개수 출력
    for i in range(cm.shape[0]):
        correct_samples = cm[i, i]
        print(f"Class {i}: {correct_samples} samples correctly predicted.")
        
    # 각 클래스별로 맞힌 확률 계산
    cm_probability = cm / cm.sum(axis=1, keepdims=True)

    # 각 클래스별 정확도 계산
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # 각 클래스별 정확도 출력
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i} Accuracy: {acc:.4f}")

    def annot_format(val):
        return f'{val:.2f}%'

    # confusion matrix 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_probability * 100, annot=True, fmt='', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes), annot_kws={'format': 'annot_format'})

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Probability %)')
    plt.show()
 

# Iteration 은 각 에포크 내에서 일어나는 단일 업데이트 단계
# 훈련 데이터가 큰 경우, 전체 데이터를 한 번에 처리하지 않고 작은 미니배치(mini-batch)로 나누어 학습을 수행. 이 미니배치에 대한 단일 업데이트 단계가 Iteration
# = > 1000개의 훈련 샘플이 있고 미니배치 크기가 100이라면, 에포크당 10개의 Iteration

# 에포크는 전체 데이터에 대한 한 번의 학습 주기이고, Iteration 은 각 학습 주기에서 모델 가중치를 업데이트하기 위한 단일 단계
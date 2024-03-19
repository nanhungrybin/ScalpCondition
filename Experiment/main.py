import torch
import torchvision.transforms as transforms
import torch.nn.init
from train import *
import test
from CustomDataset import *
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import Model
import wandb
import Lossfunc 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 랜덤시드 고정
torch.manual_seed(777)

if device =="cuda":
    torch.cuda.manual_seed_all(777)


######### parameter setting #########


CFG = {
    'num_epochs':20,
    'learning_rate':3e-4,
    'batch_size':32,
}


####################################### 데이터 업로드
### 심각도 분류하고 싶을시에는 CustomDataset_LEVEL

train_dataset = CustomDataset(csv_file='C:\\Users\\xianm\\Downloads\\HB\\Train_annotations.csv', transform=transform, train = True )
test_dataset = CustomDataset(csv_file='C:\\Users\\xianm\\Downloads\\HB\\Test_annotations.csv', transform=transform, train = False )
    
################################### 데이터 로더 생성 ###############################

dataloader_train = DataLoader(train_dataset, batch_size = CFG["batch_size"], shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size = CFG["batch_size"], shuffle=False)

###################################################################################

total_samples = len(dataloader_train.dataset)
total_batch = total_samples // CFG["batch_size"]

def wandB_main():
        
    wandb.init()
    config = wandb.config
    
    # 손실 함수 선택
    if config.loss_function == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
    elif config.loss_function == "ListwiseRankingLoss":
        loss_function = nn.ListwiseRankingLoss()
    elif config.loss_function == "TripletLoss":
        loss_function = nn.TripletLoss(margin=1.0)


    training_params = {
    'num_epochs':config.num_epochs,
    'learning_rate':config.learning_rate,
    "optimizer" : config.optimizer,
    "loss_function" :config.loss_function,
    "device": device,
    "total_batch" : total_batch 
    }
    
    ################# 학습과정에서 사용할 변수
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    loss_function = training_params["loss_function"]
    loss_function = loss_function.to(device)
    
    num_epochs = training_params["num_epochs"]
    wandb.watch(model, log='all')

    for epoch in range(num_epochs) :
        save_path = "C:\\Users\\xianm\\Downloads\\HB\\weights\\resnet50.pth"
        ### 학습 ################################################
        model, train_loss, accuracy = train(model, dataloader_train, epoch, config.num_epochs, optimizer, loss_function, total_batch, device, save_path )
        ### 최종테스트 ######################################################
        all_preds, all_labels, val_accuracy = test(model, dataloader_test, device)  
        wandb.log({"train_acc":accuracy, "train_loss":train_loss, "valid_acc":val_accuracy})

    return all_preds, all_labels

if __name__ == "__main__": # 객체를 불러 오는 것은 main함수에

    
    # output layer를 현재 data에 맞게 수정 
    num_classes = 6    

    ######## model load ############

    MODEL = Model.RESNET50(num_classes)
    # model load
    model = MODEL.to(device)

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
            "loss_function":{'values': ["CrossEntropyLoss", "ListwiseRankingLoss", "TripletLoss"]},
            "dropout": {"values": [0.1, 0.2, 0.3]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="두피 질환 유형_절차(기본)")

    ############ Sweep 작업 실행 ########################
    wandb.agent(sweep_id, wandB_main, count=2)


    # confusion matrix 계산
    cm = confusion_matrix(wandB_main[1], wandB_main[0]) # all_labels, all_preds, 


    # 각 클래스별로 맞춘 개수 출력
    for i in range(cm.shape[0]):
        correct_samples = cm[i, i]
        print(f"Class {i}: {correct_samples} samples correctly predicted.")

    # 각 클래스별 정확도 계산
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # 각 클래스별 정확도 출력
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i} Accuracy: {acc:.4f}")



    # confusion matrix 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Affectnet - Confusion Matrix')
    plt.show()
 

# Iteration 은 각 에포크 내에서 일어나는 단일 업데이트 단계
# 훈련 데이터가 큰 경우, 전체 데이터를 한 번에 처리하지 않고 작은 미니배치(mini-batch)로 나누어 학습을 수행. 이 미니배치에 대한 단일 업데이트 단계가 Iteration
# = > 1000개의 훈련 샘플이 있고 미니배치 크기가 100이라면, 에포크당 10개의 Iteration

# 에포크는 전체 데이터에 대한 한 번의 학습 주기이고, Iteration 은 각 학습 주기에서 모델 가중치를 업데이트하기 위한 단일 단계
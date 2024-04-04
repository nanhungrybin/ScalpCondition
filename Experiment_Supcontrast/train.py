from main import CFG
import time
import numpy as np
from torch.cuda import amp
from collections import defaultdict
from tqdm import tqdm
import torch
import copy


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    history = defaultdict(list)
    scaler = amp.GradScaler()

    for step, epoch in enumerate(range(1,num_epochs+1)):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','valid']:
            if(phase == 'train'):
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluation mode

            running_loss = 0.0

            # Iterate over data
            for inputs,labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(CFG.device)
                labels = labels.to(CFG.device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with amp.autocast(enabled=True):
                        _, outputs = model(inputs)  # outputs은 logit값 => 클래스에 대한 확률이 아니라 각 클래스에 대한 점수 => 로짓값(logits)은 일반적으로 소프트맥스(softmax) 함수를 거쳐 클래스에 대한 확률(probabilities)로 변환
                        loss = criterion(outputs, labels)  # 모델의 출력(logits)과 실제 레이블을 비교
                        loss = loss / CFG.n_accumulate

                    # backward only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()  # 그래디언트를 계산

                    # optimize only if in training phase
                    if phase == 'train' and (step + 1) % CFG.n_accumulate == 0:  # 그래디언트를 사용하여 모델의 가중치를 업데이트
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

                        # zero the parameter gradients
                        optimizer.zero_grad()


                running_loss += loss.item()*inputs.size(0)

            epoch_loss = running_loss/dataset_sizes[phase]
            history[phase + ' loss'].append(epoch_loss)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase=='valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                PATH = f"{best_loss}_epoch_{epoch}.bin"
                torch.save(model.state_dict(), PATH)

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss ",best_loss)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

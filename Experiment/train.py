import torch
import torch.nn as nn


def train(model, train_dataloader, epoch, num_epochs, optimizer, loss_function, total_batch, device, save_path):

    avg_cost = 0
    correct = 0
    total_samples = 0

    model.train()


    for batch_idx, (X, y) in enumerate(train_dataloader, 1):
        X = X.to(device)
        y = y.to(device)
        y = y.long()

        optimizer.zero_grad()

        hypothesis = model(X)
        hypothesis = hypothesis.float()  # Convert to float

        loss = loss_function(hypothesis, y)
        loss.backward()
        
        optimizer.step()

        avg_cost += loss.item() / total_batch  # loss.item()을 사용하여 스칼라 값 가져오기

        # accuracy 구하기
        _, predicted = torch.max(hypothesis, 1) #  hypothesis 텐서에서 지정된 차원 (1인 경우)을 따라 최대값의 인덱스를 얻는 데 사용

        total_samples += y.size(0)
        correct += (predicted == y).sum().item()


    # 정확도 계산
    accuracy = correct / total_samples

    
    print(f"[Epoch: {epoch + 1}/{num_epochs}] Loss: {avg_cost:.4f} Accuracy: {accuracy:.4f}")
    # batch_idx : 현재 배치의 인덱스
    # len(train_dataloader) : 전체 배치 수


    # 모델 가중치 저장
    if save_path is not None:
        torch.save(model.state_dict(), save_path)


    return model, loss, accuracy

"""
모델의 가중치를 저장하는 것은 모델의 학습 상태를 나중에 재사용하거나 모델을 공유하고자 할 때 유용
PyTorch에서는 모델의 가중치를 state_dict() 메서드를 사용하여 가져올 수 있으며, 이를 torch.save()를 사용하여 저장
"""
import torch
import torch.nn as nn

def test(model, test_dataloader, device):
    model.eval()
    correct = 0
    total_samples = 0  # 전체 샘플 수를 세기 위한 

    # 테스트 데이터에 대한 예측값과 정답을 저장할 리스트 초기화
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device)
            y = y.to(device)
            y = y.long()

            prediction = model(X)
            prediction = prediction.float()  # Convert to float

            correct_prediction = torch.argmax(prediction, 1) == y

            correct += correct_prediction.sum().item()  # 정확하게 분류된 샘플 수를 누적
            total_samples += y.size(0)  # 각 배치에 있는 전체 샘플 수를 누적

            accuracy = correct / total_samples  # 정확도 계산
            #print("Accuracy:", accuracy)

            # 예측값과 정답을 리스트에 추가
            all_preds.extend(torch.argmax(prediction, 1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())


    return all_preds, all_labels, accuracy
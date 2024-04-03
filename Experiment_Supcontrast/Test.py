from tqdm import tqdm
import torch
from sklearn.metrics import classification_report
import numpy as np


def evaluate_model(model, criterion, data_loader, device, extract_embeddings=True):
    model.eval()
    running_loss = 0.0
    y_pred = []  # 예측된 레이블을 저장할 리스트
    y_true = []  # 실제 레이블을 저장할 리스트
    embeddings = []  # 임베딩을 저장할 리스트

    # Iterate over data
    for inputs, labels in tqdm(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        with torch.no_grad():
            # 모델 수정에 따라 여기를 조정해야 할 수 있음
            output = model(inputs)
            if extract_embeddings:
                # 모델이 임베딩과 클래스 예측을 모두 반환하도록 가정
                embeddings_output, class_output = output
                _, preds = torch.max(class_output, 1)
                    # outputs 텐서에서, 차원 1(보통 클래스에 대한 확률을 나타내는 차원)에 대해 최대값찾기
                    # => (각 데이터 포인트(또는 배치 내 각 샘플)에 대해 가장 높은 점수를 가진 클래스의 인덱스(예측된 클래스) 찾기 )
                embeddings.extend(embeddings_output.cpu().numpy())
            else:
                class_output = output
                _, preds = torch.max(class_output, 1)

            loss = criterion(class_output, labels)

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        y_pred.extend(preds.view(-1).cpu().numpy())  # 예측값 저장 => 1차원으로 평탄화(flatten)
        y_true.extend(labels.cpu().numpy())  # 실제값 저장

    # Total loss
    total_loss = running_loss / len(data_loader.dataset)

    # classification_report를 사용한 모델 성능 평가
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(len(np.unique(y_true)))]))

    y_true = np.concatenate(y_true, axis=0) # 모든 배치 처리를 한 뒤에 numpy 배열로 합치기

    if extract_embeddings:
      embeddings = np.concatenate(embeddings, axis=0)# 모든 배치 처리를 한 뒤에 numpy 배열로 합치기
      return total_loss, y_true, y_pred, embeddings

    else:
      return total_loss, y_true, y_pred

import torch
import torch.nn as nn


# 사용자 정의 손실함수
# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # Anchor와 Positive 사이의 거리
        distance_negative = (anchor - negative).pow(2).sum(1)  # Anchor와 Negative 사이의 거리
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

    
# Listwise Ranking Loss
class ListwiseRankingLoss(nn.Module):
    def __init__(self):
        super(ListwiseRankingLoss, self).__init__()

    def forward(self, predictions, labels):
        # 예측값과 실제 레이블 간의 순서를 고려하여 손실을 계산하는 로직을 구현
        loss = torch.mean(torch.abs(predictions - labels))  # 예제 손실 계산
        return loss


# crossentropy


# cross_entropy_loss = nn.CrossEntropyLoss()
# triplet_loss = TripletLoss(margin=1.0)  # 사용자 정의 손실 함수 인스턴스화
# listwise_ranking_loss = ListwiseRankingLoss()  # 사용자 정의 손실 함수 인스턴스화


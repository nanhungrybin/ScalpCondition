import torch
import torch.nn as nn


class ScaledBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(ScaledBCEWithLogitsLoss, self).__init__()
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')  # Use 'none' to get the loss per batch element

    def forward(self, logits, targets):
        # Compute the base BCEWithLogitsLoss
        base_loss = self.base_loss(logits, targets)
        
        # Calculate probabilities using sigmoid since it's a binary classification for each label
        probabilities = torch.sigmoid(logits)
        
        # Calculate the scaling factor for each sample:
        # Using the inverse of the average predicted probability for the positive class (for simplicity)
        scaling_factors = 1 / probabilities.mean(dim=1)
        
        # Normalize scaling factors to have a mean of 1 to keep the loss scale consistent
        scaling_factors /= scaling_factors.mean()
        
        # Apply scaling factors to the loss
        scaled_loss = scaling_factors.unsqueeze(1) * base_loss  # Ensure correct broadcasting
        
        # Return the mean loss
        return scaled_loss.mean()


# crossentropy
#Focal Loss를 포함하여 어려운 샘플에 더 집중하게 만들고, 동적으로 스케일을 조정하여 쉬운 샘플로부터 시작하여 점차 어려운 샘플로 학습의 중점을 이동
class FocalDynamicScaledBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, initial_scale=1.0, scale_factor=0.1, max_scale=2.0):
        super(FocalDynamicScaledBCEWithLogitsLoss, self).__init__()
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha
        self.initial_scale = initial_scale
        self.max_scale = max_scale
        self.current_scale = initial_scale

    def forward(self, logits, targets):
        #logits = torch.mean(logits, dim=(2, 3)) ######### this line is only for swin
        base_loss = self.base_loss(logits, targets)

        probabilities = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probabilities, 1 - probabilities)
        focal_factor = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_factor * base_loss

        sample_scales = 1 / probabilities.mean(dim=1).clamp(min=0.1)
        sample_scales = sample_scales / sample_scales.mean()

        dynamic_scales = sample_scales * self.current_scale
        dynamic_scales = dynamic_scales.clamp(max=self.max_scale)

        scaled_loss = dynamic_scales.unsqueeze(1) * focal_loss
        return scaled_loss.sum() / logits.size(0)  # 각 샘플의 손실을 합산하여 배치 크기로 나눔


    def update_scale(self, epoch, total_epochs):
        self.current_scale = min(self.initial_scale + self.scale_factor * (epoch / total_epochs), self.max_scale)


import timm
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, base_model_name, pretrained=True, num_classes=None, embedding_size=512):
        super(CustomModel, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=pretrained, num_classes=0, global_pool='')  # num_classes=0, global_pool=''로 설정하여 사전 훈련된 모델의 마지막 분류 레이어를 제거합니다.
        self.num_features = self.base_model.num_features  # timm 모델에서 기본 모델의 특성 수를 얻습니다.

        # 글로벌 평균 풀링 레이어 추가
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 임베딩을 위한 레이어
        self.embedding_layer = nn.Linear(self.num_features, embedding_size)
        
        # 임베딩을 기반으로 한 최종 분류를 위한 레이어
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        embedding = self.embedding_layer(x)
        out = self.classifier(embedding)
        return embedding, out
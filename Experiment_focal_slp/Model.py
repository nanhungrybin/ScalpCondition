from torchvision import models
import torch.nn as nn
import timm
import torch


def RESNET50(num_classes):

    resnet18_pretrained = models.resnet18(weights="DEFAULT")
    num_features = resnet18_pretrained.fc.in_features
    resnet18_pretrained.fc = nn.Linear(num_features, num_classes)
    
    return resnet18_pretrained

# 224 * 224
def load_pretrained_vit(model_name='vit_small_patch16_224.augreg_in21k', pretrained=True, num_classes=7):
    model = timm.create_model(model_name, pretrained=pretrained)
    # 멀티 레이블 분류를 위해 Sigmoid를 적용할 수 있는 방식으로 분류기를 교체
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, num_classes),  # num_classes는 라벨의 수
        nn.Sigmoid()
    )
    return model

def basic_pretrained_vit(model_name='vit_base_patch32_224', pretrained=True, num_classes=7):
    # Load a pre-trained Vision Transformer model
    model = timm.create_model(model_name, pretrained=pretrained)
    
    # Replace the head of the model with a new one for fine-tuning
    num_features = model.head.in_features  # Get the number of input features of the original head
    model.head = nn.Linear(num_features, num_classes)  # Replace with a new linear layer for our number of classes
    
    return model

# 384 
# def load_pretrained_convnext(model_name='convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=7):
#     model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

#     # 기존의 Fully Connected 레이어 변경
#     model.head = nn.Sequential(
#         nn.Linear(model.head.in_features, num_classes),
#         nn.Sigmoid()
#     )
#     return model

def load_pretrained_vitres(model_name='vit_small_r26_s32_224.augreg_in21k', pretrained=True, num_classes=7):
    model = timm.create_model(model_name, pretrained=pretrained)
    # 멀티 레이블 분류를 위해 Sigmoid를 적용할 수 있는 방식으로 분류기를 교체
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, num_classes),  # num_classes는 라벨의 수
        nn.Sigmoid()
    )
    return model
def load_pretrained_convnext(model_name='convnext_base', pretrained=True, num_classes=7):
    model = timm.create_model(model_name, pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)  # 두피 유형 클래스 수에 맞게 조정
    return model


class CustomHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CustomHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def load_pretrained_swim(model_name='swin_base_patch4_window7_224.ms_in1k', pretrained=True, num_classes=7):
    model = timm.create_model(model_name, pretrained=pretrained)
    # 헤드 레이어를 새로운 형태로 교체
    in_features = model.head.in_features
    model.head = CustomHead(in_features, num_classes)
    return model

class CusHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.fc(x)

def load_pretrained_RESNET(model_name='resnetv2_50x1_bit.goog_in21k', pretrained=True, num_classes=7):
    model = timm.create_model(model_name, pretrained=pretrained)
    if hasattr(model, 'head') and hasattr(model.head, 'fc'):
        in_channels = model.head.fc.in_channels  # Conv2d의 입력 채널 수를 가져옵니다.
        model.head.fc = CusHead(in_channels, num_classes)  # CustomHead로 교체합니다.
    else:
        raise AttributeError("모델에 'head.fc' 레이어가 없습니다.")
    return model

class CUSTomHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()  # 이 부분을 수정했습니다.
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.fc(x)

def load_pretrained_RESNETX(model_name='resnext101_32x32d.fb_wsl_ig1b_ft_in1k', pretrained=True, num_classes=7):
    model = timm.create_model(model_name, pretrained=pretrained)
    in_features = model.fc.in_features  # 마지막 FC 레이어의 입력 특성을 가져옵니다.
    model.fc = CUSTomHead(in_features, num_classes)  # CustomHead로 교체합니다.
    return model


class MultiPatchViT(nn.Module):
    def __init__(self, model_names, num_classes, pretrained=True):
        super(MultiPatchViT, self).__init__()
        self.models = nn.ModuleList([
            self.create_vit(model_name, num_classes, pretrained) for model_name in model_names
        ])
        
    def create_vit(self, model_name, num_classes, pretrained):
        # Load a pre-trained Vision Transformer model
        model = timm.create_model(model_name, pretrained=pretrained)
        num_features = model.head.in_features
        # Replace the head of the model with a new one for fine-tuning
        model.head = nn.Linear(num_features, num_classes)
        return model

    def forward(self, x):
        # Forward pass for each model
        outputs = [model(x) for model in self.models]
        # Average the outputs
        output = torch.mean(torch.stack(outputs), dim=0)
        return output

# Example usage
# num_classes = 7
# model_names = ['vit_base_patch16_224', 'vit_base_patch32_224']  # Different patch sizes
# model = MultiPatchViT(model_names, num_classes)

# ViT 모델 생성 예제
def create_custom_vit(model_name='vit_base_patch16_224', pretrained=True, num_classes=1000, patch_size=64):
    # 모델 생성 시 patch_size 매개변수 전달
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        patch_size=patch_size
    )
    
    # 마지막 레이어 조정
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


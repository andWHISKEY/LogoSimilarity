import torch.nn as nn
import timm
import torch.nn.functional as F

# --------------------------------------------
# 모델 정의: Swin Transformer 백본 + Projection Head (SimCLR)
# --------------------------------------------
class ProjectionHead(nn.Module):
    """최적화된 projection head"""
    def __init__(self, input_dim, projection_dim, hidden_dim):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim))
        # 최종 출력에는 BatchNorm 제거 (대신 L2 정규화만 사용)
        
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)  # L2 정규화

class SimCLRModel(nn.Module):
    """SimCLR 모델 정의"""
    def __init__(self, backbone_name, projection_dim, hidden_dim):
        super(SimCLRModel, self).__init__()
        # pretrained Swun transformer 백본 생성
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        self.backbone.head = nn.Identity() # 마지막 FC layer 제거=분류 헤드 제거함. 그니까 특징 추출만 함
        self.feature_dim = self.backbone.num_features
        # projection head 생성
        self.projection_head = ProjectionHead(self.feature_dim, projection_dim=projection_dim, hidden_dim=hidden_dim)

    def forward_features(self, x):
        # 패치 임베딩(=백본 통해 특징 추출)
        x = self.backbone.patch_embed(x)
        if hasattr(self.backbone, 'pos_drop'):
            x = self.backbone.pos_drop(x)
        # self.backbone.layers를 순회하면서 feature 추출
        for layer in self.backbone.layers:
            x = layer(x)
        x = self.backbone.norm(x)
        # (B, N, C) -> (B, C): 토큰 차원에 대해 평균 pooling
        x = x.mean(dim=1)
        return x
    
    def forward(self, x):
        features = self.forward_features(x)
        # features의 shape가 [B, N, C] (여기서 N=7)라면,
        # 토큰 차원(N)을 평균 내어 [B, C]로 만드는거
        if features.ndim == 3:
            features = features.mean(dim=1) # [B, N, C] -> [B, C]
        projection = self.projection_head(features)
        projection = F.normalize(projection, dim=1) # projection head를 통해 고차원 특징 벡터를 저차원 임베딩 후 정규화화
        return projection
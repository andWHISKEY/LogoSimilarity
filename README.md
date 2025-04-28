# SimCLR with Swin Transformer for Logo Similarity

## 📖 프로젝트 개요
이 프로젝트는 **SimCLR**(Simple Framework for Contrastive Learning of Visual Representations)과 **Swin Transformer**를 사용하여 로고 이미지 간의 유사도를 학습하고 평가하는 모델을 구현합니다.  
Contrastive Learning을 통해 이미지의 특징 벡터를 학습하고, 이를 기반으로 유사한 이미지를 찾는 작업을 수행합니다.

---

## 📂 프로젝트 구조
```
logo/
├── README.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── model.py                # SimCLRModel, ProjectionHead 정의
│   ├── loss.py                 # nt_xent_loss_with_reweighting 함수
│   ├── datasets.py             # 데이터 로드/전처리 모듈
│   ├── train.py                # 학습 루프 및 스케줄러
│   ├── evaluate.py             # 평가 및 임베딩 시각화
│   └── utils.py                # 유틸리티 함수(imagehash 필터링 등)
└── output/                   # 결과 저장 폴더
    └── featuremap/           # featuremap 저장 폴더
```
---

## 🚀 사용 방법
### 0. 필요 library 다운
'requirements.txt' 속 library를 다운받습니다.
`pip install -r requirements.txt`

### 1. 데이터 준비
데이터셋 폴더 경로를 `params["dataset_folder"]`에 설정합니다.
데이터셋은 `.jpg` 또는 `.png` 형식의 이미지 파일이어야 합니다.

### 2. 학습 실행
train.py를 실행하여 모델을 학습합니다
`python train.py`

### 3. 특징 맵 추출
making_featuremap.py를 실행하여 입력 이미지의 특징 맵을 추출하고 
하나의 combined_featuremap.npy 파일 및 파일명만을 따로 저장한 combined_featuremap_filename.npy를 저장합니다.
`python utils.py`

### 4. 유사 이미지 검색
find_similar_logos의 find_top_k_similar_from_combined 함수를 사용하여 샘플 이미지와 유사한 이미지 상위 top k개를 찾습니다.
유사한 이미지 filename(디자인출원번호)들을
---

## 📊 주요 기능
### 1. Contrastive Learning
- SimCLR 프레임워크를 기반으로 Swin Transformer를 사용하여 이미지의 특징 벡터를 학습합니다.

### 2. NT-Xent Loss
- Contrastive Learning에서 양의 샘플(positive pair)의 유사도를 최대화하고 음의 샘플(negative pair)의 유사도를 최소화하는 손실 함수입니다.

### 3. 특징 맵 추출
- 학습된 모델을 사용하여 입력 이미지의 특징 벡터를 추출하고 .npy 파일로 저장합니다.

### 4. 유사 이미지 검색
- 샘플 이미지와 테스트 데이터셋의 모든 이미지 간 코사인 유사도를 계산하여 가장 유사한 이미지를 찾습니다.

---

## 🛠️ 주요 코드 설명
### 1. SimCLR 모델 정의 (model.py)
- Swin Transformer를 백본으로 사용하며, Projection Head를 통해 특징 벡터를 저차원 임베딩으로 변환합니다.
    ```
    class SimCLRModel(nn.Module):
        def __init__(self, backbone_name, projection_dim, hidden_dim):
            super(SimCLRModel, self).__init__()
            self.backbone = timm.create_model(backbone_name, pretrained=True)
            self.backbone.head = nn.Identity()
            self.feature_dim = self.backbone.num_features
            self.projection_head = ProjectionHead(self.feature_dim, projection_dim, hidden_dim)
    ```

### 2. NT-Xent Loss 정의 (loss.py)
- 코사인 유사도를 기반으로 크로스 엔트로피 손실을 계산합니다.
    ```
    def nt_xent_loss_with_reweighting(z_i, z_j, temperature=0.5, similarity_threshold=0.9):
        """
        NT-Xent Loss 계산 (Normalized Temperature-scaled Cross Entropy Loss)
        가까이 있는 샘플끼리 유사도 높게, 멀리 있는 샘플끼리 유사도 낮게

        Args:
            z_i (torch.Tensor): 첫 번째 뷰의 임베딩 (B, D)
            z_j (torch.Tensor): 두 번째 뷰의 임베딩 (B, D)
            temperature (float): 온도 스케일링 파라미터

        Returns:
            torch.Tensor: NT-Xent Loss 값
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # 정규화된 임베딩
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        z = F.normalize(z, dim=1)

        # 코사인 유사도 행렬 [2B, 2B]
        sim_matrix = torch.matmul(z, z.T) / temperature

        # 자기 자신은 유사도 제거
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix.masked_fill_(mask, float('-inf'))

        # positive 인덱스 설정
        pos_indices = torch.arange(batch_size, device=device)
        labels = torch.cat([pos_indices + batch_size, pos_indices], dim=0)  # (2B,)

        # Positive 쌍 마스크 설정 (이들은 항상 weight = 1이어야 함)
        positive_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = True
            positive_mask[i + batch_size, i] = True
            
        # reweighting mask 생성
        with torch.no_grad():
            fn_mask = (sim_matrix > similarity_threshold) & (~positive_mask) & (~mask)
            weights = torch.ones_like(sim_matrix)
            weights[fn_mask] = 0.7  # false negative 추정 쌍의 weight 낮춤
            # positive pair는 항상 weight = 1로 명시적으로 보장   
            weights[positive_mask] = 1.0

        # softmax 계산을 위한 weighted sim matrix
        logits = sim_matrix * weights

        # cross_entropy를 위한 softmax 대상 정렬
        loss = F.cross_entropy(logits, labels)
        return loss
    ```

### 3. 특징 맵 추출 및 저장 (utils.py)
- 입력 이미지에서 특징 벡터를 추출하고 병합한 전체 featuremap .npy와 이미지별 특징벡터에 대응하는 file명을 filename .npy 파일로 저장합니다.

---

## 📈 Neptune.ai 로깅
- 학습 및 검증 손실, 모델 체크포인트를 Neptune.ai에 로깅합니다.
- `.env` 파일에 Neptune 프로젝트 이름과 API 토큰을 설정해야 합니다.
    ```
    PROJECT_NAME=your_project_name
    API_TOKEN=your_api_token
    ```

---

## 📋 Output Featuremap 구조
featuremap/
├──  combined_featuremap_filenames.npy #featuremap 과 filename mapping
├──  combined_featuremap.npy # featuremap 전부 stack

---

## 📧 문의
프로젝트 관련 문의는 못 받으니 알아서 하세요~

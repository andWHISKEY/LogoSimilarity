import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import neptune
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable the pixel limit warning
import random
from model import SimCLRModel 
from loss import nt_xent_loss_with_reweighting  
from dataset import ContrastiveTransformations, LogoDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import EarlyStopping

# 1. Neptune 초기화 및 파라미터 설정
load_dotenv() # .env 파일에서 환경 변수 로드

# Neptune 프로젝트 및 API 토큰 불러옴
PROJECT_NAME = os.getenv('PROJECT_NAME')
API_TOKEN = os.getenv('API_TOKEN')

# Neptune 실험 초기화 및 실행
run = neptune.init_run(
    project=PROJECT_NAME,
    api_token=API_TOKEN,
    name="swin_transformer_logo_similarity", 
    tags=["server","KR","changed loss","SimCLR", "Logo"])

# hyperparameters 설정
params = {
    "batch_size": 160, #160-4gpu
    "num_epochs": 50,
    "learning_rate": 3e-4,
    "temperature": 0.5,
    "image_size": 224,
    "dataset_folder": "/home/src/markcloud-vienna-code-classifier/data/ORIGINAL_KR_FILTERED",  # 실제 데이터셋 경로로 수정
    "backbone": "swin_base_patch4_window7_224",
    "projection_dim": 128,
    "hidden_dim": 2048,
    "num_workers": 0
}

# hyperparameters 변수로 저장 (이 코드 효울적이게 짜는 법: locals().update(params)) -> debug하기 힘듬
batch_size = params["batch_size"]
num_epochs = params["num_epochs"]
learning_rate = params["learning_rate"]
temperature = params["temperature"]
image_size = params["image_size"]
dataset_folder = params["dataset_folder"]
backbone = params["backbone"]
projection_dim = params["projection_dim"]
hidden_dim = params["hidden_dim"]
num_workers = params["num_workers"]

# 데이터셋 폴더 존재 여부 확인
if not os.path.exists(params["dataset_folder"]):
    raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")

# 2. 데이터 증강 및 데이터셋 정의: Contrastive Transformations
# base_transform 정의 (기본적인 이미지 전처리)
base_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)), # 랜덤 크롭
    transforms.RandomHorizontalFlip(), # 랜덤 Horizon Flip
    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),           # ← 추가
    transforms.GaussianBlur(kernel_size=23),     # ← 추가
    transforms.ToTensor(), # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 데이터셋의 평균과 표준편차로 정규화
                         std=[0.229, 0.224, 0.225])  # 우리가 가진 데이터셋 mean, std확인하기에는 너무 많았음
])

# Contrastive Transformations 정의 (2개의 augmented된 view로 변환)
contrastive_transform = ContrastiveTransformations(base_transform, n_views=2)

# 데이터셋 경로에서 파일 리스트를 가져옴
image_paths = [os.path.join(dataset_folder, fname)
               for fname in os.listdir(dataset_folder)
               if fname.lower().endswith(('.jpg', '.png'))]

# 사전 테스트를 위해 1만장만 무작위 샘플링
# image_paths = random.sample(image_paths, 1000)

# 데이터셋을 9:1로 나눔 (train:val)
train_paths, val_paths = train_test_split(image_paths, test_size=0.1, random_state=42)

# 데이터셋 생성
train_dataset = LogoDataset(train_paths, transform=contrastive_transform)
val_dataset = LogoDataset(val_paths, transform=contrastive_transform)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 데이터셋 길이 출력
print(f"Train 데이터셋 길이: {len(train_dataset)}")
print(f"Validation 데이터셋 길이: {len(val_dataset)}")

# 3. 학습 준비
# GPU 사용 여부 확인
if torch.cuda.is_available():
    print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
    num_gpus = torch.cuda.device_count()
    print(f"사용 가능한 GPU 개수: {num_gpus}")
else:
    print("GPU 사용 불가능: CPU 사용 중")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 초기화 및 dataparallel 적용
model = SimCLRModel(backbone_name=backbone, projection_dim=projection_dim, hidden_dim=hidden_dim)
# 백본 freeze
for param in model.backbone.parameters():
    param.requires_grad = False

model = nn.DataParallel(model)
model.to(device)

print(f"torch sees {torch.cuda.device_count()} GPUs")
if isinstance(model, nn.DataParallel):
    print(f"DataParallel 사용 중, device_ids = {model.device_ids}")
else:
    print("DataParallel 사용 안 함")

# backbone & head 둘다 같은 learning_rate로로 학습할 떄
# optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer 사용

# backbone & head 다른 learning_rate로 학습할 떄
# optimizer = optim.Adam([
#     {"params": model.module.backbone.parameters(),      "lr": 1e-5},   # 예: 1e-5
#     {"params": model.module.projection_head.parameters(), "lr": 1e-3},  # 예: 1e-3
# ])

# projection head만 학습
optimizer = optim.Adam(model.module.projection_head.parameters(),lr=learning_rate)

# learning rate scheduler 설정
# CosineAnnealingLR: 학습률을 코사인 함수에 따라 조정
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
early_stopping = EarlyStopping(patience=5, verbose=True, path='models/KR_edited_loss_SimCLR_best.pth')

# 4. 학습 루프 (Neptune 로깅 포함)
warmup_epochs = 10
backbone_lr = learning_rate * 0.1   # (head lr의 1/10 정도)

for epoch in range(num_epochs):
    """ 초반 워밍업(예: 0–10 에폭) 동안에는 backbone 완전 동결 
    그 다음에는 backbone 일부(또는 전체)를 서서히 해동(fine‑tune) """
    if epoch == warmup_epochs:
        # backbone 파라미터 전부 또는 일부 해동
        for name, param in model.module.backbone.named_parameters():
            if 'layers.3' in name or 'layers.2' in name:
                param.requires_grad = True

        # ptimizer 재생성: 두 개의 param_group
        optimizer = optim.Adam([
            {'params': model.module.projection_head.parameters(),'lr': learning_rate},
            {'params': filter(lambda p: p.requires_grad, model.module.backbone.parameters()), 
                                                   'lr': backbone_lr},
        ])

    model.train()
    epoch_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for batch in train_loader_tqdm:
        views, _ = batch  # views의 shape: [batch_size, 2, C, H, W]
        view1 = views[:, 0, :, :, :].to(device)  # 첫 번째 뷰
        view2 = views[:, 1, :, :, :].to(device)  # 두 번째 뷰
        
        optimizer.zero_grad()
        z1 = model(view1)
        z2 = model(view2)
        loss = nt_xent_loss_with_reweighting(z1, z2, temperature=temperature,similarity_threshold=0.9)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # tqdm에 현재 배치 손실 표시
        train_loader_tqdm.set_postfix(loss=loss.item())
        # GPU 메모리 캐시 해제
        torch.cuda.empty_cache()

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    run["train/epoch_loss"].log(avg_loss,step=epoch + 1)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs}", unit="batch")
        for batch in val_loader_tqdm:
            views, _ = batch
            view1 = torch.stack([v[0] for v in views]).to(device)
            view2 = torch.stack([v[1] for v in views]).to(device)
            
            z1 = model(view1)
            z2 = model(view2)
            loss = nt_xent_loss_with_reweighting(z1, z2, temperature=temperature, similarity_threshold=0.9)
            val_loss += loss.item()

            # tqdm에 현재 배치 손실 표시
            val_loader_tqdm.set_postfix(loss=loss.item())
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")
    run["val/epoch_loss"].log(avg_val_loss, step=epoch + 1)

    # 최신 모델 저장
    # 모델 저장 경로 설정하고 난 다음 run을 통해 neptune에 업로드 
    latest_model_path = "models/KR_edited_loss_SimCLR_latest.pth"
    torch.save(model.module.state_dict(), latest_model_path)
    run["model/latest"].upload(latest_model_path)

    # Early stopping 체크
    early_stopping(avg_val_loss, model.module)

run.stop()
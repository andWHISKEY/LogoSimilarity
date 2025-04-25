import os
import torch
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable the pixel limit warning
from torchvision import transforms
from model import SimCLRModel
from tqdm import tqdm


# 모델 로드 함수
def load_model(model_path, backbone_name, projection_dim, hidden_dim, device):
    model = SimCLRModel(backbone_name=backbone_name, projection_dim=projection_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 평가 모드로 설정
    return model

# 이미지 전처리 함수
def get_transform(image_size=224, gray=False):
    """
    이미지 전처리 파이프라인을 반환하는 함수.
    
    Args:
        image_size (int): 이미지 크기 (정사각형 기준).
        gray (bool): True일 경우 Grayscale 전처리 적용, False면 RGB 유지.
        
    Returns:
        torchvision.transforms.Compose: 전처리 파이프라인.
    """
    if gray:
        print("⚙️ Grayscale 전처리 적용 중 (모양 중심 비교)")
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 3채널 유지
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x/255.), # 강제 [0,1] 스케일링
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        print("⚙️ RGB 전처리 적용 중 (색상 포함)")
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x/255.), # 강제 [0,1] 스케일링
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

# 특징 맵 추출 및 저장 함수
def extract_and_save_features(model, image_paths, transform, output_folder, device):
    """
    여러 이미지에서 특징 맵을 추출하고, 파일 이름을 그대로 사용하여 .npy 파일로 저장하는 함수.

    Args:
        model (torch.nn.Module): 사전 학습된 SimCLR 모델.
        image_paths (list): 처리할 이미지 파일 경로들의 리스트.
        transform (torchvision.transforms.Compose): 이미지 전처리 파이프라인.
        output_folder (str): 특징 맵을 저장할 출력 폴더 경로.
        device (torch.device): GPU 또는 CPU 장치.

    Returns:
        None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 출력 폴더 생성

    for image_path in tqdm(image_paths, desc="Extracting and Saving Features"):
        try:
            # 이미지 로드 및 전처리
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            # 특징 맵 추출
            with torch.no_grad():
                feature = model.forward_features(image_tensor)
                feature = feature.cpu().numpy().flatten()  # 1D 벡터로 변환

            # 파일 이름 그대로 .npy로 저장
            file_name = os.path.splitext(os.path.basename(image_path))[0] + ".npy"
            output_path = os.path.join(output_folder, file_name)
            np.save(output_path, feature)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def extract_feature_from_image(model, image_path, transform, device):
    """
    단일 이미지(Input image)에서 특징 맵을 추출하는 함수.

    Args:
        model (torch.nn.Module): 사전 학습된 SimCLR 모델.
        image_path (str): 입력 이미지 경로.
        transform (torchvision.transforms.Compose): 이미지 전처리 파이프라인.
        device (torch.device): GPU 또는 CPU 장치.

    Returns:
        np.ndarray: 추출된 특징 맵 (1D 벡터).
    """
    try:
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 특징 맵 추출
        with torch.no_grad():
            feature = model.forward_features(image_tensor)
            feature = feature.cpu().numpy().flatten()  # 1D 벡터로 변환

        return feature
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def create_combined_featuremap(input_folder, model, transform, device, output_folder, output_featuremap="combined_featuremap.npy", output_filenames="filenames.npy"):
    """
    전체 이미지의 featuremap을 하나의 .npy 파일로 저장하고, 파일 이름 리스트를 별도로 저장.

    Args:
        input_folder (str): 입력 이미지 폴더 경로.
        model (torch.nn.Module): SimCLR 모델.
        transform (torchvision.transforms.Compose): 이미지 전처리 파이프라인.
        device (torch.device): GPU 또는 CPU 장치.
        output_featuremap (str): 저장할 featuremap 파일 경로.
        output_filenames (str): 저장할 파일 이름 리스트 경로.

    Returns:
        None
    """
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"⚙️ 출력 폴더 생성 완료: {output_folder}")
    
    # 출력 파일 경로 설정
    output_featuremap = os.path.join(output_folder, output_featuremap)
    output_filenames = os.path.join(output_folder, output_filenames)
    print(f"⚙️ 출력 파일 경로 설정 완료: {output_featuremap}, {output_filenames}")

    features = []
    filenames = []

    # 입력 폴더의 모든 이미지 처리
    print("이미지에서 featuremap 추출 중...")
    image_paths = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.lower().endswith(('.jpg', '.png'))]

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            feature = extract_feature_from_image(model, image_path, transform, device)
            features.append(feature)
            filenames.append(os.path.splitext(os.path.basename(image_path))[0])  # 파일명에서 확장자 제거
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Featuremap과 파일 이름 저장
    features = np.array(features)
    np.save(output_featuremap, features)
    np.save(output_filenames, filenames)

    print(f"Featuremap 저장 완료: {output_featuremap}")
    print(f"파일 이름 리스트 저장 완료: {output_filenames}")

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''모델 저장'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss    

# 주요 실행 코드
if __name__ == "__main__":
    # 설정
    model_path = "models/KR_reweighting_simclr_swin_transformer_pretrained.pth"  # Best 모델 경로
    input_folder = "/home/src/markcloud-vienna-code-classifier/data/ORIGINAL_KR"  # 입력 이미지 폴더 경로
    # input_folder = "/home/markcloud-vienna-code-classifier/data/ORIGINAL_KR"
    output_folder = "markview_image/output/featuremap"  # 출력 특징 맵 저장 폴더 경로
    backbone_name = "swin_base_patch4_window7_224"
    output_featuremap = "combined_grayscale_featuremaps.npy"  # 저장할 featuremap 파일 경로
    output_filenames = "combined_grayscale_featuremap_filenames.npy"  # 저장할 파일 이름 리스트 경로
    image_path="/path/to/input.jpg" # 입력 이미지 경로
    image_size = 224
    projection_dim = 128
    hidden_dim = 2048

    # GPU 사용 여부 확인
    if torch.cuda.is_available():
        print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        num_gpus = torch.cuda.device_count()
        print(f"사용 가능한 GPU 개수: {num_gpus}")
    else:
        print("GPU 사용 불가능: CPU 사용 중")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 이미지 전처리
    transform = get_transform(image_size, gray=True)

    # 입력 이미지 경로 가져오기
    image_paths = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder)
                   if fname.lower().endswith(('.jpg', '.png'))]

    # 모델 로드
    model = load_model(model_path, backbone_name, projection_dim, hidden_dim, device)

    # 특징 맵 추출 및 저장
    # extract_and_save_features(model, image_paths, transform, output_folder, device)

    # 특징 맵만 추출
    #featuremap = extract_feature_from_image(model, image_path, transform, device)

    # 전체 Featuremap 합친 것 생성 및 저장
    create_combined_featuremap(input_folder, model, transform, device, output_folder, output_featuremap, output_filenames)
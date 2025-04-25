import os
import numpy as np
import torch
from utils import extract_feature_from_image, load_model, get_transform
from PIL import Image
import matplotlib.pyplot as plt

def find_top_k_similar_from_combined(input_image, combined_featuremap, filenames, model, transform, device, top_k=20):
    """
    입력 이미지와 combined_featuremap.npy 간의 유사도를 계산하여 상위 k개의 파일 이름을 출력.

    Args:
        input_image (str): 입력 이미지 경로.
        combined_featuremap (str): 저장된 featuremap 파일 경로.
        filenames (str): 저장된 파일 이름 리스트 경로.
        model (torch.nn.Module): SimCLR 모델.
        transform (torchvision.transforms.Compose): 이미지 전처리 파이프라인.
        device (torch.device): GPU 또는 CPU 장치.
        top_k (int): 상위 유사 featuremap 개수.

    Returns:
        list: 상위 k개의 유사한 파일 이름.
    """
    # 입력 이미지의 특징 맵 추출
    print("입력 이미지의 특징 맵 추출 중...")
    input_feature = extract_feature_from_image(model, input_image, transform, device)

    # 저장된 featuremap과 파일 이름 로드
    print("저장된 featuremap 및 파일 이름 로드 중...")
    featuremaps = np.load(combined_featuremap)
    filenames = np.load(filenames)

    # 코사인 유사도 계산
    print("코사인 유사도 계산 중...")
    input_feature = input_feature.reshape(1, -1)
    similarities = np.dot(featuremaps, input_feature.T).squeeze()

    # 상위 k개의 유사한 featuremap 찾기
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    top_k_filenames = [filenames[idx] for idx in top_k_indices]

    return top_k_filenames

def visualize_top_k_images(input_image, top_k_filenames, input_folder):
    """
    입력 이미지와 상위 k개의 유사한 이미지를 시각적으로 출력.

    Args:
        input_image (str): 입력 이미지 경로.
        top_k_filenames (list): 상위 k개의 유사한 파일 이름 리스트.
        input_folder (str): 입력 이미지가 저장된 폴더 경로.

    Returns:
        None
    """
    # 입력 이미지 로드
    input_img = Image.open(input_image).convert("RGB")

    # 상위 k개의 유사한 이미지 로드
    similar_images = []
    for fname in top_k_filenames:
        image_path_jpg = os.path.join(input_folder, fname + ".jpg")
        image_path_png = os.path.join(input_folder, fname + ".png")

        # .jpg 시도 후 없으면 .png 시도
        if os.path.exists(image_path_jpg):
            similar_images.append(Image.open(image_path_jpg).convert("RGB"))
        elif os.path.exists(image_path_png):
            similar_images.append(Image.open(image_path_png).convert("RGB"))
        else:
            print(f"Warning: File not found for {fname} (tried .jpg and .png)")

    # 시각화
    fig, axes = plt.subplots(1, len(similar_images) + 1, figsize=(15, 5))
    axes[0].imshow(input_img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    for i, img in enumerate(similar_images):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"Similar {i + 1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    # 결과 이미지를 파일로 저장
    output_file = "output/top_k_similar_images.png"
    plt.savefig(output_file)
    plt.close()
    print(f"결과 이미지가 저장되었습니다: {output_file}")

# 주요 실행 코드
if __name__ == "__main__":
    # 설정
    input_image = "input/babyshark.jpg"  # 입력 이미지 경로
    model_path = "models/KR_reweighting_simclr_swin_transformer_pretrained.pth"  # 저장된 모델 경로
    backbone_name = "swin_base_patch4_window7_224"
    projection_dim = 128
    hidden_dim = 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_featuremap = "output/featuremap/combined_featuremaps.npy"  # 저장된 featuremap 파일 경로
    filenames = "output/featuremap/combined_featuremap_filenames.npy"  # 저장된 파일 이름 리스트 경로
    input_folder = "/home/src/markcloud-vienna-code-classifier/data/ORIGINAL_KR"  # 입력 이미지가 저장된 폴더 경로
    top_k = 20  # 상위 유사 featuremap 개수

    # 이미지 전처리
    transform = get_transform(image_size=224,gray=False)

    # 모델 로드
    model = load_model(model_path, backbone_name, projection_dim, hidden_dim, device)
    # 상위 20개의 유사한 파일 이름 찾기
    top_k_filenames = find_top_k_similar_from_combined(input_image, combined_featuremap, filenames, model, transform, device, top_k)

    # 결과 출력
    print(f"입력 이미지와 가장 유사한 상위 {top_k}개의 파일 이름:")
    for i, filename in enumerate(top_k_filenames):
        print(f"{i + 1}: {filename}")

    # 상위 k개의 유사한 이미지 시각화
    visualize_top_k_images(input_image, top_k_filenames, input_folder)
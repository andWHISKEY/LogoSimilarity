import os
import shutil
from PIL import Image
import imagehash
from tqdm import tqdm

# 원본 및 대상 폴더 설정
SOURCE_FOLDER = "/home/src/markcloud-vienna-code-classifier/data/ORIGINAL_KR"
TARGET_FOLDER = "/home/src/markcloud-vienna-code-classifier/data/ORIGINAL_KR_FILTERED"

# 대상 폴더가 없으면 생성
os.makedirs(TARGET_FOLDER, exist_ok=True)

# 사용할 해시 함수 (phash, ahash, dhash 중 선택 가능)
hash_func = imagehash.phash

# 이미 처리한 해시를 저장할 집합
seen_hashes = set()

# 처리할 이미지 파일 리스트
file_list = [
    f for f in os.listdir(SOURCE_FOLDER)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

# tqdm으로 진행률 표시 (중복 이미지 메시지 없이)
for fname in tqdm(file_list, desc="Processing images"):
    src_path = os.path.join(SOURCE_FOLDER, fname)
    try:
        img = Image.open(src_path)
        img_hash = str(hash_func(img))
    except Exception as e:
        tqdm.write(f"[ERROR] {fname} 처리 중 오류: {e}")
        continue

    if img_hash not in seen_hashes:
        seen_hashes.add(img_hash)
        dst_path = os.path.join(TARGET_FOLDER, fname)
        shutil.copy2(src_path, dst_path)

print("✅ 중복 제거 후 복사 완료")
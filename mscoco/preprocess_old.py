import json
import os
import pandas as pd
import csv  # CSV 모듈 추가

# 경로 설정
IMAGE_DIR = "/data_library/mscoco/train/"
ANNOTATION_PATH = "/data_library/mscoco/annotations/instances_train2017.json"
CAPTION_PATH = "/data_library/mscoco/annotations/captions_train2017.json"
OUTPUT_CSV_PATH = "/data_library/mscoco/train_image_caption.csv"

# COCO annotation 파일 로드
with open(ANNOTATION_PATH, 'r') as f:
    coco_data = json.load(f)

with open(CAPTION_PATH, 'r') as f:
    caption_data = json.load(f)

# COCO category ID to name 매핑 생성
category_mapping = {category['id']: category['name'] for category in coco_data['categories']}

# 이미지 ID별 annotation 매핑 (가장 큰 객체 찾기)
annotations = {}
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    bbox = annotation['bbox']  # [x, y, width, height]
    area = bbox[2] * bbox[3]  # Bounding Box의 면적 계산

    if image_id not in annotations or area > annotations[image_id][1]:
        annotations[image_id] = (category_mapping.get(category_id, "Unknown"), area)  # 가장 큰 객체 저장

# 이미지 ID별 첫 번째 캡션 매핑 (리스트 활용하여 첫 번째 캡션 가져오기)
captions = {}
for caption in caption_data['annotations']:
    image_id = caption['image_id']
    if image_id not in captions:
        captions[image_id] = []  # 리스트로 캡션 저장
    clean_caption = caption['caption'].replace("\"", "").replace(",", " ").replace("\n", " ")  # " , \n 제거
    captions[image_id].append(clean_caption)

# 이미지 파일 리스트 가져오기
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])

# 이미지 파일, 가장 큰 객체 라벨 (이름), 첫 번째 캡션 매핑
data = []
for image_file in image_files:
    image_id = int(image_file.lstrip("0").split(".")[0])  # 파일명에서 image_id 추출
    label = annotations.get(image_id, ("Unknown", 0))[0]  # 해당 이미지의 가장 큰 객체 라벨(이름) 가져오기
    caption = captions.get(image_id, ["No caption available"])[0]  # 첫 번째 캡션 가져오기 (없으면 기본값)
    
    if label != "Unknown":
        data.append((image_file, label, caption))

# DataFrame으로 변환 후 CSV 저장
df = pd.DataFrame(data, columns=['image', 'label', 'caption'])
df.to_csv(OUTPUT_CSV_PATH, index=False, quoting=csv.QUOTE_MINIMAL)  # 필요한 경우만 따옴표 사용

print(f"Saved image classification labels with captions to {OUTPUT_CSV_PATH}")

# 데이터 다운로드 
from roboflow import Roboflow

roboflow_api = ''
# 참고 : https://docs.roboflow.com/api-reference/authentication

rf = Roboflow(api_key=roboflow_api)
project = rf.workspace("f22raptor").project("-ofkz6")
version = project.version(6)
dataset = version.download("folder")

# 폴더 이름 변경 
import os 
import chardet
path = '티니핑-6/train'
count = 0 

for class_name in os.listdir(path):
    detected = chardet.detect(class_name.encode())
    encoding = detected['encoding']

    try : 
        fixed_name = class_name.encode('latin1').decode(encoding)
        old_path = os.path.join(path, class_name)
        new_path = os.path.join(path, fixed_name)
        os.rename(old_path, new_path)
        count += 1 
    
    except : 
        print(f'변경 실패 : {class_name}')

print(f'변경 이름 수 : {count}')

# 학습 검증 테스트로 분리 
import os
import random 
import shutil

src_dir = '티니핑-6/train'

train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'

train_ratio = 0.8 
val_ratio = 0.1
test_ratio = 0.1 

for class_name in os.listdir(src_dir):
    src_class_path = os.path.join(src_dir, class_name)

    train_class_path = os.path.join(train_dir, class_name)
    val_class_path = os.path.join(val_dir, class_name)
    test_class_path = os.path.join(test_dir, class_name)

    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(val_class_path, exist_ok=True)
    os.makedirs(test_class_path, exist_ok=True)

    images = os.listdir(src_class_path)
    random.shuffle(images)

    num_images = len(images)
    train_end = int(num_images * train_ratio)
    val_end = int(num_images * (train_ratio+val_ratio))

    for i, image in enumerate(images): 
        src_path = os.path.join(src_class_path, image)
        if i < train_end: 
            dst_path = os.path.join(train_class_path, image)
        elif i < val_end: 
            dst_path = os.path.join(val_class_path, image)
        else: 
            dst_path = os.path.join(test_class_path, image)
        shutil.copy2(src_path, dst_path)

total_train_images = sum(len(os.listdir(os.path.join(train_dir, class_name))) for class_name in os.listdir(train_dir))
total_val_images = sum(len(os.listdir(os.path.join(val_dir, class_name))) for class_name in os.listdir(val_dir))
total_test_images = sum(len(os.listdir(os.path.join(test_dir, class_name))) for class_name in os.listdir(test_dir))

print(f'학습 데이터셋 수 : {total_train_images}')
print(f'검증 데이터셋 수 : {total_val_images}')
print(f'테스트 데이터셋 수 : {total_test_images}')
print(f'총 데이터 수 : {total_train_images+total_val_images+total_test_images}')

# 학습 데이터셋 수 : 7879
# 검증 데이터셋 수 : 986
# 테스트 데이터셋 수 : 987
# 총 데이터 수 : 9852
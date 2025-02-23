import os
import torch
import random 
from PIL import Image 
from model import teeniefier
from torchvision import transforms
from torch.nn.functional import softmax

# 추론 세팅 
test_folder = 'dataset/test'
trained_model_path = 'best_model.ckpt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 준비 
target_ping = random.sample(os.listdir(test_folder), 1)[0]
target_image = random.sample(os.listdir(os.path.join(test_folder, target_ping)), 1)[0]

image_origin = Image.open(os.path.join(test_folder, 
                                       target_ping,
                                       target_image)).convert('RGB')

# 데이터 전처리 
image_size = 28 
mean = (0.6151984, 0.51760532, 0.46836003)
std = (0.26411435, 0.24187316, 0.264022790)

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(), 
    transforms.Normalize(mean, std)
])

image = transform(image_origin)
image = image.unsqueeze(0)
image = image.to(device)

# 모델 준비 
## 모델 객체 생성 
model = teeniefier(num_teenieping=len(os.listdir(test_folder)))

## 학습 결과 불러오기 
trained_model = torch.load(trained_model_path)
model.load_state_dict(trained_model['model'])
model = model.to(device)

# 예측 진행 
predict = model(image)

# 예측 결과 시각화 
prob = softmax(predict, dim=1)
value, index = torch.max(prob, dim=1)

image_origin.show() 
print(f'사진의 티니핑 이름은 : {target_ping}')
print(f'AI 모델의 추론 결과 : {trained_model['classes'][index.item()]} ({value.item()*100:.2f}%)')
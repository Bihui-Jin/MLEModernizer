# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9811

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch # 파이토치
import random
import numpy as np
import os

seed = 50
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)                # 파이썬 난수 생성기 시드 고정
np.random.seed(seed)             # 넘파이 난수 생성기 시드 고정
torch.manual_seed(seed)          # 파이토치 난수 생성기 시드 고정 (CPU 사용 시)
torch.cuda.manual_seed(seed)     # 파이토치 난수 생성기 시드 고정 (GPU 사용 시)
torch.cuda.manual_seed_all(seed) # 파이토치 난수 생성기 시드 고정 (멀티GPU 사용 시)
torch.backends.cudnn.deterministic = True # 확정적 연산 사용
torch.backends.cudnn.benchmark = False    # 벤치마크 기능 해제
torch.backends.cudnn.enabled = False      # cudnn 사용 해제


## === cell 1
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


## === cell 2
device


## === cell 3
import pandas as pd

data_path = '/kaggle/input/aerial-cactus-identification/'

labels = pd.read_csv(data_path + 'train.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')


## === cell 4
from zipfile import ZipFile

with ZipFile(data_path + 'train.zip') as zipper:
    zipper.extractall()
    
with ZipFile(data_path + 'test.zip') as zipper:
    zipper.extractall()


## === cell 5
from sklearn.model_selection import train_test_split

train, valid = train_test_split(labels,
                                test_size = 0.1, # (1)
                                stratify = labels['has_cactus'], # (2)
                                random_state = 50)


## === cell 6
print('훈련 데이터 개수:', len(train))
print('검증 데이터 개수:', len(valid))


## === cell 7
import cv2 # OpenCV 라이브러리
from torch.utils.data import Dataset # 데이터 생성을 위한 클래스


## === cell 8
class ImageDataset(Dataset):
    def __init__(self, df, img_dir = './', transform = None):
        super().__init__() ## (2) 상속받은 Dataset의 생성자 호출
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]                  # 이미지 ID
        img_path = self.img_dir + img_id               # (6) 이미지 파일 경로
        image = cv2.imread(img_path)                   # 이미지 파일 읽기 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이미지 색상 보정
        label = self.df.iloc[idx, 1]                   # 이미지 레이블(타깃값)
        
        if self.transform is not None:
            image = self.transform(image) # (7) 변환기가 있다면 이미지 변환
        return image, label # (8)


## === cell 9
from torchvision import transforms # 이미지 변환을 위한 모듈

transform = transforms.ToTensor()


## === cell 10
dataset_train = ImageDataset(df = train, img_dir = 'train/', transform = transform)
dataset_valid = ImageDataset(df = valid, img_dir = 'train/', transform = transform)


## === cell 11
from torch.utils.data import DataLoader # 데이터 로더 클래스

loader_train = DataLoader(dataset = dataset_train, batch_size = 32, shuffle = True)
loader_valid = DataLoader(dataset = dataset_valid, batch_size = 32, shuffle = False)


## === cell 12
import torch.nn as nn # 신경망 모듈
import torch.nn.functional as F # 신경망 모듈에서 자주 사용되는 함수


## === cell 13
class Model(nn.Module):
    def __init__(self):
        super().__init__() # 상속받은 nn.Module의 __init__() 메서드 호출
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, 
                               kernel_size = 3, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, 
                               kernel_size = 3, padding = 2)
        self.max_pool = nn.MaxPool2d(kernel_size = 2)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2)
        self.fc = nn.Linear(in_features = 64 * 4 * 4, out_features = 2)
    
    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.avg_pool(x)
        x = x.view(-1, 64 * 4 * 4) # 평탄화
        x = self.fc(x)
        return x


## === cell 14
model = Model().to(device)
model


## === cell 15
criterion = nn.CrossEntropyLoss()


## === cell 16
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) ## lr : learing rate (학습률)


## === cell 17
import os

if not os.path.isdir("train"):
    candidate_dirs = [
        "/kaggle/input/aerial-cactus-identification",
        "/kaggle/data/aerial-cactus-identification",
        "/kaggle/working/aerial-cactus-identification",
        "/kaggle/working",
        "/kaggle/data",
        "/kaggle/input",
    ]
    for base in candidate_dirs:
        if os.path.isdir(os.path.join(base, "train")):
            os.chdir(base)
            break

epochs = 10  # 총 에폭
for epoch in range(epochs):
    epoch_loss = 0  # 에폭별 손실값 초기화

    for images, labels in loader_train:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()  # (7) 역전파 수행
        loss.backward()
        optimizer.step()

    print(f"에폭 [{epoch + 1}/{epochs}] - 손실값 : {epoch_loss/len(loader_train):.4f}")


## === cell 18
from sklearn.metrics import roc_auc_score # ROC AUC 점수 계산 함수 임포트
true_list = []
preds_list = []


## === cell 19
model.eval() # (1) 모델을 평가 상태로 설정

with torch.no_grad(): # (2) 기울기 계산 비활성화
    for images, labels in loader_valid: # (3)
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        preds = torch.softmax(outputs.cpu(), dim = 1)[:, 1] # (5) 예측 확률
        true = labels.cpu() # (6) 실젯값
        preds_list.extend(preds)
        true_list.extend(true)
        
print(f'검증 데이터 ROC AUC : {roc_auc_score(true_list, preds_list):.4f}')


## === cell 20
dataset_test = ImageDataset(df = submission, img_dir = 'test/', transform = transform)
loader_test = DataLoader(dataset = dataset_test, batch_size = 32, shuffle = False)


## === cell 21
model.eval() # 모델을 평가 상태로 설정

preds = [] # (1) 타깃 예측값 저장용 리스트 초기화

with torch.no_grad(): # 기울기 계산 비활성화
    for images, _ in loader_test:
        images = images.to(device)
        
        outputs = model(images)
        preds_part = torch.softmax(outputs.cpu(), dim = 1)[:, 1].tolist()
        preds.extend(preds_part)


## === cell 22
submission['has_cactus'] = preds
submission.to_csv('submission.csv', index = False)


## === cell 23
import os
import shutil

for d in ("./train", "./test"):
    if os.path.isdir(d):
        try:
            shutil.rmtree(d)
        except OSError:
            pass

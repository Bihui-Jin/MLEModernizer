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

3.9

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

0.9843

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd

data_path = "/kaggle/input/aerial-cactus-identification/"

labels = pd.read_csv(data_path + "train.csv")
submission = pd.read_csv(data_path + "sample_submission.csv")



## === cell 1
from zipfile import ZipFile

with ZipFile(data_path + "train.zip") as zipper:
    zipper.extractall()

with ZipFile(data_path + "test.zip") as zipper:
    zipper.extractall()



## === cell 2
import torch  # 파이토치
import random
import numpy as np
import os

seed = 10
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False



## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 4
device



## === cell 5
from sklearn.model_selection import train_test_split

train, valid = train_test_split(
    labels, test_size=0.1, stratify=labels["has_cactus"], random_state=10
)



## === cell 6
print("훈련 데이터 개수:", len(train))
print("검증 데이터 개수:", len(valid))



## === cell 7
import cv2  # OpenCV 라이브러리
from torch.utils.data import Dataset  # 데이터 생성을 위한 클래스


class ImageDataset(Dataset):
    def __init__(self, df, img_dir="./", transform=None):
        super().__init__()  # 상속받은 Dataset의 __init__() 메서드 호출
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]  # 이미지 ID
        img_path = self.img_dir + img_id  # 이미지 파일 경로
        image = cv2.imread(img_path)  # 이미지 파일 읽기
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정

        label = int(self.df.iloc[idx, 1])  # 이미지 레이블(타깃 값)

        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 8
from torchvision import transforms  # 이미지 변환을 위한 모듈

transform = transforms.ToTensor()



## === cell 9
dataset_train = ImageDataset(df=train, img_dir="train/", transform=transform)
dataset_valid = ImageDataset(df=valid, img_dir="train/", transform=transform)




## === cell 10
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)



## === cell 11
from torch.utils.data import DataLoader  # 데이터 로더 생성을 위한 클래스

num_workers = 2

loader_train = DataLoader(
    dataset=dataset_train,
    batch_size=32,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
loader_valid = DataLoader(
    dataset=dataset_valid,
    batch_size=32,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)



## === cell 12
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 신경망 모듈에서 자주 사용되는 함수


class Model(nn.Module):
    def __init__(self):
        super().__init__()  # 상속받은 nn.Module의 __init__() 메서드 호출

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(in_features=64 * 4 * 4, out_features=2)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.avg_pool(x)
        x = x.view(-1, 64 * 4 * 4)  # 평탄화
        x = self.fc(x)
        return x




## === cell 13
pass



## === cell 14
model = Model().to(device)

model



## === cell 15
criterion = nn.CrossEntropyLoss()



## === cell 16
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



## === cell 17
import math

math.ceil(len(train) / 32)



## === cell 18
len(loader_train)



## === cell 19
import os


def _find_extraction_root():
    candidates = [
        os.getcwd(),
        "/kaggle/working",
        "/kaggle/working/aerial-cactus-identification",
        "/kaggle/input/aerial-cactus-identification",
    ]
    for base in candidates:
        if os.path.isdir(os.path.join(base, "train")) and os.path.isdir(
            os.path.join(base, "test")
        ):
            return base

    search_root = "/kaggle/working"
    if os.path.isdir(search_root):
        for root, dirs, _files in os.walk(search_root):
            if "train" in dirs and "test" in dirs:
                return root
    return None


root = _find_extraction_root()
if root is None:
    raise FileNotFoundError(
        "Could not find extracted 'train/' and 'test/' directories. "
        "Please verify that train.zip and test.zip were extracted successfully."
    )

os.chdir(root)

epochs = 10  # 총 에폭

for epoch in range(epochs):
    model.train()
    epoch_loss = 0  # 에폭별 손실값 초기화
    for images, labels in loader_train:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # 현재 배치에서의 손실 추가
    print(f"에폭 [{epoch+1}/{epochs}] - 손실값: {epoch_loss/len(loader_train):.4f}")



## === cell 20
from sklearn.metrics import roc_auc_score  # ROC AUC 점수 계산 함수

true_list = []
preds_list = []

model.eval()  # 모델을 평가 상태로 설정

with torch.no_grad():  # 기울기 계산 비활성
    for images, labels in loader_valid:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        outputs = model(images)
        preds = (
            torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        )  # 예측 확률값
        true = labels.detach().cpu().numpy()  # 실제값
        preds_list.extend(preds.tolist())
        true_list.extend(true.tolist())

    print(f"검증 데이터 ROC AUC : {roc_auc_score(true_list, preds_list):.4f}")



## === cell 21
dataset_test = ImageDataset(df=submission, img_dir="test/", transform=transform)
loader_test = DataLoader(
    dataset=dataset_test,
    batch_size=32,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)



## === cell 22
model.eval()  # 모델을 평가 상태로 설정

preds = []  # 타깃 예측 값 저장용 변수 초기화

with torch.no_grad():  # 기울기 계산 비활성
    for images, _ in loader_test:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds_part = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().tolist()
        preds.extend(preds_part)



## === cell 23
submission["has_cactus"] = preds
submission.to_csv("submission.csv", index=False)



## === cell 24
import os
import shutil

try:
    os.chdir("/kaggle/working")
except Exception:
    pass

for d in ("./train", "./test"):
    shutil.rmtree(d, ignore_errors=True)

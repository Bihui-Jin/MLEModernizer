# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.12

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.9809

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd

data_path = "/kaggle/input/aerial-cactus-identification/"

labels = pd.read_csv(data_path + "train.csv")
submission = pd.read_csv(data_path + "sample_submission.csv")



## === cell 2
labels.head()



## === cell 3
submission.head()



## === cell 4
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("font", size=15)
plt.figure(figsize=(7, 7))

label = ["Has cactus", "Hasn't cactus"]
plt.pie(labels["has_cactus"].value_counts(), labels=label, autopct="%.1f%%")



## === cell 5
from zipfile import ZipFile
import os

WORK_DIR = "/kaggle/working"
TRAIN_DIR = os.path.join(WORK_DIR, "train")
TEST_DIR = os.path.join(WORK_DIR, "test")

os.makedirs(WORK_DIR, exist_ok=True)

with ZipFile(os.path.join(data_path, "train.zip")) as zipper:
    zipper.extractall(path=WORK_DIR)

with ZipFile(os.path.join(data_path, "test.zip")) as zipper:
    zipper.extractall(path=WORK_DIR)



## === cell 6
import os

num_train = len(os.listdir(TRAIN_DIR))
num_test = len(os.listdir(TEST_DIR))

print(f"훈련 데이터 개수: {num_train}")
print(f"테스트 데이터 개수: {num_test}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2585657594.py in <cell line: 0>()
      2 
      3 # BUGFIX: Use extracted directories under /kaggle/working.
----> 4 num_train = len(os.listdir(TRAIN_DIR))
      5 num_test = len(os.listdir(TEST_DIR))
      6 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train'

## === cell 7
import matplotlib.gridspec as gridspec
import cv2  # OpenCV library

mpl.rc("font", size=7)
plt.figure(figsize=(15, 6))
grid = gridspec.GridSpec(2, 6)  # subplot rows 2, columns 6

last_has_cactus_img_name = labels[labels["has_cactus"] == 1]["id"][-12:]

for idx, img_name in enumerate(last_has_cactus_img_name):
    img_path = os.path.join(TRAIN_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정
    ax = plt.subplot(grid[idx])
    ax.imshow(image)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1399594052.py in <cell line: 0>()
     12     image = cv2.imread(img_path)
     13     if image is None:
---> 14         raise FileNotFoundError(f"Failed to read image: {img_path}")
     15     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정
     16     ax = plt.subplot(grid[idx])

FileNotFoundError: Failed to read image: /kaggle/working/train/63d42901f6156fca193e129ec00b7d2d.jpg

## === cell 8
plt.figure(figsize=(15, 6))
grid = gridspec.GridSpec(2, 6)  # subplot rows 2, columns 6

last_has_cactus_img_name = labels[labels["has_cactus"] == 0]["id"][-12:]

for idx, img_name in enumerate(last_has_cactus_img_name):
    img_path = os.path.join(TRAIN_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정
    ax = plt.subplot(grid[idx])
    ax.imshow(image)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/906091692.py in <cell line: 0>()
      8     image = cv2.imread(img_path)
      9     if image is None:
---> 10         raise FileNotFoundError(f"Failed to read image: {img_path}")
     11     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정
     12     ax = plt.subplot(grid[idx])

FileNotFoundError: Failed to read image: /kaggle/working/train/0b7e10689867bdce972723d7b288bbb8.jpg

## === cell 9
image.shape



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3458339860.py in <cell line: 0>()
      1 # 이미지의 가로 픽셀 수, 세로 픽셀 수, 채널 수 출력(RGB)
----> 2 image.shape
      3 

AttributeError: 'NoneType' object has no attribute 'shape'

## === cell 10
import torch
import random
import numpy as np
import os

seed = 50
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True  # 확정적 연산 사용
torch.backends.cudnn.benchmark = False  # 벤치마크 기능 해제



## === cell 11
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



## === cell 12
device



## === cell 13
import pandas as pd
from zipfile import ZipFile
import os

data_path = "/kaggle/input/aerial-cactus-identification/"

labels = pd.read_csv(os.path.join(data_path, "train.csv"))
submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))

WORK_DIR = "/kaggle/working"
TRAIN_DIR = os.path.join(WORK_DIR, "train")
TEST_DIR = os.path.join(WORK_DIR, "test")
os.makedirs(WORK_DIR, exist_ok=True)

if not os.path.isdir(TRAIN_DIR) or len(os.listdir(TRAIN_DIR)) == 0:
    with ZipFile(os.path.join(data_path, "train.zip")) as zipper:
        zipper.extractall(path=WORK_DIR)

if not os.path.isdir(TEST_DIR) or len(os.listdir(TEST_DIR)) == 0:
    with ZipFile(os.path.join(data_path, "test.zip")) as zipper:
        zipper.extractall(path=WORK_DIR)



## === cell 14
from sklearn.model_selection import train_test_split

train, valid = train_test_split(
    labels, test_size=0.1, stratify=labels["has_cactus"], random_state=50
)



## === cell 15
print("훈련 데이터 개수: ", len(train))
print("테스트 데이터 개수: ", len(valid))



## === cell 16
import cv2
from torch.utils.data import Dataset  # 데이터 생성을 위한 클래스
import os


class ImageDataset(Dataset):
    def __init__(self, df, img_dir="./", transform=None):
        super().__init__()  # 상속받은 Dataset의 생성자 호출
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]  # 이미지 ID
        img_path = os.path.join(self.img_dir, img_id)  # BUGFIX: robust path join
        image = cv2.imread(img_path)  # 이미지 파일 읽기
        if image is None:
            raise FileNotFoundError(f"cv2.imread failed for path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정
        label = int(self.df.iloc[idx, 1])  # 이미지 레이블(타깃값)

        if self.transform is not None:
            image = self.transform(image)  # 변환기가 있다면 이미지 변환
        return image, label




## === cell 17
from torchvision import transforms  # 이미지 변환을 위한 모듈

transform = transforms.ToTensor()



## === cell 18
dataset_train = ImageDataset(df=train, img_dir=TRAIN_DIR, transform=transform)
dataset_valid = ImageDataset(df=valid, img_dir=TRAIN_DIR, transform=transform)



## === cell 19
from torch.utils.data import DataLoader  # 데이터 로더 클래스

loader_train = DataLoader(
    dataset=dataset_train, batch_size=32, shuffle=True, num_workers=0
)
loader_valid = DataLoader(
    dataset=dataset_valid, batch_size=32, shuffle=False, num_workers=0
)



## === cell 20
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 신경망 모듈에서 자주 사용되는 함수


class Model(nn.Module):
    def __init__(self):
        super().__init__()  # 상속받은 nn.Module의 __init__() 메서드 호출

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=2
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




## === cell 21
model = Model().to(device)

model



## === cell 22
criterion = nn.CrossEntropyLoss()



## === cell 23
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



## === cell 24
import math

math.ceil(len(train) / 32)



## === cell 25
len(loader_train)



## === cell 26
epochs = 10  # 총 에폭
for epoch in range(epochs):
    epoch_loss = 0  # 에폭별 손실값 초기화

    for images, labels_batch in loader_train:
        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"에폭 [{epoch+1}/{epochs}] - 손실값: {epoch_loss/len(loader_train):.4f}")



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3318093957.py in <cell line: 0>()
      3     epoch_loss = 0  # 에폭별 손실값 초기화
      4 
----> 5     for images, labels_batch in loader_train:
      6         images = images.to(device)
      7         labels_batch = labels_batch.to(device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_11/701289603.py in __getitem__(self, idx)
     19         image = cv2.imread(img_path)  # 이미지 파일 읽기
     20         if image is None:
---> 21             raise FileNotFoundError(f"cv2.imread failed for path: {img_path}")
     22         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정
     23         label = int(self.df.iloc[idx, 1])  # 이미지 레이블(타깃값)

FileNotFoundError: cv2.imread failed for path: /kaggle/working/train/2511e6d2a75b372f57e6af0d7cb3c370.jpg

## === cell 27
from sklearn.metrics import roc_auc_score  # ROC AUC 점수 계산 함수 임포트

true_list = []
preds_list = []



## === cell 28
model.eval()  # 모델을 평가 상태로 설정

with torch.no_grad():  # 기울기 계산 비활성화
    for images, labels_batch in loader_valid:
        images = images.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(images)
        preds = torch.softmax(outputs.detach().cpu(), dim=1)[:, 1].numpy()  # 예측 확률
        true = labels_batch.detach().cpu().numpy()  # 실제값
        preds_list.extend(preds.tolist())
        true_list.extend(true.tolist())

print(f"검증 데이터 ROC AUC : {roc_auc_score(true_list, preds_list):.4f}")



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4175224446.py in <cell line: 0>()
      2 
      3 with torch.no_grad():  # 기울기 계산 비활성화
----> 4     for images, labels_batch in loader_valid:
      5         images = images.to(device)
      6         labels_batch = labels_batch.to(device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_11/701289603.py in __getitem__(self, idx)
     19         image = cv2.imread(img_path)  # 이미지 파일 읽기
     20         if image is None:
---> 21             raise FileNotFoundError(f"cv2.imread failed for path: {img_path}")
     22         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정
     23         label = int(self.df.iloc[idx, 1])  # 이미지 레이블(타깃값)

FileNotFoundError: cv2.imread failed for path: /kaggle/working/train/0eb5c8e2ece103d136cfd51ccc891c62.jpg

## === cell 29
dataset_test = ImageDataset(df=submission, img_dir=TEST_DIR, transform=transform)
loader_test = DataLoader(
    dataset=dataset_test, batch_size=32, shuffle=False, num_workers=0
)



## === cell 30
model.eval()  # 모델을 평가 상태로 설정

preds = []  # 타깃 예측값 저장용 리스트 초기화

with torch.no_grad():  # 기울기 계산 비활성화
    for images, _ in loader_test:
        images = images.to(device)

        outputs = model(images)
        preds_part = torch.softmax(outputs.detach().cpu(), dim=1)[:, 1].tolist()
        preds.extend(preds_part)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/36334971.py in <cell line: 0>()
      4 
      5 with torch.no_grad():  # 기울기 계산 비활성화
----> 6     for images, _ in loader_test:
      7         images = images.to(device)
      8 

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_11/701289603.py in __getitem__(self, idx)
     19         image = cv2.imread(img_path)  # 이미지 파일 읽기
     20         if image is None:
---> 21             raise FileNotFoundError(f"cv2.imread failed for path: {img_path}")
     22         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정
     23         label = int(self.df.iloc[idx, 1])  # 이미지 레이블(타깃값)

FileNotFoundError: cv2.imread failed for path: /kaggle/working/test/09034a34de0e2015a8a28dfe18f423f6.jpg

## === cell 31
torch.softmax(outputs.detach().cpu(), dim=1)[:, 1]



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/430531286.py in <cell line: 0>()
      1 # Sanity check: probability tensor example (last batch)
----> 2 torch.softmax(outputs.detach().cpu(), dim=1)[:, 1]
      3 

NameError: name 'outputs' is not defined

## === cell 32
torch.softmax(outputs.detach().cpu(), dim=1)[:, 1].tolist()



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1647229366.py in <cell line: 0>()
----> 1 torch.softmax(outputs.detach().cpu(), dim=1)[:, 1].tolist()
      2 

NameError: name 'outputs' is not defined

## === cell 33
if len(preds) != len(submission):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(preds)} preds for {len(submission)} rows."
    )

submission["has_cactus"] = preds
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1673857670.py in <cell line: 0>()
      1 # BUGFIX: Ensure preds length matches submission length, then write a valid .csv submission.
      2 if len(preds) != len(submission):
----> 3     raise RuntimeError(
      4         f"Prediction length mismatch: got {len(preds)} preds for {len(submission)} rows."
      5     )

RuntimeError: Prediction length mismatch: got 0 preds for 3325 rows.

## === cell 34
import shutil
import os

if os.path.isdir(TRAIN_DIR):
    shutil.rmtree(TRAIN_DIR)
if os.path.isdir(TEST_DIR):
    shutil.rmtree(TEST_DIR)
print("Cleanup done.")

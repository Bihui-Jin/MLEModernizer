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

3.12

# 3. Installed packages

No external packages required in the script and installed.

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

0.9774

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

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
label = ["Has cactus", "Hasn't cactus"]  # 타깃값 레이블

plt.pie(labels["has_cactus"].value_counts(), labels=label, autopct="%.1f%%")



## === cell 5
from zipfile import ZipFile

with ZipFile(data_path + "train.zip") as zipper:
    zipper.extractall()

with ZipFile(data_path + "test.zip") as zipper:
    zipper.extractall()



## === cell 6
import os

candidates = [
    ".",  # expected (train/, test/)
    "aerial-cactus-identification",  # common nested extraction folder
    "/kaggle/working",  # typical working dir
    "/kaggle/working/aerial-cactus-identification",
]

base_dir = None
for c in candidates:
    if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(
        os.path.join(c, "test")
    ):
        base_dir = c
        break

if base_dir is None:
    for root, dirs, _ in os.walk("."):
        if "train" in dirs and "test" in dirs:
            base_dir = root
            break

if base_dir is None:
    raise FileNotFoundError(
        "Could not find extracted 'train/' and 'test/' directories after unzip."
    )

if os.path.abspath(os.getcwd()) != os.path.abspath(base_dir):
    os.chdir(base_dir)

num_train = len(os.listdir("train/"))
num_test = len(os.listdir("test/"))

print(f"훈련 데이터 개수 : {num_train}")
print(f"테스트 데이터 개수 : {num_test}")



## === cell 7
import matplotlib.gridspec as gridspec
import cv2  # OpenCV 라이브러리 임포트

mpl.rc("font", size=7)
plt.figure(figsize=(15, 6))  # 전체 Figure 크기 설정
grid = gridspec.GridSpec(2, 6)  # 서브플롯 배치(2행 6열로 출력)

last_has_cactus_img_name = labels[labels["has_cactus"] == 1]["id"][-12:]

for idx, img_name in enumerate(last_has_cactus_img_name):
    img_path = "train/" + img_name  # 이미지 파일 경로
    image = cv2.imread(img_path)  # 이미지 파일 읽기
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정
    ax = plt.subplot(grid[idx])
    ax.imshow(image)  # 이미지 출력



## === cell 8
plt.figure(figsize=(15, 6))
grid = gridspec.GridSpec(2, 6)

last_hasnt_cactus_img_name = labels[labels["has_cactus"] == 0]["id"][-12:]

for idx, img_name in enumerate(last_hasnt_cactus_img_name):
    img_path = "train/" + img_name  # 이미지 파일 경로
    image = cv2.imread(img_path)  # 이미지 파일 읽기
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정
    ax = plt.subplot(grid[idx])
    ax.imshow(image)  # 이미지 출력



## === cell 9
image.shape



## === cell 10
import torch
import random
import numpy as np
import os

seed = 50
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)  # 파이썬 난수 생성기 시드 고정
np.random.seed(seed)  # 넘파이 난수 생성기 시드 고정
torch.manual_seed(seed)  # 파이토치 난수 생성기 시드 고정 (CPU 사용 시)
torch.cuda.manual_seed(seed)  # 파이토치 난수 생성기 시드 고정 (GPU 사용 시)
torch.cuda.manual_seed_all(seed)  # 파이토치 난수 생성기 시드 고정 (멀티 GPU 사용 시)
torch.backends.cudnn.deterministic = True  # 확정적 연산 사용
torch.backends.cudnn.benchmark = False  # 벤치마크 기능 해제
torch.backends.cudnn.enabled = (
    False  # cudnn 사용 해제 (typo fix: 'enabled' is the correct flag)
)



## === cell 11
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



## === cell 12
device



## === cell 13
import pandas as pd

data_path = "/kaggle/input/aerial-cactus-identification/"

labels = pd.read_csv(data_path + "train.csv")
submission = pd.read_csv(data_path + "sample_submission.csv")



## === cell 14
from zipfile import ZipFile

with ZipFile(data_path + "train.zip") as zipper:
    zipper.extractall()

with ZipFile(data_path + "test.zip") as zipper:
    zipper.extractall()



## === cell 15
import os

candidates = [
    ".",  # expected (train/, test/)
    "aerial-cactus-identification",
    "/kaggle/working",
    "/kaggle/working/aerial-cactus-identification",
]
base_dir = None
for c in candidates:
    if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(
        os.path.join(c, "test")
    ):
        base_dir = c
        break
if base_dir is None:
    for root, dirs, _ in os.walk("."):
        if "train" in dirs and "test" in dirs:
            base_dir = root
            break
if base_dir is None:
    raise FileNotFoundError("Could not locate extracted train/ and test/ directories.")

if os.path.abspath(os.getcwd()) != os.path.abspath(base_dir):
    os.chdir(base_dir)

print("Using working directory:", os.getcwd())
print(
    "train dir exists:",
    os.path.isdir("train"),
    "test dir exists:",
    os.path.isdir("test"),
)



## === cell 16
from sklearn.model_selection import train_test_split

train, valid = train_test_split(
    labels, test_size=0.1, stratify=labels["has_cactus"], random_state=50
)



## === cell 17
print("훈련 데이터 개수:", len(train))
print("검증 데이터 개수:", len(valid))



## === cell 18
import cv2  # OpenCV 라이브러리
from torch.utils.data import Dataset  # 데이터 생성을 위한 클래스




## === cell 19
class ImageDataset(Dataset):
    def __init__(self, df, img_dir="./", transform=None):
        super().__init__()  # 상속받은 Dataset의 생성자 호출
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]  # 이미지 ID
        img_path = self.img_dir + img_id  # 이미지 파일 경로
        image = cv2.imread(img_path)  # 이미지 파일 읽기
        if image is None:
            raise FileNotFoundError(f"cv2.imread failed for path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정
        label = self.df.iloc[idx, 1]  # 이미지 레이블(타깃값)

        if self.transform is not None:
            image = self.transform(image)  # 변환기가 있다면 이미지 변환
        return image, label




## === cell 20
from torchvision import transforms  # 이미지 변환을 위한 모듈

transform = transforms.ToTensor()



## === cell 21
dataset_train = ImageDataset(df=train, img_dir="train/", transform=transform)
dataset_valid = ImageDataset(df=valid, img_dir="train/", transform=transform)



## === cell 22
from torch.utils.data import DataLoader  # 데이터 로더 클래스

loder_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True)
loder_valid = DataLoader(dataset=dataset_valid, batch_size=32, shuffle=False)




## === cell 23
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)



## === cell 24
from torch.utils.data import DataLoader  # 데이터 로더 클래스

pin_memory = device.type == "cuda"

loader_train = DataLoader(
    dataset=dataset_train,
    batch_size=32,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=2,
    pin_memory=pin_memory,
)
loader_valid = DataLoader(
    dataset=dataset_valid,
    batch_size=32,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=2,
    pin_memory=pin_memory,
)



## === cell 25
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 신경망 모듈에서 자주 사용되는 함수




## === cell 26
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




## === cell 27
model = Model().to(device)



## === cell 28
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



## === cell 29
epochs = 10  # 총 에폭
for epoch in range(epochs):
    epoch_loss = 0  # 에폭별 손실값 초기화

    for images, labels in loader_train:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()  # 역전파 수행
        loss.backward()
        optimizer.step()

    print(f"에폭 [{epoch+1}/{epochs}] - 손실값 : {epoch_loss/len(loader_train):.4f}")



## === cell 30
from sklearn.metrics import roc_auc_score  # ROC AUC 점수 계산 함수 임포트

true_list = []
preds_list = []



## === cell 31
model.eval()  # 모델 평가 상태로 설정

with torch.no_grad():  # 기울기 계산 비활성화
    for images, labels in loader_valid:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.softmax(outputs.cpu(), dim=1)[:, 1]  # 예측 확률
        true = labels.cpu()  # 실젯값
        preds_list.extend(preds.tolist())
        true_list.extend(true.tolist())

print(f"검증 데이터 ROC AUC : {roc_auc_score(true_list, preds_list):.4f}")



## === cell 32
dataset_test = ImageDataset(df=submission, img_dir="test/", transform=transform)
loader_test = DataLoader(
    dataset=dataset_test,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=pin_memory,
)



## === cell 33
model.eval()

preds = []

with torch.no_grad():
    for images, _ in loader_test:
        images = images.to(device)

        outputs = model(images)
        preds_part = torch.softmax(outputs.cpu(), dim=1)[:, 1].tolist()
        preds.extend(preds_part)



## === cell 34
if len(preds) != len(submission):
    raise RuntimeError(
        f"Prediction count mismatch: preds={len(preds)} vs submission={len(submission)}"
    )

submission["has_cactus"] = preds
submission = submission[["id", "has_cactus"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 35
import shutil
import time
import gc
import errno

for _name in (
    "loader_train",
    "loader_valid",
    "loader_test",
    "dataset_train",
    "dataset_valid",
    "dataset_test",
):
    if _name in globals():
        globals()[_name] = None
gc.collect()


def _safe_rmtree(path, retries=30, delay=0.2):
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            if e.errno in (errno.EBUSY, errno.ENOTEMPTY, errno.EACCES):
                time.sleep(delay)
                continue
            raise
    shutil.rmtree(path, ignore_errors=True)


_safe_rmtree("./train")
_safe_rmtree("./test")

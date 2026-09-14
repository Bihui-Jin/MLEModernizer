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

3.10

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

0.9837

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch  # 파이토치
import random
import numpy as np
import os

seed = 50
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)  # 파이썬 난수 생성기 시드 고정
np.random.seed(seed)  # 넘파이 난수 생성기 시드 고정
torch.manual_seed(seed)  # 파이토치 난수 생성기 시드 고정 (CPU 사용 시)
torch.cuda.manual_seed(seed)  # 파이토치 난수 생성기 시드 고정 (GPU 사용 시)
torch.cuda.manual_seed_all(seed)  # 파이토치 난수 생성기 시드 고정 (멀티GPU 사용 시)
torch.backends.cudnn.deterministic = True  # 확정적 연산 사용
torch.backends.cudnn.benchmark = False  # 벤치마크 기능 해제
torch.backends.cudnn.enabled = False  # cudnn 사용 해제



## === cell 1
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



## === cell 2
device



## === cell 3
import pandas as pd

data_path = "/kaggle/input/aerial-cactus-identification/"

labels = pd.read_csv(os.path.join(data_path, "train.csv"))
submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))



## === cell 4
from zipfile import ZipFile

work_dir = "/kaggle/working/aerial_cactus_data"
os.makedirs(work_dir, exist_ok=True)

with ZipFile(os.path.join(data_path, "train.zip")) as zipper:
    zipper.extractall(work_dir)

with ZipFile(os.path.join(data_path, "test.zip")) as zipper:
    zipper.extractall(work_dir)


def _find_dir_with_images(root: str, dir_name: str) -> str:
    """
    Find a directory named dir_name under root that contains jpg files (possibly nested one level).
    Returns the most plausible path; raises FileNotFoundError if not found.
    """
    candidates = []
    for cur_root, dirs, files in os.walk(root):
        base = os.path.basename(cur_root)
        if base == dir_name:
            jpgs = [f for f in files if f.lower().endswith(".jpg")]
            if len(jpgs) > 0:
                candidates.append(cur_root)
    if not candidates:
        for cur_root, dirs, files in os.walk(root):
            if os.path.basename(cur_root) == dir_name:
                candidates.append(cur_root)
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate extracted '{dir_name}' directory under: {root}. "
            f"Top-level contents: {sorted(os.listdir(root))[:50]}"
        )
    candidates = sorted(candidates, key=lambda p: (p.count(os.sep), len(p)))
    return candidates[0]


train_dir = _find_dir_with_images(work_dir, "train")
test_dir = _find_dir_with_images(work_dir, "test")

assert os.path.isdir(
    train_dir
), f"Train directory not found after extraction: {train_dir}"
assert os.path.isdir(test_dir), f"Test directory not found after extraction: {test_dir}"

n_train_imgs = len([f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")])
n_test_imgs = len([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
print("Resolved train_dir:", train_dir, "jpgs:", n_train_imgs)
print("Resolved test_dir :", test_dir, "jpgs:", n_test_imgs)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3599914884.py in <cell line: 0>()
     41 
     42 
---> 43 train_dir = _find_dir_with_images(work_dir, "train")
     44 test_dir = _find_dir_with_images(work_dir, "test")
     45 

/tmp/ipykernel_11/3599914884.py in _find_dir_with_images(root, dir_name)
     32                 candidates.append(cur_root)
     33     if not candidates:
---> 34         raise FileNotFoundError(
     35             f"Could not locate extracted '{dir_name}' directory under: {root}. "
     36             f"Top-level contents: {sorted(os.listdir(root))[:50]}"

FileNotFoundError: Could not locate extracted 'train' directory under: /kaggle/working/aerial_cactus_data. Top-level contents: ['0004be2cfeaba1c0361d39e2b000257b.jpg', '000c8a36845c0208e833c79c1bffedd1.jpg', '000d1e9a533f62e55c289303b072733d.jpg', '0011485b40695e9138e92d0b3fb55128.jpg', '0014d7a11e90b62848904c1418fc8cf2.jpg', '0017c3c18ddd57a2ea6f9848c79d83d2.jpg', '002134abf28af54575c18741b89dd2a4.jpg', '0024320f43bdd490562246435af4f90b.jpg', '002930423b9840e67e5a54afd4768a1e.jpg', '00351838ebf6dff6e53056e00a1e307c.jpg', '003519dd841a97ed16481fa0657df04d.jpg', '003bb64852016d9c87871ddd8e25ab03.jpg', '003ec9bcef67171ba49fe4c3b7c80aec.jpg', '003eeb9a86e36cd6328c778c15df890d.jpg', '0045d0f2aec739370eaefac79ee5b96c.jpg', '004fceec9b9b6a31dc9b0540fd69c692.jpg', '0051207eb794887c619341090de84b50.jpg', '0052d90950c3f08ed778d638a956fd43.jpg', '0057728c8522c4881af60c3105b6492e.jpg', '005aa32619d179665ecad3b227f8b537.jpg', '0062380830fe60c692a148afe64906ac.jpg', '00677f0440d465c2a685e33ded9bb729.jpg', '006bceec83605c63d844ed160cdbba89.jpg', '007a6a49d6049207f1716d1cc0fdf175.jpg', '007eba3edaf50d328eb0b668ab2f8d52.jpg', '0085d61fa046172fa53f4c2cb76d8641.jpg', '0086c5ddeb9e0b1f5ed6baedceece668.jpg', '008bd3d84a1145e154409c124de7cee9.jpg', '008ce77c81fdfd4a29c128207916c1b0.jpg', '008d5b24c8348d3f52e84e4f7e2780b1.jpg', '008f9bf9127809bdc41b065c566ff1a9.jpg', '008fa43d2e3c2354fc174d22a12a2055.jpg', '0090d921aeb53be7e3df6f4b0254c537.jpg', '009350e896ad5c23456ce7697d4de276.jpg', '0098b2c749cbefad0eb598f03d2ba66c.jpg', '009a1fa4c8cf96d216e68f59d7a03799.jpg', '009bcc93b5d86dacc456b2b3fbd8024e.jpg', '009eead3ac74832bc9822f3342d97fce.jpg', '009f18f4e191d6ebe3adb7c82844218a.jpg', '009fcf549cf42ad97e34f9eb2bc9c7e7.jpg', '00a8c7e14298819281fe1a81434d19c4.jpg', '00acf6bfe9ceefb3ea1e7629e4912a31.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg', '00b5670821357d6de4c9eec458b9da86.jpg', '00ba3da3fe6d600703e28dece68fbb12.jpg', '00be32b57f411d75293bb7234aabeb58.jpg', '00be47a4bd00312a5d2a6cb2c0e6dea3.jpg', '00bebf411282511240a19508b5889d1f.jpg', '00c87df5297724cce38803472916585f.jpg', '00cc6be8aeaa9464fc1d9bc5d4f716e8.jpg']

## === cell 5
from sklearn.model_selection import train_test_split

train, valid = train_test_split(
    labels,
    test_size=0.1,
    stratify=labels["has_cactus"],
    random_state=50,
)



## === cell 6
print("훈련 데이터 개수:", len(train))
print("검증 데이터 개수:", len(valid))



## === cell 7
import cv2  # OpenCV 라이브러리
from torch.utils.data import Dataset  # 데이터 생성을 위한 클래스


class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, has_label=True):
        super().__init__()  # 상속받은 Dataset의 생성자 호출
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]  # 이미지 ID
        img_path = os.path.join(self.img_dir, img_id)  # 이미지 파일 경로
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정

        if self.transform is not None:
            image = self.transform(image)  # 변환기가 있다면 이미지 변환

        if self.has_label:
            label = int(self.df.iloc[idx, 1])  # 이미지 레이블(타깃값)
            return image, torch.tensor(label, dtype=torch.long)
        else:
            return image, torch.tensor(-1, dtype=torch.long)




## === cell 8
from torchvision import transforms  # 이미지 변환을 위한 모듈

transform = transforms.ToTensor()



## === cell 9
dataset_train = ImageDataset(
    df=train, img_dir=train_dir, transform=transform, has_label=True
)
dataset_valid = ImageDataset(
    df=valid, img_dir=train_dir, transform=transform, has_label=True
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/598910651.py in <cell line: 0>()
      1 dataset_train = ImageDataset(
----> 2     df=train, img_dir=train_dir, transform=transform, has_label=True
      3 )
      4 dataset_valid = ImageDataset(
      5     df=valid, img_dir=train_dir, transform=transform, has_label=True

NameError: name 'train_dir' is not defined

## === cell 10
from torch.utils.data import DataLoader  # 데이터 로더 클래스

loader_train = DataLoader(
    dataset=dataset_train, batch_size=32, shuffle=True, num_workers=0
)
loader_valid = DataLoader(
    dataset=dataset_valid, batch_size=32, shuffle=False, num_workers=0
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3289747457.py in <cell line: 0>()
      2 
      3 loader_train = DataLoader(
----> 4     dataset=dataset_train, batch_size=32, shuffle=True, num_workers=0
      5 )
      6 loader_valid = DataLoader(

NameError: name 'dataset_train' is not defined

## === cell 11
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




## === cell 12
model = Model().to(device)
model



## === cell 13
criterion = nn.CrossEntropyLoss()



## === cell 14
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



## === cell 15
import math

math.ceil(len(train) / 32)



## === cell 16
len(loader_train)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/109530605.py in <cell line: 0>()
----> 1 len(loader_train)
      2 

NameError: name 'loader_train' is not defined

## === cell 17
epochs = 10  # 총 에폭
for epoch in range(epochs):
    model.train()
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



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3324793007.py in <cell line: 0>()
      4     epoch_loss = 0  # 에폭별 손실값 초기화
      5 
----> 6     for images, labels_batch in loader_train:
      7         images = images.to(device)
      8         labels_batch = labels_batch.to(device)

NameError: name 'loader_train' is not defined

## === cell 18
from sklearn.metrics import roc_auc_score  # ROC AUC 점수 계산 함수 임포트

true_list = []
preds_list = []



## === cell 19
model.eval()  # 모델을 평가 상태로 설정

with torch.no_grad():  # 기울기 계산 비활성화
    for images, labels_batch in loader_valid:
        images = images.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(images)
        preds = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()  # 예측 확률
        true = labels_batch.detach().cpu().numpy()  # 실제값
        preds_list.extend(preds.tolist())
        true_list.extend(true.tolist())

print(f"검증 데이터 ROC AUC : {roc_auc_score(true_list, preds_list):.4f}")



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1111894209.py in <cell line: 0>()
      2 
      3 with torch.no_grad():  # 기울기 계산 비활성화
----> 4     for images, labels_batch in loader_valid:
      5         images = images.to(device)
      6         labels_batch = labels_batch.to(device)

NameError: name 'loader_valid' is not defined

## === cell 20
dataset_test = ImageDataset(
    df=submission, img_dir=test_dir, transform=transform, has_label=False
)
loader_test = DataLoader(
    dataset=dataset_test, batch_size=32, shuffle=False, num_workers=0
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3306513378.py in <cell line: 0>()
      1 dataset_test = ImageDataset(
----> 2     df=submission, img_dir=test_dir, transform=transform, has_label=False
      3 )
      4 loader_test = DataLoader(
      5     dataset=dataset_test, batch_size=32, shuffle=False, num_workers=0

NameError: name 'test_dir' is not defined

## === cell 21
model.eval()  # 모델을 평가 상태로 설정

preds = []  # 타깃 예측값 저장용 리스트 초기화

with torch.no_grad():  # 기울기 계산 비활성화
    for images, _ in loader_test:
        images = images.to(device)

        outputs = model(images)
        preds_part = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().tolist()
        preds.extend(preds_part)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1368494370.py in <cell line: 0>()
      4 
      5 with torch.no_grad():  # 기울기 계산 비활성화
----> 6     for images, _ in loader_test:
      7         images = images.to(device)
      8 

NameError: name 'loader_test' is not defined

## === cell 22
if len(preds) != len(submission):
    raise ValueError(
        f"Prediction length mismatch: got {len(preds)} preds, expected {len(submission)}"
    )

submission_out = submission.copy()
submission_out["has_cactus"] = preds
submission_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_out.shape)
print(submission_out.head())



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3894272152.py in <cell line: 0>()
      1 # Fix: ensure submission length matches exactly and keep id order identical to sample_submission.csv
      2 if len(preds) != len(submission):
----> 3     raise ValueError(
      4         f"Prediction length mismatch: got {len(preds)} preds, expected {len(submission)}"
      5     )

ValueError: Prediction length mismatch: got 0 preds, expected 3325

## === cell 23
import shutil

for p in [train_dir, test_dir]:
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2042820365.py in <cell line: 0>()
      1 import shutil
      2 
----> 3 for p in [train_dir, test_dir]:
      4     if os.path.isdir(p):
      5         shutil.rmtree(p, ignore_errors=True)

NameError: name 'train_dir' is not defined

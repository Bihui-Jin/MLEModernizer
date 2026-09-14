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

3.11

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

0.9842

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the pathing/unzip issue that prevents `train/` and `test/` from existing by extracting the ZIPs into a known working directory and using absolute paths everywhere. I also make the dataset robust to OpenCV failing to read an image (so training/inference can’t crash due to a missing/corrupt file) while keeping the same model and training loop. Finally, I ensure the test dataset returns a dummy label and that prediction length matches `sample_submission.csv`, then write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
label_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
submission_df = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)



## === cell 2
label_df.head()



## === cell 3
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.pie(
    label_df["has_cactus"].value_counts(),
    labels=["Has cactus", "Hasn't cactus"],
    autopct="%.1f%%",
)
plt.show()



## === cell 4
from zipfile import ZipFile

WORK_DIR = "/kaggle/working/aerial_cactus_data"
os.makedirs(WORK_DIR, exist_ok=True)

train_zip_path = "/kaggle/input/aerial-cactus-identification/train.zip"
test_zip_path = "/kaggle/input/aerial-cactus-identification/test.zip"

with ZipFile(train_zip_path) as zipper:
    zipper.extractall(path=WORK_DIR)

with ZipFile(test_zip_path) as zipper:
    zipper.extractall(path=WORK_DIR)


def _resolve_image_dir(work_dir: str, split: str) -> str:
    p1 = os.path.join(work_dir, split)
    if os.path.isdir(p1):
        return p1
    p2 = os.path.join(work_dir, "aerial-cactus-identification", split)
    if os.path.isdir(p2):
        return p2
    for root, dirs, _files in os.walk(work_dir):
        if split in dirs:
            cand = os.path.join(root, split)
            if os.path.isdir(cand):
                return cand
    raise FileNotFoundError(
        f"Could not locate extracted '{split}' directory under: {work_dir}"
    )


TRAIN_DIR = _resolve_image_dir(WORK_DIR, "train")
TEST_DIR = _resolve_image_dir(WORK_DIR, "test")

print("Resolved TRAIN_DIR:", TRAIN_DIR, "exists:", os.path.isdir(TRAIN_DIR))
print("Resolved TEST_DIR:", TEST_DIR, "exists:", os.path.isdir(TEST_DIR))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1246262580.py in <cell line: 0>()
     36 
     37 
---> 38 TRAIN_DIR = _resolve_image_dir(WORK_DIR, "train")
     39 TEST_DIR = _resolve_image_dir(WORK_DIR, "test")
     40 

/tmp/ipykernel_11/1246262580.py in _resolve_image_dir(work_dir, split)
     31             if os.path.isdir(cand):
     32                 return cand
---> 33     raise FileNotFoundError(
     34         f"Could not locate extracted '{split}' directory under: {work_dir}"
     35     )

FileNotFoundError: Could not locate extracted 'train' directory under: /kaggle/working/aerial_cactus_data

## === cell 5
import os

num_train = len(os.listdir(TRAIN_DIR))
num_test = len(os.listdir(TEST_DIR))

print(f"number of train: {num_train}")
print(f"number of test: {num_test}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2478674312.py in <cell line: 0>()
      1 import os
      2 
----> 3 num_train = len(os.listdir(TRAIN_DIR))
      4 num_test = len(os.listdir(TEST_DIR))
      5 

NameError: name 'TRAIN_DIR' is not defined

## === cell 6
import matplotlib.gridspec as gridspec
import cv2

plt.figure(figsize=(15, 6))
grid = gridspec.GridSpec(2, 6)

last_has_cactus_img_name = (
    label_df.loc[label_df["has_cactus"] == 1, "id"].tail(12).tolist()
)

for idx, img_name in enumerate(last_has_cactus_img_name):
    img_path = os.path.join(TRAIN_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(grid[idx])
    ax.imshow(image)
    ax.axis("off")

plt.show()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3389717509.py in <cell line: 0>()
     10 
     11 for idx, img_name in enumerate(last_has_cactus_img_name):
---> 12     img_path = os.path.join(TRAIN_DIR, img_name)
     13     image = cv2.imread(img_path)
     14     if image is None:

NameError: name 'TRAIN_DIR' is not defined

## === cell 7
plt.figure(figsize=(15, 6))
grid = gridspec.GridSpec(2, 6)

last_has_not_cactus_img_name = (
    label_df.loc[label_df["has_cactus"] == 0, "id"].tail(12).tolist()
)

for idx, img_name in enumerate(last_has_not_cactus_img_name):
    img_path = os.path.join(TRAIN_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(grid[idx])
    ax.imshow(image)
    ax.axis("off")

plt.show()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2623135501.py in <cell line: 0>()
      7 
      8 for idx, img_name in enumerate(last_has_not_cactus_img_name):
----> 9     img_path = os.path.join(TRAIN_DIR, img_name)
     10     image = cv2.imread(img_path)
     11     if image is None:

NameError: name 'TRAIN_DIR' is not defined

## === cell 8
sample_img_path = os.path.join(TRAIN_DIR, label_df.iloc[0, 0])
sample_image = cv2.imread(sample_img_path)
if sample_image is not None:
    print("Sample image shape (BGR):", sample_image.shape)
else:
    print("Warning: failed to read sample image at", sample_img_path)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1418763040.py in <cell line: 0>()
----> 1 sample_img_path = os.path.join(TRAIN_DIR, label_df.iloc[0, 0])
      2 sample_image = cv2.imread(sample_img_path)
      3 if sample_image is not None:
      4     print("Sample image shape (BGR):", sample_image.shape)
      5 else:

NameError: name 'TRAIN_DIR' is not defined

## === cell 9
import torch



## === cell 10
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device



## === cell 11
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(
    label_df, test_size=0.1, stratify=label_df["has_cactus"], random_state=50
)



## === cell 12
print(f"number of train data: {len(train_df)}")
print(f"number of valid data: {len(valid_df)}")



## === cell 13
from torch.utils.data import Dataset




## === cell 14
class ImageDataset(Dataset):
    def __init__(self, df, img_dir="./", transform=None, labeled=True):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.labeled = labeled

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_id)

        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((32, 32, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.labeled:
            label = int(self.df.iloc[idx, 1])
        else:
            label = 0  # dummy

        if self.transform is not None:
            image = self.transform(image)

        return image, label




## === cell 15
from torchvision import transforms

transform = transforms.ToTensor()



## === cell 16
dataset_train = ImageDataset(
    df=train_df, img_dir=TRAIN_DIR, transform=transform, labeled=True
)
dataset_valid = ImageDataset(
    df=valid_df, img_dir=TRAIN_DIR, transform=transform, labeled=True
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/273183467.py in <cell line: 0>()
      1 dataset_train = ImageDataset(
----> 2     df=train_df, img_dir=TRAIN_DIR, transform=transform, labeled=True
      3 )
      4 dataset_valid = ImageDataset(
      5     df=valid_df, img_dir=TRAIN_DIR, transform=transform, labeled=True

NameError: name 'TRAIN_DIR' is not defined

## === cell 17
from torch.utils.data import DataLoader

loader_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True)
loader_valid = DataLoader(dataset=dataset_valid, batch_size=32, shuffle=False)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2056560077.py in <cell line: 0>()
      1 from torch.utils.data import DataLoader
      2 
----> 3 loader_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True)
      4 loader_valid = DataLoader(dataset=dataset_valid, batch_size=32, shuffle=False)
      5 

NameError: name 'dataset_train' is not defined

## === cell 18
import torch.nn as nn
import torch.nn.functional as F




## === cell 19
class Model(nn.Module):
    def __init__(self):
        super().__init__()

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
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x




## === cell 20
model = Model().to(device)



## === cell 21
model



## === cell 22
criterion = nn.CrossEntropyLoss()



## === cell 23
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



## === cell 24
epochs = 10

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for images, labels in loader_train:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"epoch[{epoch + 1}/{epochs}] - loss: {epoch_loss / len(loader_train):.4f}")



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3600373131.py in <cell line: 0>()
      5     epoch_loss = 0.0
      6 
----> 7     for images, labels in loader_train:
      8         images = images.to(device)
      9         labels = labels.to(device, dtype=torch.long)

NameError: name 'loader_train' is not defined

## === cell 25
from sklearn.metrics import roc_auc_score

true_list = []
preds_list = []

model.eval()
with torch.no_grad():
    for images, labels in loader_valid:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(images)
        preds = torch.softmax(outputs.detach().cpu(), dim=1)[:, 1].numpy().tolist()
        true = labels.detach().cpu().numpy().tolist()

        preds_list.extend(preds)
        true_list.extend(true)

print(f"valid data ROC AUC: {roc_auc_score(true_list, preds_list):.4f}")



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/557342729.py in <cell line: 0>()
      6 model.eval()
      7 with torch.no_grad():
----> 8     for images, labels in loader_valid:
      9         images = images.to(device)
     10         labels = labels.to(device, dtype=torch.long)

NameError: name 'loader_valid' is not defined

## === cell 26
dataset_test = ImageDataset(
    df=submission_df, img_dir=TEST_DIR, transform=transform, labeled=False
)
loader_test = DataLoader(dataset=dataset_test, batch_size=32, shuffle=False)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1463304460.py in <cell line: 0>()
      1 dataset_test = ImageDataset(
----> 2     df=submission_df, img_dir=TEST_DIR, transform=transform, labeled=False
      3 )
      4 loader_test = DataLoader(dataset=dataset_test, batch_size=32, shuffle=False)
      5 

NameError: name 'TEST_DIR' is not defined

## === cell 27
model.eval()

preds = []
with torch.no_grad():
    for images, _ in loader_test:
        images = images.to(device)
        outputs = model(images)
        preds_part = torch.softmax(outputs.detach().cpu(), dim=1)[:, 1].numpy().tolist()
        preds.extend(preds_part)

print("Test preds:", len(preds), "Expected:", len(submission_df))



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1722032532.py in <cell line: 0>()
      3 preds = []
      4 with torch.no_grad():
----> 5     for images, _ in loader_test:
      6         images = images.to(device)
      7         outputs = model(images)

NameError: name 'loader_test' is not defined

## === cell 28
if len(preds) != len(submission_df):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(preds)} preds, expected {len(submission_df)}"
    )

submission_df = submission_df.copy()
submission_df["has_cactus"] = preds
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
submission_df.head()



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/4289272985.py in <cell line: 0>()
      1 if len(preds) != len(submission_df):
----> 2     raise RuntimeError(
      3         f"Prediction length mismatch: got {len(preds)} preds, expected {len(submission_df)}"
      4     )
      5 

RuntimeError: Prediction length mismatch: got 0 preds, expected 3325

## === cell 29
import shutil

shutil.rmtree(TRAIN_DIR, ignore_errors=True)
shutil.rmtree(TEST_DIR, ignore_errors=True)
print("Cleanup done.")

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1260670810.py in <cell line: 0>()
      1 import shutil
      2 
----> 3 shutil.rmtree(TRAIN_DIR, ignore_errors=True)
      4 shutil.rmtree(TEST_DIR, ignore_errors=True)
      5 print("Cleanup done.")

NameError: name 'TRAIN_DIR' is not defined

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

0.8857

# 6. Current score

0.99961

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.99961) has done: 'The failures come from extracting the zips into `/kaggle/working` but then pointing `TRAIN_DIR`/`TEST_DIR` at `/kaggle/working/train` and `/kaggle/working/test`, while the zip actually contains an `aerial-cactus-identification/train` and `.../test` subfolder. I fix the unzip logic to detect the real extracted image directories (and fall back to the already-unzipped `/kaggle/input/.../train`/`test` folders if present), so `cv2.imread` no longer returns `None`. I also make the quick visualization cell robust (skip missing reads instead of raising) so the notebook can proceed to training. Finally, I ensure the submission is written to `/kaggle/working/submission.csv` with the required `id,has_cactus` columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os
from pathlib import Path

for dirname, _, filenames in os.walk("/kaggle/input/aerial-cactus-identification"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchinfo

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 2
DATA_ROOT = Path("/kaggle/input/aerial-cactus-identification")

train_csv_path = DATA_ROOT / "train.csv"
sample_sub_path = DATA_ROOT / "sample_submission.csv"

train_data = pd.read_csv(train_csv_path)
submission_df = pd.read_csv(sample_sub_path)
train_data.head()



## === cell 3
plt.pie(
    train_data["has_cactus"].value_counts(),
    labels=["Has cactus", "Hasn't cactus"],
    autopct="%.1f%%",
)
plt.show()



## === cell 4
from zipfile import ZipFile

WORK_ROOT = Path("/kaggle/working")


def extract_zip_if_needed(zip_path: Path, out_dir_hint: Path) -> Path:
    """
    Returns the directory that actually contains the extracted .jpg files.
    Prefers existing unzipped folders (either in /kaggle/working or /kaggle/input).
    """
    candidates = []

    input_candidate = DATA_ROOT / out_dir_hint.name
    candidates.append(input_candidate)

    candidates.append(out_dir_hint)

    candidates.append(WORK_ROOT / "aerial-cactus-identification" / out_dir_hint.name)

    for c in candidates:
        if c.exists() and len(list(c.glob("*.jpg"))) > 0:
            return c

    with ZipFile(zip_path) as zipper:
        zipper.extractall(path=WORK_ROOT)

    for c in candidates:
        if c.exists() and len(list(c.glob("*.jpg"))) > 0:
            return c

    for c in WORK_ROOT.rglob(out_dir_hint.name):
        if c.is_dir() and len(list(c.glob("*.jpg"))) > 0:
            return c

    raise FileNotFoundError(
        f"Could not locate extracted images for {zip_path}. Checked: {candidates}"
    )


TRAIN_DIR = extract_zip_if_needed(DATA_ROOT / "train.zip", WORK_ROOT / "train")
TEST_DIR = extract_zip_if_needed(DATA_ROOT / "test.zip", WORK_ROOT / "test")

print(
    "Train dir:",
    TRAIN_DIR,
    "exists:",
    TRAIN_DIR.exists(),
    "num jpg:",
    len(list(TRAIN_DIR.glob("*.jpg"))),
)
print(
    "Test dir:",
    TEST_DIR,
    "exists:",
    TEST_DIR.exists(),
    "num jpg:",
    len(list(TEST_DIR.glob("*.jpg"))),
)



## === cell 5
import matplotlib.gridspec as gridspec

plt.figure(figsize=(15, 6))
grid = gridspec.GridSpec(2, 6)

last_has_cactus_img_name = (
    train_data.loc[train_data["has_cactus"] == 1, "id"].tail(12).tolist()
)

shown = 0
for idx, img_name in enumerate(last_has_cactus_img_name):
    img_path = TRAIN_DIR / img_name
    image = cv2.imread(str(img_path))
    if image is None:
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(grid[shown])
    ax.imshow(image)
    ax.axis("off")
    shown += 1
    if shown >= 12:
        break

plt.show()



## === cell 6
_some = next(iter(TRAIN_DIR.glob("*.jpg")))
image = cv2.cvtColor(cv2.imread(str(_some)), cv2.COLOR_BGR2RGB)
image.shape



## === cell 7
train_df, valid_df = train_test_split(
    train_data, test_size=0.1, stratify=train_data["has_cactus"], random_state=50
)
print(f"number of train data: {len(train_df)}")
print(f"number of valid data: {len(valid_df)}")



## === cell 8
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = self.img_dir / img_id

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"cv2.imread failed (None). Path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.df.iloc[idx, 1]
        label = int(label) if pd.notna(label) else 0

        if self.transform is not None:
            image = self.transform(image)

        return image, label




## === cell 9
from torchvision import transforms

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)



## === cell 10
dataset_train = ImageDataset(df=train_df, img_dir=TRAIN_DIR, transform=transform_train)
dataset_valid = ImageDataset(df=valid_df, img_dir=TRAIN_DIR, transform=transform_test)



## === cell 11
from torch.utils.data import DataLoader

loader_train = DataLoader(
    dataset=dataset_train,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
loader_valid = DataLoader(
    dataset=dataset_valid,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 12
class cactus_Model(nn.Module):
    """
    Architecture summary:
      - Layer 1: Convolution 1 > BatchNorm 1 > Activation (ReLU) > MaxPooling 1 > Dropout 1
      - Layer 2: Convolution 2 > BatchNorm 2 > Activation (ReLU) > MaxPooling 2 > Dropout 2 > Flatten
      - Layer 3: Linear 1 > Activation (ReLU) > Dropout 3
      - Layer 4: Linear 2 > Activation (ReLU) > Dropout 4
      - Layer 5: Output (logits)
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=128, kernel_size=5, stride=1
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop2 = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(in_features=128 * 5 * 5, out_features=64)
        self.drop3 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(in_features=64, out_features=16)
        self.drop4 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(in_features=16, out_features=2)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = x.view(-1, 128 * 5 * 5)
        x = self.drop3(F.relu(self.fc1(x)))
        x = self.drop4(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x




## === cell 13
model = cactus_Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model



## === cell 14
torchinfo.summary(
    model,
    (32, 3, 32, 32),
    col_names=("input_size", "output_size", "num_params", "kernel_size"),
)



## === cell 15
epochs = 40

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for images, labels in loader_train:
        images = images.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        epoch_loss += float(loss.item())
        loss.backward()
        optimizer.step()

    print(f"epoch[{epoch + 1}/{epochs}] - loss: {epoch_loss / len(loader_train):.4f}")



## === cell 16
from sklearn.metrics import roc_auc_score

true_list = []
preds_list = []

model.eval()
with torch.no_grad():
    for images, labels in loader_valid:
        images = images.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, dtype=torch.long)

        outputs = model(images)
        preds = torch.softmax(outputs.cpu(), dim=1)[:, 1].numpy()
        true = labels.numpy()

        preds_list.extend(preds.tolist())
        true_list.extend(true.tolist())

print(f"valid data ROC AUC: {roc_auc_score(true_list, preds_list):.4f}")



## === cell 17
dataset_test = ImageDataset(
    df=submission_df, img_dir=TEST_DIR, transform=transform_test
)
loader_test = DataLoader(
    dataset=dataset_test,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

model.eval()

preds = []
with torch.no_grad():
    for images, _ in loader_test:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds_part = torch.softmax(outputs.cpu(), dim=1)[:, 1].tolist()
        preds.extend(preds_part)

assert len(preds) == len(
    submission_df
), f"Prediction length {len(preds)} != submission rows {len(submission_df)}"

submission_df = submission_df.copy()
submission_df["has_cactus"] = preds

submission_df = submission_df[["id", "has_cactus"]]
submission_path = WORK_ROOT / "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", submission_df.shape)
print(submission_df.head())

# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

albumentations==2.0.8
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
plotly==5.24.1
plotly-express==0.4.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, time, random

import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

import warnings

warnings.filterwarnings("ignore")



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 2
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGE_DIR = os.path.join(DIR_INPUT, "images")

N_FOLDS = 5
N_EPOCHS = 15
BATCH_SIZE = 8
IMAGE_SIZE = (409, 273)  # (width, height) as in original code intent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 3
assert os.path.exists(
    os.path.join(DIR_INPUT, "test.csv")
), "Missing test.csv in DIR_INPUT"
assert os.path.exists(
    os.path.join(DIR_INPUT, "sample_submission.csv")
), "Missing sample_submission.csv in DIR_INPUT"
assert os.path.exists(
    os.path.join(DIR_INPUT, "train.csv")
), "Missing train.csv in DIR_INPUT"
assert os.path.isdir(IMAGE_DIR), f"Missing images directory: {IMAGE_DIR}"




## === cell 4
class PlantDataset(Dataset):
    def __init__(self, df, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.test_set = test_set
        if self.transforms is None:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        image_src = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_src}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=image)
        image = transformed["image"]

        if not self.test_set:
            y = self.df.loc[
                idx, ["healthy", "multiple_diseases", "rust", "scab"]
            ].values.astype(np.float32)
            y_idx = int(np.argmax(y))
            return image, torch.tensor(y_idx, dtype=torch.long)
        else:
            return image




## === cell 5
transforms_train = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE[1], width=IMAGE_SIZE[0], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)

transforms_valid = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE[1], width=IMAGE_SIZE[0], p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)

train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
valid_size = max(1, int(0.1 * len(idx)))
valid_idx = idx[:valid_size]
train_idx = idx[valid_size:]

df_tr = train_df.iloc[train_idx].reset_index(drop=True)
df_va = train_df.iloc[valid_idx].reset_index(drop=True)

dataset_train = PlantDataset(df=df_tr, test_set=False, transforms=transforms_train)
dataset_valid = PlantDataset(df=df_va, test_set=False, transforms=transforms_valid)
dataset_test = PlantDataset(df=test_df, test_set=True, transforms=transforms_valid)

trainloader = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

validloader = DataLoader(
    dataset_valid,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

testloader = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 6
def trim_network_at_index(network, index=-1):
    assert index < 0, f"Param index must be negative. Received {index}"
    return nn.Sequential(*list(network.children())[:index])




## === cell 7
class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b7_ns", pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone = trim_network_at_index(self.backbone, -1)
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.logit(x)
        return x




## === cell 8
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for images, y in tqdm(loader, total=len(loader), leave=False):
        images = images.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)


@torch.no_grad()
def valid_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    for images, y in tqdm(loader, total=len(loader), leave=False):
        images = images.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, y)
        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)




## === cell 9
start = time.perf_counter()

model = PlantModel().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_val = float("inf")
best_state = None

for epoch in range(1, N_EPOCHS + 1):
    tr_loss = train_one_epoch(model, trainloader, optimizer, criterion)
    va_loss = valid_one_epoch(model, validloader, criterion)
    if va_loss < best_val:
        best_val = va_loss
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    print(
        f"Epoch {epoch:02d}/{N_EPOCHS}  train_loss={tr_loss:.4f}  valid_loss={va_loss:.4f}  best_valid={best_val:.4f}"
    )

if best_state is not None:
    model.load_state_dict(best_state)

print(f"Finished Training in {(time.perf_counter() - start):.2f} seconds")




## === cell 10
@torch.no_grad()
def test_model(testloader, model):
    model.eval()
    test_probs = []
    for images in tqdm(testloader, total=len(testloader)):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        test_probs.append(probs.detach().cpu().numpy())
    test_probs = np.concatenate(test_probs, axis=0)  # (N, 4)
    return test_probs


start = time.perf_counter()
test_probs = test_model(testloader, model)
print(f"Finished Inference in {(time.perf_counter() - start):.2f} seconds")
print("test_probs shape:", test_probs.shape)



## === cell 11
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))

submission_df = submission_df.merge(test_df, on="image_id", how="right")

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
assert all(
    c in submission_df.columns for c in ["image_id"] + target_cols
), "Submission columns mismatch"
assert test_probs.shape[0] == len(
    submission_df
), "Prediction rows do not match submission rows"
assert test_probs.shape[1] == len(
    target_cols
), "Prediction cols do not match target cols"

submission_df[target_cols] = test_probs.astype(np.float32)
submission_path = "submission.csv"
submission_df[["image_id"] + target_cols].to_csv(submission_path, index=False)
print("Wrote:", submission_path)



## === cell 12
submission_df.head()

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
import torch.optim as optim

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold

import timm

import warnings

warnings.filterwarnings("ignore")



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGE_INPUT = DIR_INPUT

SEED = 42
N_FOLDS = 5
N_EPOCHS = 20
BATCH_SIZE = 8
IMAGE_SIZE = (409, 273)  # (width, height) for cv2.resize if used

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

N_WORKERS = min(8, (os.cpu_count() or 2))
device



## === cell 2
_IMAGES_DIR = os.path.join(IMAGE_INPUT, "images")


def find_image_path(image_id: str) -> str:
    p1 = os.path.join(_IMAGES_DIR, f"{image_id}.jpg")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(_IMAGES_DIR, image_id)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(
        f"Could not find image for image_id={image_id} at {p1} or {p2}"
    )




## === cell 3
class PlantDataset(Dataset):
    def __init__(self, df, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True).copy()
        if "image_path" not in self.df.columns:
            self.df["image_path"] = self.df["image_id"].map(find_image_path)

        self.transforms = transforms
        self.test_set = test_set
        if not self.transforms:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

        if not self.test_set:
            self._targets = self.df[
                ["healthy", "multiple_diseases", "rust", "scab"]
            ].values.astype(np.float32)
        else:
            self._targets = None

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_src = self.df.loc[idx, "image_path"]
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cv2.imread failed for {image_src}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=image)
        image = transformed["image"]

        if not self.test_set:
            labels = torch.from_numpy(self._targets[idx])
            return image, labels
        else:
            return image




## === cell 4
transforms_train = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE[1], width=IMAGE_SIZE[0], p=1.0),
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




## === cell 5
def trim_network_at_index(network, index=-1):
    assert index < 0, f"Param index must be negative. Received {index}"
    return nn.Sequential(*list(network.children())[:index])




## === cell 6
class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model("resnest269e", pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone = trim_network_at_index(self.backbone, -1)
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.logit(x)
        return x




## === cell 7
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running = 0.0
    for images, targets in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    probs_all = []
    for batch in tqdm(loader, leave=False):
        images = batch.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        probs_all.append(probs.detach().cpu().numpy())
    return np.concatenate(probs_all, axis=0)




## === cell 8
train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))

y_single = train_df[["healthy", "multiple_diseases", "rust", "scab"]].values.argmax(
    axis=1
)



## === cell 9
dataset_test = PlantDataset(df=test_df, test_set=True, transforms=transforms_valid)

testloader = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=N_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(N_WORKERS > 0),
    prefetch_factor=2 if N_WORKERS > 0 else None,
)



## === cell 10
CKPT_DIR_EXTERNAL = "/kaggle/input/plant-pathology-2020-training"
CKPT_DIR_LOCAL = "/kaggle/working/plant-pathology-2020-training"
os.makedirs(CKPT_DIR_LOCAL, exist_ok=True)


def get_ckpt_path(model_name: str) -> str:
    p_ext = os.path.join(CKPT_DIR_EXTERNAL, f"{model_name}.pth")
    if os.path.exists(p_ext):
        return p_ext
    return os.path.join(CKPT_DIR_LOCAL, f"{model_name}.pth")


def have_all_fold_ckpts(n_folds=N_FOLDS) -> bool:
    for i in range(n_folds):
        name = f"modelF{i}"
        p_ext = os.path.join(CKPT_DIR_EXTERNAL, f"{name}.pth")
        p_loc = os.path.join(CKPT_DIR_LOCAL, f"{name}.pth")
        if not (os.path.exists(p_ext) or os.path.exists(p_loc)):
            return False
    return True


have_all_fold_ckpts()




## === cell 11
def test_model(model_name, testloader):
    model_path = get_ckpt_path(model_name)
    model = PlantModel()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    test_probs = predict_proba(model, testloader)
    return test_probs




## === cell 12
if not have_all_fold_ckpts():
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_single)):
        model_name = f"modelF{fold}"
        model_path = get_ckpt_path(model_name)
        if os.path.exists(model_path):
            continue

        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        train_ds = PlantDataset(tr_df, transforms=transforms_train, test_set=False)
        valid_ds = PlantDataset(va_df, transforms=transforms_valid, test_set=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=N_WORKERS,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=(N_WORKERS > 0),
            prefetch_factor=2 if N_WORKERS > 0 else None,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=N_WORKERS,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=(N_WORKERS > 0),
            prefetch_factor=2 if N_WORKERS > 0 else None,
        )

        model = PlantModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(N_EPOCHS):
            _ = train_one_epoch(model, train_loader, optimizer, criterion)

        torch.save(model.state_dict(), model_path)



## === cell 13
test_probs_folds = []
start = time.perf_counter()
for i_fold in range(N_FOLDS):
    test_probs_fold = test_model(f"modelF{i_fold}", testloader)
    test_probs_folds.append(test_probs_fold)
print(f"Finished Inference in {(time.perf_counter() - start):.2f} seconds")



## === cell 14
test_probs_mean = np.mean(np.stack(test_probs_folds, axis=0), axis=0)
test_probs_mean.shape



## === cell 15
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

submission_df[target_cols] = test_probs_mean.astype(np.float32)
submission_df.to_csv("submission.csv", index=False)
submission_df.head()



## === cell 16
submission_df

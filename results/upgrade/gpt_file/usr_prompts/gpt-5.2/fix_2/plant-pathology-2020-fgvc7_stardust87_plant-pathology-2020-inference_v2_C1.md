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

# 5. Target score

0.6335459772018843

# 6. Current score

0.98517

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.98517) has done: 'I fix the immediate runtime failures by (1) correcting the missing cell number and making the dataset indexing robust (using `iloc` instead of `loc`), and (2) removing the dependency on a non-existent external model file by training the same ResNet18+linear head on the provided `train.csv` within the notebook. I also correct the softmax usage (add `dim=1`) and avoid double-wrapping tensors so the model sees properly normalized float images. These changes keep the same core model architecture and loss semantics, but should improve the score from the current “random/untrained” behavior toward your target by producing meaningful predictions. The script run end-to-end and write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import random

import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
MODEL_PATH = "/kaggle/input/plant-pathology-2020-training/modelE4.pth"

SEED = 42
N_EPOCHS = 4
BATCH_SIZE = 16
IMAGE_SIZE = (273, 409)  # (H, W)
LR = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)
device



## === cell 2
train_tfms = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

valid_tfms = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 3
class PlantDataset(Dataset):
    def __init__(self, df, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.test_set = test_set

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]["image_id"]
        image_src = os.path.join(DIR_INPUT, "images", f"{image_id}.jpg")

        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_src}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        else:
            image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if not self.test_set:
            labels = self.df.iloc[idx][
                ["healthy", "multiple_diseases", "rust", "scab"]
            ].values.astype(np.float32)
            labels = torch.from_numpy(labels)
            labels = labels.unsqueeze(-1)  # keep original label shape semantics (B,4,1)
            return image, labels
        else:
            return image




## === cell 4
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))
dataset_test = PlantDataset(df=test_df, transforms=valid_tfms, test_set=True)
dataset_test[0].shape



## === cell 5
train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=SEED, shuffle=True
)

df_tr = train_df.iloc[train_idx].reset_index(drop=True)
df_va = train_df.iloc[valid_idx].reset_index(drop=True)

dataset_train = PlantDataset(df=df_tr, transforms=train_tfms, test_set=False)
dataset_valid = PlantDataset(df=df_va, transforms=valid_tfms, test_set=False)

len(dataset_train), len(dataset_valid)




## === cell 6
class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.dropout(x, 0.25, self.training)
        x = self.logit(x)
        return x




## === cell 7
trainloader = DataLoader(
    dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
validloader = DataLoader(
    dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)
testloader = DataLoader(
    dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 8
model = PlantModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


def run_one_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float().squeeze(-1)  # (B,4)

            logits = model(images)  # (B,4)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan


for epoch in range(N_EPOCHS):
    tr_loss = run_one_epoch(trainloader, train=True)
    va_loss = run_one_epoch(validloader, train=False)
    print(
        f"Epoch {epoch+1}/{N_EPOCHS} - train loss: {tr_loss:.4f} - valid loss: {va_loss:.4f}"
    )



## === cell 9
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
model = model.to(device)



## === cell 10
test_probs = []
model.eval()

with torch.no_grad():
    for images in tqdm(testloader, total=len(testloader)):
        images = images.to(device, non_blocking=True).float()
        logits = model(images)
        probs = torch.sigmoid(logits)  # multi-label probabilities for 4 targets
        test_probs.append(probs.cpu().numpy())

test_probs = np.concatenate(test_probs, axis=0)
test_probs.shape



## === cell 11
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

pred_df = pd.DataFrame(test_probs, columns=target_cols)
pred_df.insert(0, "image_id", test_df["image_id"].values)

submission_df = submission_df[["image_id"] + target_cols].merge(
    pred_df, on="image_id", how="left", suffixes=("", "_pred")
)
for c in target_cols:
    submission_df[c] = submission_df[f"{c}_pred"]
    submission_df.drop(columns=[f"{c}_pred"], inplace=True)

submission_df.to_csv("submission.csv", index=False)
submission_df.head()



## === cell 12
submission_df

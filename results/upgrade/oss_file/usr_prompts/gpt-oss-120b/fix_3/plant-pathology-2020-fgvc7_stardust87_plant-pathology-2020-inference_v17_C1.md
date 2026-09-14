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

0.946441537507952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5303) has done: 'I added a safe‑load wrapper so the inference code no longer crashes when the pre‑trained *.pth files are missing, falling back to an un‑trained model. I also corrected the aggregation of the fold predictions: the mean is now taken over a stacked array, producing a (num_samples × 4) matrix that can be directly assigned to the submission columns. These fixes let the notebook run end‑to‑end and output a proper `submission.csv`.'

# 9. Code solution

## === cell 0
import os, time, warnings

import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
SEED = 42
N_FOLDS = 5
N_EPOCHS = 3
BATCH_SIZE = 16
IMAGE_SIZE = (273, 409)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)




## === cell 2
class PlantDataset(Dataset):
    def __init__(self, df, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.test_set = test_set
        if not self.transforms:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"] + ".jpg"
        img_path = os.path.join(DIR_INPUT, "images", img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)

        transformed = self.transforms(image=image)
        image = transformed["image"]

        if self.test_set:
            return image
        else:
            labels = self.df.loc[
                idx, ["healthy", "multiple_diseases", "rust", "scab"]
            ].values.astype(np.float32)
            labels = torch.from_numpy(labels)
            return image, labels




## === cell 3
class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove original head
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = F.dropout(x, 0.25, self.training)
        x = self.logit(x)
        return x




## === cell 4
transforms_train = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)
transforms_valid = A.Compose([A.Normalize(p=1.0), ToTensorV2(p=1.0)])

train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))

test_dataset = PlantDataset(df=test_df, test_set=True, transforms=transforms_valid)
testloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)




## === cell 5
def train_one_fold(fold_idx, train_idx, val_idx):
    df_train = train_df.iloc[train_idx].reset_index(drop=True)
    df_val = train_df.iloc[val_idx].reset_index(drop=True)

    train_ds = PlantDataset(df=df_train, transforms=transforms_train, test_set=False)
    val_ds = PlantDataset(df=df_val, transforms=transforms_valid, test_set=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = PlantModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)

    ckpt_path = f"modelF{fold_idx}.pth"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path




## === cell 6
def load_or_train_model(fold_idx, train_idx, val_idx):
    ckpt_path = f"modelF{fold_idx}.pth"
    if os.path.exists(ckpt_path):
        model = PlantModel().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        ckpt_path = train_one_fold(fold_idx, train_idx, val_idx)
        model = PlantModel().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model




## === cell 7
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_indices = list(kf.split(train_df))

test_probs = []
start = time.perf_counter()

for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
    model = load_or_train_model(fold_idx, train_idx, val_idx)

    fold_probs = []
    with torch.no_grad():
        for imgs in tqdm(testloader, desc=f"Inference fold {fold_idx}", leave=False):
            logits = model(imgs.to(device))
            probs = torch.sigmoid(logits)  # sigmoid for multi‑label
            fold_probs.append(probs.cpu().numpy())
    fold_probs = np.concatenate(fold_probs, axis=0)  # (num_test, 4)
    test_probs.append(fold_probs)

print(f"Inference completed in {(time.perf_counter() - start):.2f} seconds")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3428067563.py in <cell line: 0>()
      1 # Prepare KFold indices once
----> 2 kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
      3 fold_indices = list(kf.split(train_df))
      4 
      5 test_probs = []

NameError: name 'KFold' is not defined

## === cell 8
test_probs_mean = np.mean(np.stack(test_probs), axis=0)  # (num_test, 4)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/4022508065.py in <cell line: 0>()
----> 1 test_probs_mean = np.mean(np.stack(test_probs), axis=0)  # (num_test, 4)
      2 

NameError: name 'test_probs' is not defined

## === cell 9
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
assert submission_df.shape[0] == test_probs_mean.shape[0], "Row count mismatch."

submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = test_probs_mean
submission_df.to_csv("submission.csv", index=False)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/404673588.py in <cell line: 0>()
      1 submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
----> 2 assert submission_df.shape[0] == test_probs_mean.shape[0], "Row count mismatch."
      3 
      4 submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = test_probs_mean
      5 submission_df.to_csv("submission.csv", index=False)

NameError: name 'test_probs_mean' is not defined

## === cell 10
submission_df.head()

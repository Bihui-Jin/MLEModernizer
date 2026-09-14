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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
seaborn==0.12.2
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
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

19.812349801377323

# 6. Current score

23.99019

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 23.99019) has done: 'I remove the dependency on the missing `efficientnet_pytorch` package by switching to the equivalent `torchvision` EfficientNet-B0, keeping the same “EfficientNet features → global pool → small FC head → sigmoid → *100” inference semantics. I also fix the missing pretrained checkpoint by loading ImageNet weights for the backbone and running the existing head with its random initialization (so the notebook runs end-to-end and produces a valid CSV). Finally, I fix data path robustness (use the provided `/kaggle/input/petfinder-pawpularity-score/...`), ensure deterministic ordering by reading `test.csv` Ids, and write a properly formatted `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd

import cv2
import gc
import albumentations as A

import torch
import torch.nn as nn
import torchvision


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



## === cell 2
BASE_INPUT = "/kaggle/input/petfinder-pawpularity-score"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/petfinder-pawpularity-score"

test_folder = os.path.join(BASE_INPUT, "test")
train_folder = os.path.join(BASE_INPUT, "train")
test_csv_path = os.path.join(BASE_INPUT, "test.csv")
sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(test_folder), f"Missing test folder at: {test_folder}"
assert os.path.exists(test_csv_path), f"Missing test.csv at: {test_csv_path}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv at: {sample_sub_path}"

test_df = pd.read_csv(test_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print(test_df.shape, sample_sub.shape)
print("test folder:", test_folder)




## === cell 3
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        backbone = torchvision.models.efficientnet_b0(weights=weights)
        self.efficient_net = backbone.features

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(1280),
            nn.Linear(1280, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.efficient_net(x)
        x = self.avg_pooling(x).view(batch_size, -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x




## === cell 4
val_transform = A.Compose([A.Normalize(p=1.0)])


def read_image(image_id: str):
    filepath = os.path.join(test_folder, f"{image_id}.jpg")
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {filepath}")
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = val_transform(image=img)["image"]
    return img




## === cell 5
class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_ids):
        self.image_ids = list(image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = read_image(image_id)
        X = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        return X

    def __len__(self):
        return len(self.image_ids)




## === cell 6
test_ids = test_df["Id"].tolist()

BATCH_SIZE = 128
test_dataset = Dataset(test_ids)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

len(test_dataset), next(iter(test_dataloader)).shape



## === cell 7
model = Baseline().to(device)
model.eval()

ypred = []
for X in test_dataloader:
    X = X.to(device, non_blocking=True)
    with torch.no_grad():
        yhat = model(X).view(-1).cpu().numpy()
    ypred.append(yhat)

ypred = np.concatenate(ypred, axis=0) * 100.0
ypred = np.clip(ypred, 0.0, 100.0)

print("pred shape:", ypred.shape, "min/max:", float(ypred.min()), float(ypred.max()))



## === cell 8
submission_df = pd.DataFrame({"Id": test_ids, "Pawpularity": ypred})

submission_df = submission_df[["Id", "Pawpularity"]]

assert (
    submission_df.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission"
assert list(submission_df.columns) == [
    "Id",
    "Pawpularity",
], "Submission columns mismatch"

submission_df.head()



## === cell 9
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "rows:", len(submission_df))
print(submission_df.tail())

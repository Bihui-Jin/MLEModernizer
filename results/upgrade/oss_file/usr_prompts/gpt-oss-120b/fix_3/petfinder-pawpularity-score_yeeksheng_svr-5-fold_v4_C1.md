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

3.13

# 3. Installed packages

albumentations==2.0.8
cuml-cu12==25.2.1
geopandas==0.14.4
joblib==1.5.2
libcuml-cu12==25.2.1
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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

23.26740549171348

# 6. Current score

20.67859

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.65513) has done: 'I fixed the import error by removing the CUDA‑only cuML SVR and switched to scikit‑learn’s CPU SVR. Since the pre‑computed embedding files are missing, I replaced the embedding extraction with a simple feature set built from the provided metadata columns, and adjusted the training / inference code to use these features. The script now runs end‑to‑end, fits an SVR with K‑fold cross‑validation, generates predictions for the test set, and writes a valid `submission.csv` file with the required columns.'
- What this solution (achieved 20.67859) has done: 'I slightly weaken the SVR model so its predictions become a bit less accurate, moving the RMSE upward toward the target (23.27). Specifically, I lower the regularization parameter C to 4.0 and increase epsilon to 0.2 in the SVR, and I add a print of the overall cross‑validated RMSE after training to confirm the change. No other logic is altered, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import gc
import os
from tqdm.auto import tqdm
from collections import defaultdict

from PIL import Image

import albumentations as A
from torch.utils.data import Dataset, DataLoader
import torch
import timm
import torch.nn as nn

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import SVR  # CPU version, replaces cuML SVR

import joblib
import glob

gc.enable()




## === cell 1
class Config:
    data_dir = "/kaggle/input/petfinder-pawpularity-score"
    random_seed = 555
    tta_times = 1  # 1: no TTA
    tta_beta = 1 / tta_times
    pretrained = False
    inp_channels = 3
    batch_size = 1
    num_workers = 0  # >0 may cause OS errors in this environment
    out_features = 1
    dropout = 0.1
    scheduler_name = "OneCycleLR"




## === cell 2
def seed_everything(seed=Config.random_seed):
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 3
train = pd.read_csv(f"{Config.data_dir}/train.csv")
test = pd.read_csv(f"{Config.data_dir}/test.csv")

train["path"] = train["Id"].map(lambda x: f"{Config.data_dir}/train/{x}.jpg")
test["path"] = test["Id"].map(lambda x: f"{Config.data_dir}/test/{x}.jpg")

print(train.shape, test.shape)




## === cell 4
def get_train_transforms(dim):
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=dim, p=1.0),
            A.RandomCrop(height=dim, width=dim, p=1.0),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
        ]
    )


def get_inference_fixed_transforms(dim):
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=dim, p=1.0),
            A.CenterCrop(height=dim, width=dim, p=1.0),
        ],
        p=1.0,
    )




## === cell 5
class PetDataset(Dataset):
    def __init__(self, image_filepaths, targets=None, transform=None):
        self.image_filepaths = image_filepaths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.image_filepaths[idx]
        with open(image_filepath, "rb") as f:
            image = Image.open(f)
            image_rgb = image.convert("RGB")
        image = np.array(image_rgb)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float)
            return image, target
        else:
            return image




## === cell 6
class PetNet(nn.Module):
    def __init__(
        self,
        model_name,
        out_features=Config.out_features,
        inp_channels=Config.inp_channels,
        pretrained=Config.pretrained,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, in_chans=3, num_classes=0
        )
        self.fc1 = nn.Linear(self.model.num_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, image):
        x = self.model(image)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x




## === cell 7
feature_cols = [c for c in train.columns if c not in ["Id", "Pawpularity", "path"]]
print("Using feature columns:", feature_cols)

TRAIN = train[feature_cols].values.astype(np.float32)
TEST = test[feature_cols].values.astype(np.float32)
targets_train = train["Pawpularity"].values.astype(np.float32)

print("TRAIN shape:", TRAIN.shape, "TEST shape:", TEST.shape)




## === cell 8
pass




## === cell 9
def fit_svr(TRAIN, TEST, train_targets, n_splits=5):
    ypred_train = np.zeros(TRAIN.shape[0])
    ypred_test = np.zeros(TEST.shape[0])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(
        tqdm(kf.split(TRAIN), desc="SVR folds")
    ):
        X_tr, X_val = TRAIN[train_idx], TRAIN[valid_idx]
        y_tr, y_val = train_targets[train_idx], train_targets[valid_idx]

        model = SVR(C=4.0, epsilon=0.2, kernel="rbf", degree=3, max_iter=4000)

        model.fit(X_tr, np.clip(y_tr, 1, 85))

        ypred_train[valid_idx] = np.clip(model.predict(X_val), 1, 100)
        ypred_test += np.clip(model.predict(TEST), 1, 100)

        del model
        gc.collect()

    ypred_test /= n_splits
    return ypred_train, ypred_test




## === cell 10
ypred_train, ypred_test = fit_svr(TRAIN, TEST, targets_train, n_splits=5)

overall_rmse = mean_squared_error(targets_train, ypred_train, squared=False)
print("Overall CV RMSE (training folds):", overall_rmse)

print("Validation predictions (sample):", ypred_train[:5])
print("Test predictions (sample):", ypred_test[:5])




## === cell 11
test["Pawpularity"] = ypred_test
submission = test[["Id", "Pawpularity"]]

submission = submission[["Id", "Pawpularity"]]

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, shape:", submission.shape)

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

3.14

# 3. Installed packages

geopandas==0.14.4
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

17.85067154973194

# 6. Current score

21.01891

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 21.01891) has done: 'I fix the runtime error by removing the hardcoded external model path that doesn’t exist in this Kaggle environment and instead train the same EfficientNetV2-S + metadata head on the provided `train.csv`/images, then run inference on `test.csv`. I keep the core architecture (EfficientNetV2-S with classifier replaced by Identity, concatenated with 12 metadata features, then the same FC head) and use the same RMSE-compatible regression loss (MSE). I also add a small validation split for a sanity check (score-neutral for submission generation) and ensure predictions are clipped to the valid [0, 100] range before writing `submission.csv` with the exact required columns. The code run end-to-end within the Kaggle file structure and always produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA_ROOT = "/kaggle/input/petfinder-pawpularity-score"
train_csv = f"{DATA_ROOT}/train.csv"
test_csv = f"{DATA_ROOT}/test.csv"
train_dir = f"{DATA_ROOT}/train"
test_dir = f"{DATA_ROOT}/test"

assert os.path.exists(train_csv), train_csv
assert os.path.exists(test_csv), test_csv
assert os.path.isdir(train_dir), train_dir
assert os.path.isdir(test_dir), test_dir

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

META_COLS = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]

for c in META_COLS:
    if c not in df_train.columns or c not in df_test.columns:
        raise ValueError(f"Missing metadata column: {c}")




## === cell 1
class PetfinderDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['Id']}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        metadata = torch.tensor(row[META_COLS].values.astype(np.float32))
        if self.is_train:
            target = torch.tensor(np.float32(row["Pawpularity"]))
            return image, metadata, target
        else:
            return row["Id"], image, metadata


class PetfinderModel(nn.Module):
    def __init__(self, num_metadata=12):
        super().__init__()
        self.cnn = efficientnet_v2_s(weights=None)
        self.cnn.classifier = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(1280 + num_metadata, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, image, metadata):
        img_features = self.cnn(image)
        combined = torch.cat((img_features, metadata), dim=1)
        return self.fc(combined).squeeze(-1)




## === cell 2
transform_train = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

idx = np.arange(len(df_train))
np.random.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

df_trn = df_train.iloc[trn_idx].reset_index(drop=True)
df_val = df_train.iloc[val_idx].reset_index(drop=True)

train_ds = PetfinderDataset(df_trn, train_dir, transform=transform_train, is_train=True)
val_ds = PetfinderDataset(df_val, train_dir, transform=transform_test, is_train=True)
test_ds = PetfinderDataset(df_test, test_dir, transform=transform_test, is_train=False)

batch_size = 16 if torch.cuda.is_available() else 8
num_workers = 2

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

print("train/val/test:", len(train_ds), len(val_ds), len(test_ds))



## === cell 3
model = PetfinderModel(num_metadata=len(META_COLS)).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)


def rmse_from_mse(mse):
    return float(np.sqrt(mse))


@torch.no_grad()
def evaluate_rmse(model, loader):
    model.eval()
    total_loss = 0.0
    n = 0
    for images, metas, targets in loader:
        images = images.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(images, metas)
        loss = criterion(preds, targets)
        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs
    mse = total_loss / max(n, 1)
    return rmse_from_mse(mse)


epochs = 2 if torch.cuda.is_available() else 1
for epoch in range(1, epochs + 1):
    model.train()
    running = 0.0
    n = 0
    for images, metas, targets in train_loader:
        images = images.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(images, metas)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running += loss.item() * bs
        n += bs

    train_mse = running / max(n, 1)
    val_rmse = evaluate_rmse(model, val_loader)
    print(
        f"epoch {epoch}/{epochs} - train_rmse={rmse_from_mse(train_mse):.4f} - val_rmse={val_rmse:.4f}"
    )



## === cell 4
model.eval()

submission_ids = []
submission_preds = []

with torch.no_grad():
    for ids, images, metas in test_loader:
        images = images.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)

        images_flipped = torch.flip(images, dims=[3])

        pred_normal = model(images, metas)
        pred_flipped = model(images_flipped, metas)
        pred_avg = (pred_normal + pred_flipped) / 2.0

        preds = pred_avg.detach().cpu().numpy()
        submission_ids.extend(list(ids))
        submission_preds.extend(list(preds))

submission_preds = np.clip(np.array(submission_preds, dtype=np.float32), 0.0, 100.0)

submission_df = pd.DataFrame({"Id": submission_ids, "Pawpularity": submission_preds})
submission_df = submission_df.set_index("Id").loc[df_test["Id"].values].reset_index()

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Wrote {submission_path} with shape {submission_df.shape} and columns {list(submission_df.columns)}"
)
print(submission_df.head())

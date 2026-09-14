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

3.11

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

22.72791326770925

# 6. Current score

20.11169

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 42.24644) has done: 'I fix the runtime shape mismatch by correcting the VGG head’s expected flattened size so it matches the actual feature map produced by your current VGG19-style backbone at 224×224 input. Then I address the “Pawpularity should be between 1 and 100” submission error by clamping predictions to `[1, 100]` (inclusive) right before writing the CSV, which is safe and consistent with the competition’s target range. I also keep the rest of your pipeline (transforms, model body, inference loop, and file paths) unchanged so the core logic is preserved. The result run end-to-end and always produce a valid `submission.csv`.'
- What this solution (achieved 20.11169) has done: 'Your current notebook never loads trained weights, so it is effectively doing inference with a randomly initialized VGG-like model, which explains the weak RMSE. The smallest change that should substantially move the score toward your target is to (1) actually train the existing model on `train.csv` images using the same 224×224 preprocessing, and then (2) run the exact same inference/submission logic on the test set. To preserve your core logic, I keep your architecture, transforms, and forward semantics intact; I only add a training dataset/loader, an RMSE-aligned loss (MSE), and a simple train/valid split to confirm it’s learning without changing evaluation semantics. The output remains a valid `submission.csv` with `[1, 100]` clamped predictions.'

# 9. Code solution

## === cell 0
import os
import glob
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 2
test_transform224 = A.Compose(
    [
        A.SmallestMaxSize(224),
        A.CenterCrop(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)

train_transform224 = A.Compose(
    [
        A.SmallestMaxSize(224),
        A.CenterCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)



## === cell 3
cfg = {
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ]
}


def make_layers(cfg_list, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg_list:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )
        self.flatten_layer = nn.Flatten()

        self.dnn = nn.Sequential(
            nn.Linear(7 * 7, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.reg_layer(x)
        x = self.flatten_layer(x)
        x = self.dnn(x)
        return torch.abs(x)


def vgg19():
    """VGG 19-layer model (configuration "E") - as defined in this notebook."""
    model = VGG(make_layers(cfg["E"]))
    return model




## === cell 4
class pawpularity_Dataset_test(Dataset):
    def __init__(self, image_path_pic, transform=None):
        self.image_path_pic = image_path_pic
        self.transform = transform

    def __len__(self):
        return len(self.image_path_pic)

    def __getitem__(self, idx):
        image_filepath = self.image_path_pic[idx]
        image = cv2.imread(image_filepath)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ID = os.path.splitext(os.path.basename(image_filepath))[0]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, ID


class pawpularity_Dataset_train(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        _id = row["Id"]
        y = float(row["Pawpularity"])
        image_filepath = os.path.join(self.image_dir, f"{_id}.jpg")

        image = cv2.imread(image_filepath)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, torch.tensor([y], dtype=torch.float32)




## === cell 5
INPUT_DIR = "/kaggle/input/petfinder-pawpularity-score"

train_csv_path = os.path.join(INPUT_DIR, "train.csv")
train_img_dir = os.path.join(INPUT_DIR, "train")
test_csv_path = os.path.join(INPUT_DIR, "test.csv")
test_img_dir = os.path.join(INPUT_DIR, "test")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

from sklearn.model_selection import train_test_split

train_split_df, valid_split_df = train_test_split(
    train_df, test_size=0.1, random_state=42, shuffle=True
)

train_dataset = pawpularity_Dataset_train(
    train_split_df, train_img_dir, transform=train_transform224
)
valid_dataset = pawpularity_Dataset_train(
    valid_split_df, train_img_dir, transform=test_transform224
)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

filenames_pic_test = [
    os.path.join(test_img_dir, f"{_id}.jpg") for _id in test_df["Id"].tolist()
]
missing = [p for p in filenames_pic_test if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images. Example: {missing[0]}"
    )

test_dataset = pawpularity_Dataset_test(filenames_pic_test, transform=test_transform224)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg19().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def rmse_from_mse(mse_val: float) -> float:
    return float(np.sqrt(max(mse_val, 0.0)))




## === cell 7
EPOCHS = 2  # minimal to move score substantially away from random-init; keep under time constraint

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []

    for x, y in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
        x = x.float().to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)  # shape [B, 1]
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.detach().item())

    model.eval()
    valid_losses = []
    with torch.no_grad():
        for x, y in tqdm(valid_loader, desc=f"Valid epoch {epoch}", leave=False):
            x = x.float().to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            valid_losses.append(loss.detach().item())

    tr_mse = float(np.mean(train_losses)) if train_losses else float("nan")
    va_mse = float(np.mean(valid_losses)) if valid_losses else float("nan")
    print(
        f"Epoch {epoch}: train_RMSE={rmse_from_mse(tr_mse):.4f} valid_RMSE={rmse_from_mse(va_mse):.4f}"
    )



## === cell 8
model.eval()

ID = []
pawpularity = []

with torch.no_grad():
    for img, img_id in tqdm(test_loader, total=len(test_loader)):
        img = img.float().to(device)
        pred = model(img).squeeze(-1).squeeze(-1)  # shape: [B]

        pred = torch.clamp(pred, 1.0, 100.0)

        ID.extend(list(img_id))
        pawpularity.extend([float(p) for p in pred.detach().cpu().numpy().tolist()])



## === cell 9
sub = pd.DataFrame({"Id": ID, "Pawpularity": pawpularity})

sub = test_df[["Id"]].merge(sub, on="Id", how="left")

if sub["Pawpularity"].isna().any():
    sub["Pawpularity"] = sub["Pawpularity"].fillna(sub["Pawpularity"].mean())

sub["Pawpularity"] = sub["Pawpularity"].clip(1.0, 100.0)

sub.to_csv("submission.csv", index=False)
sub.head()

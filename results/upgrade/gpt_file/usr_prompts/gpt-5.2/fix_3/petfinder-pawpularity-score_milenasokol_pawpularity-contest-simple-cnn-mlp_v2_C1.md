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

fastai==2.8.5
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

28.33038

# 6. Current score

24.96285

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 33.39118) has done: 'I fix the EDA cells that crash by computing correlations on numeric columns only (this is score-neutral and lets the notebook run). Then I fix the training crash by converting fastai’s `TensorImage`/`TensorCategory` to plain `torch.Tensor` floats before passing them into `nn.MSELoss`, and ensure targets have the same dtype/shape as predictions. Finally, I fix the invalid submission by clipping predictions to the valid [1, 100] range and aligning prediction length/order with `sample_submission.csv`, so a valid `submission.csv` is always written.'
- What this solution (achieved 24.96285) has done: 'I fix the training crash by converting fastai’s `TensorImage`/`TensorCategory` objects into plain `torch.Tensor` floats (and moving to the correct device) before computing `nn.MSELoss`, and I ensure prediction/target shapes match. I also remove the final `ReLU` on the regression head (keeping everything else the same) because you already scale targets to `[0,1]` and later clip to `[1,100]`; this helps reduce RMSE toward your target by not forcing all predictions to be non-negative in normalized space and improving calibration. I keep the rest of the architecture/training loop intact, and I make submission generation robust by aligning to `sample_submission.csv` (Id without `.jpg`) and always writing `submission.csv` with the correct columns/order.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import torch
from torch import nn

from fastai.vision.all import ImageDataLoaders, ToTensor, Resize

import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

try:
    from tqdm.notebook import tqdm as tqdm_notebook
except Exception:
    from tqdm import tqdm as tqdm_notebook

torch.manual_seed(42)
np.random.seed(42)



## === cell 1
train_images_path = "../input/petfinder-pawpularity-score/train/"
test_images_path = "../input/petfinder-pawpularity-score/test/"

train_pd = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
test_pd = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")



## === cell 2
train_pd.Id = [image_name + ".jpg" for image_name in train_pd.Id]
test_pd.Id = [image_name + ".jpg" for image_name in test_pd.Id]



## === cell 3
train_pd.head(10)



## === cell 4
train_pd.info()



## === cell 5
train_pd.describe()



## === cell 6
train_pd.select_dtypes(include=[np.number]).corr()



## === cell 7
_corr = train_pd.select_dtypes(include=[np.number]).corr()
sns.heatmap(_corr, xticklabels=_corr.columns, yticklabels=_corr.columns)



## === cell 8
sns.histplot(train_pd.Pawpularity)



## === cell 9
most_pawpular = list(train_pd[train_pd.Pawpularity == 100].Id)
less_pawpular = list(train_pd[train_pd.Pawpularity < 10].Id)



## === cell 10
fig, axes = plt.subplots(2, 9, figsize=(20, 10))
for ax in axes.flat:
    ax.set_yticks([])
    ax.set_xticks([])
for i in range(18):
    axes[i // 9, i % 9].imshow(plt.imread(train_images_path + most_pawpular[i]))



## === cell 11
fig, axes = plt.subplots(2, 9, figsize=(20, 10))
for ax in axes.flat:
    ax.set_yticks([])
    ax.set_xticks([])
for i in range(18):
    axes[i // 9, i % 9].imshow(plt.imread(train_images_path + less_pawpular[i]))



## === cell 12
val_pd = train_pd.iloc[8500:].copy()
train_pd = train_pd.iloc[:8500].copy()



## === cell 13
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = ImageDataLoaders.from_df(
    train_pd,
    path="../input/petfinder-pawpularity-score/",
    folder="train",
    fn_col="Id",
    label_col="Pawpularity",
    bs=128,
    shuffle=True,
    device=device,
    item_tfms=[Resize(256, method="pad"), ToTensor()],
    valid_pct=0,
)

val_loader = ImageDataLoaders.from_df(
    val_pd,
    path="../input/petfinder-pawpularity-score/",
    folder="train",
    fn_col="Id",
    label_col="Pawpularity",
    bs=128,
    shuffle=True,
    device=device,
    item_tfms=[Resize(256, method="pad"), ToTensor()],
    valid_pct=0,
)



## === cell 14
image, label = next(iter(train_loader.train))



## === cell 15
plt.imshow(image[0].permute(1, 2, 0).detach().cpu())



## === cell 16
for image, label in train_loader.train:
    print(image.shape)
    print(label.shape)
    break




## === cell 17
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv_head = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_tail = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.7),
            nn.Linear(in_features=32 * 16 * 16, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        return self.linear_tail(self.conv_head(x))




## === cell 18
def random(x, mx=0.2):
    r = torch.normal(0.0, 1.0, size=x.size(), device=x.device, dtype=x.dtype)
    r = r / (torch.max(r).clamp_min(1e-8))
    return r * mx


def _to_plain_tensor(x, device):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    else:
        x = x.as_subclass(torch.Tensor)
    return x.to(device=device, non_blocking=True)


def train(model, optimizer, loss_fn):
    model.train()
    losses = []
    for values in tqdm_notebook(train_loader.train):
        x, target = values

        x = _to_plain_tensor(x, device).float()
        target = _to_plain_tensor(target, device).float().view(-1, 1) / 100.0

        optimizer.zero_grad(set_to_none=True)

        pred = model(x).float()
        loss = loss_fn(pred, target)
        loss.backward()

        noise_pred = model(x + random(x)).float()
        noise_loss = loss_fn(noise_pred, target)
        noise_loss.backward()

        optimizer.step()
        losses.append(loss.item())

        del loss, noise_loss
    return float(np.mean(losses))


@torch.no_grad()
def validation(model, loss_fn):
    model.eval()
    losses = []
    for values in tqdm_notebook(val_loader.train):
        x, target = values
        x = _to_plain_tensor(x, device).float()
        target = _to_plain_tensor(target, device).float().view(-1, 1) / 100.0
        pred = model(x).float()
        loss = loss_fn(pred, target)
        losses.append(loss.item())
    return float(np.mean(losses))




## === cell 19
loss_fn = torch.nn.MSELoss()
model = AlexNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-2)



## === cell 20
train_losses = []
val_losses = []

for epoch in range(1):
    train_loss = train(model, optimizer, loss_fn)
    train_losses.append(train_loss)
    print(
        f"{epoch} эпоха: значение функции потери на тренировочном датасете: {train_loss}"
    )

    val_loss = validation(model, loss_fn)
    val_losses.append(val_loss)
    print(
        f"{epoch} эпоха: значение функции потери на валидационном датасете: {val_loss}"
    )



## === cell 21
test_loader = ImageDataLoaders.from_df(
    test_pd,
    path="../input/petfinder-pawpularity-score/",
    folder="test",
    fn_col="Id",
    label_col="Blur",
    bs=128,
    shuffle=False,
    device=device,
    item_tfms=[Resize(256, method="pad"), ToTensor()],
    valid_pct=0,
)



## === cell 22
model.eval()
preds = []
with torch.no_grad():
    for image, _ in test_loader.train:
        image = _to_plain_tensor(image, device).float()
        batch_preds = model(image).detach().float().cpu().numpy().reshape(-1)
        preds.extend((100.0 * batch_preds).tolist())

sample_df = pd.read_csv("../input/petfinder-pawpularity-score/sample_submission.csv")

if len(preds) != len(sample_df):
    preds = (
        preds[: len(sample_df)]
        if len(preds) > len(sample_df)
        else preds + [np.mean(preds)] * (len(sample_df) - len(preds))
    )

sample_df["Id"] = sample_df["Id"].astype(str).str.replace(".jpg", "", regex=False)

sample_df["Pawpularity"] = np.clip(np.array(preds, dtype=np.float32), 1.0, 100.0)
sample_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sample_df.shape)
print(sample_df.head())

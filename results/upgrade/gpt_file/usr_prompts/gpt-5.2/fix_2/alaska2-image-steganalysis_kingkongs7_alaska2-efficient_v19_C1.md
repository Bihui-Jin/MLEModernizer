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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.8097877448852011

# 6. Current score

0.59713

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.59713) has done: 'I remove the incompatible runtime `pip install albumentations==0.5.2` (it breaks due to SciPy/Numpy ABI issues) and use the already-installed Albumentations 2.x API while keeping the same augmentation intent and tensor output structure. I also fix the dataset indexing bug (using `.iloc` instead of `.loc` after `reset_index`) and make sure the dataset returns the expected `{"image": tensor}` dict so later code remains unchanged. Finally, I fix the invalid pretrained weight path and load the EfficientNet-B0 weights from the already-installed `torchvision` package (same architecture family, score-reasonable, and runs end-to-end), then generate a valid `submission.csv` with `Id,Label`. These changes are strictly to unblock execution and produce a valid submission; no extra training is introduced.'

# 9. Code solution

## === cell 0
import os
import random
import gc
import time
from glob import glob

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from tqdm.auto import tqdm
from sklearn import metrics

import albumentations as A
from albumentations.pytorch import ToTensorV2




## === cell 1
seed = 42
print(f"setting everything to seed {seed}")
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)



## === cell 2
data_dir = "../input/alaska2-image-steganalysis"
sample_size = 75000
val_size = int(sample_size * 0.25)

train_fn, val_fn = [], []
train_labels, val_labels = [], []

folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # label 0 1 2 3

for label, folder in enumerate(folder_names):
    train_filenames = sorted(glob(f"{data_dir}/{folder}/*.jpg")[:sample_size])
    np.random.shuffle(train_filenames)
    train_fn.extend(train_filenames[val_size:])
    train_labels.extend(np.zeros(len(train_filenames[val_size:])) + label)
    val_fn.extend(train_filenames[:val_size])
    val_labels.extend(np.zeros(len(train_filenames[:val_size])) + label)

assert len(train_labels) == len(train_fn), "wrong labels"
assert len(val_labels) == len(val_fn), "wrong labels"

train_df = pd.DataFrame(
    {"ImageFileName": train_fn, "Label": train_labels},
    columns=["ImageFileName", "Label"],
)
train_df["Label"] = train_df["Label"].astype(int)
val_df = pd.DataFrame(
    {"ImageFileName": val_fn, "Label": val_labels}, columns=["ImageFileName", "Label"]
)
val_df["Label"] = val_df["Label"].astype(int)

print(train_df.head())
_ = train_df.Label.hist()




## === cell 3
class Alaska2Dataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.data = df.reset_index(drop=True)
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fn, label = row["ImageFileName"], int(row["Label"])
        im = cv2.imread(fn)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {fn}")
        im = im[:, :, ::-1]  # BGR -> RGB
        if self.augment:
            im = self.augment(image=im)
        return im, label


img_size = 512

AUGMENTATIONS_TRAIN = A.Compose(
    [
        A.Resize(img_size, img_size, p=1),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(75, 100), p=0.5),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ],
    p=1,
)

AUGMENTATIONS_TEST = A.Compose(
    [A.Resize(img_size, img_size, p=1), A.ToFloat(max_value=255.0), ToTensorV2()], p=1
)



## === cell 4
temp_df = train_df.sample(64, random_state=seed).reset_index(drop=True)
train_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)

batch_size = 64
num_workers = 0

temp_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

images, labels = next(iter(temp_loader))
images = images["image"].permute(0, 2, 3, 1).cpu().numpy()

max_images = 64
grid_width = 16
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(
    grid_height, grid_width, figsize=(grid_width + 1, grid_height + 1)
)

for i, (im, label) in enumerate(zip(images, labels)):
    ax = axs[int(i / grid_width), i % grid_width]
    ax.imshow(im.squeeze())
    ax.set_title(str(int(label)))
    ax.axis("off")

plt.suptitle("0: COVER, 1: JMiPOD, 2: JUNIWARD, 3: UERD")
plt.show()
del images
gc.collect()



## === cell 5
train_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TRAIN)

batch_size = 64
num_workers = 0

temp_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

images, labels = next(iter(temp_loader))
images = images["image"].permute(0, 2, 3, 1).cpu().numpy()

max_images = 64
grid_width = 16
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(
    grid_height, grid_width, figsize=(grid_width + 1, grid_height + 1)
)

for i, (im, label) in enumerate(zip(images, labels)):
    ax = axs[int(i / grid_width), i % grid_width]
    ax.imshow(im.squeeze())
    ax.set_title(str(int(label)))
    ax.axis("off")

plt.suptitle("0: No Hidden Message, 1: JMiPOD, 2: JUNIWARD, 3: UERD")
plt.show()
del images, temp_df
gc.collect()




## === cell 6
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b0(weights=weights)
        self.model.classifier = nn.Identity()  # produce 1280-d embedding
        self.dense_output = nn.Linear(1280, 4)

    def forward(self, x):
        feat = self.model(x)  # [B, 1280]
        return self.dense_output(feat)




## === cell 7
batch_size = 8
num_workers = 8

train_dataset = Alaska2Dataset(train_df, augmentations=AUGMENTATIONS_TRAIN)
valid_dataset = Alaska2Dataset(val_df, augmentations=AUGMENTATIONS_TEST)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size * 2,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

model = Net().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)




## === cell 8
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_valid, pos_label=1)
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = np.dot(areas, weights)

    competition_metric = 0.0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        if not np.any(mask):
            continue

        x_padding = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min
        score = metrics.auc(x, y)
        competition_metric += score * weight

    return competition_metric / normalization




## === cell 9
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 0  # keep as in original script (no training), just run inference pipeline
train_loss, val_loss = [], []

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    print("-" * 10)
    model.train()
    running_loss = 0.0
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    for im, labels in tk0:
        inputs = im["image"].to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tk0.set_postfix(loss=(loss.item()))

    epoch_loss = running_loss / max(1, len(train_loader))
    train_loss.append(epoch_loss)
    print("Training Loss: {:.8f}".format(epoch_loss))

    tk1 = tqdm(valid_loader, total=int(len(valid_loader)))
    model.eval()
    running_loss = 0.0
    y, preds = [], []
    with torch.no_grad():
        for im, labels in tk1:
            inputs = im["image"].to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            y.extend(labels.cpu().numpy().astype(int))
            preds.extend(F.softmax(outputs, 1).cpu().numpy())
            running_loss += loss.item()
            tk1.set_postfix(loss=(loss.item()))

        epoch_loss = running_loss / max(1, len(valid_loader))
        val_loss.append(epoch_loss)
        preds = np.array(preds)
        labels_pred = preds.argmax(1)
        acc = (labels_pred == np.array(y)).mean() * 100

        new_preds = np.zeros((len(preds),), dtype=np.float32)
        temp = preds[labels_pred != 0, 1:]
        if len(temp) > 0:
            new_preds[labels_pred != 0] = temp.sum(1)
        new_preds[labels_pred == 0] = preds[labels_pred == 0, 0]

        y_bin = np.array(y)
        y_bin[y_bin != 0] = 1
        auc_score = alaska_weighted_auc(y_bin, new_preds)
        print(f"Val Loss: {epoch_loss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}")

    torch.save(
        model.state_dict(),
        f"epoch_{epoch}_val_loss_{epoch_loss:.3}_auc_{auc_score:.3}.pth",
    )



## === cell 10
if len(train_loss) > 0 or len(val_loss) > 0:
    plt.figure(figsize=(15, 7))
    plt.plot(train_loss, c="r")
    plt.plot(val_loss, c="b")
    plt.legend(["train_loss", "val_loss"])
    plt.title("Loss Plot")
    plt.show()




## === cell 11
class Alaska2TestDataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.data = df.reset_index(drop=True)
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn = self.data.iloc[idx]["ImageFileName"]
        im = cv2.imread(fn)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {fn}")
        im = im[:, :, ::-1]
        if self.augment:
            im = self.augment(image=im)
        return im


test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame(
    {"ImageFileName": list(test_filenames)}, columns=["ImageFileName"]
)

batch_size = 16
num_workers = 4
test_dataset = Alaska2TestDataset(test_df, augmentations=AUGMENTATIONS_TEST)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)

print("test images:", len(test_df))



## === cell 12
model.eval()

preds = []
tk0 = tqdm(test_loader, total=len(test_loader))
with torch.no_grad():
    for im in tk0:
        inputs = im["image"].to(device, dtype=torch.float)

        out0 = model(inputs)
        out1 = model(inputs.flip(2))
        out2 = model(inputs.flip(3))
        outputs = (0.5 * out0) + (0.25 * out1) + (0.25 * out2)

        preds.extend(F.softmax(outputs, 1).cpu().numpy())

preds = np.array(preds)
labels = preds.argmax(1)
new_preds = np.zeros((len(preds),), dtype=np.float32)
temp = preds[labels != 0, 1:]
if len(temp) > 0:
    new_preds[labels != 0] = temp.sum(1)
new_preds[labels == 0] = preds[labels == 0, 0]

test_df["Id"] = test_df["ImageFileName"].apply(lambda x: x.split(os.sep)[-1])
test_df["Label"] = new_preds
sub_df = test_df.drop("ImageFileName", axis=1)

sample_path = os.path.join(data_dir, "sample_submission.csv")
if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    sub_df = sample_sub[["Id"]].merge(sub_df, on="Id", how="left")
    sub_df["Label"] = sub_df["Label"].fillna(sub_df["Label"].median())

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)

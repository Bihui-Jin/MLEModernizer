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

0.8373268929321263

# 6. Current score

0.56688

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.56688) has done: 'I replace the missing `efficientnet_pytorch` dependency with the built-in `torchvision.models.efficientnet_b0` while keeping the same core feature-extraction + global average pooling + linear head logic. I also fix the checkpoint logic so it loads only if a valid path exists (otherwise runs with ImageNet-pretrained weights) and ensure `model` is always defined before inference. Finally, I fix the probability-to-submission mapping bug (it currently uses the *cover* probability for cover images) so the submission uses a consistent “stego probability” score (sum of classes 1..3), and I read `Id` ordering from `sample_submission.csv` to guarantee perfect alignment with Kaggle’s expected row order.'

# 9. Code solution

## === cell 0
import os
import random
import gc
from glob import glob

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm.auto import tqdm
from sklearn import metrics

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision

print("Versions:")
print("python: 3.10")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("albumentations:", A.__version__)



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



## === cell 2
data_dir = "../input/alaska2-image-steganalysis"
sample_size = 75000
val_size = int(sample_size * 0.25)

train_fn, val_fn = [], []
train_labels, val_labels = [], []

folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # label 0 1 2 3

for label, folder in enumerate(folder_names):
    all_files = sorted(glob(f"{data_dir}/{folder}/*.jpg"))
    all_files = all_files[: min(sample_size, len(all_files))]
    np.random.shuffle(all_files)

    train_part = all_files[val_size:]
    val_part = all_files[:val_size]

    train_fn.extend(train_part)
    train_labels.extend((np.zeros(len(train_part)) + label).tolist())

    val_fn.extend(val_part)
    val_labels.extend((np.zeros(len(val_part)) + label).tolist())

train_df = pd.DataFrame(
    {"ImageFileName": train_fn, "Label": train_labels},
    columns=["ImageFileName", "Label"],
)
train_df["Label"] = train_df["Label"].astype(int)

val_df = pd.DataFrame(
    {"ImageFileName": val_fn, "Label": val_labels}, columns=["ImageFileName", "Label"]
)
val_df["Label"] = val_df["Label"].astype(int)

assert len(train_df) == len(train_fn)
assert len(val_df) == len(val_fn)

print(train_df.head())
print("train size:", len(train_df), "val size:", len(val_df))




## === cell 3
class Alaska2Dataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.data = df.reset_index(drop=True)
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fn = row["ImageFileName"]
        label = int(row["Label"])

        im = cv2.imread(fn)
        if im is None:
            raise FileNotFoundError(f"Failed to read image: {fn}")
        im = im[:, :, ::-1]  # BGR->RGB

        if self.augment:
            im = self.augment(image=im)

        return im, label


img_size = 512

AUGMENTATIONS_TRAIN = A.Compose(
    [
        A.Resize(img_size, img_size, p=1.0),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(75, 100), p=0.5),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ],
    p=1.0,
)

AUGMENTATIONS_TEST = A.Compose(
    [A.Resize(img_size, img_size, p=1.0), A.ToFloat(max_value=255.0), ToTensorV2()],
    p=1.0,
)



## === cell 4
temp_df = train_df.sample(16, random_state=seed).reset_index(drop=True)
temp_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)
temp_loader = torch.utils.data.DataLoader(
    temp_dataset, batch_size=16, num_workers=0, shuffle=False
)

images, labels = next(iter(temp_loader))
images = images["image"].permute(0, 2, 3, 1).numpy()

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
axs = axs.reshape(-1)
for i in range(16):
    axs[i].imshow(images[i])
    axs[i].set_title(str(int(labels[i])))
    axs[i].axis("off")
plt.suptitle("0: COVER, 1: JMiPOD, 2: JUNIWARD, 3: UERD")
plt.tight_layout()
plt.show()

del images, labels, temp_dataset, temp_loader
gc.collect()




## === cell 5
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.features = self.backbone.features  # conv feature extractor
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dense_output = nn.Linear(1280, 4)

    def forward(self, x):
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)  # (B, 1280)
        return self.dense_output(feat)




## === cell 6
batch_size = 8
num_workers = 4  # keep reasonable for Kaggle

train_dataset = Alaska2Dataset(train_df, augmentations=AUGMENTATIONS_TRAIN)
valid_dataset = Alaska2Dataset(val_df, augmentations=AUGMENTATIONS_TEST)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=True,
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size * 2,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(device)

possible_ckpts = [
    "../input/alaska/epoch_13_val_loss_6.67_auc_0.819.pth",
    "../input/epoch_13_val_loss_6.67_auc_0.819.pth",
    "./epoch_13_val_loss_6.67_auc_0.819.pth",
]
ckpt_path = next((p for p in possible_ckpts if os.path.exists(p)), None)
if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    print("Loaded checkpoint:", ckpt_path)
else:
    print("Checkpoint not found; using ImageNet-pretrained EfficientNet weights only.")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)




## === cell 7
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
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




## === cell 8
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 0
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
        tk0.set_postfix(loss=float(loss.item()))

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
            y.extend(labels.cpu().numpy().astype(int).tolist())
            preds.extend(F.softmax(outputs, 1).cpu().numpy().tolist())
            running_loss += loss.item()
            tk1.set_postfix(loss=float(loss.item()))

        epoch_loss = running_loss / max(1, len(valid_loader))
        val_loss.append(epoch_loss)

        preds = np.array(preds)
        labels_pred = preds.argmax(1)
        acc = (labels_pred == np.array(y)).mean() * 100.0

        new_preds = np.zeros((len(preds),), dtype=np.float32)
        temp = preds[labels_pred != 0, 1:]
        new_preds[labels_pred != 0] = temp.sum(1)
        new_preds[labels_pred == 0] = preds[labels_pred == 0, 0]

        y_bin = np.array(y)
        y_bin[y_bin != 0] = 1
        auc_score = alaska_weighted_auc(y_bin, new_preds)
        print(f"Val Loss: {epoch_loss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}")

    torch.save(
        model.state_dict(),
        f"epoch_{epoch+1}_val_loss_{epoch_loss:.3}_auc_{auc_score:.3}.pth",
    )



## === cell 9
if len(train_loss) > 0:
    plt.figure(figsize=(15, 7))
    plt.plot(train_loss, c="r")
    plt.plot(val_loss, c="b")
    plt.legend(["train_loss", "val_loss"])
    plt.title("Loss Plot")
    plt.show()




## === cell 10
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
            raise FileNotFoundError(f"Failed to read image: {fn}")
        im = im[:, :, ::-1]
        if self.augment:
            im = self.augment(image=im)
        return im


sample_path = f"{data_dir}/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)
id_list = sample_sub["Id"].tolist()
test_filenames = [f"{data_dir}/Test/{img_id}" for img_id in id_list]

test_df = pd.DataFrame({"ImageFileName": test_filenames}, columns=["ImageFileName"])

batch_size = 16
num_workers = 4

test_dataset = Alaska2TestDataset(test_df, augmentations=AUGMENTATIONS_TEST)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

print("Test images:", len(test_df))
assert len(test_df) == len(sample_sub)



## === cell 11
model.eval()

preds = []
tk0 = tqdm(test_loader, total=int(len(test_loader)))
with torch.no_grad():
    for im in tk0:
        inputs = im["image"].to(device, dtype=torch.float)

        out1 = model(inputs.flip(2))
        out2 = model(inputs.flip(3))
        out0 = model(inputs)
        outputs = 0.25 * out1 + 0.25 * out2 + 0.5 * out0

        preds.extend(F.softmax(outputs, 1).detach().cpu().numpy())

preds = np.array(preds)

stego_prob = preds[:, 1:].sum(1).astype(np.float32)

submission = pd.DataFrame({"Id": id_list, "Label": stego_prob})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
assert submission.shape[0] == 5000 and list(submission.columns) == ["Id", "Label"]

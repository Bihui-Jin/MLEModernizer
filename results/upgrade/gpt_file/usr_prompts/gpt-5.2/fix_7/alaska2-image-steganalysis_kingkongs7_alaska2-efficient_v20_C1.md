# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import random
import gc
from glob import glob
import time

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn import metrics
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(max(1, os.cpu_count() // 2))



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


def _seed_worker(worker_id: int):
    worker_seed = (seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




## === cell 2
data_dir = "../input/alaska2-image-steganalysis"
sample_size = 75000
val_size = int(sample_size * 0.25)

train_fn, val_fn = [], []
train_labels, val_labels = [], []

folder_names = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]  # label 0 1 2 3
rng = np.random.RandomState(seed)
for label, folder in enumerate(folder_names):
    filenames = glob(f"{data_dir}/{folder}/*.jpg")
    if len(filenames) == 0:
        raise FileNotFoundError(f"No jpg files found in: {data_dir}/{folder}")

    filenames = np.array(filenames, dtype=object)
    rng.shuffle(filenames)
    filenames = filenames[:sample_size]

    val_files = filenames[:val_size].tolist()
    trn_files = filenames[val_size:].tolist()

    train_fn.extend(trn_files)
    train_labels.extend([label] * len(trn_files))
    val_fn.extend(val_files)
    val_labels.extend([label] * len(val_files))

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
        self.fns = self.data["ImageFileName"].to_numpy()
        self.labels = self.data["Label"].to_numpy(dtype=np.int64, copy=False)
        self.augment = augmentations

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        fn = self.fns[idx]
        label = int(self.labels[idx])

        im = cv2.imread(fn, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {fn}")
        im = im[:, :, ::-1]  # BGR -> RGB

        if self.augment is not None:
            out = self.augment(image=im)
            im = out["image"]

        return im, label


img_size = 512

AUGMENTATIONS_TRAIN = A.Compose(
    [
        A.Resize(img_size, img_size),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(75, 100), p=0.5),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

AUGMENTATIONS_TEST = A.Compose(
    [
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ]
)



## === cell 4
if False:
    temp_df = train_df.sample(32, random_state=seed).reset_index(drop=True)
    train_dataset_vis = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)

    batch_size = 32
    num_workers = 0

    temp_loader = torch.utils.data.DataLoader(
        train_dataset_vis, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    images, labels = next(iter(temp_loader))
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    images = np.clip(images, 0, 1)

    grid_width = 8
    grid_height = int(np.ceil(len(images) / grid_width))
    fig, axs = plt.subplots(
        grid_height, grid_width, figsize=(grid_width + 2, grid_height + 2)
    )
    axs = np.array(axs).reshape(grid_height, grid_width)

    for i in range(grid_height * grid_width):
        ax = axs[i // grid_width, i % grid_width]
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(str(int(labels[i].item())))
        ax.axis("off")

    plt.suptitle("0: COVER, 1: JMiPOD, 2: JUNIWARD, 3: UERD")
    plt.show()
    del images, labels, temp_df, train_dataset_vis, temp_loader
    gc.collect()




## === cell 5
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = torchvision.models.efficientnet_b0(weights=weights)
        self.dense_output = nn.Linear(1280, 4)

    def forward(self, x):
        feat = self.model.features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        return self.dense_output(feat)




## === cell 6
batch_size = 8

num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))

train_dataset = Alaska2Dataset(train_df, augmentations=AUGMENTATIONS_TRAIN)
valid_dataset = Alaska2Dataset(val_df, augmentations=AUGMENTATIONS_TEST)

g = torch.Generator()
g.manual_seed(seed)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,  # less memory pressure vs 4
    worker_init_fn=_seed_worker,
    generator=g,
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size * 2,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    worker_init_fn=_seed_worker,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(device)

candidate_ckpts = [
    "../input/alaska/epoch_10_val_loss_6.73_auc_0.815.pth",
    "../input/alaska2-image-steganalysis/epoch_10_val_loss_6.73_auc_0.815.pth",
    "../input/alaska2-image-steganalysis/alaska2-image-steganalysis/epoch_10_val_loss_6.73_auc_0.815.pth",
    "/kaggle/input/alaska/epoch_10_val_loss_6.73_auc_0.815.pth",
    "/kaggle/input/alaska2-image-steganalysis/epoch_10_val_loss_6.73_auc_0.815.pth",
]
loaded = False
for ckpt_path in candidate_ckpts:
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint: {ckpt_path}")
        loaded = True
        break
if not loaded:
    print(
        "Checkpoint not found. Proceeding with ImageNet-pretrained EfficientNet-B0 backbone + fresh head."
    )

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)




## === cell 7
def alaska_weighted_auc(y_true, y_score):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)

    areas = np.diff(np.array(tpr_thresholds))
    normalization = float(np.dot(areas, weights))

    score = 0.0
    for i, w in enumerate(weights):
        y_min, y_max = tpr_thresholds[i], tpr_thresholds[i + 1]
        tpr_c = np.clip(tpr, y_min, y_max) - y_min
        score += w * metrics.auc(fpr, tpr_c)

    return score / normalization




## === cell 8
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 0 if loaded else 2

train_loss, val_loss = [], []

log_every = 500

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    print("-" * 10)
    model.train()
    running_loss = 0.0

    tk0 = tqdm(train_loader, total=len(train_loader))
    for step, (im, labels) in enumerate(tk0):
        inputs = im.to(device, dtype=torch.float, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_scalar = float(loss.detach().cpu().item())
        running_loss += loss_scalar
        if (step % log_every) == 0:
            tk0.set_postfix(loss=loss_scalar)

    epoch_loss = running_loss / max(1, len(train_loader))
    train_loss.append(epoch_loss)
    print("Training Loss: {:.8f}".format(epoch_loss))

    model.eval()
    running_loss = 0.0

    y_arr = np.empty(len(valid_dataset), dtype=np.int64)
    preds_arr = np.empty((len(valid_dataset), 4), dtype=np.float32)
    ofs = 0

    tk1 = tqdm(valid_loader, total=len(valid_loader))
    with torch.inference_mode():
        for step, (im, labels) in enumerate(tk1):
            b = labels.size(0)
            inputs = im.to(device, dtype=torch.float, non_blocking=True)
            labels_dev = labels.to(device, dtype=torch.long, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels_dev)

            running_loss += float(loss.detach().cpu().item())

            y_arr[ofs : ofs + b] = labels.numpy().astype(np.int64, copy=False)
            preds_arr[ofs : ofs + b] = F.softmax(outputs, 1).detach().cpu().numpy()
            ofs += b

            if (step % log_every) == 0:
                tk1.set_postfix(loss=float(loss.detach().cpu().item()))

    epoch_loss = running_loss / max(1, len(valid_loader))
    val_loss.append(epoch_loss)

    preds = preds_arr[:ofs]
    y = y_arr[:ofs]

    pred_class = preds.argmax(1)
    acc = (pred_class == y).mean() * 100.0

    new_preds = (1.0 - preds[:, 0]).astype(np.float32)

    y_bin = (y != 0).astype(int)
    auc_score = alaska_weighted_auc(y_bin, new_preds)
    print(f"Val Loss: {epoch_loss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}")

    torch.save(
        model.state_dict(),
        f"epoch_{epoch+1}_val_loss_{epoch_loss:.3}_auc_{auc_score:.3}.pth",
    )



## === cell 9
if False and len(train_loss) > 0:
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
        self.fns = self.data["ImageFileName"].to_numpy()
        self.augment = augmentations

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        fn = self.fns[idx]
        im = cv2.imread(fn, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {fn}")
        im = im[:, :, ::-1]

        if self.augment is not None:
            out = self.augment(image=im)
            im = out["image"]
        return im


test_filenames = glob(f"{data_dir}/Test/*.jpg")
if len(test_filenames) == 0:
    raise FileNotFoundError(f"No test jpg files found in: {data_dir}/Test")

test_df = pd.DataFrame(
    {"ImageFileName": list(test_filenames)}, columns=["ImageFileName"]
)

batch_size = 32  # Speed: larger batch for inference, same math.
num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
test_dataset = Alaska2TestDataset(test_df, augmentations=AUGMENTATIONS_TEST)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    worker_init_fn=_seed_worker,
)

print("Test images:", len(test_df))



## === cell 11
model.eval()

preds_arr = np.empty((len(test_dataset), 4), dtype=np.float32)
ofs = 0

tk0 = tqdm(test_loader, total=len(test_loader))
with torch.inference_mode():
    for im in tk0:
        b = im.size(0)
        inputs = im.to(device, dtype=torch.float, non_blocking=True)

        out0 = model(inputs)
        out1 = model(inputs.flip(2))
        out2 = model(inputs.flip(3))

        outputs = 0.5 * out0 + 0.25 * out1 + 0.25 * out2
        preds_arr[ofs : ofs + b] = F.softmax(outputs, 1).detach().cpu().numpy()
        ofs += b

preds = preds_arr[:ofs]

new_preds = (1.0 - preds[:, 0]).astype(np.float32)

test_df["Id"] = test_df["ImageFileName"].map(os.path.basename)
test_df["Label"] = new_preds.astype(float)

submission = test_df[["Id", "Label"]].copy()
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))

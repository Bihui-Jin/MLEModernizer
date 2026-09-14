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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.6687791547669983

# 6. Current score

0.8888

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.03655) has done: 'I remove the hard dependency on a missing `../input/checkpoint4` by loading a checkpoint only if it exists, otherwise instantiating the same kind of torchvision model and running with its available weights so the notebook completes. I also fix path robustness by auto-detecting the dataset base folder (it can be under `../input/aptos2019-blindness-detection` or `../input/aptos2019-blindness-detection/aptos2019-blindness-detection`) and ensure images are loaded as RGB safely. Finally, I make inference device-safe (CPU/GPU), run in `torch.no_grad()`, and write a valid `submission.csv` with the exact required columns aligned to `test.csv` order.'
- What this solution (achieved 0.8888) has done: 'Your current score is far below the target because the model you submit is effectively untrained for this task (random 5-class head on ImageNet features), so predictions collapse to near-random labels and QWK goes negative. To move toward the target with minimal changes and without altering the core architecture/training paradigm, I add a short, deterministic fine-tuning step on the provided `train.csv` + `train_images` using the same ResNet18 backbone and a standard CrossEntropyLoss. I keep preprocessing consistent with your current inference pipeline, add a small validation split and compute QWK to pick the best epoch (no early stopping; we still run all epochs), then generate `submission.csv` in the exact required format aligned to `test.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
os.listdir("../input")



## === cell 2
ckpt_dir = "../input/checkpoint4"
ckpt_path = os.path.join(ckpt_dir, "checkpoint_epoch_4.pt")

has_ckpt = os.path.exists(ckpt_path)
has_ckpt, ckpt_path



## === cell 3
checkpoint = None
if has_ckpt:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
checkpoint is not None




## === cell 4
def build_model(num_classes: int = 5) -> nn.Module:
    try:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        m = models.resnet18(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


if checkpoint is not None and isinstance(checkpoint, dict) and "model" in checkpoint:
    model = checkpoint["model"]
else:
    model = build_model(num_classes=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

type(model), device




## === cell 5
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = sorted(
            [f for f in os.listdir(self.root_dir) if f.lower().endswith(".png")]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        path = os.path.join(self.root_dir, fname)
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, fname




## === cell 6
candidate_bases = [
    "../input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
]
base_dir = None
for b in candidate_bases:
    if os.path.exists(os.path.join(b, "test_images")) and os.path.exists(
        os.path.join(b, "test.csv")
    ):
        base_dir = b
        break

if base_dir is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection data folder. "
        f"Tried: {candidate_bases}"
    )

test_data_dir = os.path.join(base_dir, "test_images")
test_csv_path = os.path.join(base_dir, "test.csv")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")

train_data_dir = os.path.join(base_dir, "train_images")
train_csv_path = os.path.join(base_dir, "train.csv")

img_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_dataset = TestDataset(test_data_dir, img_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=20,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

len(test_dataset), test_data_dir



## === cell 7


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < num_classes and 0 <= b < num_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id_code"]
        label = int(row["diagnosis"])
        path = os.path.join(self.root_dir, f"{img_id}.png")
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


train_df_full = pd.read_csv(train_csv_path)

perm = np.random.RandomState(SEED).permutation(len(train_df_full))
val_size = max(1, int(0.15 * len(train_df_full)))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

train_df = train_df_full.iloc[trn_idx].reset_index(drop=True)
val_df = train_df_full.iloc[val_idx].reset_index(drop=True)

train_dataset = TrainDataset(train_df, train_data_dir, img_transform)
val_dataset = TrainDataset(val_df, train_data_dir, img_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=24,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

criterion = nn.CrossEntropyLoss()

do_train = not (
    checkpoint is not None and isinstance(checkpoint, dict) and "model" in checkpoint
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 2  # small, to move toward target while respecting 600s constraint

best_state = None
best_val_kappa = -1e9

if do_train:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * imgs.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.numpy().tolist())

        val_kappa = quadratic_weighted_kappa(y_true, y_pred, num_classes=5)

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"epoch={epoch+1}/{num_epochs} train_loss={train_loss:.4f} val_qwk={val_kappa:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

model.eval()
best_val_kappa



## === cell 8
id_codes = []
diags = []

with torch.no_grad():
    for imgs, files in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)
        diagnosis = torch.argmax(preds, dim=1)
        for fn, diag in zip(files, diagnosis):
            id_codes.append(fn.replace(".png", ""))
            diags.append(int(diag.item()))

pred_df = pd.DataFrame({"id_code": id_codes, "diagnosis": diags})

test_df = pd.read_csv(test_csv_path)
sub_df = test_df.merge(pred_df, on="id_code", how="left")
sub_df["diagnosis"] = sub_df["diagnosis"].fillna(0).astype(int)

sub_path = "./submission.csv"
sub_df.to_csv(sub_path, index=False)

sub_path, sub_df.shape, sub_df.head()



## === cell 9
sub_df.head()

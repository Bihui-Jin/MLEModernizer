# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        input/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
            test_images/
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
            train_images/
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> input/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> working/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import shutil
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn import metrics
from functools import partial
import scipy as sp
import scipy.optimize
import re
import math

os.makedirs("models", exist_ok=True)
src_ckpt = "../input/kaggle-public/abcdef.pth"
dst_ckpt = "models/abcdef.pth"  # corrected extension
if os.path.exists(src_ckpt):
    shutil.copy(src_ckpt, dst_ckpt)




## === cell 1
def locate_path(*relative_parts):
    """Return the first existing path by trying several common Kaggle root prefixes."""
    candidates = [
        os.path.join(*relative_parts),  # relative to cwd
        os.path.join("input", *relative_parts),  # typical competition folder
        os.path.join("/kaggle", "input", *relative_parts),  # absolute kaggle input
        os.path.join("/", *relative_parts),  # root absolute
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not locate {'/'.join(relative_parts)} in any known location."
    )


class TestDataset(Dataset):
    """Pre‑load all test images as tensors to avoid per‑batch I/O."""

    def __init__(self, csv_path, img_root, transform=None):
        df = pd.read_csv(csv_path)
        self.ids = df["id_code"].tolist()
        self.transform = transform
        tensors = []
        for id_code in self.ids:
            img_path = os.path.join(img_root, f"{id_code}.png")
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            tensors.append(img)
        self.tensors = torch.stack(tensors)  # shape (N, C, H, W)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.tensors[idx], self.ids[idx]


class TrainDataset(Dataset):
    """Pre‑load all training images, labels and ids as tensors."""

    def __init__(self, csv_path, img_root, transform=None):
        df = pd.read_csv(csv_path)
        self.ids = df["id_code"].tolist()
        self.labels = df["diagnosis"].astype(int).tolist()
        self.transform = transform
        tensors = []
        for id_code in self.ids:
            img_path = os.path.join(img_root, f"{id_code}.png")
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            tensors.append(img)
        self.tensors = torch.stack(tensors)  # (N, C, H, W)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx], self.ids[idx]


img_size = 224
test_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

base_dir = locate_path("aptos2019-blindness-detection")
test_csv_path = locate_path("aptos2019-blindness-detection", "test.csv")
test_images_dir = locate_path("aptos2019-blindness-detection", "test_images")
train_csv_path = locate_path("aptos2019-blindness-detection", "train.csv")
train_images_dir = locate_path("aptos2019-blindness-detection", "train_images")

test_dataset = TestDataset(
    csv_path=test_csv_path,
    img_root=test_images_dir,
    transform=test_transform,
)

worker_count = min(4, os.cpu_count() or 1)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=worker_count,
    pin_memory=True,
    persistent_workers=True,  # keep workers alive between epochs
)




## === cell 2
class OptimizedRounder(object):
    """Vectorized version of threshold optimisation for quadratic weighted kappa."""

    def __init__(self):
        self.coef_ = None

    def _kappa_loss(self, coef, X, y):
        thresholds = np.array(coef)
        X_p = np.digitize(X, thresholds, right=False)
        return -metrics.cohen_kappa_score(y, X_p, weights="quadratic")

    def fit(self, X, y):
        loss = partial(self._kappa_loss, X=X, y=y)
        init_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss, init_coef, method="nelder-mead")
        print("Optimized kappa:", -loss(self.coef_["x"]))

    def predict(self, X, coef=None):
        if coef is None:
            coef = self.coef_["x"] if self.coef_ is not None else [0.5, 1.5, 2.5, 3.5]
        thresholds = np.array(coef)
        X_p = np.digitize(X, thresholds, right=False)
        return X_p.astype(int)




## === cell 3
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b5(pretrained=True)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)
model = model.to(device)

ckpt_path = "models/abcdef.pth"  # corrected path
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
else:
    print("Checkpoint not found – using default pretrained weights.")

model.eval()

full_dataset = TrainDataset(
    csv_path=train_csv_path,
    img_root=train_images_dir,
    transform=test_transform,
)

val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

train_loader = DataLoader(
    train_subset,
    batch_size=64,
    shuffle=True,
    num_workers=worker_count,
    pin_memory=True,
    persistent_workers=True,
)

scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

model.train()
max_epochs = 8 if device.type == "cuda" else 2
for epoch in range(max_epochs):
    epoch_losses = []
    for imgs, labels, _ in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        epoch_losses.append(loss.item())
    print(f"Fine‑tune epoch {epoch+1}/{max_epochs}, loss: {np.mean(epoch_losses):.4f}")

model.eval()

val_loader = DataLoader(
    val_subset,
    batch_size=64,
    shuffle=False,
    num_workers=worker_count,
    pin_memory=True,
    persistent_workers=True,
)

val_preds = []
val_labels = []
with torch.no_grad():
    for imgs, labels, _ in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs).squeeze(1).cpu().numpy()
        val_preds.extend(logits)
        val_labels.extend(labels.numpy())

opt = OptimizedRounder()
opt.fit(np.array(val_preds), np.array(val_labels))

all_preds = []
ids = []
with torch.no_grad():
    for imgs, img_ids in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs).squeeze(1)
        all_preds.extend(logits.cpu().numpy())
        ids.extend(img_ids)

final_preds = opt.predict(np.array(all_preds))

submission = pd.DataFrame({"id_code": ids, "diagnosis": final_preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.9080228716816332

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import cohen_kappa_score
import scipy.optimize as sp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)




## === cell 1
class RetinaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.paths = df["path"].values
        self.labels = df["diagnosis"].values if "diagnosis" in df.columns else None
        self.transform = transform

        self.cached_imgs = []
        for p in self.paths:
            img = torchvision.io.read_image(p).float() / 255.0  # C,H,W as float32
            if self.transform:
                img = self.transform(img)
            self.cached_imgs.append(img)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.cached_imgs[idx]
        if self.labels is not None:
            label = int(self.labels[idx])
            return img, label
        else:
            return img




## === cell 2
def get_dataframes():
    base_dir = os.path.join("..", "input", "aptos2019-blindness-detection")
    train_csv = os.path.join(base_dir, "train.csv")
    test_csv = os.path.join(base_dir, "sample_submission.csv")
    train_img_dir = os.path.join(base_dir, "train_images")
    test_img_dir = os.path.join(base_dir, "test_images")
    train_df = pd.read_csv(train_csv)
    train_df["path"] = train_df["id_code"].apply(
        lambda x: os.path.join(train_img_dir, f"{x}.png")
    )
    test_df = pd.read_csv(test_csv)
    test_df["path"] = test_df["id_code"].apply(
        lambda x: os.path.join(test_img_dir, f"{x}.png")
    )
    return train_df, test_df


train_df, test_df = get_dataframes()




## === cell 3
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    stratify=train_df["diagnosis"],
    random_state=42,
)

df_train = train_df.iloc[train_idx].reset_index(drop=True)
df_val = train_df.iloc[val_idx].reset_index(drop=True)

size = 224
train_tfms = T.Compose(
    [
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=15),
    ]
)
val_tfms = T.Compose(
    [
        T.Resize((size, size)),
    ]
)

batch_size = 32
train_ds = RetinaDataset(df_train, img_dir="", transform=train_tfms)
val_ds = RetinaDataset(df_val, img_dir="", transform=val_tfms)
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
)




## === cell 4
def get_model(num_classes=5):
    try:
        model = torchvision.models.efficientnet_b5(pretrained=True)
    except AttributeError:
        model = torchvision.models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


model = get_model(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
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
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
            else:
                logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_true)




## === cell 5
epochs = 5
for epoch in range(epochs):
    loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_pred, val_true = evaluate(model, val_loader)
    qk = cohen_kappa_score(val_pred, val_true, weights="quadratic")
    print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f} - val QWK: {qk:.4f}")




## === cell 6
class OptimizedRounder:
    def __init__(self):
        self.coef_ = None

    def _kappa_loss(self, coef, preds, labels):
        preds_adj = np.digitize(preds, coef)
        return -cohen_kappa_score(labels, preds_adj, weights="quadratic")

    def fit(self, preds, labels):
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        bounds = [(0, 4)] * 4
        result = sp.minimize(
            self._kappa_loss, initial_coef, args=(preds, labels), method="nelder-mead"
        )
        self.coef_ = result.x

    def predict(self, preds):
        return np.digitize(preds, self.coef_)


model.eval()
val_probs = []
val_labels = []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        val_probs.append(probs)
        val_labels.append(labels.numpy())
val_probs = np.concatenate(val_probs)
val_labels = np.concatenate(val_labels)

opt_round = OptimizedRounder()
opt_round.fit(val_probs, val_labels)




## === cell 7
test_ds = RetinaDataset(test_df, img_dir="", transform=val_tfms)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
)

model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        if isinstance(batch, list):
            imgs = torch.stack(batch)
        else:
            imgs = batch
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        test_preds.append(probs)
test_preds = np.concatenate(test_preds)

final_pred = opt_round.predict(test_preds).astype(int)

submission = pd.DataFrame({"id_code": test_df["id_code"], "diagnosis": final_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/273041359.py in <cell line: 0>()
     10         # The batch may be a list of tensors (if collate could not stack); ensure a tensor before moving to device
     11         if isinstance(batch, list):
---> 12             imgs = torch.stack(batch)
     13         else:
     14             imgs = batch

RuntimeError: stack expects each tensor to be equal size, but got [32, 3, 224, 224] at entry 0 and [32] at entry 1

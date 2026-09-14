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

No external packages required in the script and installed.

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

0.8321871294659671

# 6. Current score

0.66597

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the unavailable fastai imports, replace them with a lightweight PyTorch‑only data pipeline, and add the missing os import and scipy optimizer import. The core EfficientNet model definition stays unchanged, and a simple training loop with a validation split is added so the script runs end‑to‑end and writes a valid submission.csv file.'
- What this solution (achieved 0.0) has done: 'Implemented fixes to correctly build train/validation DataFrames from `random_split` Subsets (using their indices) and updated the training loop to run 5 epochs for a modest performance gain. These changes resolve the DataFrame construction error, ensure loaders are defined, and produce a valid `submission.csv` while moving the QWK score above zero.'
- What this solution (achieved 0.66597) has done: 'The changes focus on eliminating the expensive repeated soft‑max computation inside the threshold optimisation loop and removing the overhead of spawning workers for the modest‑size image data. By pre‑computing class probabilities and expected scores once, the optimiser evaluates the loss much faster while yielding identical thresholds. Setting `num_workers=0` avoids unnecessary process creation during DataLoader iteration, which also speeds up training without altering any model logic.'

# 9. Code solution

## === cell 0
import warnings, os, math, re, collections, pathlib
from functools import partial
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score
import scipy.optimize as sp
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




## === cell 1
class DRDataset(Dataset):
    def __init__(self, dataframe, transform=None, is_test=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx in self._cache:
            if self.is_test:
                return self._cache[idx]
            else:
                img_tensor = self._cache[idx]
                label = int(self.df.loc[idx, "diagnosis"])
                return img_tensor, label

        img_path = self.df.loc[idx, "path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        self._cache[idx] = image

        if self.is_test:
            return image
        label = int(self.df.loc[idx, "diagnosis"])
        return image, label




## === cell 2
from torchvision.models import efficientnet_b0

model = efficientnet_b0(pretrained=False, num_classes=5)


def get_base_path():
    possible = ["./input", "/kaggle/input"]
    for p in possible:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("Could not locate the input directory.")


def get_df():
    base = os.path.join(get_base_path(), "aptos2019-blindness-detection")
    train_dir = os.path.join(base, "train_images")
    df = pd.read_csv(os.path.join(base, "train.csv"))
    df["path"] = df["id_code"].apply(lambda x: os.path.join(train_dir, f"{x}.png"))
    test_df = pd.read_csv(os.path.join(base, "sample_submission.csv"))
    return df, test_df


df, test_df = get_df()

tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_frac = 0.2
train_idx, val_idx = train_test_split(
    np.arange(len(df)), test_size=val_frac, random_state=42, stratify=df["diagnosis"]
)

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

train_dataset = DRDataset(train_df, transform=tfms, is_test=False)
val_dataset = DRDataset(val_df, transform=tfms, is_test=False)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
)




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def qk_score(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


val_logits_all = []
val_true_all = []

for epoch in range(12):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
    model.eval()
    all_true, all_pred = [], []
    epoch_val_logits = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(yb.numpy())
            epoch_val_logits.append(logits.cpu())
    val_kappa = qk_score(all_true, all_pred)
    print(f"Epoch {epoch+1}, Val QWK: {val_kappa:.4f}")

    val_logits_all = np.concatenate([l.numpy() for l in epoch_val_logits], axis=0)
    val_true_all = np.array(all_true)


def apply_thresholds(logits, thresholds):
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    scores = np.dot(probs, np.arange(5))  # expected value as continuous prediction
    preds = np.digitize(scores, thresholds)
    return preds


def optimise_thresholds(logits, true):
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    scores = np.dot(probs, np.arange(5))

    init_thr = np.array([0.5, 1.5, 2.5, 3.5])

    def loss(thr):
        thr = np.sort(thr)
        pred = np.digitize(scores, thr)
        return -qk_score(true, pred)

    bounds = [(0, 4)] * 4
    result = sp.minimize(loss, init_thr, bounds=bounds, method="L-BFGS-B")
    return np.sort(result.x)


opt_thresholds = optimise_thresholds(val_logits_all, val_true_all)
print("Optimised thresholds:", opt_thresholds)




## === cell 4
test_dataset = DRDataset(
    test_df.assign(
        path=test_df["id_code"].apply(
            lambda x: os.path.join(
                get_base_path(),
                "aptos2019-blindness-detection",
                "test_images",
                f"{x}.png",
            )
        )
    ),
    transform=tfms,
    is_test=True,
)

test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True
)

model.eval()
test_logits = []
with torch.no_grad():
    for xb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        test_logits.append(logits.cpu())
test_logits = np.concatenate([l.numpy() for l in test_logits], axis=0)

test_preds = apply_thresholds(test_logits, opt_thresholds)
test_df["diagnosis"] = test_preds.astype(int)
submission_path = "submission.csv"
test_df[["id_code", "diagnosis"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

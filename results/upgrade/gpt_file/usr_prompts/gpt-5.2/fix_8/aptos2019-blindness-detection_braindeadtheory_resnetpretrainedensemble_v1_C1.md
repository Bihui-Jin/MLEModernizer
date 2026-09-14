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

3.8

# 3. Installed packages

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
pillow==11.3.0
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

0.8545481310495799

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.18561) has done: 'I fix the runtime failure by removing the dependency on a missing external weight file (`../input/resnet1/FinalResnet0.pt`) and instead load the ResNet101 ImageNet pretrained weights that are available via torchvision in Kaggle. I also correct the model initialization order (define `fc` before loading weights) and make inference robust by ensuring proper CPU conversion and using a standard `tqdm` progress bar (the current notebook-specific import can break in script runs). Finally, I keep the same core prediction logic (argmax over 5-class logits) and write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import random

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import functional as TF




## === cell 1
transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)


def _resolve_comp_root():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/data/input/aptos2019-blindness-detection",
        "../input/aptos2019-blindness-detection",
        "../data/aptos2019-blindness-detection",
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
    ]

    def is_valid_root(root):
        return (
            os.path.exists(os.path.join(root, "train.csv"))
            and os.path.exists(os.path.join(root, "test.csv"))
            and (
                os.path.isdir(os.path.join(root, "train_images"))
                or os.path.isdir(
                    os.path.join(root, "aptos2019-blindness-detection", "train_images")
                )
            )
        )

    for c in candidates:
        if os.path.exists(c):
            if is_valid_root(c):
                return c
            nested = os.path.join(c, "aptos2019-blindness-detection")
            if is_valid_root(nested):
                return nested

    return "../input/aptos2019-blindness-detection"


COMP_ROOT = _resolve_comp_root()

TRAIN_IMG_DIR = os.path.join(COMP_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(COMP_ROOT, "test_images")
if not os.path.isdir(TRAIN_IMG_DIR):
    alt = os.path.join(COMP_ROOT, "aptos2019-blindness-detection", "train_images")
    if os.path.isdir(alt):
        TRAIN_IMG_DIR = alt
if not os.path.isdir(TEST_IMG_DIR):
    alt = os.path.join(COMP_ROOT, "aptos2019-blindness-detection", "test_images")
    if os.path.isdir(alt):
        TEST_IMG_DIR = alt


class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None, df=None):
        if df is None:
            eye_frame = pd.read_csv(csv_file)
        else:
            eye_frame = df

        self.filetype = filetype
        self.transform = transform

        id_codes = eye_frame["id_code"].astype(str).to_numpy()
        if self.filetype == "train":
            self.labels = eye_frame["diagnosis"].astype(np.int64).to_numpy()
            base_dir = TRAIN_IMG_DIR
        else:
            self.labels = None
            base_dir = TEST_IMG_DIR

        self.id_codes = id_codes
        self.img_paths = np.char.add(np.char.add(base_dir + os.sep, id_codes), ".png")

        self._mean = torch.tensor([0.460, 0.247, 0.080], dtype=torch.float32).view(
            3, 1, 1
        )
        self._std = torch.tensor([0.249, 0.138, 0.081], dtype=torch.float32).view(
            3, 1, 1
        )

    def __len__(self):
        return self.id_codes.shape[0]

    def _load_image_tensor(self, img_path: str) -> torch.Tensor:
        img = read_image(img_path, mode=ImageReadMode.RGB).to(torch.float32).div_(255.0)
        img = TF.resize(
            img, [320, 320], interpolation=TF.InterpolationMode.BILINEAR, antialias=True
        )
        img = (img - self._mean) / self._std
        return img

    def __getitem__(self, idx):
        img_path = str(self.img_paths[idx])
        image = self._load_image_tensor(img_path)

        if self.filetype == "train":
            return image, int(self.labels[idx])
        else:
            return image, str(self.id_codes[idx])




## === cell 2
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

weights = torchvision.models.ResNet101_Weights.DEFAULT
model = torchvision.models.resnet101(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model = model.to(device)

for name, p in model.named_parameters():
    p.requires_grad = False
for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True




## === cell 3
train_df = pd.read_csv(os.path.join(COMP_ROOT, "train.csv"))

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["diagnosis"].values,
)

train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
val_split_df = train_df.iloc[val_idx].reset_index(drop=True)

train_split_path = "train_split.csv"
val_split_path = "val_split.csv"
train_split_df.to_csv(train_split_path, index=False)
val_split_df.to_csv(val_split_path, index=False)

train_dataset = APTOSDataset(
    train_split_path, filetype="train", transform=transform, df=train_split_df
)
val_dataset = APTOSDataset(
    val_split_path, filetype="train", transform=transform, df=val_split_df
)

pin_memory = torch.cuda.is_available()
num_workers = 2  # keep as-is to preserve throughput and core behavior

loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=24,
    shuffle=True,
    drop_last=False,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=24,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(model.layer4.parameters()) + list(model.fc.parameters()),
    lr=1e-4,  # kept
)


def eval_kappa(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=pin_memory)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


EPOCHS = 8

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
        inputs = inputs.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    kappa = eval_kappa(model, val_loader, device)
    print(f"epoch={epoch} train_loss={avg_loss:.4f} val_qwk={kappa:.4f}")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3538850293.py in <cell line: 0>()
     17 val_split_df.to_csv(val_split_path, index=False)
     18 
---> 19 train_dataset = APTOSDataset(
     20     train_split_path, filetype="train", transform=transform, df=train_split_df
     21 )

/tmp/ipykernel_55/3816634200.py in __init__(self, csv_file, filetype, transform, df)
     81 
     82         self.id_codes = id_codes
---> 83         self.img_paths = np.char.add(np.char.add(base_dir + os.sep, id_codes), ".png")
     84 
     85         # Pre-store normalization tensors for fast tensor path

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U57' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 4
test_dataset = APTOSDataset(
    csv_file=os.path.join(COMP_ROOT, "test.csv"),
    filetype="test",
    transform=transform,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=24,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2501184344.py in <cell line: 0>()
----> 1 test_dataset = APTOSDataset(
      2     csv_file=os.path.join(COMP_ROOT, "test.csv"),
      3     filetype="test",
      4     transform=transform,
      5 )

/tmp/ipykernel_55/3816634200.py in __init__(self, csv_file, filetype, transform, df)
     81 
     82         self.id_codes = id_codes
---> 83         self.img_paths = np.char.add(np.char.add(base_dir + os.sep, id_codes), ".png")
     84 
     85         # Pre-store normalization tensors for fast tensor path

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U56' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 5
def _expected_severity_from_logits(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits, dim=1)
    classes = torch.arange(5, device=probs.device, dtype=probs.dtype).view(1, -1)
    exp = (probs * classes).sum(dim=1)
    return exp.detach().cpu().numpy()


def _apply_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    t0, t1, t2, t3 = thresholds
    return np.digitize(scores, bins=[t0, t1, t2, t3]).astype(int)


def fit_qwk_thresholds(val_scores: np.ndarray, val_labels: np.ndarray) -> np.ndarray:
    thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)

    min_s, max_s = float(val_scores.min()), float(val_scores.max())
    grid = np.linspace(min_s, max_s, 61)

    def qwk_for(thr):
        pred = _apply_thresholds(val_scores, thr)
        return cohen_kappa_score(val_labels, pred, weights="quadratic")

    best = qwk_for(thresholds)

    for _ in range(6):
        improved = False
        for k in range(4):
            best_k = thresholds[k]
            best_local = best
            for candidate in grid:
                thr = thresholds.copy()
                thr[k] = candidate
                if not (thr[0] < thr[1] < thr[2] < thr[3]):
                    continue
                score = qwk_for(thr)
                if score > best_local:
                    best_local = score
                    best_k = candidate
            if best_local > best:
                thresholds[k] = best_k
                best = best_local
                improved = True
        if not improved:
            break

    return thresholds


def get_val_scores_and_labels(model, data_loader, device):
    model.eval()
    scores_list, y_list = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Collect val scores", leave=False):
            inputs = inputs.to(device, non_blocking=pin_memory)
            outputs = model(inputs)
            scores = _expected_severity_from_logits(outputs)
            scores_list.append(scores)
            y_list.append(labels.numpy())
    return np.concatenate(scores_list), np.concatenate(y_list)


val_scores, val_labels = get_val_scores_and_labels(model, val_loader, device)
thresholds = fit_qwk_thresholds(val_scores, val_labels)
val_pred_thr = _apply_thresholds(val_scores, thresholds)
val_qwk_thr = cohen_kappa_score(val_labels, val_pred_thr, weights="quadratic")
print("Fitted thresholds:", thresholds)
print(f"val_qwk_after_thresholding={val_qwk_thr:.4f}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3824602419.py in <cell line: 0>()
     60 
     61 
---> 62 val_scores, val_labels = get_val_scores_and_labels(model, val_loader, device)
     63 thresholds = fit_qwk_thresholds(val_scores, val_labels)
     64 val_pred_thr = _apply_thresholds(val_scores, thresholds)

NameError: name 'val_loader' is not defined

## === cell 6
def compute_predictions(model, model_type, data_loader, device, thresholds=None):
    if model_type == "train":
        model.eval()
        predictions = []
        correct_pred, num_examples = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(
                data_loader, desc="Predict (train)", leave=False
            ):
                inputs = inputs.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.append(preds.detach().cpu())
                num_examples += labels.size(0)
                correct_pred += (preds == labels).sum()
        return predictions, correct_pred.item() / num_examples * 100.0
    else:
        model.eval()
        predictions = []
        img_ids = []
        with torch.no_grad():
            for inputs, img_id in tqdm(data_loader, desc="Predict (test)"):
                inputs = inputs.to(device, non_blocking=pin_memory)
                outputs = model(inputs)

                if thresholds is not None:
                    scores = _expected_severity_from_logits(outputs)
                    preds = _apply_thresholds(scores, thresholds).tolist()
                else:
                    _, preds_t = torch.max(outputs, 1)
                    preds = preds_t.detach().cpu().tolist()

                predictions.extend(preds)
                img_ids.extend(list(img_id))

        pred_df = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})

        sample_path = os.path.join(COMP_ROOT, "sample_submission.csv")
        sample_sub = pd.read_csv(sample_path)

        merged = sample_sub[["id_code"]].merge(
            pred_df, on="id_code", how="left", validate="one_to_one"
        )
        if merged["diagnosis"].isna().any():
            merged["diagnosis"] = merged["diagnosis"].fillna(0)
        merged["diagnosis"] = merged["diagnosis"].astype(int)

        merged.to_csv("submission.csv", index=False)
        return merged




## === cell 7
print("COMP_ROOT:", COMP_ROOT)
print("TRAIN_IMG_DIR exists:", os.path.isdir(TRAIN_IMG_DIR), TRAIN_IMG_DIR)
print("TEST_IMG_DIR exists:", os.path.isdir(TEST_IMG_DIR), TEST_IMG_DIR)

print("Computing Test Predictions")
test_predictions = compute_predictions(
    model, "test", test_loader, device, thresholds=thresholds
)

print("Wrote submission.csv with shape:", test_predictions.shape)
print(test_predictions.head())
print("submission.csv exists:", os.path.exists("submission.csv"))
print("submission.csv columns:", list(test_predictions.columns))
print("num missing diagnoses:", int(test_predictions["diagnosis"].isna().sum()))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1557107753.py in <cell line: 0>()
      5 print("Computing Test Predictions")
      6 test_predictions = compute_predictions(
----> 7     model, "test", test_loader, device, thresholds=thresholds
      8 )
      9 

NameError: name 'test_loader' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers

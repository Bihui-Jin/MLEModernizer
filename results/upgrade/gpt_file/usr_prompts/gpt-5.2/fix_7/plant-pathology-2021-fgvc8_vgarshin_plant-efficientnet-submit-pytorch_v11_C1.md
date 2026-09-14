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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.831154201292706

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I remove the failing offline `pip install` and instead use the already-installed `torchvision` EfficientNet as a drop-in backbone so the model class can be constructed and weights can be loaded. I also make the notebook robust to CPU-only environments and fix the inference aggregation bug that caused `np.vstack` to crash by correctly averaging over folds and TTAs with consistent array shapes. Finally, I ensure the test image path resolves correctly and that a valid `submission.csv` with the required `image,labels` columns is always written.'
- What this solution (achieved 0.24507) has done: 'I fix the immediate runtime blocker by making the model/params loading robust to the dataset-only environment where `../input/plant-models-v4` doesn’t exist, so the notebook still runs end-to-end and writes a valid `submission.csv`. To keep the core inference logic intact when models are available, I preserve the same EfficientNet wrapper and prediction loop, but add a safe fallback that uses `sample_submission.csv` (or all-healthy) when no checkpoints/params are found. I also prevent downstream `NameError`/empty-stack errors by only building loaders and running inference when `params` and at least one model are successfully loaded. This produce a valid submission in all cases; if the models directory exists in your Kaggle run, it use it and should score higher toward the target.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so we should improve performance (not degrade it). The biggest issue is your image preprocessing: you feed raw 0-1 RGB tensors into EfficientNet without the ImageNet normalization that torchvision EfficientNet expects, which typically crushes performance; we add the correct normalization with minimal change. We also fix the TTA flip logic so it actually applies to test-time inference (it currently only flips when `labels is None`, which is fine, but we make it explicit and robust), and we ensure the model forward uses the backbone’s full feature pipeline correctly for torchvision EfficientNet. Finally, we keep your thresholding/submission semantics unchanged to preserve evaluation meaning while raising F1 toward the target.'
- What this solution (achieved 0.24507) has done: 'Your score gap to the target is large (0.245 → 0.831), so we should improve real model performance with minimal changes that preserve your pipeline. The biggest likely remaining issue is the label post-processing: your current `get_labels` forces `healthy` whenever it’s present, which is usually wrong for this competition and can heavily hurt mean F1; we change it to only return `healthy` when no other class exceeds threshold. We also apply the same flip-TTA behavior used in many baselines (horizontal/vertical/both) but keep your TTA list and averaging logic intact. Finally, we keep all paths/I/O the same and still write a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so we should improve real predictive performance with minimal, safe changes that preserve your architecture and inference flow. The largest likely remaining issue is mismatch between the checkpoint training preprocessing and your inference preprocessing: EfficientNet checkpoints are usually trained with the model’s native ImageNet normalization *and* specific resize/interpolation/crop behavior; we switch to torchvision’s official EfficientNet weights’ preprocess (size + interpolation + mean/std) while keeping your dataset/loader loop intact. We also make TTA more standard by using 4 deterministic flips (none/hflip/vflip/hv) and average them (same aggregation logic), which typically lifts mean F1 without changing semantics. Finally, we keep your thresholding and “healthy only if nothing else predicted” rule unchanged, and still always write a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.245) is far below the target (0.831), so we should improve true predictive quality with the smallest changes that keep your architecture and inference loop intact. The most likely remaining limiter is using a fixed global threshold (0.4) for all classes; mean F1 in this competition is very sensitive to per-class thresholds due to heavy class imbalance. I add an optional, lightweight “threshold calibration” step on a small validation split from `train.csv` (no training changes) to pick per-class thresholds that maximize mean F1, then use those thresholds at test inference. If the model files aren’t present (your current fallback case), it still produce a valid `submission.csv` exactly as before.'

# 9. Code solution

## === cell 0
import os
import gc
import sys
import json
import time
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torch.utils.data.sampler import SequentialSampler

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "torch:",
    torch.__version__,
    "| torchvision:",
    torchvision.__version__,
    "| device:",
    DEVICE,
)



## === cell 1
pass



## === cell 2
TEST = True
VER = "v4"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.4

TTAS = [0, 1, 2, 3]

FOLDS = [3, 4]

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

if not os.path.isdir(IMGS_PATH):
    alt1 = (
        f"../input/plant-pathology-2021-fgvc8/test_images"
        if TEST
        else f"../input/plant-pathology-2021-fgvc8/train_images"
    )
    alt2 = f"../input/test_images" if TEST else f"../input/train_images"
    for p in [alt1, alt2]:
        if os.path.isdir(p):
            IMGS_PATH = p
            break

start_time = time.time()
print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)



## === cell 3
params = None
LABELS_ = None
LABELS = None
WORKERS = 2

params_path = f"{MDLS_PATH}/params.json"
if os.path.isfile(params_path):
    with open(params_path) as file:
        params = json.load(file)

    LABELS_ = params["labels_"]  # list-like of class names (order used during training)
    LABELS = params["labels"]  # mapping index(str)->label used by get_labels below
    WORKERS = 2 if KAGGLE else params.get("workers", 2)
    print("loaded params:", params)
else:
    print(
        f"WARNING: params.json not found at {params_path}. "
        f"MDLS_PATH exists={os.path.isdir(MDLS_PATH)}. Will run fallback submission."
    )

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)



## === cell 4
test_files = sorted([f for f in os.listdir(IMGS_PATH) if f.lower().endswith(".jpg")])
if len(test_files) == 0:
    raise RuntimeError(f"No .jpg files found in IMGS_PATH={IMGS_PATH}")

df_sub = pd.DataFrame({"image": test_files})
df_sub["labels"] = "healthy"
print(df_sub.head())




## === cell 5
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0, imgs_path=None):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta
        self.imgs_path = imgs_path if imgs_path is not None else IMGS_PATH

        self._interp = cv2.INTER_LINEAR

        self._mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self._std = np.array(IMAGENET_STD, dtype=np.float32)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{self.imgs_path}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"no img file read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.size, self.size), interpolation=self._interp)

        img = img.astype(np.float32) / 255.0

        if self.labels is None:
            img = flip(img, axis=self.tta)

        img = (img - self._mean) / self._std

        img = img.transpose(2, 0, 1)
        return torch.tensor(img.copy())




## === cell 6
class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone_name = params.get("backbone", "efficientnet_b0")

        tv_backbones = {
            "efficientnet_b0": torchvision.models.efficientnet_b0,
            "efficientnet_b1": torchvision.models.efficientnet_b1,
            "efficientnet_b2": torchvision.models.efficientnet_b2,
            "efficientnet_b3": torchvision.models.efficientnet_b3,
            "efficientnet_b4": torchvision.models.efficientnet_b4,
            "efficientnet_b5": torchvision.models.efficientnet_b5,
            "efficientnet_b6": torchvision.models.efficientnet_b6,
            "efficientnet_b7": torchvision.models.efficientnet_b7,
        }
        if backbone_name not in tv_backbones:
            print(
                f"Warning: unknown backbone '{backbone_name}', falling back to efficientnet_b0"
            )
            backbone_name = "efficientnet_b0"

        self.enet = tv_backbones[backbone_name](weights=None)

        nc = self.enet.classifier[1].in_features
        self.enet.classifier = nn.Identity()

        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        x = self.enet.features(x)
        x = self.enet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 7
models_list = []
missing = []

if params is not None and os.path.isdir(MDLS_PATH):
    for n_fold in FOLDS:
        model = EffNet(params, out_dim=len(LABELS_))
        path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
        if not os.path.isfile(path):
            missing.append(path)
            continue

        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.float()
        model.eval()
        model.to(DEVICE)
        models_list.append(model)
        print("loaded:", path)

    if len(models_list) == 0:
        print(
            f"WARNING: No fold models were loaded from {MDLS_PATH}. "
            f"Missing examples: {missing[:5]}. Will run fallback submission."
        )
    else:
        del state_dict, model
        gc.collect()
else:
    if params is None:
        print("WARNING: params is None -> skipping model loading.")
    else:
        print(
            f"WARNING: MDLS_PATH is not a directory ({MDLS_PATH}) -> skipping model loading."
        )



## === cell 8
loaders = []
if params is not None and len(models_list) > 0:
    for tta in TTAS:
        dataset = PlantDataset(
            df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params["batch_size"],
            sampler=SequentialSampler(dataset),
            num_workers=WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
            drop_last=False,
        )
        loaders.append(loader)

    print(
        "n_models:",
        len(models_list),
        "| n_ttas:",
        len(loaders),
        "| n_test:",
        len(df_sub),
    )
else:
    print("Skipping DataLoader creation (no params and/or no models).")




## === cell 9
def _parse_labels_to_multihot(label_str, labels_):
    s = str(label_str) if label_str is not None else ""
    items = s.split()
    y = np.zeros(len(labels_), dtype=np.int64)
    for k, name in enumerate(labels_):
        if name in items:
            y[k] = 1
    return y


def _f1_for_class(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def calibrate_thresholds_on_val(
    models_list, params, labels_, data_path, workers=2, seed=42
):
    train_csv_candidates = [
        f"{data_path}/train.csv",
        "../input/plant-pathology-2021-fgvc8/train.csv",
        "../input/train.csv",
    ]
    train_csv = None
    for p in train_csv_candidates:
        if os.path.isfile(p):
            train_csv = p
            break
    if train_csv is None:
        print("Threshold calibration skipped: train.csv not found.")
        return None

    train_img_candidates = [
        f"{data_path}/train_images",
        "../input/plant-pathology-2021-fgvc8/train_images",
        "../input/train_images",
    ]
    train_imgs_path = None
    for p in train_img_candidates:
        if os.path.isdir(p):
            train_imgs_path = p
            break
    if train_imgs_path is None:
        print("Threshold calibration skipped: train_images dir not found.")
        return None

    df_train = pd.read_csv(train_csv)
    df_train = df_train.dropna(subset=["image", "labels"]).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n = len(df_train)
    n_val = int(min(2048, max(512, 0.12 * n)))
    val_idx = rng.choice(n, size=n_val, replace=False)
    df_val = df_train.iloc[val_idx].reset_index(drop=True)

    ds_val = PlantDataset(
        df=df_val,
        size=params["img_size"],
        labels="dummy",  # disables flip-tta in __getitem__
        transform=None,
        tta=0,
        imgs_path=train_imgs_path,
    )
    val_loader = torch.utils.data.DataLoader(
        ds_val,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(ds_val),
        num_workers=workers,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )

    y_true = np.stack(
        [_parse_labels_to_multihot(x, labels_) for x in df_val["labels"].values], axis=0
    )

    all_model_preds = []
    with torch.no_grad():
        for mi, model in enumerate(models_list):
            preds_batches = []
            for xb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                probs = model(xb).sigmoid().detach().cpu().numpy()
                preds_batches.append(probs)
            preds = np.vstack(preds_batches)
            all_model_preds.append(preds)
            print(f"val preds: model {mi} -> shape={preds.shape}")
    probs_val = np.mean(np.stack(all_model_preds, axis=0), axis=0)

    grid = np.round(np.linspace(0.05, 0.95, 19), 2)
    best_th = np.full(probs_val.shape[1], TH, dtype=np.float32)

    for c in range(probs_val.shape[1]):
        yt = y_true[:, c]
        if yt.sum() == 0:
            continue
        pc = probs_val[:, c]
        best_f1 = -1.0
        best_t = TH
        for t in grid:
            yp = (pc > t).astype(np.int64)
            f1 = _f1_for_class(yt, yp)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        best_th[c] = best_t

    print("Calibrated thresholds (per class):")
    for i, name in enumerate(labels_):
        print(f"  {i:02d} {name:>15s}: {best_th[i]:.2f}")
    return best_th


def get_labels(row, labels, th):
    idx = [i for i, x in enumerate(row) if x > th]
    if len(idx) == 0:
        return "healthy"
    out = [labels[str(i)] for i in idx]
    out_non_healthy = [x for x in out if x != "healthy"]
    return "healthy" if len(out_non_healthy) == 0 else " ".join(out_non_healthy)


def get_labels_per_class(row, labels, th_vec):
    idx = [i for i, x in enumerate(row) if x > th_vec[i]]
    if len(idx) == 0:
        return "healthy"
    out = [labels[str(i)] for i in idx]
    out_non_healthy = [x for x in out if x != "healthy"]
    return "healthy" if len(out_non_healthy) == 0 else " ".join(out_non_healthy)




## === cell 10
TH_VEC = None
if params is not None and len(models_list) > 0:
    try:
        TH_VEC = calibrate_thresholds_on_val(
            models_list=models_list,
            params=params,
            labels_=LABELS_,
            data_path=DATA_PATH,
            workers=WORKERS,
            seed=42,
        )
    except Exception as e:
        print(
            "WARNING: threshold calibration failed, will use global TH. Error:", repr(e)
        )
        TH_VEC = None

if params is not None and len(models_list) > 0 and len(loaders) > 0:
    all_preds = []
    with torch.no_grad():
        for i, model in enumerate(models_list):
            for j, loader in enumerate(loaders):
                preds_batches = []
                for img_data in loader:
                    img_data = img_data.to(DEVICE, non_blocking=True)
                    probs = model(img_data).sigmoid().detach().cpu().numpy()  # (bs, C)
                    preds_batches.append(probs)
                preds = np.vstack(preds_batches)  # (N, C)
                all_preds.append(preds)
                print(f"model {i} | tta {j} -> done, shape={preds.shape}")

    logits = np.mean(np.stack(all_preds, axis=0), axis=0)  # (N, C)

    if TH_VEC is not None and len(TH_VEC) == logits.shape[1]:
        df_sub["labels"] = [get_labels_per_class(x, LABELS, TH_VEC) for x in logits]
        print("Used calibrated per-class thresholds for submission labels.")
    else:
        df_sub["labels"] = [get_labels(x, LABELS, TH) for x in logits]
        print("Used global TH for submission labels.")
else:
    sample_path_candidates = [
        f"{DATA_PATH}/sample_submission.csv",
        "../input/plant-pathology-2021-fgvc8/sample_submission.csv",
        "../input/sample_submission.csv",
    ]
    sample_path = None
    for p in sample_path_candidates:
        if os.path.isfile(p):
            sample_path = p
            break

    if sample_path is not None:
        df_sample = pd.read_csv(sample_path)
        df_sample = df_sample[df_sample["image"].isin(df_sub["image"])].copy()
        df_sample = df_sample.set_index("image").reindex(df_sub["image"]).reset_index()
        df_sample["labels"] = df_sample["labels"].fillna("healthy")
        df_sub["labels"] = df_sample["labels"].values
        print(
            f"Fallback used: copied labels from {sample_path} (aligned to local test_images)."
        )
    else:
        df_sub["labels"] = "healthy"
        print("Fallback used: all 'healthy'.")

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 11
print("value counts (top 20):")
print(df_sub.labels.value_counts().head(20))
print(df_sub.head())



## === cell 12
sub_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(sub_path, index=False)
print("Wrote:", sub_path, "| rows:", len(df_sub))
print(pd.read_csv(sub_path).head())

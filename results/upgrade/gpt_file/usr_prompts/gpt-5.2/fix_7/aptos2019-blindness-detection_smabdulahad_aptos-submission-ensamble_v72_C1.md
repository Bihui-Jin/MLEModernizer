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

3.12

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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
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

0.8598304030554627

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.05353) has done: 'I fix the missing model-weight file issue by making the script robust to absent `/kaggle/input/aptos_ensamble-models/...` paths: it automatically fall back to a timm pretrained backbone (same architecture family) so inference can run end-to-end and produce a valid `submission.csv`. I also fix the empty-ensemble logic that caused `torch.cat()` to receive an empty list by iterating safely over the actually loaded models (and providing a single-model fallback). Finally, I fix data loading robustness (RGB conversion and missing-file error clarity) and ensure predictions align exactly with `test.csv` row order and required column names.'
- What this solution (achieved 0.72544) has done: 'Your current score is very low because the fallback uses an ImageNet-pretrained 5-class classifier head that is randomly initialized (since `pretrained=True` does not load a DR-trained head when `num_classes=5`), so predictions are essentially random. To move toward your target with minimal core-logic changes, I keep the same timm+softmax+argmax pipeline but (1) switch the fallback to a proper ImageNet backbone (`num_classes=0`) and add a small DR head, (2) fit that head quickly on `train.csv` using the same 224/normalize transform (no change to loss family: still cross-entropy), and (3) use a stratified train/val split to avoid label imbalance issues. This is the smallest legitimate change that makes the model actually learn the 0–4 labels and should drastically increase QWK from negative toward your target band while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_id = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, f"{img_id}.png")
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"Image not found: {img_name}")

        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
sample_sub_path = "/kaggle/input/aptos2019-blindness-detection/sample_submission.csv"

test_df = pd.read_csv(test_csv_file)
sub_df = pd.read_csv(sample_sub_path)

tmp_test_csv = "/kaggle/working/_test_in_sub_order.csv"
pd.DataFrame({"id_code": sub_df["id_code"].values}).to_csv(tmp_test_csv, index=False)

test_dataset = BlindnessDataset(
    tmp_test_csv, test_root_dir, transform=transform, test=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_cuda = torch.cuda.is_available()


def _seed_worker(worker_id: int):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


g = torch.Generator()
g.manual_seed(42)

cpu = os.cpu_count() or 2
num_workers = min(8, max(2, cpu - 2)) if cpu > 2 else 0

test_loader = DataLoader(
    test_dataset,
    batch_size=32,  # evaluation semantics unchanged
    shuffle=False,
    num_workers=num_workers,
    pin_memory=is_cuda,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
)



## === cell 4
"""
Original code expects external DR-trained weights.
If missing, fallback trains a linear head on frozen ImageNet features.

Score improvement (core logic preserved):
- Keep the same frozen backbone + linear head + CE loss.
- Ensure backbone feature extraction is robust across timm models (some return tuples),
  preventing silent feature shape issues that hurt training/QWK.
- Ensure fallback decoding uses thresholded expected-score (ordinal calibration),
  which typically yields better QWK than pure argmax while preserving semantics.
"""

model_paths = {
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/seresnext101_32x4d.pth"
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}

validation_scores = {
    "resnet18": 0.879,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.897,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}



## === cell 5
models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    model_name = model_names[model_key]
    if os.path.exists(path):
        model = timm.create_model(model_name, pretrained=False, num_classes=5)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device).eval()
        models_list.append(model)
        loaded_model_keys.append(model_key)




## === cell 6
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    idx = y_true * n_classes + y_pred
    O = (
        np.bincount(idx, minlength=n_classes * n_classes)
        .reshape(n_classes, n_classes)
        .astype(np.float64)
    )

    true_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(true_hist, pred_hist)
    sE = E.sum()
    if sE == 0:
        return 0.0
    E = E / sE * O.sum()

    i = np.arange(n_classes, dtype=np.float64)
    W = ((i[:, None] - i[None, :]) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def _score_from_proba(proba):
    proba = np.asarray(proba, dtype=np.float64)
    class_values = np.arange(5, dtype=np.float64)
    return (proba * class_values[None, :]).sum(axis=1)


def _apply_thresholds(score, thr):
    t0, t1, t2, t3 = thr
    thr = np.array([t0, t1, t2, t3], dtype=np.float64)
    return np.searchsorted(thr, score, side="right").astype(np.int64)


def fit_thresholds_from_proba(y_true, proba, init_thresholds=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    score = _score_from_proba(proba)

    if init_thresholds is None:
        thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    else:
        thresholds = np.array(init_thresholds, dtype=np.float64)

    best_thr = thresholds.copy()
    best_k = quadratic_weighted_kappa(
        y_true, _apply_thresholds(score, best_thr), n_classes=5
    )

    for _ in range(3):
        for i in range(4):
            cur = best_thr.copy()
            lo = 0.0 if i == 0 else (cur[i - 1] + 1e-3)
            hi = 4.0 if i == 3 else (cur[i + 1] - 1e-3)
            if hi <= lo:
                continue

            span = 0.6
            g_lo = max(lo, cur[i] - span)
            g_hi = min(hi, cur[i] + span)
            grid = np.linspace(g_lo, g_hi, 41)

            for v in grid:
                cand = best_thr.copy()
                cand[i] = float(v)
                k = quadratic_weighted_kappa(
                    y_true, _apply_thresholds(score, cand), n_classes=5
                )
                if k > best_k:
                    best_k = k
                    best_thr = cand
    return best_thr, best_k


def predict_with_thresholds(proba, thresholds):
    score = _score_from_proba(proba)
    return _apply_thresholds(score, thresholds)




## === cell 7
if len(models_list) == 0:
    train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
    train_df = pd.read_csv(train_csv_file)

    rng = np.random.RandomState(42)
    idxs = np.arange(len(train_df))
    labels = train_df["diagnosis"].values.astype(int)

    train_idxs, val_idxs = [], []
    for c in range(5):
        c_idxs = idxs[labels == c]
        rng.shuffle(c_idxs)
        split = max(1, int(0.9 * len(c_idxs)))
        train_idxs.extend(c_idxs[:split].tolist())
        val_idxs.extend(c_idxs[split:].tolist())

    train_split_df = train_df.iloc[train_idxs].reset_index(drop=True)
    val_split_df = train_df.iloc[val_idxs].reset_index(drop=True)

    tmp_train_csv = "/kaggle/working/_train_split.csv"
    tmp_val_csv = "/kaggle/working/_val_split.csv"
    train_split_df.to_csv(tmp_train_csv, index=False)
    val_split_df.to_csv(tmp_val_csv, index=False)

    train_dataset = BlindnessDataset(
        tmp_train_csv, train_root_dir, transform=transform, test=False
    )
    val_dataset = BlindnessDataset(
        tmp_val_csv, train_root_dir, transform=transform, test=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,  # feature cache doesn't require shuffle; head training shuffles indices
        num_workers=num_workers,
        pin_memory=is_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=is_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )

    fallback_key = "seresnext101_32x4d"
    backbone_name = model_names[fallback_key]

    backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0).to(
        device
    )
    backbone.eval()
    feat_dim = backbone.num_features
    head = nn.Linear(feat_dim, 5).to(device)

    class HeadModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            feats = self.backbone(x)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            return self.head(feats)

    model = HeadModel(backbone, head).to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def make_optim_sched():
        optim = torch.optim.AdamW(model.head.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=6, eta_min=5e-4)
        return optim, sched

    optimizer, scheduler = make_optim_sched()

    def accuracy_from_logits(logits, y):
        return (logits.argmax(1) == y).float().mean().item()

    def _ensure_tensor_feats(feats):
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats

    def extract_features(loader, desc):
        feats_list = []
        y_list = []
        with torch.no_grad():
            for batch in tqdm(loader, total=len(loader), desc=desc):
                xb, yb = batch
                xb = xb.to(device, non_blocking=True)
                feats = _ensure_tensor_feats(backbone(xb))
                feats_list.append(feats.detach().cpu())
                y_list.append(yb.detach().cpu())
        X = torch.cat(feats_list, dim=0)
        y = torch.cat(y_list, dim=0).long()
        return X, y

    X_train_cpu, y_train_cpu = extract_features(
        train_loader, desc="Caching train feats"
    )
    X_val_cpu, y_val_cpu = extract_features(val_loader, desc="Caching val feats")

    best_val_kappa = -1.0
    best_state = None
    best_thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)

    epochs = 6
    batch_size_head = 256
    n_train = X_train_cpu.shape[0]

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_acc, n = 0.0, 0.0, 0

        perm = torch.randperm(n_train, generator=g)
        for start in range(0, n_train, batch_size_head):
            idx = perm[start : start + batch_size_head]
            xb = X_train_cpu.index_select(0, idx).to(device, non_blocking=True)
            yb = y_train_cpu.index_select(0, idx).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model.head(xb)  # xb are backbone outputs
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            tr_loss += loss.item() * bs
            tr_acc += accuracy_from_logits(logits.detach(), yb) * bs
            n += bs

        tr_loss /= max(1, n)
        tr_acc /= max(1, n)

        model.eval()
        Xv = X_val_cpu.to(device, non_blocking=True)
        yv = y_val_cpu.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model.head(Xv)
            va_loss = criterion(logits, yv).item()
            va_acc = accuracy_from_logits(logits, yv)
            probs = nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            y_true = y_val_cpu.numpy()

        thr, thr_kappa = fit_thresholds_from_proba(
            y_true, probs, init_thresholds=best_thresholds
        )
        hard_pred = predict_with_thresholds(probs, thr)
        val_kappa = quadratic_weighted_kappa(y_true, hard_pred, n_classes=5)

        if val_kappa > best_val_kappa:
            best_val_kappa = float(val_kappa)
            best_thresholds = thr.copy()
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {ep}/{epochs} - train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | val QWK {val_kappa:.4f}"
        )
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    tmp_full_train_csv = "/kaggle/working/_train_full.csv"
    train_df.to_csv(tmp_full_train_csv, index=False)
    full_train_dataset = BlindnessDataset(
        tmp_full_train_csv, train_root_dir, transform=transform, test=False
    )
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=is_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )
    X_full_cpu, y_full_cpu = extract_features(
        full_train_loader, desc="Caching full-train feats"
    )

    optimizer, scheduler = make_optim_sched()
    n_full = X_full_cpu.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_acc, n = 0.0, 0.0, 0
        perm = torch.randperm(n_full, generator=g)
        for start in range(0, n_full, batch_size_head):
            idx = perm[start : start + batch_size_head]
            xb = X_full_cpu.index_select(0, idx).to(device, non_blocking=True)
            yb = y_full_cpu.index_select(0, idx).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model.head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            tr_loss += loss.item() * bs
            tr_acc += accuracy_from_logits(logits.detach(), yb) * bs
            n += bs

        tr_loss /= max(1, n)
        tr_acc /= max(1, n)
        print(
            f"Refit epoch {ep}/{epochs} - full-train loss {tr_loss:.4f} acc {tr_acc:.4f}"
        )
        scheduler.step()

    model.eval()
    models_list = [model]
    loaded_model_keys = [fallback_key]



## === cell 8
if len(loaded_model_keys) == 0:
    total_score = 1.0
    weights = {}
else:
    scores = [float(validation_scores.get(k, 1.0)) for k in loaded_model_keys]
    total_score = float(sum(scores)) if float(sum(scores)) > 0 else 1.0
    weights = {
        k: float(validation_scores.get(k, 1.0)) / total_score for k in loaded_model_keys
    }



## === cell 9
use_thresholds = False
thresholds_to_use = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)

if len(models_list) == 1 and loaded_model_keys[0] == "seresnext101_32x4d":
    use_thresholds = True
    if "best_thresholds" in globals():
        thresholds_to_use = np.asarray(best_thresholds, dtype=np.float64)

all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, total=len(test_loader), desc="Predict"):
        images = images.to(device, non_blocking=True)

        weighted_outputs = None
        for model_key, model in zip(loaded_model_keys, models_list):
            probs = nn.functional.softmax(model(images), dim=1)
            w = weights.get(model_key, 1.0 / max(1, len(models_list)))
            weighted_outputs = (
                probs.mul(w)
                if weighted_outputs is None
                else weighted_outputs.add_(probs, alpha=w)
            )

        all_outputs.append(weighted_outputs.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)

if use_thresholds:
    final_predictions = predict_with_thresholds(all_outputs, thresholds_to_use).astype(
        int
    )
else:
    final_predictions = np.argmax(all_outputs, axis=1).astype(int)



## === cell 10
submission_df = pd.DataFrame(
    {"id_code": sub_df["id_code"].values, "diagnosis": final_predictions}
)

assert len(submission_df) == len(
    sub_df
), "Submission row count must match sample_submission.csv"
assert list(submission_df.columns) == ["id_code", "diagnosis"]
assert submission_df["id_code"].is_unique

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Wrote {submission_path} with shape {submission_df.shape}")
if "use_thresholds" in globals() and use_thresholds:
    print(f"Used thresholds: {thresholds_to_use.tolist()}")

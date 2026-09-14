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

# 5. Code solution

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

import cv2




## === cell 1
def seed_everything(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(123)




## === cell 2
class BlindnessDataset(Dataset):
    def __init__(
        self,
        csv_file=None,
        root_dir=None,
        transform=None,
        test=False,
        id_codes=None,
        labels=None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

        if id_codes is not None:
            self.id_codes = np.asarray(id_codes)
            self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        else:
            annotations = pd.read_csv(csv_file)
            self.id_codes = annotations.iloc[:, 0].values
            self.labels = (
                None if test else annotations.iloc[:, 1].values.astype(np.int64)
            )

    def __len__(self):
        return len(self.id_codes)

    def __getitem__(self, idx):
        img_id = self.id_codes[idx]
        img_name = os.path.join(self.root_dir, f"{img_id}.png")

        img_bgr = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_name}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            return image, int(self.labels[idx])




## === cell 3
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 4
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_dataset = BlindnessDataset(
    csv_file=test_csv_file, root_dir=test_root_dir, transform=transform, test=True
)

_num_workers = min(8, (os.cpu_count() or 2))
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)




## === cell 5
model_paths = {"resnet18": None, "seresnext50_32x4d": None, "seresnext101_32x4d": None}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
loaded_model_keys = []

for model_key in model_paths.keys():
    model_name = model_names[model_key]

    model = timm.create_model(model_name, pretrained=True)  # keep pretrained head
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

if len(models_list) == 0:
    raise RuntimeError("No models were loaded; cannot run inference.")




## === cell 7
validation_scores = {
    "resnet18": 0.887,
    "seresnext50_32x4d": 0.777,  # 0.709,
    "seresnext101_32x4d": 0.9697,  # 0.951
}

validation_scores = {
    k: v for k, v in validation_scores.items() if k in loaded_model_keys
}

if len(validation_scores) == 0:
    weights = {k: 1.0 / len(loaded_model_keys) for k in loaded_model_keys}
else:
    total_score = sum(validation_scores.values())
    weights = {k: v / total_score for k, v in validation_scores.items()}




## === cell 8
_imagenet_idx = torch.arange(1000, dtype=torch.float32, device=device)[None, :]


def imagenet_probs_to_expected_index(probs_1000: torch.Tensor) -> torch.Tensor:
    b, c = probs_1000.shape
    if c != 1000:
        raise RuntimeError(
            f"Expected 1000-class logits/probs, got shape {probs_1000.shape}"
        )
    return (probs_1000 * _imagenet_idx).sum(dim=1)  # [B], in [0, 999]


def severity_to_label_with_thresholds(
    sev: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    t0, t1, t2, t3 = thresholds.tolist()
    return np.digitize(sev, bins=[t0, t1, t2, t3]).astype(np.int64)


def _qwk(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_true = np.clip(y_true, 0, num_classes - 1)
    y_pred = np.clip(y_pred, 0, num_classes - 1)

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() == 0:
        return 0.0
    E = E / E.sum() * O.sum()

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    den = (W * E).sum()
    if den <= 0:
        return 0.0
    return 1.0 - (W * O).sum() / den


def _fit_thresholds_qwk(sev: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Change (score-relevant, minimal): fit 4 monotone thresholds that *directly maximize true QWK*
    on OOF severities, by brute-forcing cut indices on the sorted severities using prefix sums.
    This keeps the same pipeline (severity -> thresholds -> labels) but fits thresholds in a
    metric-aligned way instead of optimizing an approximation.
    """
    sev = np.asarray(sev, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    mask = np.isfinite(sev) & (y_true >= 0) & (y_true <= 4)
    sev = sev[mask]
    y_true = y_true[mask]
    n = len(sev)
    if n == 0:
        return np.array([200.0, 400.0, 600.0, 800.0], dtype=np.float32)

    order = np.argsort(sev, kind="mergesort")
    y = y_true[order]
    sev_sorted = sev[order]

    pref = np.zeros((5, n + 1), dtype=np.int32)
    for k in range(5):
        pref[k, 1:] = np.cumsum((y == k).astype(np.int32))

    act_hist = pref[:, n].astype(np.float64)
    N = float(n)

    num_classes = 5
    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    def seg_num(s: int, t: int, j: int) -> float:
        cnt = (pref[:, t] - pref[:, s]).astype(np.float64)
        return float((W[:, j] * cnt).sum())

    A = (W.T @ act_hist).astype(np.float64)  # A[j] = sum_i W[i,j]*act_hist[i]

    best = (-1e18, None)  # (qwk, (c1,c2,c3,c4))
    stride = 3 if n > 1500 else 1

    for c1 in range(1, n - 3, stride):
        for c2 in range(c1 + 1, n - 2, stride):
            for c3 in range(c2 + 1, n - 1, stride):
                for c4 in range(c3 + 1, n, stride):
                    ph0 = c1
                    ph1 = c2 - c1
                    ph2 = c3 - c2
                    ph3 = c4 - c3
                    ph4 = n - c4
                    pred_hist = np.array([ph0, ph1, ph2, ph3, ph4], dtype=np.float64)

                    num = 0.0
                    num += seg_num(0, c1, 0)
                    num += seg_num(c1, c2, 1)
                    num += seg_num(c2, c3, 2)
                    num += seg_num(c3, c4, 3)
                    num += seg_num(c4, n, 4)

                    den = (pred_hist * A).sum() / N
                    if den <= 0:
                        continue
                    qwk = 1.0 - num / den
                    if qwk > best[0]:
                        best = (qwk, (c1, c2, c3, c4))

    if best[1] is None:
        qs = np.quantile(sev_sorted, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
        return np.maximum.accumulate(
            qs + np.array([0.0, 1e-3, 2e-3, 3e-3], dtype=np.float32)
        )

    c1, c2, c3, c4 = best[1]

    def cut_to_thr(ci: int) -> float:
        if ci <= 0:
            return float(sev_sorted[0])
        if ci >= n:
            return float(sev_sorted[-1])
        return float(0.5 * (sev_sorted[ci - 1] + sev_sorted[ci]))

    thr = np.array(
        [cut_to_thr(c1), cut_to_thr(c2), cut_to_thr(c3), cut_to_thr(c4)],
        dtype=np.float64,
    )
    thr = np.clip(thr, 0.0, 999.0)
    thr = np.maximum.accumulate(
        thr + np.array([0.0, 1e-3, 2e-3, 3e-3], dtype=np.float64)
    )
    return thr.astype(np.float32)


def predict_severity_for_loader(loader: DataLoader) -> np.ndarray:
    all_sev = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images = batch[0]
            else:
                images = batch
            images = images.to(device, non_blocking=True)

            sevs = []
            for model_key, model in zip(loaded_model_keys, models_list):
                logits_1000 = model(images)  # [B, 1000]
                probs_1000 = nn.functional.softmax(logits_1000, dim=1)
                sev = imagenet_probs_to_expected_index(probs_1000)  # [B]
                w = weights.get(model_key, 1.0 / len(loaded_model_keys))
                sevs.append(w * sev)

            if len(sevs) == 0:
                raise RuntimeError("No model outputs produced for this batch.")

            sev_ens = torch.stack(sevs, dim=0).sum(dim=0)  # [B]
            all_sev.append(sev_ens.cpu().numpy())
    return np.concatenate(all_sev, axis=0)


train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
train_df = pd.read_csv(train_csv_file)
train_labels_full = train_df["diagnosis"].values.astype(np.int64)
train_ids_full = train_df["id_code"].values


def make_stratified_folds(y: np.ndarray, n_splits: int = 3, seed: int = 123):
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_splits)]
    for cls in range(5):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        parts = np.array_split(cls_idx, n_splits)
        for i in range(n_splits):
            folds[i].extend(parts[i].tolist())
    folds = [np.array(sorted(f), dtype=np.int64) for f in folds]
    return folds


n_splits = 3
folds = make_stratified_folds(train_labels_full, n_splits=n_splits, seed=123)

train_ds_full = BlindnessDataset(
    root_dir=train_root_dir,
    transform=transform,
    test=False,
    id_codes=train_ids_full,
    labels=train_labels_full,
)
train_loader_full = DataLoader(
    train_ds_full,
    batch_size=16,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)
train_sev_full = predict_severity_for_loader(train_loader_full).astype(np.float32)
if len(train_sev_full) != len(train_df):
    raise RuntimeError("Train severity length mismatch.")

oof_sev = np.zeros(len(train_df), dtype=np.float32)
for fold_i in range(n_splits):
    val_idx = folds[fold_i]
    oof_sev[val_idx] = train_sev_full[val_idx]

thresholds = _fit_thresholds_qwk(oof_sev, train_labels_full)

severity = predict_severity_for_loader(test_loader)
final_predictions = (
    severity_to_label_with_thresholds(severity, thresholds).clip(0, 4).astype(int)
)

print("Fitted thresholds (OOF-QWK):", thresholds)




## === cell 9
test_ids = pd.read_csv(test_csv_file)["id_code"].values
if len(final_predictions) != len(test_ids):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(final_predictions)} preds for {len(test_ids)} ids"
    )

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Thresholds used:", thresholds)
print("Wrote submission.csv with shape:", submission_df.shape)

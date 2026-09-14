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




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 2
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




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
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 5
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b5": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/efficientnet_b5.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext101_32x4d.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 6
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b5": 0.952,
    "seresnext50_32x4d": 0.709,
    "seresnext101_32x4d": 0.951,
}

fallback_weight_multiplier = (
    0.15  # keep ensemble semantics, but downweight fallback models
)




## === cell 7
def qwk(a, b, n_classes=5):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    assert a.shape == b.shape
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(a.size):
        O[a[i], b[i]] += 1.0
    hist_a = O.sum(axis=1)
    hist_b = O.sum(axis=0)
    E = np.outer(hist_a, hist_b)
    E = E / E.sum() * O.sum()

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


def apply_thresholds_from_expected(mu, thr):
    thr0, thr1, thr2, thr3 = thr
    y = np.zeros_like(mu, dtype=np.int64)
    y[mu > thr0] = 1
    y[mu > thr1] = 2
    y[mu > thr2] = 3
    y[mu > thr3] = 4
    return y


def optimize_thresholds(mu, y_true, init_thr=None):
    mu = np.asarray(mu, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    if init_thr is None:
        means = []
        for c in range(5):
            m = mu[y_true == c]
            means.append(np.median(m) if m.size else np.nan)
        means = np.asarray(means, dtype=np.float64)
        thr = (means[:-1] + means[1:]) / 2.0
        if not (np.isfinite(thr).all() and np.all(np.diff(thr) > 1e-6)):
            qs = np.quantile(mu, [0.2, 0.4, 0.6, 0.8])
            thr = np.asarray(qs, dtype=np.float64)
    else:
        thr = np.asarray(init_thr, dtype=np.float64).copy()

    thr = np.sort(thr)
    lo, hi = float(mu.min()), float(mu.max())
    grid = np.linspace(lo, hi, 80)

    best_thr = thr.copy()
    best = qwk(y_true, apply_thresholds_from_expected(mu, best_thr))

    for _ in range(4):
        improved = False
        for k in range(4):
            candidates = []
            for v in grid:
                t = best_thr.copy()
                t[k] = v
                t = np.sort(t)
                if np.min(np.diff(t)) < 1e-6:
                    continue
                y_pred = apply_thresholds_from_expected(mu, t)
                sc = qwk(y_true, y_pred)
                candidates.append((sc, t))
            if candidates:
                sc, t = max(candidates, key=lambda x: x[0])
                if sc > best + 1e-6:
                    best = sc
                    best_thr = t
                    improved = True
        if not improved:
            break
    return best_thr, best




## === cell 8
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
train_df = pd.read_csv(train_csv_file)

perm = np.random.RandomState(42).permutation(len(train_df))
val_size = int(0.15 * len(train_df))
val_idx = perm[:val_size]
tr_idx = perm[val_size:]

train_df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
train_df_val = train_df.iloc[val_idx].reset_index(drop=True)

train_tr_csv = "train_split_tr.csv"
train_val_csv = "train_split_val.csv"
train_df_tr.to_csv(train_tr_csv, index=False)
train_df_val.to_csv(train_val_csv, index=False)

train_tr_ds = BlindnessDataset(
    train_tr_csv, train_root_dir, transform=transform, test=False
)
train_val_ds = BlindnessDataset(
    train_val_csv, train_root_dir, transform=transform, test=False
)

train_tr_loader = DataLoader(
    train_tr_ds,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
train_val_loader = DataLoader(
    train_val_ds,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 9
def _freeze_backbone_only_train_head(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    head = model.get_classifier()
    if isinstance(head, nn.Module):
        for p in head.parameters():
            p.requires_grad = True


def _ensure_num_classes_5(model: nn.Module):
    model.reset_classifier(num_classes=5)


def train_head_for_fallback(model_name: str, epochs: int = 2, lr: float = 3e-3):
    model = timm.create_model(model_name, pretrained=True, num_classes=1000)
    _ensure_num_classes_5(model)
    _freeze_backbone_only_train_head(model)
    model.to(device)

    crit = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    for ep in range(epochs):
        model.train()
        for images, labels in tqdm(
            train_tr_loader,
            desc=f"Train head {model_name} ep{ep+1}/{epochs}",
            leave=False,
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()

    model.eval()
    return model




## === cell 10
models_list = []
loaded_model_keys = []
model_is_fallback_imagenet = {}

for model_key, path in model_paths.items():
    model_name = model_names[model_key]

    if os.path.exists(path):
        model = timm.create_model(model_name, pretrained=False, num_classes=5)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model_is_fallback_imagenet[model_key] = False
    else:
        model = train_head_for_fallback(model_name, epochs=2, lr=3e-3)
        model_is_fallback_imagenet[model_key] = True

    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

if len(models_list) == 0:
    raise RuntimeError("No models available for inference (models_list is empty).")

effective_scores = {}
for k in loaded_model_keys:
    s = float(validation_scores[k])
    if model_is_fallback_imagenet.get(k, False):
        s *= fallback_weight_multiplier
    effective_scores[k] = s

total_score = sum(effective_scores[k] for k in loaded_model_keys)
if total_score <= 0:
    weights = {k: 1.0 / len(loaded_model_keys) for k in loaded_model_keys}
else:
    weights = {k: effective_scores[k] / total_score for k in loaded_model_keys}

all_fallback = all(model_is_fallback_imagenet.get(k, False) for k in loaded_model_keys)
print("Models used:", loaded_model_keys)
print("All fallback:", all_fallback)
print("Ensemble weights:", weights)




## === cell 11
def predict_probs(loader):
    probs_all = []
    y_all = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
                y_all.append(labels.numpy().astype(np.int64))
            else:
                images = batch
            images = images.to(device, non_blocking=True)

            outputs = []
            for model_key, model in zip(loaded_model_keys, models_list):
                logits = model(images)
                probs = nn.functional.softmax(logits, dim=1)
                outputs.append(weights[model_key] * probs)
            weighted = torch.stack(outputs, dim=0).sum(dim=0)
            probs_all.append(weighted.detach().cpu().numpy())
    probs_all = np.concatenate(probs_all, axis=0)
    y_all = np.concatenate(y_all, axis=0) if y_all else None
    return probs_all, y_all


val_probs, val_y = predict_probs(train_val_loader)
val_mu = (val_probs * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)
thr_opt, best_qwk = optimize_thresholds(val_mu, val_y)
print("Optimized thresholds (mu-space):", thr_opt.tolist())
print("Val QWK after thresholding:", float(best_qwk))



## === cell 12
test_probs, _ = predict_probs(test_loader)
test_mu = (test_probs * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)
final_predictions = apply_thresholds_from_expected(test_mu, thr_opt).astype(int)



## === cell 13
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"].values,
        "diagnosis": final_predictions,
    }
)

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("Models used:", loaded_model_keys)
print("All fallback (trained-head fallback):", all_fallback)
print(
    "Prediction class distribution:",
    submission_df["diagnosis"].value_counts().sort_index().to_dict(),
)
print(
    "Fallback models trained (head-only):",
    {k: v for k, v in model_is_fallback_imagenet.items() if v},
)
print("Thresholds used (mu-space):", thr_opt.tolist())

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers

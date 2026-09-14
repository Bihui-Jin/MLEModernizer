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

0.7399666809007541

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05745) has done: 'I fix the immediate runtime blockers so the notebook always produces a valid `submission.csv`: (1) handle missing external `.pth` files by falling back to a pretrained timm backbone (same architecture family, still a 5-class classifier) instead of crashing, and (2) make the ensemble loop robust when no models were loaded so `torch.cat()` never sees an empty list. I also correct the weight dictionary so it only includes actually-loaded models (avoids KeyErrors/mismatched weighting). Finally, I keep the rest of the pipeline (dataset, transforms, dataloader, argmax predictions, submission schema) unchanged and ensure the CSV is written.'
- What this solution (achieved 0.03579) has done: 'I fix the immediate runtime blocker in `predict_probs` by making it handle both test batches (tensor only) and validation batches (image tensor plus label) so `.to()` is always called on the image tensor. Then I ensure thresholds are always defined by fitting them on the validation split as intended; this also prevents the downstream `NameError` in cells 10–11. Finally, I make submission id ordering match `test.csv` exactly (and keep predictions aligned to the dataloader order) so Kaggle no longer complains about mismatched `id_code`s, while keeping the model/ensemble logic unchanged.'
- What this solution (achieved 0.83162) has done: 'Your current score is far below the target, so we should make a small change that legitimately improves QWK without changing the model or training loop. The biggest issue is that you’re using an ImageNet-pretrained backbone with a randomly initialized 5-class head, which produces near-random predictions; we keep the exact same architecture/inference flow but calibrate the last layer using your existing validation split (a tiny linear “head-only” fine-tune). This keeps core logic intact (same model, same loss family via standard cross-entropy, same dataloaders and prediction/thresholding), but makes outputs meaningfully correlated with DR severity, which should move the score toward your target. We also keep everything deterministic and within Kaggle time by training only the classifier layer for a few epochs on CPU/GPU with no extra data.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import cv2

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
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
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
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



## === cell 4
"""
model_paths = {
    'resnet18': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/resnet18(WD_1e-3)_aptos.pth",
    'efficientnet_b5': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/efficientnet_b5.pth",
    'inception_resnet_v2': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_resnet_v2.pth",
    'inception_v4': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_v4.pth",
    'seresnext50_32x4d': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext50_32x4d.pth",
    'seresnext101_32x4d': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext101_32x4d.pth"
}
"""
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/resnet18.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
loaded_model_keys = []
loaded_from_checkpoint = {}  # key -> bool

for model_key, path in model_paths.items():
    if model_key not in model_names:
        raise KeyError(f"model_key '{model_key}' not found in model_names mapping.")

    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)

    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        loaded_model_keys.append(model_key)
        loaded_from_checkpoint[model_key] = True
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=5)
        loaded_model_keys.append(model_key)
        loaded_from_checkpoint[model_key] = False

    model.to(device)
    model.eval()
    models_list.append(model)

assert len(models_list) > 0, "No models available for inference."



## === cell 6
validation_scores = {
    "resnet18": 0.879,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.897,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,  # 0.888,
    "seresnext50_32x4d": 0.8652,  # 0.709,
    "seresnext101_32x4d": 0.9083,  # 0.951
}



## === cell 7
used_scores = {k: validation_scores.get(k, 1.0) for k in loaded_model_keys}
total_score = sum(used_scores.values())
weights = {k: v / total_score for k, v in used_scores.items()}




## === cell 8
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    denom = float((n_classes - 1) ** 2)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / denom

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def apply_thresholds(score, thresholds):
    t0, t1, t2, t3 = thresholds
    return np.where(
        score < t0,
        0,
        np.where(score < t1, 1, np.where(score < t2, 2, np.where(score < t3, 3, 4))),
    ).astype(np.int64)


def fit_thresholds(y_true, score, n_classes=5, n_iter=1, grid_size=11):
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)

    qs = [20, 40, 60, 80]
    thr = np.percentile(score, qs).astype(np.float64)

    s_min, s_max = float(score.min()), float(score.max())
    span = max(1e-6, s_max - s_min)

    def clamp_and_sort(t):
        t = np.clip(t, s_min, s_max)
        t = np.sort(t)
        eps = 1e-6 * span
        for i in range(1, len(t)):
            if t[i] <= t[i - 1] + eps:
                t[i] = min(s_max, t[i - 1] + eps)
        return t

    thr = clamp_and_sort(thr)
    best_thr = thr.copy()
    best_k = quadratic_weighted_kappa(
        y_true, apply_thresholds(score, best_thr), n_classes=n_classes
    )

    for _ in range(n_iter):
        for j in range(n_classes - 1):
            center = best_thr[j]
            window = 0.10 * span
            grid = np.linspace(
                center - window, center + window, grid_size, dtype=np.float64
            )
            for v in grid:
                cand = best_thr.copy()
                cand[j] = v
                cand = clamp_and_sort(cand)
                k = quadratic_weighted_kappa(
                    y_true, apply_thresholds(score, cand), n_classes=n_classes
                )
                if k > best_k:
                    best_k = k
                    best_thr = cand
    return best_thr, best_k


def infer_expected_score_from_probs(probs):
    w = np.arange(probs.shape[1], dtype=np.float64)
    return (probs * w[None, :]).sum(axis=1)


def predict_probs(dataloader):
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Infer"):
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                images = batch[0]
            else:
                images = batch

            images = images.to(device, non_blocking=True)

            outputs_list = []
            for model_key, model in zip(loaded_model_keys, models_list):
                probs = nn.functional.softmax(model(images), dim=1)
                outputs_list.append(weights[model_key] * probs.unsqueeze(0))

            if len(outputs_list) == 0:
                raise RuntimeError(
                    "No model outputs were produced; check model loading."
                )

            outputs = torch.cat(outputs_list, dim=0)
            weighted_outputs = torch.sum(outputs, dim=0)
            all_probs.append(weighted_outputs.detach().cpu().numpy())
    return np.concatenate(all_probs, axis=0)




## === cell 9
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
train_df = pd.read_csv(train_csv_file)


def stable_hash_01(s):
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return (h % 10_000_000) / 10_000_000.0


val_mask = train_df["id_code"].apply(stable_hash_01) < 0.20
val_df = train_df[val_mask].reset_index(drop=True)
tr_df = train_df[~val_mask].reset_index(drop=True)

val_csv_path = "/kaggle/working/_val_split.csv"
val_df.to_csv(val_csv_path, index=False)

tr_csv_path = "/kaggle/working/_train_split.csv"
tr_df.to_csv(tr_csv_path, index=False)

val_dataset = BlindnessDataset(
    val_csv_path, train_root_dir, transform=transform, test=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

tr_dataset = BlindnessDataset(
    tr_csv_path, train_root_dir, transform=transform, test=False
)
tr_loader = DataLoader(
    tr_dataset,
    batch_size=16,
    shuffle=True,  # standard for training; does not affect inference ordering
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

need_head_calibration = any(
    not loaded_from_checkpoint.get(k, True) for k in loaded_model_keys
)


def freeze_all_but_classifier(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    head = None
    try:
        head = model.get_classifier()
    except Exception:
        head = None
    if head is None:
        for attr in ["fc", "classifier", "head"]:
            if hasattr(model, attr):
                head = getattr(model, attr)
                break
    if head is None:
        raise RuntimeError(
            "Could not locate classifier layer for head-only calibration."
        )
    for p in head.parameters():
        p.requires_grad = True
    return head


if need_head_calibration:
    calibrate_idx = None
    for i, k in enumerate(loaded_model_keys):
        if not loaded_from_checkpoint.get(k, True):
            calibrate_idx = i
            break
    if calibrate_idx is None:
        calibrate_idx = 0

    model0 = models_list[calibrate_idx]
    model0.train()
    head = freeze_all_but_classifier(model0)

    mse = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model0.parameters()),
        lr=3e-3,
        weight_decay=1e-4,
    )

    w = torch.arange(5, device=device, dtype=torch.float32).view(1, 5)

    n_epochs = 3  # unchanged
    for ep in range(n_epochs):
        running = 0.0
        n_seen = 0
        for images, labels in tqdm(
            tr_loader, desc=f"Head calibrate ep {ep+1}/{n_epochs}"
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            logits = model0(images)
            probs = torch.softmax(logits, dim=1)
            exp_score = (probs * w).sum(dim=1)
            loss = mse(exp_score, labels)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * labels.size(0)
            n_seen += labels.size(0)
        print(f"Epoch {ep+1}: train loss {running / max(1,n_seen):.4f}")

    model0.eval()

val_probs = predict_probs(val_loader)
val_score = infer_expected_score_from_probs(val_probs)
val_y = val_df["diagnosis"].values.astype(np.int64)

thresholds, best_k = fit_thresholds(
    val_y, val_score, n_classes=5, n_iter=1, grid_size=11
)
print("Fitted thresholds:", thresholds)
print("Val QWK after thresholding:", best_k)



## === cell 10
test_probs = predict_probs(test_loader)
test_score = infer_expected_score_from_probs(test_probs)
final_predictions = apply_thresholds(test_score, thresholds).astype(int)

test_ids = pd.read_csv(test_csv_file)["id_code"].values
assert len(test_ids) == len(final_predictions), "Test ids/predictions length mismatch."

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())

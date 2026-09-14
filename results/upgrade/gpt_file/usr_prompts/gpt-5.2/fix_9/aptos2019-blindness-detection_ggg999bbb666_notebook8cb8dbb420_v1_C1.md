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

3.9

# 3. Installed packages

geopandas==0.14.4
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

0.8536147975702988

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00254) has done: 'I remove the failing dependency on a missing `pretrainedmodels` package and instead use the equivalent `torchvision` ResNet50 backbone, keeping the same GeM pooling and single-output regression head. I also fix file/path issues by using the provided competition directory under `/kaggle/input/aptos2019-blindness-detection/` and ensure predictions are generated in the exact `test.csv` order (not filesystem glob order), which prevents submission misalignment. Since the provided external weights file (`../input/mymodels/model15.pth`) is not present in your environment, I make the checkpoint loading optional (safe fallback) so the notebook always produces a valid `submission.csv`. Finally, I keep your original thresholding logic (score-neutral vs your intent) and ensure the output CSV has the required columns and `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'Your current 0.00254 score is consistent with running a randomly initialized ResNet50 because the intended checkpoint isn’t available; the smallest legitimate step toward the 0.8536 target is to use ImageNet pretrained weights (same architecture/head/pooling) so the model produces non-random features. I keep your GeM + single-output regression head and the exact same thresholding logic, only switching `pretrain=None` to `pretrain="imagenet"` and making the test-time preprocessing match standard ResNet inference a bit better by using a center crop after resize (still deterministic, no TTA). This should move the score substantially upward without changing the core approach or training loop (there is none). The script still write `submission.csv` in the required format and order.'
- What this solution (achieved -0.06045) has done: 'Your current 0.0 score is consistent with a submission-format or row-alignment problem rather than model quality, so the smallest change toward the target is to guarantee the `submission.csv` exactly matches `sample_submission.csv` row order and schema. I also make inference deterministic and robust by forcing CPU/GPU eval settings and ensuring we never reorder or drop any `id_code`. Finally, I clip/round predictions defensively to valid class integers 0–4 after your existing thresholding, without changing your model or preprocessing, so Kaggle always receives valid labels.'
- What this solution (achieved -0.02634) has done: 'You currently calibrate thresholds on the same data you later “evaluate” them on (train-on-train), which tends to overfit the thresholds and usually hurts public/private QWK; the smallest legitimate improvement toward your target is to calibrate thresholds on an out-of-fold (OOF) set while keeping your exact model/inference and thresholding approach. I add a deterministic stratified split (no new training, no architecture changes) and optimize thresholds only on the held-out fold, then use those thresholds for test predictions. I also add simple batched inference (same deterministic preprocessing, same model outputs) to reduce runtime and keep the script well within the 600s limit, which indirectly helps ensure you can iterate without timeouts. The output `submission.csv` format, ordering, and your GeM+ResNet50 regression head remain unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np

import torch
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import models
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = "/kaggle/input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_IMAGE_PATH = os.path.join(DATA_DIR, "test_images")
TRAIN_IMAGE_PATH = os.path.join(DATA_DIR, "train_images")

MODEL_PATH = "/kaggle/input/mymodels/model15.pth"  # may not exist in this environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("device:", device)
print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print("TRAIN_IMAGE_PATH exists:", os.path.exists(TRAIN_IMAGE_PATH))
print("TEST_IMAGE_PATH exists:", os.path.exists(TEST_IMAGE_PATH))
print("MODEL_PATH exists:", os.path.exists(MODEL_PATH))
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("TEST_CSV exists:", os.path.exists(TEST_CSV))
print("SAMPLE_SUB_CSV exists:", os.path.exists(SAMPLE_SUB_CSV))




## === cell 1
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_resnet50_gem(pretrain=None):
    if pretrain == "imagenet":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None

    model = models.resnet50(weights=weights)
    model.avgpool = GeM()
    model.fc = nn.Linear(2048, 1)
    return model


model = get_resnet50_gem(pretrain="imagenet")
model.to(device)

if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location=device)
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k[7:] if k.startswith("module.") else k
                new_state[nk] = v
            state = new_state
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded MODEL_PATH with strict=False")
        print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    except Exception as e:
        print("Warning: failed to load checkpoint:", repr(e))
        print("Proceeding with ImageNet-pretrained model.")
else:
    print("Checkpoint not found; using ImageNet-pretrained model.")




## === cell 2
def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

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


def apply_thresholds(raw_scores, t0, t1, t2, t3):
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    y = np.zeros_like(raw_scores, dtype=int)
    y[raw_scores >= t0] = 1
    y[raw_scores >= t1] = 2
    y[raw_scores >= t2] = 3
    y[raw_scores >= t3] = 4
    return np.clip(y, 0, 4)


def stratified_kfold_indices(y, n_splits=5, seed=42, num_classes=5):
    y = np.asarray(y, dtype=int)
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_splits)]
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[i % n_splits].append(int(ix))
    folds = [np.array(f, dtype=int) for f in folds]
    return folds


class IdImageDataset(Dataset):
    def __init__(self, id_list, image_dir, tfm):
        self.id_list = list(id_list)
        self.image_dir = image_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        id_code = self.id_list[idx]
        im_path = os.path.join(self.image_dir, f"{id_code}.png")
        if not os.path.exists(im_path):
            raise FileNotFoundError(f"Missing image: {im_path}")
        image = Image.open(im_path).convert("RGB")
        image = self.tfm(image)
        return image, id_code


class TrainImageDataset(Dataset):
    def __init__(self, df, image_dir, tfm):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_code = row["id_code"]
        y = float(row["diagnosis"])
        im_path = os.path.join(self.image_dir, f"{id_code}.png")
        if not os.path.exists(im_path):
            raise FileNotFoundError(f"Missing image: {im_path}")
        image = Image.open(im_path).convert("RGB")
        image = self.tfm(image)
        return image, torch.tensor([y], dtype=torch.float32)


@torch.no_grad()
def predict_ids_batched(id_list, image_dir, tfm, batch_size=32, num_workers=2):
    use_cuda = torch.cuda.is_available()
    ds = IdImageDataset(id_list, image_dir, tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # must preserve id_list order
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )
    preds = np.zeros((len(id_list),), dtype=np.float64)
    offset = 0
    model.eval()
    for bi, (x, _) in enumerate(dl):
        if bi % 20 == 0:
            print(f"batch {bi}/{len(dl)}")
        x = x.to(device, non_blocking=True)
        out = model(x).squeeze(1).detach().cpu().numpy().astype(np.float64)
        preds[offset : offset + len(out)] = out
        offset += len(out)
    return preds


weights = models.ResNet50_Weights.IMAGENET1K_V2
tfm = weights.transforms()

train_df = pd.read_csv(TRAIN_CSV)
assert list(train_df.columns) == ["id_code", "diagnosis"]
train_ids = train_df["id_code"].tolist()
y_train = train_df["diagnosis"].astype(int).to_numpy()




## === cell 3
def train_one_epoch_regression(model, dl, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        n += bs
    return total_loss / max(1, n)


BATCH_SIZE = 16
NUM_WORKERS = min(4, (os.cpu_count() or 2))
EPOCHS = 2
LR = 1e-4

train_ds = TrainImageDataset(train_df, TRAIN_IMAGE_PATH, tfm)
train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
    drop_last=False,
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("\nTraining (fine-tuning) on full train set...")
for ep in range(EPOCHS):
    avg_loss = train_one_epoch_regression(model, train_dl, optimizer, loss_fn)
    print(f"epoch {ep+1}/{EPOCHS} - train_mse: {avg_loss:.6f}")

model.eval()




## === cell 4
def optimize_thresholds(raw_scores, y_true, initial=(0.7, 1.5, 2.5, 3.5), iters=6):
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)

    lo = float(np.percentile(raw_scores, 1))
    hi = float(np.percentile(raw_scores, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(raw_scores.min()), float(raw_scores.max())

    t = np.array(initial, dtype=np.float64)

    def score_for(tvec):
        t0, t1, t2, t3 = tvec
        if not (t0 < t1 < t2 < t3):
            return -1e9
        pred = apply_thresholds(raw_scores, t0, t1, t2, t3)
        return quadratic_weighted_kappa(y_true, pred, num_classes=5)

    best = score_for(t)

    for k in range(iters):
        step = (hi - lo) * (0.08 / (2**k))
        for idx in range(4):
            candidates = t[idx] + step * np.array([-2, -1, 0, 1, 2], dtype=np.float64)
            best_local_t = t[idx]
            best_local = best
            for c in candidates:
                t_try = t.copy()
                t_try[idx] = float(c)
                if not (t_try[0] < t_try[1] < t_try[2] < t_try[3]):
                    continue
                s = score_for(t_try)
                if s > best_local:
                    best_local = s
                    best_local_t = float(c)
            t[idx] = best_local_t
            best = best_local

        t = np.clip(t, lo, hi)
        eps = 1e-4
        t[1] = max(t[1], t[0] + eps)
        t[2] = max(t[2], t[1] + eps)
        t[3] = max(t[3], t[2] + eps)

    return tuple(map(float, t)), float(best)


folds = stratified_kfold_indices(y_train, n_splits=5, seed=seed, num_classes=5)
va_idx = folds[0]  # deterministic held-out fold
va_ids = [train_ids[i] for i in va_idx]
y_valid = y_train[va_idx]

print(f"\nCalibrating thresholds on 1 held-out fold (size={len(va_idx)})...")
raw_valid = predict_ids_batched(
    va_ids,
    TRAIN_IMAGE_PATH,
    tfm,
    batch_size=(
        48 if torch.cuda.is_available() else 24
    ),  # change: faster inference, same outputs
    num_workers=NUM_WORKERS,
)

best_t, qwk_valid = optimize_thresholds(
    raw_valid, y_valid, initial=(0.7, 1.5, 2.5, 3.5), iters=6
)
print(f"Holdout QWK (for threshold calibration sanity-check): {qwk_valid:.6f}")
print("Thresholds used for test:", best_t)



## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
assert list(sample_sub.columns) == ["id_code", "diagnosis"]
test_df = pd.read_csv(TEST_CSV)
assert "id_code" in test_df.columns

id_list = sample_sub["id_code"].tolist()

test_ids = set(test_df["id_code"].tolist())
missing_in_test = [x for x in id_list if x not in test_ids]
if len(missing_in_test) > 0:
    raise ValueError(
        f"Some sample_submission ids are missing in test.csv, example: {missing_in_test[:5]}"
    )

print("Predicting test set...")
raw_test = predict_ids_batched(
    id_list,
    TEST_IMAGE_PATH,
    tfm,
    batch_size=(
        48 if torch.cuda.is_available() else 24
    ),  # change: faster inference, same outputs
    num_workers=NUM_WORKERS,
)

raw_df = pd.DataFrame({"id_code": id_list, "diagnosis": raw_test.tolist()})
raw_df.to_csv("submission_raw_value.csv", index=False)
print(raw_df.head())



## === cell 6
t0, t1, t2, t3 = best_t
pred_labels = apply_thresholds(raw_test, t0, t1, t2, t3).astype(int)

submission = pd.DataFrame({"id_code": id_list, "diagnosis": pred_labels})
submission["diagnosis"] = submission["diagnosis"].astype(int).clip(0, 4)

submission = submission[["id_code", "diagnosis"]]
submission = sample_sub[["id_code"]].merge(submission, on="id_code", how="left")
if submission["diagnosis"].isna().any():
    raise ValueError(
        "NaNs found in diagnosis after merge; would create invalid submission."
    )
submission["diagnosis"] = submission["diagnosis"].astype(int)

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Unique labels:", sorted(submission["diagnosis"].unique().tolist()))
print("Used thresholds:", best_t)

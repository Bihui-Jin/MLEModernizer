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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8991451080771339

# 6. Current score

0.73056

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0896) has done: 'I fix the immediate runtime blocker by removing the missing `../input/weights/...` dependency and instead loading ImageNet pretrained weights for the same EfficientNet-B5 backbone so inference can run end-to-end. I also make device selection robust (CPU fallback) and ensure test image paths exist in this environment, while keeping the model architecture and inference loop semantics the same. Finally, I guarantee a valid `submission.csv` is always written with the required columns and integer class labels in `[0,4]`. This should produce a valid submission and a reasonable score (likely below your original custom-weight model, but it run and yield a score rather than failing).'
- What this solution (achieved 0.11119) has done: 'Your current score is far below the target, and the biggest issue is that the inference-time mapping from the regression output to classes is inconsistent with the provided thresholds (you’re currently using `round`, while `regress2class()` exists but isn’t used). I make the smallest change to use your existing `regress2class()` thresholding for converting the regressed severity into class labels, which is directly aligned with quadratic weighted kappa behavior in this common APTOS setup. I also fix a small device/dtype issue inside `regress2class()` so it always returns an integer tensor of the right shape without silently mixing CPU/GPU. Everything else (model, weights source, transforms, inference loop) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.10714) has done: 'I keep your model/inference pipeline intact and focus on the highest-impact score lever available without “changing the approach”: calibrating the fixed regression→class thresholds. Your current fixed thresholds are very likely mismatched to the ImageNet-pretrained regressor head (which is essentially untrained), causing low QWK. I estimate better thresholds from the training set by running the same model on train images (no training), then fitting thresholds that maximize quadratic weighted kappa on those out-of-fold-like predictions (on the full train, since we’re only calibrating post-processing). This preserves architecture, forward pass, transforms, and loss/training (still none), but should move your score substantially toward the target by fixing ordinal mapping.'
- What this solution (achieved 0.73056) has done: 'Your current gap to the target is large, and the main score limiter is that the regressor head is essentially untrained (random) because we’re using ImageNet-pretrained EfficientNet features but a fresh `nn.Linear(1000,1)`. Keeping the same architecture and inference semantics, the smallest legitimate improvement is to quickly fit only that final linear layer on the provided training images (backbone frozen) using a simple regression objective, then keep your existing QWK-based threshold calibration step. This preserves the model, forward pass, transforms, and overall pipeline, but replaces the “random head” with a minimally trained head that should move QWK substantially toward the target. I also batch inference/training for speed and to stay within the 600s budget, without changing what is computed.'

# 9. Code solution

## === cell 0
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from PIL import Image

import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor, thr=None) -> torch.Tensor:
    """
    Convert regression output to ordinal class in {0..4} using thresholds.
    """
    if thr is None:
        thr = threshold

    if out.ndim == 2 and out.size(1) == 1:
        out = out.squeeze(1)

    out_cpu = out.detach().to("cpu")
    prediction = torch.zeros(out_cpu.size(0), dtype=torch.long)

    for t in thr:
        prediction += (out_cpu >= t).to(torch.long)

    return prediction




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

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


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=True)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out




## === cell 3
from sklearn.metrics import cohen_kappa_score


def apply_thresholds_numpy(preds: np.ndarray, thr) -> np.ndarray:
    preds = preds.reshape(-1)
    cls = np.zeros(preds.shape[0], dtype=np.int64)
    for t in thr:
        cls += (preds >= t).astype(np.int64)
    return cls


def fit_thresholds_by_qwk(
    preds: np.ndarray, y_true: np.ndarray, iters: int = 6
) -> list:
    """
    Fit 4 thresholds to maximize QWK on train predictions (coordinate ascent).
    """
    preds = preds.reshape(-1).astype(np.float64)
    y_true = y_true.astype(np.int64)

    qs = []
    for k in [0.2, 0.4, 0.6, 0.8]:
        qs.append(float(np.quantile(preds, k)))
    thr = sorted(qs)

    def qwk_for(thr_list):
        y_pred = apply_thresholds_numpy(preds, thr_list)
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")

    best = qwk_for(thr)

    pmin, pmax = float(preds.min()), float(preds.max())
    span = max(1e-6, pmax - pmin)

    for it in range(iters):
        improved_any = False
        for i in range(4):
            lo = pmin if i == 0 else thr[i - 1] + 1e-6
            hi = pmax if i == 3 else thr[i + 1] - 1e-6
            if hi <= lo:
                continue

            step = span / (40.0 * (1 + it))
            left = max(lo, thr[i] - 6 * step)
            right = min(hi, thr[i] + 6 * step)

            candidates = np.linspace(left, right, 25)
            local_best_thr = thr[i]
            local_best = best
            for c in candidates:
                trial = list(thr)
                trial[i] = float(c)
                trial = sorted(trial)
                score = qwk_for(trial)
                if score > local_best:
                    local_best = score
                    local_best_thr = float(c)

            if local_best > best:
                thr[i] = local_best_thr
                thr = sorted(thr)
                best = local_best
                improved_any = True

        if not improved_any:
            break

    return thr




## === cell 4
CANDIDATE_ROOTS = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
]
DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(r):
        DATA_ROOT = r
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find aptos2019-blindness-detection in any of: {CANDIDATE_ROOTS}"
    )

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
train_img_dir = os.path.join(DATA_ROOT, "train_images")
test_csv_path = os.path.join(DATA_ROOT, "test.csv")
test_img_dir = os.path.join(DATA_ROOT, "test_images")

train_df = pd.read_csv(train_csv_path)
test_ids_df = pd.read_csv(test_csv_path)
test_ids = np.squeeze(test_ids_df["id_code"].values)

input_size = 384

tranforms = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = Regressor().to(device)
net.eval()



## === cell 5
torch.set_grad_enabled(True)
for p in net.backbone.parameters():
    p.requires_grad = False
for p in net.regressor.parameters():
    p.requires_grad = True

optimizer = torch.optim.Adam(net.regressor.parameters(), lr=3e-3)
loss_fn = nn.MSELoss()

train_pairs = list(
    zip(
        train_df["id_code"].values.tolist(),
        train_df["diagnosis"].values.astype(int).tolist(),
    )
)

batch_size = 16 if device.startswith("cuda") else 8
max_epochs = (
    1  # minimal training to move score up without changing approach substantially
)

t0 = time.time()
net.train()
for epoch in range(max_epochs):
    random.shuffle(train_pairs)
    running = 0.0
    seen = 0
    missing = 0

    for start in range(0, len(train_pairs), batch_size):
        batch = train_pairs[start : start + batch_size]
        imgs = []
        ys = []
        for idx, y in batch:
            image_name = os.path.join(train_img_dir, f"{idx}.png")
            if not os.path.exists(image_name):
                missing += 1
                continue
            img = Image.open(image_name).convert("RGB")
            img = tranforms(img)
            imgs.append(img)
            ys.append(float(y))

        if len(imgs) == 0:
            continue

        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
        y = torch.tensor(ys, dtype=torch.float32, device=device).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        out = net(x)  # already sigmoid*4.5
        y_scaled = y * (4.5 / 4.0)
        loss = loss_fn(out, y_scaled)
        loss.backward()
        optimizer.step()

        running += float(loss.detach().cpu().item()) * x.size(0)
        seen += x.size(0)

        if (start // batch_size) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"train head epoch={epoch+1}/{max_epochs} step={start//batch_size} "
                f"seen={seen} mse={running/max(seen,1):.4f} missing={missing} elapsed={elapsed:.1f}s"
            )

net.eval()
torch.set_grad_enabled(False)



## === cell 6
train_preds = []
train_labels = []
missing = 0

t0 = time.time()
bs = 16 if device.startswith("cuda") else 8

ids = train_df["id_code"].values.tolist()
ys = train_df["diagnosis"].values.astype(int)

for start in range(0, len(ids), bs):
    if start % (250) == 0:
        print(f"calib train {start}/{len(ids)} elapsed={time.time()-t0:.1f}s")

    batch_ids = ids[start : start + bs]
    batch_y = ys[start : start + bs]

    imgs = []
    labels = []
    for idx, y in zip(batch_ids, batch_y):
        image_name = os.path.join(train_img_dir, f"{idx}.png")
        if not os.path.exists(image_name):
            missing += 1
            continue
        img = Image.open(image_name).convert("RGB")
        imgs.append(tranforms(img))
        labels.append(int(y))

    if len(imgs) == 0:
        continue

    x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
    r_out = net(x).detach().cpu().numpy().reshape(-1)

    train_preds.extend([float(v) for v in r_out])
    train_labels.extend(labels)

train_preds = np.array(train_preds, dtype=np.float64)
train_labels = np.array(train_labels, dtype=np.int64)

if len(train_preds) < 100:
    raise RuntimeError(
        f"Too few train images found for calibration: {len(train_preds)}"
    )

print("Calibration: found", len(train_preds), "train images; missing:", missing)

calib_thr = fit_thresholds_by_qwk(train_preds, train_labels, iters=8)
calib_qwk = cohen_kappa_score(
    train_labels, apply_thresholds_numpy(train_preds, calib_thr), weights="quadratic"
)

print("Old thresholds:", threshold)
print("Calibrated thresholds:", calib_thr)
print("Train QWK with calibrated thresholds:", float(calib_qwk))



## === cell 7
submission_rows = []
bs = 16 if device.startswith("cuda") else 8

t0 = time.time()
for start in range(0, len(test_ids), bs):
    if start % 50 == 0:
        print(f"test {start}/{len(test_ids)} elapsed={time.time()-t0:.1f}s")

    batch_ids = test_ids[start : start + bs]
    imgs = []
    kept_ids = []
    for idx in batch_ids:
        image_name = os.path.join(test_img_dir, f"{idx}.png")
        if not os.path.exists(image_name):
            raise FileNotFoundError(f"Missing test image: {image_name}")
        img = Image.open(image_name).convert("RGB")
        imgs.append(tranforms(img))
        kept_ids.append(idx)

    x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
    r_out = net(x)

    pred = regress2class(r_out, thr=calib_thr).numpy().reshape(-1)
    for idx, p in zip(kept_ids, pred):
        p_int = int(p)
        p_int = max(0, min(4, p_int))
        submission_rows.append([idx, p_int])

df = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])
df["diagnosis"] = df["diagnosis"].astype(int)
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
assert df.shape[0] == len(test_ids), "Submission row count mismatch vs test.csv"

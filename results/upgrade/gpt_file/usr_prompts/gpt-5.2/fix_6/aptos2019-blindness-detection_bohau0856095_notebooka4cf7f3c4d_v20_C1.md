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

0.03912

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0896) has done: 'I fix the immediate runtime blocker by removing the missing `../input/weights/...` dependency and instead loading ImageNet pretrained weights for the same EfficientNet-B5 backbone so inference can run end-to-end. I also make device selection robust (CPU fallback) and ensure test image paths exist in this environment, while keeping the model architecture and inference loop semantics the same. Finally, I guarantee a valid `submission.csv` is always written with the required columns and integer class labels in `[0,4]`. This should produce a valid submission and a reasonable score (likely below your original custom-weight model, but it run and yield a score rather than failing).'
- What this solution (achieved 0.11119) has done: 'Your current score is far below the target, and the biggest issue is that the inference-time mapping from the regression output to classes is inconsistent with the provided thresholds (you’re currently using `round`, while `regress2class()` exists but isn’t used). I make the smallest change to use your existing `regress2class()` thresholding for converting the regressed severity into class labels, which is directly aligned with quadratic weighted kappa behavior in this common APTOS setup. I also fix a small device/dtype issue inside `regress2class()` so it always returns an integer tensor of the right shape without silently mixing CPU/GPU. Everything else (model, weights source, transforms, inference loop) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.10714) has done: 'I keep your model/inference pipeline intact and focus on the highest-impact score lever available without “changing the approach”: calibrating the fixed regression→class thresholds. Your current fixed thresholds are very likely mismatched to the ImageNet-pretrained regressor head (which is essentially untrained), causing low QWK. I estimate better thresholds from the training set by running the same model on train images (no training), then fitting thresholds that maximize quadratic weighted kappa on those out-of-fold-like predictions (on the full train, since we’re only calibrating post-processing). This preserves architecture, forward pass, transforms, and loss/training (still none), but should move your score substantially toward the target by fixing ordinal mapping.'
- What this solution (achieved 0.73056) has done: 'Your current gap to the target is large, and the main score limiter is that the regressor head is essentially untrained (random) because we’re using ImageNet-pretrained EfficientNet features but a fresh `nn.Linear(1000,1)`. Keeping the same architecture and inference semantics, the smallest legitimate improvement is to quickly fit only that final linear layer on the provided training images (backbone frozen) using a simple regression objective, then keep your existing QWK-based threshold calibration step. This preserves the model, forward pass, transforms, and overall pipeline, but replaces the “random head” with a minimally trained head that should move QWK substantially toward the target. I also batch inference/training for speed and to stay within the 600s budget, without changing what is computed.'
- What this solution (achieved 0.03912) has done: 'Your current score (0.73056) is well below the target (0.8991), so we should improve it with minimal, low-risk changes that preserve your approach (freeze backbone, train only the linear head, then QWK-threshold calibration). The biggest limiter in your setup is that the regression head is trained for only 1 epoch, which is likely underfitting; increasing this slightly (without changing architecture or objective) should move QWK upward toward the target. To keep this stable and within time, I also add a lightweight train/validation split for monitoring (no early stopping) and fix determinism + input pipeline efficiency (DataLoader, pinned memory) without changing what is computed. Finally, I keep your threshold calibration logic but fit thresholds on the validation split (not the same exact images used to fit the head) to reduce overfitting and improve generalization to test.'

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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



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
    Fit 4 thresholds to maximize QWK on provided predictions (coordinate ascent).
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
from torch.utils.data import Dataset, DataLoader

ids_all = train_df["id_code"].values
y_all = train_df["diagnosis"].values.astype(int)

rng = np.random.RandomState(42)
perm = rng.permutation(len(ids_all))
val_size = max(200, int(0.15 * len(ids_all)))
val_idx = perm[:val_size]
tr_idx = perm[val_size:]

train_ids = ids_all[tr_idx]
train_y = y_all[tr_idx]
val_ids = ids_all[val_idx]
val_y = y_all[val_idx]


class AptosDataset(Dataset):
    def __init__(self, ids, ys, img_dir, tfm):
        self.ids = ids
        self.ys = ys
        self.img_dir = img_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        path = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        if self.ys is None:
            return idx, x
        y = float(self.ys[i])
        return idx, x, y


train_ds = AptosDataset(train_ids, train_y, train_img_dir, tranforms)
val_ds = AptosDataset(val_ids, val_y, train_img_dir, tranforms)

batch_size = 24 if device.startswith("cuda") else 8
num_workers = 2 if device.startswith("cuda") else 0

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=device.startswith("cuda"),
    drop_last=False,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=device.startswith("cuda"),
    drop_last=False,
)



## === cell 6
torch.set_grad_enabled(True)
for p in net.backbone.parameters():
    p.requires_grad = False
for p in net.regressor.parameters():
    p.requires_grad = True

optimizer = torch.optim.Adam(net.regressor.parameters(), lr=3e-3)
loss_fn = nn.MSELoss()

max_epochs = 3  # minimal increase from 1 epoch to move score toward target

t0 = time.time()
for epoch in range(max_epochs):
    net.train()
    running = 0.0
    seen = 0

    for step, batch in enumerate(train_loader):
        _, x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        out = net(x)  # sigmoid*4.5
        y_scaled = y * (4.5 / 4.0)
        loss = loss_fn(out, y_scaled)
        loss.backward()
        optimizer.step()

        running += float(loss.detach().cpu().item()) * x.size(0)
        seen += x.size(0)

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"train head epoch={epoch+1}/{max_epochs} step={step} "
                f"seen={seen} mse={running/max(seen,1):.4f} elapsed={elapsed:.1f}s"
            )

    net.eval()
    with torch.no_grad():
        vpred = []
        vytrue = []
        for _, x, y in val_loader:
            x = x.to(device, non_blocking=True)
            out = net(x).detach().cpu().numpy().reshape(-1)
            vpred.append(out)
            vytrue.append(y.numpy().astype(np.int64))
        vpred = np.concatenate(vpred, axis=0)
        vytrue = np.concatenate(vytrue, axis=0)
        thr_tmp = fit_thresholds_by_qwk(vpred, vytrue, iters=6)
        vqwk = cohen_kappa_score(
            vytrue, apply_thresholds_numpy(vpred, thr_tmp), weights="quadratic"
        )
        print(
            f"monitor val QWK (thr-fit-on-val) after epoch {epoch+1}: {float(vqwk):.5f}"
        )

net.eval()
torch.set_grad_enabled(False)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/754781264.py in <cell line: 0>()
     27         y_scaled = y * (4.5 / 4.0)
     28         loss = loss_fn(out, y_scaled)
---> 29         loss.backward()
     30         optimizer.step()
     31 

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

RuntimeError: Found dtype Double but expected Float

## === cell 7
val_preds = []
val_labels = []

t0 = time.time()
with torch.no_grad():
    for step, batch in enumerate(val_loader):
        if step % 25 == 0:
            print(
                f"calib val step {step}/{len(val_loader)} elapsed={time.time()-t0:.1f}s"
            )
        _, x, y = batch
        x = x.to(device, non_blocking=True)
        r_out = net(x).detach().cpu().numpy().reshape(-1)
        val_preds.append(r_out)
        val_labels.append(y.numpy().astype(np.int64))

val_preds = np.concatenate(val_preds, axis=0).astype(np.float64)
val_labels = np.concatenate(val_labels, axis=0).astype(np.int64)

calib_thr = fit_thresholds_by_qwk(val_preds, val_labels, iters=8)
calib_qwk = cohen_kappa_score(
    val_labels, apply_thresholds_numpy(val_preds, calib_thr), weights="quadratic"
)

print("Old thresholds:", threshold)
print("Calibrated thresholds (fit on val):", calib_thr)
print("Val QWK with calibrated thresholds:", float(calib_qwk))



## === cell 8
test_ds = AptosDataset(test_ids, None, test_img_dir, tranforms)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=device.startswith("cuda"),
    drop_last=False,
)

submission_rows = []
t0 = time.time()

net.eval()
with torch.no_grad():
    for step, batch in enumerate(test_loader):
        if step % 10 == 0:
            print(f"test step {step}/{len(test_loader)} elapsed={time.time()-t0:.1f}s")

        kept_ids, x = batch
        x = x.to(device, non_blocking=True)
        r_out = net(x)

        pred = regress2class(r_out, thr=calib_thr).numpy().reshape(-1)
        for idx, p in zip(list(kept_ids), pred):
            p_int = int(p)
            p_int = max(0, min(4, p_int))
            submission_rows.append([idx, p_int])

df = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])
df["diagnosis"] = df["diagnosis"].astype(int)
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
assert df.shape[0] == len(test_ids), "Submission row count mismatch vs test.csv"

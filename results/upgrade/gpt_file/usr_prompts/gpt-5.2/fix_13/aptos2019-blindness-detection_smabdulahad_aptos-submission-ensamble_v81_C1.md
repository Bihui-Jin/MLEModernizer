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

0.8946705026979583

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.06699) has done: 'I first fix the runtime blocker by making model checkpoint loading robust: if the external ensemble weight files are missing (as in your environment), the code fall back to using the same timm architectures with ImageNet pretrained weights so inference can run end-to-end and produce a valid `submission.csv`. Next, I fix the ensemble weighting logic bug by ensuring `weights` contains keys for exactly the models being ensembled (previously it used a larger unrelated dict, causing key mismatch/empty concatenation). Finally, I make inference stable in Kaggle by adding `map_location`, RGB conversion, and safe DataLoader settings so it runs without hanging and always defines `final_predictions` before writing the submission.'
- What this solution (achieved 0.72901) has done: 'The timeout is dominated by (1) fine-tuning three heavy CNNs sequentially per batch (3 forward+backward passes per step) and (2) slow single-process image loading/decoding. To preserve identical model/loss/training semantics while reducing wall time, I increase DataLoader parallelism with pinned memory + persistent workers, enable cudnn benchmarking for fixed input size, and (most importantly) fuse the three separate model backward passes into a single backward over the summed loss so the expensive autograd graph traversal happens once per batch (still the same per-model loss contributions and optimizer steps). I also make inference faster via AMP autocast (inference-only) and avoid redundant CPU/GPU syncs inside the training loop.'

# 9. Code solution

## === cell 0
import os
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import timm

import timm.data




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

        self._ids = self.annotations.iloc[:, 0].astype(str).to_numpy()
        if not test and self.annotations.shape[1] > 1:
            self._labels = self.annotations.iloc[:, 1].astype(np.int64).to_numpy()
        else:
            self._labels = None

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        img_id = self._ids[idx]
        img_name = os.path.join(self.root_dir, f"{img_id}.png")

        with Image.open(img_name) as im:
            image = im.convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self._labels[idx])
            return image, label




## === cell 2
def build_model_transform(model_name: str):
    m = timm.create_model(model_name, pretrained=True)
    cfg = timm.data.resolve_model_data_config(m)
    tfm = timm.data.create_transform(**cfg, is_training=False)
    return tfm




## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

_cuda = torch.cuda.is_available()
_num_workers = min(4, os.cpu_count() or 1)



## === cell 4
model_paths = {
    "efficientnet_b2": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
    "efficientnet_b3": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos-ensamble-models/pytorch/ensamble_v2/2/seresnext101_32x4d.pth",
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



## === cell 5
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.9127,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False




## === cell 7
def _clean_state_dict_keys(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def _create_model_with_best_available_weights(
    model_name: str, ckpt_path: str, num_classes: int = 5
):
    has_ckpt = os.path.exists(ckpt_path)

    if has_ckpt:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        state = torch.load(ckpt_path, map_location="cpu")
        state = _clean_state_dict_keys(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            print(
                f"[load_state_dict] {model_name}: missing={len(missing)} unexpected={len(unexpected)}"
            )
        return model, True

    try:
        pretrained_models = timm.list_models(model_name, pretrained=True)
    except Exception:
        pretrained_models = []

    aptos_candidates = [m for m in pretrained_models if "aptos" in m.lower()]
    if len(aptos_candidates) > 0:
        chosen = aptos_candidates[0]
        model = timm.create_model(chosen, pretrained=True, num_classes=num_classes)
        print(
            f"[checkpoint missing] {model_name}: using timm pretrained weights '{chosen}'"
        )
        return model, False

    model = timm.create_model(model_name, pretrained=True)
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=num_classes)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    print(
        f"[checkpoint missing] {model_name}: using ImageNet pretrained backbone and resetting classifier to {num_classes} classes"
    )
    return model, False


models_list = []
active_model_keys = []

for model_key, path in model_paths.items():
    model_name = model_names[model_key]
    model, _ = _create_model_with_best_available_weights(
        model_name, path, num_classes=5
    )

    model.to(device)
    model.eval()

    models_list.append(model)
    active_model_keys.append(model_key)

if len(models_list) == 0:
    raise RuntimeError(
        "No models available for inference. Check model_paths and environment inputs."
    )

model_transforms = {k: build_model_transform(model_names[k]) for k in active_model_keys}



## === cell 8
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"

train_df_full = pd.read_csv(train_csv_file)
val_frac = 0.15
rng = np.random.RandomState(42)
perm = rng.permutation(len(train_df_full))
val_size = int(round(len(train_df_full) * val_frac))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

train_df = train_df_full.iloc[trn_idx].reset_index(drop=True)
val_df = train_df_full.iloc[val_idx].reset_index(drop=True)

_train_csv_tmp = "train_split.csv"
_val_csv_tmp = "val_split.csv"
train_df.to_csv(_train_csv_tmp, index=False)
val_df.to_csv(_val_csv_tmp, index=False)

train_transform = model_transforms[active_model_keys[0]]
val_transform = model_transforms[active_model_keys[0]]

train_dataset = BlindnessDataset(
    _train_csv_tmp, train_root_dir, transform=train_transform, test=False
)
val_dataset = BlindnessDataset(
    _val_csv_tmp, train_root_dir, transform=val_transform, test=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=_cuda,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=_cuda,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
    drop_last=False,
)



## === cell 9
criterion = nn.CrossEntropyLoss()

optimizers = []
for m in models_list:
    for p in m.parameters():
        p.requires_grad = True
    optimizers.append(torch.optim.Adam(m.parameters(), lr=1e-4))



## === cell 10
epochs = 2  # keep as-is
TRAIN_TIME_BUDGET_SEC = 260  # leave time for val inference + test inference + CSV write
train_start_time = time.time()
trained_any_batches = False

for epoch in range(epochs):
    for m in models_list:
        m.train()

    pbar = tqdm(train_loader, desc=f"Fine-tune epoch {epoch+1}/{epochs}", leave=False)
    for images, labels in pbar:
        if time.time() - train_start_time > TRAIN_TIME_BUDGET_SEC:
            pbar.close()
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        total_loss = None
        losses_for_log = []

        for m in models_list:
            logits = m(images)
            loss = criterion(logits, labels)
            losses_for_log.append(loss.detach())
            total_loss = loss if total_loss is None else (total_loss + loss)

        total_loss.backward()

        for opt in optimizers:
            opt.step()

        trained_any_batches = True
        mean_loss = torch.stack(losses_for_log).mean().item()
        pbar.set_postfix({"loss": mean_loss})

    if time.time() - train_start_time > TRAIN_TIME_BUDGET_SEC:
        break

for m in models_list:
    m.eval()

print("Trained any batches:", trained_any_batches)



## === cell 11
active_scores = {k: float(validation_scores.get(k, 1.0)) for k in active_model_keys}
total_score = sum(active_scores.values())
weights = {k: v / total_score for k, v in active_scores.items()}




## === cell 12
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def apply_thresholds_to_score(score, thresholds):
    t0, t1, t2, t3 = thresholds
    return np.where(
        score < t0,
        0,
        np.where(score < t1, 1, np.where(score < t2, 2, np.where(score < t3, 3, 4))),
    ).astype(np.int64)


def tune_thresholds(y_true, score, init_thresholds=None, iters=4):
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)

    if init_thresholds is None:
        init_thresholds = np.quantile(score, [0.2, 0.4, 0.6, 0.8]).astype(np.float64)

    thr = np.array(init_thresholds, dtype=np.float64)

    for stage in range(iters):
        step = 0.25 / (2**stage)
        for k in range(4):
            best_thr_k = thr[k]
            best_kappa = quadratic_weighted_kappa(
                y_true, apply_thresholds_to_score(score, thr)
            )

            for delta in [-3, -2, -1, 1, 2, 3]:
                cand = thr.copy()
                cand[k] = cand[k] + delta * step

                if k > 0 and cand[k] <= cand[k - 1] + 1e-6:
                    continue
                if k < 3 and cand[k] >= cand[k + 1] - 1e-6:
                    continue

                kappa = quadratic_weighted_kappa(
                    y_true, apply_thresholds_to_score(score, cand)
                )
                if kappa > best_kappa:
                    best_kappa = kappa
                    best_thr_k = cand[k]

            thr[k] = best_thr_k

    return thr




## === cell 13
def make_loader_for_model(csv_file, root_dir, transform, test, batch_size):
    ds = BlindnessDataset(csv_file, root_dir, transform=transform, test=test)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=_cuda,
        persistent_workers=(_num_workers > 0),
        prefetch_factor=2 if _num_workers > 0 else None,
        drop_last=False,
    )


def predict_single_model_probs(model, loader):
    all_probs = []

    use_amp_infer = bool(_cuda)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp_infer
        else torch.cpu.amp.autocast(enabled=False)
    )

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, _ = batch
            else:
                images = batch
            images = images.to(device, non_blocking=True)

            with autocast_ctx:
                logits = model(images)
                probs = nn.functional.softmax(logits, dim=1)

            all_probs.append(probs.detach().float().cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def predict_ensemble_probs(csv_file, root_dir, test, batch_size):
    probs_sum = None
    for model_key, model in zip(active_model_keys, models_list):
        loader = make_loader_for_model(
            csv_file=csv_file,
            root_dir=root_dir,
            transform=model_transforms[model_key],
            test=test,
            batch_size=batch_size,
        )
        probs = predict_single_model_probs(model, loader)
        w = weights[model_key]
        probs_sum = (w * probs) if probs_sum is None else (probs_sum + w * probs)
    return probs_sum




## === cell 14
val_probs = predict_ensemble_probs(
    _val_csv_tmp, train_root_dir, test=False, batch_size=32
)
val_labels = val_df["diagnosis"].astype(np.int64).to_numpy()

val_score = (val_probs * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)

init_thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
best_thr = tune_thresholds(val_labels, val_score, init_thresholds=init_thr, iters=6)

val_pred_argmax = np.argmax(val_probs, axis=1).astype(np.int64)
val_kappa_argmax = quadratic_weighted_kappa(val_labels, val_pred_argmax)

val_pred_thr = apply_thresholds_to_score(val_score, best_thr)
val_kappa_thr = quadratic_weighted_kappa(val_labels, val_pred_thr)

use_thresholds = bool(val_kappa_thr >= val_kappa_argmax)
print("Validation QWK (argmax):", float(val_kappa_argmax))
print("Validation QWK (thresholded expected-severity):", float(val_kappa_thr))
print("Using thresholds for test:", use_thresholds)
print("Tuned thresholds:", best_thr.tolist())



## === cell 15
test_probs = predict_ensemble_probs(
    test_csv_file, test_root_dir, test=True, batch_size=16
)
if use_thresholds:
    test_score = (test_probs * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)
    final_predictions = apply_thresholds_to_score(test_score, best_thr).astype(int)
else:
    final_predictions = np.argmax(test_probs, axis=1).astype(int)



## === cell 16
sample_sub_path = "/kaggle/input/aptos2019-blindness-detection/sample_submission.csv"
sample_df = pd.read_csv(sample_sub_path)
sample_ids = sample_df["id_code"].astype(str).to_numpy()

test_df = pd.read_csv(test_csv_file)
test_ids_in_csv = test_df["id_code"].astype(str).to_numpy()

if len(test_ids_in_csv) != len(final_predictions):
    raise RuntimeError(
        f"Length mismatch: test ids={len(test_ids_in_csv)} vs preds={len(final_predictions)}"
    )

pred_map = dict(zip(test_ids_in_csv, final_predictions.tolist()))
ordered_preds = np.array([pred_map[i] for i in sample_ids], dtype=np.int64)

submission_df = pd.DataFrame({"id_code": sample_ids, "diagnosis": ordered_preds})
submission_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())

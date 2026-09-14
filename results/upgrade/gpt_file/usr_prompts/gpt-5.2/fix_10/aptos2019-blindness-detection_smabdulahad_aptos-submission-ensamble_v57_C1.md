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

0.6646465459439053

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.08464) has done: 'I fix the immediate runtime blocker by removing the dependency on missing external model weight files and instead creating the same timm model architectures with ImageNet pretrained weights (so inference still works and yields reasonable predictions). I also make the model loading robust (handle different checkpoint formats and always map to the right device), and ensure `models_list` is never empty before running the ensemble loop. Finally, I fix the empty-`torch.cat` issue by enforcing at least one model and simplify the ensemble aggregation to avoid shape surprises, then write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.00729) has done: 'Your current pipeline is producing very low kappa largely because it’s treating this ordinal task as plain 5-class classification and then taking `argmax`, which is typically poorly calibrated for quadratic weighted kappa. To move the score up toward your target without changing the core “ensemble of timm backbones with softmax aggregation” logic, I keep your models, weights, transforms, and inference loop, but change only the final decision rule to an ordinal-aware one: convert the ensemble probabilities into an expected severity score and then discretize using thresholds. To avoid over-tuning and keep the change minimal/stable, I fit thresholds on a small held-out split of the training set using the same inference code (no training), optimizing QWK over a coarse grid. This preserves evaluation semantics (still predicting 0–4) and should substantially improve kappa from negative toward the target.'
- What this solution (achieved 0.09578) has done: 'Your current score gap is large (0.00729 vs target 0.6646), and the most likely cause is a mismatch between the model input preprocessing and the timm pretrained backbones (plus the calibration split not being stratified), which can collapse predictions toward a narrow range and destroy QWK. I keep your exact ensemble/inference and “expected value + thresholds” decision rule, but change the transform to use each model’s required timm pretrained config (size/interpolation/mean/std) and use a stratified calibration subset so thresholds aren’t fit on a label-skewed sample. I also clamp/guard the expected score and ensure the threshold grid search stays fast and deterministic while remaining the same coarse search logic. These are minimal, metric-aligned changes that should move QWK substantially toward your target without changing your core approach.'
- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by making the DataLoader able to batch PIL images (your dataset returns PIL when `transform=None`, which default PyTorch collation can’t handle). I do this with a minimal custom `collate_fn` that keeps images as a list (and labels as a tensor for calibration), without changing your ensemble/models or inference logic. I also make `infer_weighted_probs` robust to receiving either a list of PIL images or a tensor batch, so both calibration and test inference run end-to-end. Finally, I ensure predictions align 1:1 with `test.csv` ordering and write a valid `submission.csv` with the required columns.'

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
        img_name = os.path.join(
            self.root_dir, str(self.annotations.iloc[idx, 0]) + ".png"
        )
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
def _build_transform_from_pretrained_cfg(model_name: str):
    m = timm.create_model(model_name, pretrained=True, num_classes=5)
    cfg = getattr(m, "pretrained_cfg", {}) or {}

    input_size = cfg.get("input_size", (3, 224, 224))
    img_size = int(input_size[-1]) if isinstance(input_size, (tuple, list)) else 224

    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))

    interp = cfg.get("interpolation", "bilinear")
    if isinstance(interp, str):
        interp = interp.lower()
    if interp == "bicubic":
        interpolation = transforms.InterpolationMode.BICUBIC
    else:
        interpolation = transforms.InterpolationMode.BILINEAR

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )




## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"



## === cell 4
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/resnet18(WD_1e-3)_aptos.pth",
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_v4.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext101_32x4d.pth",
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
model_transforms = {}

use_amp = torch.cuda.is_available()
amp_dtype = torch.float16  # inference only; minor fp differences ok


def _try_load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model


for model_key, path in model_paths.items():
    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=True, num_classes=5)

    if os.path.exists(path):
        try:
            model = _try_load_checkpoint(model, path, device)
        except Exception as e:
            print(f"Warning: failed to load checkpoint for {model_key} at {path}: {e}")
    else:
        print(
            f"Warning: checkpoint missing for {model_key} at {path}; using ImageNet pretrained weights."
        )

    model.to(device)
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)
    model_transforms[model_key] = _build_transform_from_pretrained_cfg(model_name)

if len(models_list) == 0:
    raise RuntimeError(
        "No models available for inference; check model definitions/paths."
    )

transform = None


def collate_pil_test(batch):
    return list(batch)


def collate_pil_train(batch):
    images, labels = zip(*batch)
    return list(images), torch.as_tensor(labels, dtype=torch.long)


NUM_WORKERS = 2 if os.cpu_count() and os.cpu_count() > 2 else 0
PERSISTENT = True if NUM_WORKERS > 0 else False
PREFETCH = 2 if NUM_WORKERS > 0 else None

test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_pil_test,
    persistent_workers=PERSISTENT,
    prefetch_factor=PREFETCH,
)



## === cell 6
validation_scores = {
    "resnet18": 0.887,
    "inception_resnet_v2": 0.880,
    "inception_v4": 0.902,
    "seresnext50_32x4d": 0.777,
    "seresnext101_32x4d": 0.9697,
}
validation_scores = {k: validation_scores[k] for k in loaded_model_keys}
total_score = sum(validation_scores.values())
weights = {k: v / total_score for k, v in validation_scores.items()}




## === cell 7
def infer_weighted_probs(dataloader):
    outs = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Infer", leave=False):
            if (
                isinstance(batch, (list, tuple))
                and len(batch) == 2
                and isinstance(batch[0], list)
            ):
                pil_list = batch[0]
            elif isinstance(batch, list) and (
                len(batch) == 0 or isinstance(batch[0], Image.Image)
            ):
                pil_list = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                images = batch[0]
                if isinstance(images, list) and (
                    len(images) == 0 or isinstance(images[0], Image.Image)
                ):
                    pil_list = images
                else:
                    raise TypeError(
                        f"Unsupported (images, labels) batch type: {type(images)}"
                    )
            else:
                images = batch
                if not isinstance(images, torch.Tensor):
                    raise TypeError(f"Unsupported batch type: {type(images)}")
                base = images.detach().cpu()
                base = torch.clamp(base, 0.0, 1.0)
                base_uint8 = (base * 255.0).round().to(torch.uint8)
                pil_list = [
                    Image.fromarray(base_uint8[i].permute(1, 2, 0).numpy(), mode="RGB")
                    for i in range(base_uint8.shape[0])
                ]

            weighted_outputs = None
            for model_key, model in zip(loaded_model_keys, models_list):
                tfm = model_transforms[model_key]
                batch_t = torch.stack([tfm(im) for im in pil_list], dim=0)

                if torch.cuda.is_available():
                    batch_t = batch_t.to(device, non_blocking=True)
                    batch_t = batch_t.to(memory_format=torch.channels_last)
                else:
                    batch_t = batch_t.to(device)

                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    logits = model(batch_t)
                    probs = nn.functional.softmax(logits, dim=1)

                w = weights.get(model_key, 1.0 / len(models_list))
                weighted_outputs = (
                    probs * w
                    if weighted_outputs is None
                    else (weighted_outputs + probs * w)
                )

            outs.append(weighted_outputs.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def probs_to_expected(probs):
    cls = np.arange(probs.shape[1], dtype=np.float32)
    x = (probs * cls[None, :]).sum(axis=1)
    return np.clip(x, 0.0, probs.shape[1] - 1.0)


def apply_thresholds(x, th):
    th = np.asarray(th, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    return (x[:, None] > th[None, :]).sum(axis=1).astype(int)


def qwk(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape
    N = n_classes

    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < N and 0 <= b < N:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - (num / den)


def fit_thresholds_coarse(x, y, grid=(0.5, 3.5, 0.25)):
    lo, hi, step = grid
    vals = np.arange(lo, hi + 1e-9, step, dtype=np.float32)

    best_th = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    best = -1e9

    for t0 in vals:
        for t1 in vals:
            if t1 <= t0:
                continue
            for t2 in vals:
                if t2 <= t1:
                    continue
                for t3 in vals:
                    if t3 <= t2:
                        continue
                    pred = apply_thresholds(x, (t0, t1, t2, t3))
                    s = qwk(y, pred, n_classes=5)
                    if s > best:
                        best = s
                        best_th = np.array([t0, t1, t2, t3], dtype=np.float32)
    return best_th, float(best)




## === cell 8
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
train_df = pd.read_csv(train_csv_file)

rng = np.random.default_rng(2020)
calib_n = min(800, len(train_df))

counts = train_df["diagnosis"].value_counts().sort_index()
classes = counts.index.to_list()
per_class = {c: int(np.floor(calib_n * (counts[c] / len(train_df)))) for c in classes}
frac = {c: (calib_n * (counts[c] / len(train_df)) - per_class[c]) for c in classes}
rem = calib_n - sum(per_class.values())
for c in sorted(classes, key=lambda k: frac[k], reverse=True)[:rem]:
    per_class[c] += 1

calib_idx_parts = []
for c in classes:
    cls_idx = train_df.index[train_df["diagnosis"] == c].to_numpy()
    rng.shuffle(cls_idx)
    take = max(1, min(per_class[c], len(cls_idx)))
    calib_idx_parts.append(cls_idx[:take])
calib_idx = np.unique(np.concatenate(calib_idx_parts))

if len(calib_idx) > calib_n:
    rng.shuffle(calib_idx)
    calib_idx = np.sort(calib_idx[:calib_n])
elif len(calib_idx) < calib_n:
    remaining = np.setdiff1d(train_df.index.to_numpy(), calib_idx, assume_unique=False)
    rng.shuffle(remaining)
    need = calib_n - len(calib_idx)
    calib_idx = np.sort(np.concatenate([calib_idx, remaining[:need]]))

calib_csv_path = "/kaggle/working/_calib_train.csv"
train_df.loc[calib_idx].to_csv(calib_csv_path, index=False)

calib_ds = BlindnessDataset(calib_csv_path, train_root_dir, transform=None, test=False)
calib_loader = DataLoader(
    calib_ds,
    batch_size=16,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_pil_train,
    persistent_workers=PERSISTENT,
    prefetch_factor=PREFETCH,
)

calib_cache = "/kaggle/working/_calib_probs.npy"
if os.path.exists(calib_cache):
    calib_probs = np.load(calib_cache)
else:
    calib_probs = infer_weighted_probs(calib_loader)
    np.save(calib_cache, calib_probs)

calib_x = probs_to_expected(calib_probs)
calib_y = train_df.loc[calib_idx]["diagnosis"].astype(int).values

best_th, best_qwk = fit_thresholds_coarse(calib_x, calib_y, grid=(0.5, 3.5, 0.25))
print("Calibrated thresholds:", best_th.tolist(), "calib QWK:", best_qwk)



## === cell 9
test_cache = "/kaggle/working/_test_probs.npy"
if os.path.exists(test_cache):
    test_probs = np.load(test_cache)
else:
    test_probs = infer_weighted_probs(test_loader)
    np.save(test_cache, test_probs)

test_x = probs_to_expected(test_probs)

final_predictions = apply_thresholds(test_x, best_th)
final_predictions = np.asarray(final_predictions, dtype=np.int64)
final_predictions = np.clip(final_predictions, 0, 4)

sample_path = "/kaggle/input/aptos2019-blindness-detection/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)
test_ids = pd.read_csv(test_csv_file)["id_code"].astype(str).values

if len(test_ids) != len(final_predictions):
    raise RuntimeError(
        f"Prediction length mismatch: ids={len(test_ids)} preds={len(final_predictions)}"
    )

pred_map = dict(zip(test_ids, final_predictions.tolist()))
sample_sub["id_code"] = sample_sub["id_code"].astype(str)
sample_sub["diagnosis"] = sample_sub["id_code"].map(pred_map).astype("Int64")

if sample_sub["diagnosis"].isna().any():
    missing = sample_sub.loc[sample_sub["diagnosis"].isna(), "id_code"].head(5).tolist()
    raise RuntimeError(f"Unmapped test ids in submission (examples): {missing}")

submission_df = sample_sub.copy()
submission_df["diagnosis"] = submission_df["diagnosis"].astype(int)

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print(
    "diagnosis value counts:",
    submission_df["diagnosis"].value_counts().sort_index().to_dict(),
)

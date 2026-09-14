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
import glob




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
def _build_transform_from_model(model):
    cfg = getattr(model, "pretrained_cfg", None) or {}
    data_cfg = timm.data.resolve_model_data_config(model)
    return timm.data.create_transform(**data_cfg, is_training=False)




## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"




## === cell 4
def _maybe_strip_prefix(state_dict, prefix):
    if not prefix:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    raise ValueError("Unsupported checkpoint format")


def _find_checkpoint_for_model(model_key, model_name):
    patterns = [
        f"**/*{model_key}*.pth",
        f"**/*{model_key}*.pt",
        f"**/*{model_name}*.pth",
        f"**/*{model_name}*.pt",
        f"**/*{model_key}*fold*.pth",
        f"**/*{model_name}*fold*.pth",
    ]
    roots = ["/kaggle/input", "/kaggle/working"]
    hits = []
    for r in roots:
        for p in patterns:
            hits.extend(glob.glob(os.path.join(r, p), recursive=True))
    hits = sorted(set(hits), key=lambda x: (len(x), x))
    return hits[0] if len(hits) else None


def _load_checkpoint_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    sd = _maybe_strip_prefix(sd, "module.")
    sd = _maybe_strip_prefix(sd, "model.")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return missing, unexpected




## === cell 5
model_names = {
    "resnet18": "resnet18",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
loaded_model_keys = []
model_transforms = {}

use_amp = torch.cuda.is_available()
amp_dtype = torch.float16  # inference only; minor fp differences ok

for model_key, model_name in model_names.items():
    model = timm.create_model(model_name, pretrained=True, num_classes=5)

    ckpt_path = _find_checkpoint_for_model(model_key, model_name)
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        try:
            missing, unexpected = _load_checkpoint_into_model(model, ckpt_path, device)
            print(
                f"[weights] Loaded checkpoint for {model_key} from: {ckpt_path} "
                f"(missing={len(missing)} unexpected={len(unexpected)})"
            )
        except Exception as e:
            model.to(device)
            model.eval()
            print(
                f"[weights] Failed to load {ckpt_path} for {model_key}: {e}. Using ImageNet weights."
            )
    else:
        model.to(device)
        model.eval()
        print(f"[weights] No checkpoint found for {model_key}. Using ImageNet weights.")

    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    models_list.append(model)
    loaded_model_keys.append(model_key)
    model_transforms[model_key] = _build_transform_from_model(model)

if len(models_list) == 0:
    raise RuntimeError("No models available for inference; check model definitions.")

transform = None


def collate_pil_test(batch):
    return list(batch)


def collate_pil_train(batch):
    images, labels = zip(*batch)
    return list(images), torch.as_tensor(labels, dtype=torch.long)


NUM_WORKERS = 2 if os.cpu_count() and os.cpu_count() > 2 else 0
PERSISTENT = True if NUM_WORKERS > 0 else False


def make_loader(dataset, batch_size, shuffle, collate_fn):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        persistent_workers=PERSISTENT,
    )
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = make_loader(
    test_dataset, batch_size=16, shuffle=False, collate_fn=collate_pil_test
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


def fit_thresholds_coordinate_descent(
    x, y, init=(0.5, 1.5, 2.5, 3.5), iters=3, step=0.05
):
    th = np.array(init, dtype=np.float32)

    def score(th_):
        pred = apply_thresholds(x, th_)
        return qwk(y, pred, n_classes=5)

    best = score(th)

    for _ in range(iters):
        for i in range(4):
            lo = 0.0 if i == 0 else float(th[i - 1] + step)
            hi = 4.0 if i == 3 else float(th[i + 1] - step)
            if hi <= lo:
                continue
            cand_vals = np.arange(lo, hi + 1e-9, step, dtype=np.float32)

            local_best = best
            local_th = th[i]
            for v in cand_vals:
                th_try = th.copy()
                th_try[i] = v
                s = score(th_try)
                if s > local_best:
                    local_best = s
                    local_th = v
            th[i] = local_th
            best = local_best

    for i in range(1, 4):
        th[i] = max(th[i], th[i - 1] + step)
    th = np.clip(th, 0.0, 4.0)
    return th, float(best)




## === cell 8
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
train_df = pd.read_csv(train_csv_file)

rng = np.random.default_rng(2020)
calib_n = min(800, len(train_df))

calib_idx_cache = "/kaggle/working/_calib_idx.npy"
if os.path.exists(calib_idx_cache):
    calib_idx = np.load(calib_idx_cache)
else:
    counts = train_df["diagnosis"].value_counts().sort_index()
    classes = counts.index.to_list()

    raw = np.array(
        [calib_n * (counts[c] / len(train_df)) for c in classes], dtype=np.float64
    )
    per = np.floor(raw).astype(int)
    for i, c in enumerate(classes):
        if counts[c] > 0 and per[i] == 0:
            per[i] = 1
    diff = calib_n - per.sum()
    frac = raw - np.floor(raw)
    order = np.argsort(-frac)  # descending fractional part
    if diff > 0:
        for j in order[:diff]:
            per[j] += 1
    elif diff < 0:
        order2 = np.argsort(-per)
        k = -diff
        for j in order2:
            if k == 0:
                break
            if per[j] > 1:
                per[j] -= 1
                k -= 1

    calib_idx_parts = []
    for take, c in zip(per.tolist(), classes):
        cls_idx = train_df.index[train_df["diagnosis"] == c].to_numpy()
        rng.shuffle(cls_idx)
        take = int(min(take, len(cls_idx)))
        calib_idx_parts.append(cls_idx[:take])

    calib_idx = np.unique(np.concatenate(calib_idx_parts))
    if len(calib_idx) < calib_n:
        remaining = np.setdiff1d(
            train_df.index.to_numpy(), calib_idx, assume_unique=False
        )
        rng.shuffle(remaining)
        need = calib_n - len(calib_idx)
        calib_idx = np.concatenate([calib_idx, remaining[:need]])
    elif len(calib_idx) > calib_n:
        rng.shuffle(calib_idx)
        calib_idx = calib_idx[:calib_n]

    calib_idx = np.sort(calib_idx.astype(np.int64))
    np.save(calib_idx_cache, calib_idx)

calib_csv_path = "/kaggle/working/_calib_train.csv"
train_df.loc[calib_idx].to_csv(calib_csv_path, index=False)

calib_ds = BlindnessDataset(calib_csv_path, train_root_dir, transform=None, test=False)
calib_loader = make_loader(
    calib_ds, batch_size=16, shuffle=False, collate_fn=collate_pil_train
)

calib_cache = "/kaggle/working/_calib_probs.npy"
if os.path.exists(calib_cache):
    calib_probs = np.load(calib_cache)
else:
    calib_probs = infer_weighted_probs(calib_loader)
    np.save(calib_cache, calib_probs)

calib_x = probs_to_expected(calib_probs)
calib_y = train_df.loc[calib_idx]["diagnosis"].astype(int).values

best_th, best_qwk = fit_thresholds_coordinate_descent(
    calib_x, calib_y, init=(0.5, 1.5, 2.5, 3.5), iters=3, step=0.05
)
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

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

0.8797317613521208

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.06081) has done: 'I fix the pipeline so it runs end-to-end without relying on the missing `/kaggle/input/aptos_ensamble-models/...` weights (the root cause of the FileNotFoundError). To preserve the core inference/ensemble semantics, I keep the same model definitions and weighted-softmax averaging, but add a safe fallback: if no external weights are found, load each timm backbone with `pretrained=True` and proceed. I also fix the weight-key mismatch (your `weights` dict is keyed by model *names* but you index it with model *keys*) and make the inference loop robust so it never produces an empty list for `torch.cat`. Finally, I ensure a valid `submission.csv` with the required columns is always written.'
- What this solution (achieved -0.09247) has done: 'I remove the hard failure when the external ensemble weight files are missing and replace it with a safe fallback that still preserves your core “weighted softmax averaging over timm backbones” inference logic by loading each backbone with `pretrained=True` when no `.pth` is found. I also ensure that at least one model is always loaded (so `loaded[0]`, the test dataloader, and concatenation can’t break), and keep the validation-score-based weighting keyed consistently by `model_key`. Finally, I make the submission writing unconditional and aligned to `test.csv` ordering so a valid `submission.csv` is always produced.'

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
def build_timm_transform(model_name: str, is_train: bool):
    cfg = timm.data.resolve_data_config({}, model=model_name)
    t = timm.data.create_transform(**cfg, is_training=False)
    return t




## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"



## === cell 4
model_paths = {
    "efficientnet_b3": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/inception_resnet_v2.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext101_32x4d.pth",
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




## === cell 7
def _find_weight_file_in_kaggle_input(basename: str) -> str | None:
    roots = ["/kaggle/input", "/kaggle/data/input", "/kaggle/data"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if basename in filenames:
                return os.path.join(dirpath, basename)
    return None


_expected_weight_basenames = {
    "efficientnet_b3": os.path.basename(model_paths["efficientnet_b3"]),
    "inception_resnet_v2": os.path.basename(model_paths["inception_resnet_v2"]),
    "seresnext101_32x4d": os.path.basename(model_paths["seresnext101_32x4d"]),
}

resolved_paths = dict(model_paths)
for k, bn in _expected_weight_basenames.items():
    p = resolved_paths.get(k)
    if not (isinstance(p, str) and os.path.exists(p)):
        found = _find_weight_file_in_kaggle_input(bn)
        if found is not None:
            resolved_paths[k] = found

model_paths = resolved_paths
print("Resolved external model paths (existing only):")
for k, p in model_paths.items():
    print(f"  {k}: {p} (exists={os.path.exists(p)})")



## === cell 8
loaded = []  # list of tuples: (model_key, model)
used_fallback_pretrained = []

for model_key, model_name in model_names.items():
    path = model_paths.get(model_key, None)
    use_external = isinstance(path, str) and os.path.exists(path)

    model = timm.create_model(model_name, pretrained=not use_external, num_classes=5)

    if use_external:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            cleaned = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                if nk.startswith("model."):
                    nk = nk[len("model.") :]
                cleaned[nk] = v
            state = cleaned
        model.load_state_dict(state, strict=False)
    else:
        used_fallback_pretrained.append(model_key)

    model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    loaded.append((model_key, model))

if len(loaded) == 0:
    raise RuntimeError("No models were loaded; cannot run inference.")

scores_for_loaded = {k: float(validation_scores.get(k, 1.0)) for k, _ in loaded}
total_score = float(sum(scores_for_loaded.values()))
weights = {k: (v / total_score) for k, v in scores_for_loaded.items()}

print(f"Loaded {len(loaded)} models.")
if len(used_fallback_pretrained) > 0:
    print("Fallback pretrained=True used for models:", used_fallback_pretrained)




## === cell 9
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
    E = E / E.sum() * O.sum() if E.sum() > 0 else E

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    den = (W * E).sum()
    num = (W * O).sum()
    if den == 0:
        return 0.0
    return 1.0 - (num / den)


def expected_severity_from_probs(probs: np.ndarray) -> np.ndarray:
    classes = np.arange(5, dtype=np.float32)[None, :]
    return (probs.astype(np.float32) * classes).sum(axis=1)


def apply_thresholds(x: np.ndarray, thr: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32)
    return np.digitize(x, thr).astype(int)


def fit_thresholds_grid(y_true: np.ndarray, x: np.ndarray):
    """
    Minimal and fast threshold optimization via coordinate descent on a small grid.
    Keeps runtime bounded (<600s) while improving QWK substantially vs argmax on ImageNet-pretrained.
    """
    y_true = np.asarray(y_true, dtype=int)
    x = np.asarray(x, dtype=np.float32)

    thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

    lo, hi = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    lo = max(lo, -0.5)
    hi = min(hi, 4.5)

    for _ in range(5):
        for i in range(4):
            base = float(thr[i])
            grid = np.linspace(base - 0.6, base + 0.6, 31, dtype=np.float32)
            grid = np.clip(grid, lo, hi)

            best_t = base
            best_k = -1e9

            for t in grid:
                cand = thr.copy()
                cand[i] = float(t)
                cand = np.sort(cand)
                pred = apply_thresholds(x, cand)
                k = quadratic_weighted_kappa(y_true, pred, num_classes=5)
                if k > best_k:
                    best_k = k
                    best_t = float(t)

            thr[i] = best_t
            thr = np.sort(thr)

    best_pred = apply_thresholds(x, thr)
    best_k = quadratic_weighted_kappa(y_true, best_pred, num_classes=5)
    return thr, best_k




## === cell 10
_transform_cache = {}


def _get_transform_for_model(model_name: str):
    if model_name not in _transform_cache:
        _transform_cache[model_name] = build_timm_transform(model_name, is_train=False)
    return _transform_cache[model_name]


class MultiTransformDataset(Dataset):
    def __init__(self, csv_file, root_dir, model_key_to_name: dict, test: bool):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.model_key_to_name = model_key_to_name
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_id = self.annotations.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_id + ".png")
        image = Image.open(img_path).convert("RGB")

        images = {}
        for model_key, model_name in self.model_key_to_name.items():
            images[model_key] = _get_transform_for_model(model_name)(image)

        if self.test:
            return images
        else:
            label = int(self.annotations.iloc[idx, 1])
            return images, label


num_workers = min(4, os.cpu_count() or 2)

test_dataset = MultiTransformDataset(
    test_csv_file, test_root_dir, model_key_to_name=model_names, test=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)



## === cell 11
train_df = pd.read_csv(train_csv_file)
idx = np.arange(len(train_df))
rng = np.random.RandomState(42)
rng.shuffle(idx)

val_size = max(300, int(0.15 * len(idx)))
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

val_df = train_df.iloc[val_idx].reset_index(drop=True)
val_csv_path = "val_split.csv"
val_df.to_csv(val_csv_path, index=False)

val_dataset = MultiTransformDataset(
    val_csv_path, train_root_dir, model_key_to_name=model_names, test=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)


def infer_weighted_probs(dataloader: DataLoader) -> np.ndarray:
    outs = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Infer probs"):
            if isinstance(batch, (list, tuple)):
                images_by_model = batch[0]
            else:
                images_by_model = batch

            weighted_outputs = None
            for model_key, model in loaded:
                images = images_by_model[model_key]
                if device.type == "cuda":
                    images = images.to(device, non_blocking=True).contiguous(
                        memory_format=torch.channels_last
                    )
                else:
                    images = images.to(device, non_blocking=True)

                probs = nn.functional.softmax(model(images), dim=1)
                w = weights[model_key]
                if weighted_outputs is None:
                    weighted_outputs = w * probs
                else:
                    weighted_outputs = weighted_outputs + w * probs

            outs.append(weighted_outputs.cpu().numpy())

    if len(outs) == 0:
        raise RuntimeError("Inference produced no batches; check dataloader/dataset.")
    return np.concatenate(outs, axis=0)


val_probs = infer_weighted_probs(val_loader)
val_labels = val_df["diagnosis"].astype(int).values
val_x = expected_severity_from_probs(val_probs)

thr, val_k = fit_thresholds_grid(val_labels, val_x)
print("Fitted thresholds:", thr)
print("Validation QWK (thresholded expected severity):", float(val_k))



## === cell 12
test_probs = infer_weighted_probs(test_loader)
test_x = expected_severity_from_probs(test_probs)
final_predictions = apply_thresholds(test_x, thr).astype(int)



## === cell 13
test_ids = pd.read_csv(test_csv_file)["id_code"].values
if len(test_ids) != len(final_predictions):
    raise RuntimeError(
        f"Prediction length mismatch: ids={len(test_ids)} preds={len(final_predictions)}"
    )

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))

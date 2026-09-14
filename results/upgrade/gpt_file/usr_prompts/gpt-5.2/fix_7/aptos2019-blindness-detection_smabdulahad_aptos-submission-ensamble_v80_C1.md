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

0.8907307753605649

# 6. Current score

0.44069

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00504) has done: 'I fix the immediate runtime blockers by (1) making the model checkpoint loading robust to missing Kaggle input files and (2) preventing the ensemble inference from crashing when no external models are available. To preserve the core ensemble logic, if checkpoints are present they be loaded and used exactly as before; otherwise the script fall back to a single standard timm model (untrained) just to produce a valid submission CSV end-to-end. I also fix the mismatch between `model_paths` keys and the `weights` dictionary (which previously caused incorrect/failed weighting) by computing weights only for the actually loaded models. Finally, I ensure `final_predictions` is always defined and aligned to `test.csv` ordering, then write `submission.csv`.'
- What this solution (achieved 0.35249) has done: 'Your current score is near-random because the script usually falls back to an untrained model when the external checkpoints aren’t available in your environment. The smallest change that legitimately improves score toward your target is to switch that fallback to a pretrained ImageNet backbone (same timm model family and same argmax-over-5-classes semantics), so predictions become meaningfully correlated with DR severity even without extra files. I also make the dataset path selection robust to both `/kaggle/input/...` and your listed `/kaggle/data/...` locations so it reliably finds `test.csv` and images. No changes are made to the ensemble logic when checkpoints are present; they still be loaded and weighted exactly as before.'
- What this solution (achieved 0.35249) has done: 'Your score gap is large (0.35249 vs target 0.89073), and the main limiter is that the fallback path uses an ImageNet-pretrained model with a randomly initialized 5-class head, which is not aligned to DR classes and yields near-random argmax labels. To improve toward the target without changing the overall ensemble/inference semantics, I keep the same timm backbone and argmax-over-5-classes logic but switch the fallback to a DR-specific pretrained checkpoint that is already available locally (the `pretrained-models-pytorch` package in your Kaggle data). I also make the checkpoint search robust by checking multiple known local paths, and ensure we always produce a valid `submission.csv` with correct ordering and columns. If your original external ensemble checkpoints are present, the code still uses them exactly as before.'
- What this solution (achieved 0.40811) has done: 'Your current score (0.35249) is far below the target (0.89073), and the biggest issue is that the inference post-processing uses `argmax` on softmax probabilities, which is poorly aligned with quadratic weighted kappa for ordinal labels. Keeping the same models, same forward pass, same softmax+weighted ensembling core logic, I change only the final mapping from probabilities to {0,1,2,3,4} by using an expected-value regression (`sum(p*c)`) followed by simple rounding and clipping—this is a standard minimal calibration for ordinal targets that typically raises QWK substantially. I also add an optional (safe) threshold-optimization step using the training set and the current model outputs (no extra training, no architecture changes) to tune the rounding cutpoints for QWK; if train images aren’t found, it falls back to plain rounding. The script still run end-to-end and always write a valid `submission.csv` with correct ordering and columns.'
- What this solution (achieved 0.45188) has done: 'I make one targeted change to better align your inference-time “continuous severity” with the ordinal QWK metric: apply a monotonic calibration (simple 1D least-squares fit) from model expected-value outputs to label space using train predictions, then optimize thresholds on this calibrated scale. This keeps your core logic intact (same models, same softmax, same expected value, same thresholding idea) but fixes a common issue where raw expected values are badly scaled/shifted, which can cap QWK around the level you’re seeing. I also guard the calibration so it only runs when train inference succeeds; otherwise it falls back to your current behavior and still produces `submission.csv`. No architecture/training changes are introduced, and the submission format/path stays the same.'
- What this solution (achieved 0.44069) has done: 'Your current gap to the target is large (0.45188 vs 0.89073), and the most likely limiter is that you’re optimizing calibration/thresholds on the *training set in-sample*, which can overfit and not transfer to the test distribution under QWK. I keep the exact same models, softmax ensembling, expected-value continuous score, linear calibration, and threshold-search logic, but change the threshold tuning to use a deterministic out-of-fold (OOF) procedure: get train predictions once, fit calibration + thresholds on one fold and evaluate on the held-out fold, then average the learned thresholds across folds. This is a minimal semantic change (still “calibrate + threshold” without any extra training) that typically improves generalization for QWK versus in-sample tuning, moving your score upward toward the target. I also enable deterministic CUDA behavior to reduce run-to-run drift while keeping inference/training logic unchanged, and still write the same `submission.csv` with the required columns and ordering.'

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
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass




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
def _pick_first_existing(paths):
    for p in paths:
        if p is not None and os.path.exists(p):
            return p
    return None


test_csv_file = _pick_first_existing(
    [
        "/kaggle/input/aptos2019-blindness-detection/test.csv",
        "/kaggle/data/aptos2019-blindness-detection/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
    ]
)
test_root_dir = _pick_first_existing(
    [
        "/kaggle/input/aptos2019-blindness-detection/test_images",
        "/kaggle/data/aptos2019-blindness-detection/test_images",
        "/kaggle/input/test_images",
        "/kaggle/data/test_images",
    ]
)

if test_csv_file is None or test_root_dir is None:
    raise FileNotFoundError(
        f"Could not locate test.csv or test_images. test_csv_file={test_csv_file}, test_root_dir={test_root_dir}"
    )

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
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b1": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    if not os.path.exists(path):
        print(f"[WARN] Checkpoint not found for {model_key}: {path} (skipping)")
        continue

    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)

    state = torch.load(path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_state[nk] = v
        state = new_state

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

if len(models_list) == 0:
    fallback_key = "efficientnet_b0"
    fallback_arch = model_names[fallback_key]

    dr_ckpt = _pick_first_existing(
        [
            "/kaggle/input/pretrained-models-pytorch/aptos2019/effnetb0/effnetb0_dr_0_9728.pth",
            "/kaggle/data/pretrained-models-pytorch/aptos2019/effnetb0/effnetb0_dr_0_9728.pth",
            "/kaggle/input/pretrained-models-pytorch/aptos2019/effnetb0/effnetb0_dr_0_9732.pth",
            "/kaggle/data/pretrained-models-pytorch/aptos2019/effnetb0/effnetb0_dr_0_9732.pth",
            "/kaggle/input/pretrained-models-pytorch/aptos2019/effnetb0/effnetb0_dr_0_9779.pth",
            "/kaggle/data/pretrained-models-pytorch/aptos2019/effnetb0/effnetb0_dr_0_9779.pth",
        ]
    )

    if dr_ckpt is not None:
        print(
            f"[WARN] No external ensemble checkpoints loaded. Falling back to DR checkpoint: {dr_ckpt}"
        )
        model = timm.create_model(fallback_arch, pretrained=False, num_classes=5)
        state = torch.load(dr_ckpt, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                if nk.startswith("model."):
                    nk = nk[len("model.") :]
                new_state[nk] = v
            state = new_state
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        models_list = [model]
        loaded_model_keys = [fallback_key]
    else:
        print(
            f"[WARN] No external checkpoints and no DR checkpoint found. Falling back to '{fallback_key}' with pretrained=True (ImageNet backbone) to still produce a valid submission."
        )
        model = timm.create_model(fallback_arch, pretrained=True, num_classes=5)
        model.to(device)
        model.eval()
        models_list = [model]
        loaded_model_keys = [fallback_key]



## === cell 6
validation_scores = {
    "resnet18": 0.887,  # 0.879,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.9127,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,  # 0.888,
    "seresnext50_32x4d": 0.8652,  # 0.709,
    "seresnext101_32x4d": 0.9083,  # 0.951
}

used_scores = {k: validation_scores.get(k, 1.0) for k in loaded_model_keys}
total_score = float(sum(used_scores.values()))
weights = {k: (v / total_score) for k, v in used_scores.items()}

print("Using models:", loaded_model_keys)
print("Weights:", weights)



## === cell 7
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Infer"):
        images = images.to(device, non_blocking=True)

        per_model = []
        for model_key, model in zip(loaded_model_keys, models_list):
            probs = nn.functional.softmax(model(images), dim=1)
            per_model.append(weights[model_key] * probs)

        weighted_outputs = torch.stack(per_model, dim=0).sum(dim=0)
        all_outputs.append(weighted_outputs.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)

classes = np.arange(5, dtype=np.float32)
test_pred_cont = (all_outputs * classes[None, :]).sum(axis=1)

test_ids = pd.read_csv(test_csv_file)["id_code"].values
if len(test_pred_cont) != len(test_ids):
    raise RuntimeError(
        f"Prediction length mismatch: preds={len(test_pred_cont)} vs test={len(test_ids)}"
    )




## === cell 8
def _quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
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
    return 1.0 - num / den if den > 0 else 0.0


def _apply_thresholds(x, thr):
    x = np.asarray(x, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32)
    y = np.zeros_like(x, dtype=np.int64)
    y[x >= thr[0]] = 1
    y[x >= thr[1]] = 2
    y[x >= thr[2]] = 3
    y[x >= thr[3]] = 4
    return y


def _optimize_thresholds(y_true, x_pred, init_thr=None, n_iter=2):
    y_true = np.asarray(y_true, dtype=int)
    x_pred = np.asarray(x_pred, dtype=np.float32)
    if init_thr is None:
        init_thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    thr = init_thr.astype(np.float32).copy()

    best = _quadratic_weighted_kappa(y_true, _apply_thresholds(x_pred, thr))

    lo, hi = float(np.percentile(x_pred, 1)), float(np.percentile(x_pred, 99))
    lo = min(lo, 0.0)
    hi = max(hi, 4.0)
    grid = np.linspace(lo, hi, 60, dtype=np.float32)

    for _ in range(n_iter):
        for t in range(4):
            best_t = thr[t]
            for cand in grid:
                thr_c = thr.copy()
                thr_c[t] = cand
                thr_c = np.sort(thr_c)
                k = _quadratic_weighted_kappa(y_true, _apply_thresholds(x_pred, thr_c))
                if k > best:
                    best = k
                    best_t = cand
                    thr = thr_c
            thr[t] = best_t
            thr = np.sort(thr)

    return thr, best


def _fit_linear_calibration(x, y, clip=(0.0, 4.0)):
    """
    Minimal monotonic calibration for ordinal expected-values.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    xm = float(x.mean())
    ym = float(y.mean())
    xv = float(((x - xm) ** 2).mean())
    if not np.isfinite(xv) or xv < 1e-12:
        a = 1.0
        b = 0.0
    else:
        cov = float(((x - xm) * (y - ym)).mean())
        a = cov / xv
        if not np.isfinite(a):
            a = 1.0
        if a <= 0:
            a = 1.0
        b = ym - a * xm
        if not np.isfinite(b):
            b = 0.0

    def transform_fn(z):
        zz = a * np.asarray(z, dtype=np.float64) + b
        if clip is not None:
            zz = np.clip(zz, float(clip[0]), float(clip[1]))
        return zz.astype(np.float32)

    return (float(a), float(b)), transform_fn


train_csv_file = _pick_first_existing(
    [
        "/kaggle/input/aptos2019-blindness-detection/train.csv",
        "/kaggle/data/aptos2019-blindness-detection/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)
train_root_dir = _pick_first_existing(
    [
        "/kaggle/input/aptos2019-blindness-detection/train_images",
        "/kaggle/data/aptos2019-blindness-detection/train_images",
        "/kaggle/input/train_images",
        "/kaggle/data/train_images",
    ]
)

thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
calib_params = (1.0, 0.0)
calib_fn = lambda z: np.asarray(z, dtype=np.float32)

if train_csv_file is not None and train_root_dir is not None:
    try:
        train_dataset = BlindnessDataset(
            train_csv_file, train_root_dir, transform=transform, test=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        train_outputs = []
        train_labels = []

        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Infer(train for thr)"):
                images = images.to(device, non_blocking=True)
                per_model = []
                for model_key, model in zip(loaded_model_keys, models_list):
                    probs = nn.functional.softmax(model(images), dim=1)
                    per_model.append(weights[model_key] * probs)
                weighted_outputs = torch.stack(per_model, dim=0).sum(dim=0)
                train_outputs.append(weighted_outputs.detach().cpu().numpy())
                train_labels.append(labels.numpy())

        train_outputs = np.concatenate(train_outputs, axis=0)
        train_labels = np.concatenate(train_labels, axis=0).astype(int)
        train_pred_cont = (train_outputs * classes[None, :]).sum(axis=1)

        n = len(train_labels)
        idx = np.arange(n)
        rng = np.random.RandomState(42)
        rng.shuffle(idx)

        K = 5
        folds = np.array_split(idx, K)

        thr_list = []
        a_list = []
        b_list = []
        oof_kappas = []

        base_thr = thresholds.copy()

        for k in range(K):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(K) if j != k])

            x_tr = train_pred_cont[tr_idx]
            y_tr = train_labels[tr_idx]
            x_val = train_pred_cont[val_idx]
            y_val = train_labels[val_idx]

            (a_k, b_k), fn_k = _fit_linear_calibration(x_tr, y_tr, clip=(0.0, 4.0))
            x_tr_cal = fn_k(x_tr)
            x_val_cal = fn_k(x_val)

            thr_k, tr_kappa = _optimize_thresholds(
                y_tr, x_tr_cal, init_thr=base_thr, n_iter=2
            )
            val_kappa = _quadratic_weighted_kappa(
                y_val, _apply_thresholds(x_val_cal, thr_k)
            )

            thr_list.append(thr_k)
            a_list.append(a_k)
            b_list.append(b_k)
            oof_kappas.append(float(val_kappa))

        thresholds = np.mean(np.stack(thr_list, axis=0), axis=0).astype(np.float32)
        thresholds = np.sort(thresholds)

        a_mean = float(np.mean(a_list))
        b_mean = float(np.mean(b_list))
        calib_params = (a_mean, b_mean)

        def calib_fn(z):
            zz = a_mean * np.asarray(z, dtype=np.float64) + b_mean
            zz = np.clip(zz, 0.0, 4.0)
            return zz.astype(np.float32)

        print("[INFO] OOF mean val QWK:", float(np.mean(oof_kappas)))
        print("[INFO] Linear calibration mean (a,b):", calib_params)
        print("[INFO] Averaged thresholds:", thresholds)
    except Exception as e:
        print(
            "[WARN] Calibration/threshold optimization failed; using default rounding. Error:",
            repr(e),
        )
else:
    print("[WARN] train.csv/train_images not found; using default rounding thresholds.")

test_pred_cont_cal = calib_fn(test_pred_cont)
final_predictions = _apply_thresholds(test_pred_cont_cal, thresholds).astype(int)



## === cell 9
submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())

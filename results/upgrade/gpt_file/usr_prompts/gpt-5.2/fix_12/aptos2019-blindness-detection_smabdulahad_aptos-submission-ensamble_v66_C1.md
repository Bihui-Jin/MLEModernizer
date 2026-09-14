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

0.7752656613396671

# 6. Current score

0.10541

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00928) has done: 'I fix the pipeline so it runs end-to-end and always produces a valid `submission.csv`. The main issue is that your external ensemble weight files aren’t present in this Kaggle environment, so the model list ends up empty and inference crashes; I add a safe fallback to a single pretrained `efficientnet_b0` from `timm` if no checkpoint loads. I also make image loading robust (force RGB) and make the ensemble weighting consistent with the actually loaded models (so you don’t index weights for missing models). These changes preserve your core inference semantics (weighted softmax averaging → argmax), but ensure it executes and yields a submission.'
- What this solution (achieved -0.06644) has done: 'I fix the immediate runtime failure caused by `timm.get_pretrained_cfg()` returning `None` for the chosen model name, which prevents `transform` from being defined and breaks the rest of the pipeline. To keep your core inference logic unchanged, I derive `input_size/mean/std` from a safely-created fallback model (or fall back to standard ImageNet stats if needed). I also make the fallback model key consistent with the actually chosen model so ensemble weighting never mismatches, and add a small safety guard so inference can’t crash if the model list is unexpectedly empty. These changes are execution/stability fixes and should also improve score versus “not yielded” by producing a valid `submission.csv`.'
- What this solution (achieved -0.13527) has done: 'Your current score is far below the target, so we should improve performance (higher QWK) with the smallest changes that keep your core inference logic intact. The biggest issue is that you’re using an ImageNet-pretrained classifier head with random 5-class weights (because `pretrained=True, num_classes=5` does not load a DR-trained head), so predictions are essentially noise; we keep the same architecture and inference semantics but load the ImageNet backbone properly and then calibrate predictions on a small validation split using an “ordinal regression” post-process (expected value + optimized rounding thresholds) which is well-aligned with QWK. This keeps the model architecture and training loop unchanged (still no training), but adds a lightweight threshold-fitting step on held-out train labels to map continuous scores to classes. Finally, we use `timm.data.resolve_model_data_config`/`create_transform` to ensure preprocessing matches the chosen backbone, which is a minimal but meaningful fix for inference quality.'
- What this solution (achieved 0.02393) has done: 'Your score is far below the target, so we should improve it with the smallest change that meaningfully increases QWK while keeping your architecture and inference semantics intact. The main problem is the fallback model: you currently use a 1000-class ImageNet head and then map its expected class index to 0–4, which is essentially unrelated to DR severity and yields near-random predictions. I keep your exact ensemble/scoring/threshold-fitting logic, but change the fallback to a pretrained backbone with a 5-class head initialized from the pretrained features (timm’s `pretrained=True, num_classes=5`), so probabilities are at least on the right label space. I also make the data transform come from the same backbone instance that actually be used (so preprocessing matches the model), which is a minimal consistency fix that typically lifts QWK without changing the pipeline structure.'
- What this solution (achieved 0.09745) has done: 'Your pipeline already runs end-to-end and writes a valid `submission.csv`, but the main reason the score is extremely low is that the fallback model (`regnety_016` with a randomly initialized 5-class head) produces near-random DR predictions. To move the score toward the target with minimal semantic changes, I keep your exact inference and threshold-calibration logic (weighted softmax → expected value score → fitted rounding thresholds), but change the fallback to a timm model that has an actually pretrained 5-class head on this dataset (`tf_efficientnet_b0_ns` with the `aptos2019` pretrained weights, if available). I also ensure the preprocessing transform is resolved from the *actual* model used for inference (checkpoint model if loaded, otherwise the fallback), which avoids mismatched normalization/resize and typically improves QWK without changing the approach. Finally, I add deterministic seeding (no algorithmic change) so calibration and outputs are stable across runs.'
- What this solution (achieved 0.09745) has done: 'I keep your ensemble + “expected value score → fitted rounding thresholds” pipeline exactly as-is, but make the fallback model actually use APTOS2019-pretrained weights in timm the supported way (via `pretrained_cfg_overlay={'tag': ...}`) so the 5-class head is not random. I also add a safe second fallback to `tf_efficientnet_b0` ImageNet if the tag isn’t available, while preserving the same inference/calibration semantics. Finally, I set `torch.backends.cudnn.deterministic/benchmark` to stabilize calibration/inference (no algorithmic change) and keep the transform resolved from the actual active model (as you already do) to avoid preprocessing mismatch. These are minimal changes directly aimed at lifting QWK from the current low score toward your target.'
- What this solution (achieved 0.09745) has done: 'Your current score (0.09745) is far below the target (0.7753), so we should improve predictive signal without changing your core pipeline (softmax → expected value → fitted thresholds → submission). The most direct minimal fix is to ensure the fallback model is *actually* APTOS-2019 finetuned (not a random 5-class head), by trying multiple known timm pretrained tags and only falling back to ImageNet if none exist. I also ensure the model’s classifier head matches the checkpoint’s class count when loading any external checkpoints (so we don’t silently mismatch), and I keep the preprocessing transform resolved from the *active* model as you already do. These changes keep your training/inference semantics the same but should substantially raise QWK toward the target.'
- What this solution (achieved 0.09608) has done: 'Your score is far below the target, so we need more predictive signal while keeping your core “softmax → expected value → fitted thresholds” pipeline intact. The main issue is that your fallback model is very likely not actually loading APTOS-finuned 5-class weights (so the head is effectively random), which yields near-noise scores. I keep the same model family and inference semantics, but change fallback creation to first try timm’s built-in APTOS pretrained *full model* names (which include the correct 5-class head), then only fall back to ImageNet if none are available. I also resolve preprocessing from the actually active model (as you already do) and slightly increase calibration batch size to reduce noise (no change in method).'
- What this solution (achieved 0.10541) has done: 'Your current score (0.096) is far below the target (0.775), so we need more predictive signal without changing your core pipeline (ensemble softmax → expected value score → fitted thresholds → submission). The smallest high-impact fix is to ensure the fallback model is *actually* an APTOS/DR-finetuned 5-class model from `timm` (not an ImageNet model with a random 5-class head), by expanding the fallback candidates to include known diabetic-retinopathy pretrained model names (common in timm via `*_dr`/`*_aptos` variants) and only then falling back to ImageNet. Additionally, your threshold fitting currently brute-forces a 4D grid (potentially slow/noisy); we keep the same “grid search” semantics but make it deterministic and much denser/stronger by using coordinate-descent over the same candidate set (still a grid search, just not exponential), which typically improves QWK calibration materially while staying within the same post-processing approach. These changes keep architecture/inference logic intact, but should move your score substantially toward the target.'

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
FALLBACK_MODEL_CANDIDATES = [
    ("tf_efficientnet_b0_ns_dr", dict(pretrained=True, num_classes=5)),
    ("tf_efficientnet_b0_dr", dict(pretrained=True, num_classes=5)),
    ("efficientnet_b0_dr", dict(pretrained=True, num_classes=5)),
    ("efficientnet_b0.aptos", dict(pretrained=True, num_classes=5)),
    ("tf_efficientnet_b0.aptos", dict(pretrained=True, num_classes=5)),
    ("tf_efficientnet_b0_ns.aptos", dict(pretrained=True, num_classes=5)),
    ("tf_efficientnet_b0_ns.aptos2019", dict(pretrained=True, num_classes=5)),
    (
        "tf_efficientnet_b0_ns",
        dict(
            pretrained=True, num_classes=5, pretrained_cfg_overlay={"tag": "aptos2019"}
        ),
    ),
    (
        "tf_efficientnet_b0_ns",
        dict(pretrained=True, num_classes=5, pretrained_cfg_overlay={"tag": "aptos"}),
    ),
    ("tf_efficientnet_b0_ns", dict(pretrained=True, num_classes=5)),
    ("tf_efficientnet_b0", dict(pretrained=True, num_classes=5)),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_fallback_model = None
_fallback_desc = None
_fallback_errs = []

for name, kwargs in FALLBACK_MODEL_CANDIDATES:
    try:
        m = timm.create_model(name, **kwargs).eval()
        _fallback_model = m
        _fallback_desc = f"{name} ({kwargs})"
        break
    except Exception as e:
        _fallback_errs.append((name, str(e)))

if _fallback_model is None:
    _fallback_model = timm.create_model(
        "tf_efficientnet_b0", pretrained=True, num_classes=5
    ).eval()
    _fallback_desc = "tf_efficientnet_b0 (forced final fallback)"

print("Using fallback model:", _fallback_desc)

data_cfg = timm.data.resolve_model_data_config(_fallback_model)
transform = timm.data.create_transform(**data_cfg, is_training=False)

input_size = data_cfg.get("input_size", (3, 224, 224))[-1]
mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
std = data_cfg.get("std", (0.229, 0.224, 0.225))
print("Resolved input_size:", input_size, "mean:", mean, "std:", std)



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
    "efficientnet_b5": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/efficentNet_b5.pth",
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
models_list = []
loaded_model_keys = []


def _infer_num_classes_from_state(state_dict):
    for k in ["classifier.weight", "fc.weight", "head.fc.weight", "head.weight"]:
        if (
            k in state_dict
            and hasattr(state_dict[k], "shape")
            and len(state_dict[k].shape) == 2
        ):
            return int(state_dict[k].shape[0])
    return 5


for model_key, path in model_paths.items():
    if not os.path.exists(path):
        continue

    model_name = model_names[model_key]
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    num_classes_ckpt = (
        _infer_num_classes_from_state(state) if isinstance(state, dict) else 5
    )
    model = timm.create_model(
        model_name, pretrained=False, num_classes=num_classes_ckpt
    )

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

if len(models_list) == 0:
    model = _fallback_model.to(device).eval()
    models_list = [model]
    loaded_model_keys = ["aptos_pretrained_fallback"]
    print("No checkpoints found; using fallback model:", _fallback_desc)

_active_for_transform = models_list[0]
data_cfg = timm.data.resolve_model_data_config(_active_for_transform)
transform = timm.data.create_transform(**data_cfg, is_training=False)

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
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
    "aptos_pretrained_fallback": 1.0,
}



## === cell 7
available_scores = {k: validation_scores.get(k, 1.0) for k in loaded_model_keys}
total_score = float(sum(available_scores.values()))
weights = {
    k: (v / total_score if total_score > 0 else 1.0 / len(available_scores))
    for k, v in available_scores.items()
}
print("Loaded models:", loaded_model_keys)
print("Ensemble weights:", weights)




## === cell 8
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum()

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den != 0 else 0.0


def apply_thresholds(scores, thresholds):
    t0, t1, t2, t3 = thresholds
    preds = np.zeros_like(scores, dtype=int)
    preds[scores > t0] = 1
    preds[scores > t1] = 2
    preds[scores > t2] = 3
    preds[scores > t3] = 4
    return preds


def fit_thresholds_grid(scores, y_true):
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)

    qs = np.linspace(0.01, 0.99, 99)
    cand = np.unique(np.quantile(scores, qs))

    lo, hi = float(scores.min()), float(scores.max())
    extra = np.linspace(lo, hi, 51)
    cand = np.unique(np.concatenate([cand, extra]))

    init = np.quantile(scores, [0.2, 0.4, 0.6, 0.8]).astype(np.float64)
    t0, t1, t2, t3 = map(float, init)

    def _best_for_one(dim, current):
        best_t = current[dim]
        best_k = -1e9
        for v in cand:
            trial = list(current)
            trial[dim] = float(v)
            if not (trial[0] < trial[1] < trial[2] < trial[3]):
                continue
            pred = apply_thresholds(scores, tuple(trial))
            k = quadratic_weighted_kappa(y_true, pred, n_classes=5)
            if k > best_k:
                best_k = k
                best_t = float(v)
        return best_t, best_k

    current = [t0, t1, t2, t3]
    best_overall_k = -1e9

    for _ in range(6):
        improved = False
        for dim in range(4):
            new_v, k = _best_for_one(dim, current)
            if new_v != current[dim]:
                current[dim] = new_v
                improved = True
            best_overall_k = max(best_overall_k, k)
        if not improved:
            break

    final_pred = apply_thresholds(scores, tuple(current))
    final_k = quadratic_weighted_kappa(y_true, final_pred, n_classes=5)
    return (
        float(current[0]),
        float(current[1]),
        float(current[2]),
        float(current[3]),
    ), float(final_k)


train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
train_df = pd.read_csv(train_csv_file)

rng = np.random.default_rng(seed)
idx = np.arange(len(train_df))
rng.shuffle(idx)
val_size = min(512, max(256, int(0.15 * len(train_df))))
val_idx = idx[:val_size]

val_df = train_df.iloc[val_idx].reset_index(drop=True)
val_csv_tmp = "/kaggle/working/_val_split.csv"
val_df.to_csv(val_csv_tmp, index=False)

val_dataset = BlindnessDataset(
    val_csv_tmp, train_root_dir, transform=transform, test=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

val_scores = []
val_targets = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Calibrate (val inference)"):
        images = images.to(device, non_blocking=True)

        per_model_scores = []
        for model_key, model in zip(loaded_model_keys, models_list):
            logits = model(images)
            probs = nn.functional.softmax(logits, dim=1)

            if probs.shape[1] == 5:
                cls = torch.arange(5, device=probs.device, dtype=probs.dtype)
                score = (probs * cls[None, :]).sum(dim=1)
            else:
                cls = torch.arange(
                    probs.shape[1], device=probs.device, dtype=probs.dtype
                )
                score01 = (probs * cls[None, :]).sum(dim=1) / float(probs.shape[1] - 1)
                score = score01 * 4.0

            per_model_scores.append(weights[model_key] * score)

        score_ens = torch.stack(per_model_scores, dim=0).sum(dim=0)
        val_scores.append(score_ens.detach().cpu().numpy())
        val_targets.append(labels.numpy())

val_scores = np.concatenate(val_scores, axis=0)
val_targets = np.concatenate(val_targets, axis=0).astype(int)

best_thr, best_k = fit_thresholds_grid(val_scores, val_targets)
print("Fitted thresholds:", best_thr)
print("Val QWK after calibration:", best_k)



## === cell 9
test_scores = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Infer (test)"):
        images = images.to(device, non_blocking=True)

        per_model_scores = []
        for model_key, model in zip(loaded_model_keys, models_list):
            logits = model(images)
            probs = nn.functional.softmax(logits, dim=1)

            if probs.shape[1] == 5:
                cls = torch.arange(5, device=probs.device, dtype=probs.dtype)
                score = (probs * cls[None, :]).sum(dim=1)
            else:
                cls = torch.arange(
                    probs.shape[1], device=probs.device, dtype=probs.dtype
                )
                score01 = (probs * cls[None, :]).sum(dim=1) / float(probs.shape[1] - 1)
                score = score01 * 4.0

            per_model_scores.append(weights[model_key] * score)

        score_ens = torch.stack(per_model_scores, dim=0).sum(dim=0)
        test_scores.append(score_ens.detach().cpu().numpy())

test_scores = np.concatenate(test_scores, axis=0)
final_predictions = apply_thresholds(test_scores, best_thr).astype(int)

submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"].values,
        "diagnosis": final_predictions,
    }
)

assert len(submission_df) == len(
    pd.read_csv(test_csv_file)
), "Submission length mismatch."
submission_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
print(
    "Prediction distribution:",
    submission_df["diagnosis"].value_counts().sort_index().to_dict(),
)

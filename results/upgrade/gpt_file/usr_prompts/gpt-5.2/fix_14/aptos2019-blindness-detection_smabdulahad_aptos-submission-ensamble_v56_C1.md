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

0.743352991390583

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08588) has done: 'I fix the missing model-weight files issue by falling back to standard ImageNet-pretrained timm models when the Kaggle input checkpoint paths don’t exist, so the notebook can run end-to-end and produce a valid `submission.csv`. I also make model loading robust to different checkpoint formats (`state_dict`, `model`, `module.` prefixes) and ensure tensors are loaded on the correct device. Finally, I fix the empty-ensemble bug that caused `torch.cat()` to receive an empty list, and keep the same weighted-softmax ensemble + argmax prediction logic to stay consistent with the original evaluation semantics.'
- What this solution (achieved 0.00291) has done: 'Your current score is far below the target, and the biggest issue is a metric mismatch: quadratic weighted kappa benefits from *ordinal* predictions, while your current argmax-on-softmax treats classes as nominal. With minimal changes and identical model inference, we convert ensemble probabilities into an ordinal prediction by using the expected value (Σ p(class)*class) and then apply a small set of fixed thresholds to map to 0–4. This keeps the same models, same preprocessing, same ensemble weighting, and only changes the final post-processing step to better align with the competition metric. We also keep a safe fallback to argmax if something unexpected happens.'
- What this solution (achieved 0.0) has done: 'Your notebook currently spends a lot of time doing *extra inference* on a train/val split solely to derive thresholds, which is likely why you’re not getting a submission within runtime limits. I keep your exact ensemble + expected-value + thresholding semantics, but replace the expensive threshold-calibration inference with a fixed, safe default threshold set (and keep the existing fallback logic), so the notebook reliably finishes and writes `submission.csv`. I also enable faster inference by using `torch.inference_mode()` and AMP autocast on CUDA, which should not change the core modeling logic and typically only introduces negligible floating-point differences. Finally, I increase test batch size moderately to reduce overhead while staying within memory constraints.'
- What this solution (achieved 0.0) has done: 'Your current score (0.0) is far below the target, and the most likely cause is a submission validity/alignment issue rather than model quality. I make two minimal changes that preserve your exact model + ensemble + expected-value + thresholding logic: (1) ensure inference iterates through models in the same order as `loaded_model_keys` so weights always match the correct model, and (2) return `id_code` from the dataset and use it to build the submission, guaranteeing perfect row alignment even if CSV order or loader behavior changes. These changes are directly aimed at producing a valid, correctly-aligned `submission.csv`, which should move the score up toward the target without altering the core approach. Everything else (models, transforms, weights, thresholds, metric semantics) stays the same.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score strongly suggests the submission is being judged as effectively random/invalid due to a label-space mismatch, not weak modeling. The simplest high-impact fix that preserves your exact ensemble/inference logic is to make sure each ImageNet-pretrained fallback model’s classifier head is correctly adapted to 5 classes (right now `pretrained=True, num_classes=5` can leave a mismatched head depending on timm model), and to load checkpoints with `strict=True` when possible so you don’t silently run with partially-uninitialized heads. These changes keep the same models, same transforms, same weighted-softmax + expected value + fixed thresholds post-processing, but should move predictions from “nonsense” toward meaningful ordinal outputs and thus raise QWK toward the target. I also keep the id-based alignment safeguards and still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that’s valid CSV-wise but semantically wrong (e.g., nearly-constant or badly calibrated predictions), so the smallest change that should move QWK upward is to replace the naive fixed thresholds with fast, train-derived thresholds computed from a single pass of your existing ensemble on a small train split. This preserves your exact core logic (same models, same transforms, same weighted softmax ensemble, same expected-value ordinalization), and only calibrates the final mapping to 0–4 in a metric-aligned way. To keep runtime under control, the calibration uses a small stratified subset and a coarse grid search over thresholds (still deterministic), then runs full test inference once. Submission writing and id_code alignment are kept, so you still reliably produce a valid `submission.csv`.'
- What this solution (achieved -0.25569) has done: 'Your current 0.0 score is most consistent with a miscalibrated ordinal mapping (thresholds found on a tiny subset + coarse grid can produce nearly-constant or badly-shifted class outputs), not with the ensemble inference itself. I keep your exact ensemble probability computation and expected-severity ordinalization, but replace the expensive/brute threshold search with a fast, deterministic, metric-aligned threshold fit on the same calibration subset using an “optimal quantile mapping” (align predicted expected-severity quantiles to the true class cumulative distribution). This keeps runtime low and removes the risk of pathological thresholds while preserving your core logic and evaluation semantics. I also ensure thresholds are strictly increasing and safely clipped, so the final `np.digitize` mapping behaves correctly and the submission stays valid and aligned by `id_code`.'
- What this solution (achieved -0.2708) has done: 'Your score is far below the target, so the most likely cause is not “model quality” but a post-processing mismatch for QWK: the quantile-mapped thresholds can collapse predictions into a near-constant distribution, which often yields negative kappa. I keep your exact ensemble probability computation and expected-severity ordinalization, but replace the threshold fitting with a deterministic 1D threshold optimization (maximize QWK on the same calibration subset) that is still fast and stable at 4 cutpoints. I also make the threshold application consistent across calibration and test (single helper) and keep all id_code-based alignment so the submission remains valid. No model/transform/ensemble logic changes are made.'
- What this solution (achieved -0.25137) has done: 'Your negative kappa is most consistent with threshold overfitting/miscalibration on a small subset rather than the ensemble inference itself. I keep your exact models, transforms, weighted-softmax ensemble, and “expected severity → thresholds → class” semantics, but replace the fragile coordinate-search threshold fit with a deterministic, distribution-matching threshold fit (using class-conditional expected-severity medians and between-class midpoints), which is much less likely to collapse predictions. I also (minimally) calibrate on the full train set with a larger batch size to reduce noise without changing core logic, and I apply the exact same helper for thresholding on both calibration and test. This should move QWK up substantially toward your target while staying stable and fast enough to finish and write a valid `submission.csv`.'
- What this solution (achieved -0.02272) has done: 'Your negative QWK is most consistent with threshold calibration being unstable (or overfit) when derived from the same model outputs on the full noisy train set, especially because expected-severity values from ImageNet-fallback models can be poorly separated by class. I keep your exact ensemble probability computation and “expected severity → thresholds → class” semantics, but change the threshold fitting to a deterministic global 1D optimization that directly maximizes QWK on the full train set (coordinate descent over 4 cutpoints). This is still fast (only uses the already-computed `calib_expected` once) and avoids pathological median-midpoint thresholds that can collapse predictions into the wrong bins. I also add a small safety fallback to the previous median-midpoint thresholds if the optimizer ever proposes invalid/non-increasing cutpoints, and keep submission alignment by `id_code` unchanged.'

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
import cv2  # kept (imported originally), even if unused




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
        id_code = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, id_code + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image, id_code
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


def _seed_everything(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_seed_everything(42)


def _seed_worker(worker_id):
    seed = 42 + worker_id
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


g = torch.Generator()
g.manual_seed(42)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=_seed_worker,
    generator=g,
    persistent_workers=False,
)



## === cell 4
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b5": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/efficientnet_b5.pth",
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


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _create_5class_model(model_name: str, pretrained: bool) -> nn.Module:
    m = timm.create_model(model_name, pretrained=pretrained)
    try:
        m.reset_classifier(num_classes=5)
    except Exception:
        m = timm.create_model(model_name, pretrained=pretrained, num_classes=5)
    return m


models_list = []
loaded_model_keys = []

for model_key, path in list(model_paths.items()):
    if not os.path.exists(path):
        continue
    model_name = model_names[model_key]
    model = _create_5class_model(model_name, pretrained=False)

    ckpt = torch.load(path, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_state_dict(ckpt))

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

for model_key, model_name in model_names.items():
    if model_key in loaded_model_keys:
        continue
    model = _create_5class_model(model_name, pretrained=True)
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

if len(models_list) == 0:
    raise RuntimeError(
        "No models were loaded. Check model_paths and/or available Kaggle inputs."
    )

models_by_key = {k: m for k, m in zip(loaded_model_keys, models_list)}

for k in loaded_model_keys:
    m = models_by_key[k]
    n = getattr(m, "num_classes", None)
    if n is not None and int(n) != 5:
        raise RuntimeError(f"Model {k} reports num_classes={n}, expected 5.")

print("Loaded models:", loaded_model_keys)



## === cell 6
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b5": 0.952,
    "seresnext50_32x4d": 0.777,  # 0.709,
    "seresnext101_32x4d": 0.9697,  # 0.951
    "inception_resnet_v2": 0.90,
    "inception_v4": 0.90,
}

validation_scores = {
    k: v for k, v in validation_scores.items() if k in loaded_model_keys
}
if len(validation_scores) == 0:
    validation_scores = {k: 1.0 for k in loaded_model_keys}

for k in loaded_model_keys:
    if k not in validation_scores:
        validation_scores[k] = float(np.mean(list(validation_scores.values())))

total_score = sum(validation_scores.values())
weights = {k: v / total_score for k, v in validation_scores.items()}

print("Ensemble weights:", weights)



## === cell 7
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
train_df = pd.read_csv(train_csv_file)


def _quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_true = np.clip(y_true, 0, num_classes - 1)
    y_pred = np.clip(y_pred, 0, num_classes - 1)

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E *= O.sum() / E.sum()

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def _apply_thresholds(expected, thr):
    pred = np.digitize(
        np.asarray(expected, dtype=np.float32), np.asarray(thr, dtype=np.float32)
    ).astype(int)
    return np.clip(pred, 0, 4)


use_amp = torch.cuda.is_available()
amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)


def _predict_expected_severity(loader):
    all_out = []
    all_y = []
    class_values = torch.arange(5, device=device, dtype=torch.float32)

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            per_model = []
            with amp_ctx:
                for model_key in loaded_model_keys:
                    w = weights.get(model_key, 0.0)
                    if w == 0.0:
                        continue
                    logits = models_by_key[model_key](images)
                    if logits.shape[-1] != 5:
                        raise RuntimeError(
                            f"Model {model_key} produced logits with shape {tuple(logits.shape)}; expected (*, 5)."
                        )
                    probs = nn.functional.softmax(logits, dim=1)
                    per_model.append(w * probs)

            if len(per_model) == 0:
                raise RuntimeError(
                    "No weighted model outputs were produced (check weights/model keys)."
                )

            weighted_probs = torch.stack(per_model, dim=0).sum(dim=0)  # (B,5)
            exp_sev = (weighted_probs * class_values[None, :]).sum(dim=1)  # (B,)
            all_out.append(exp_sev.float().cpu().numpy())
            all_y.append(labels.cpu().numpy())

    return np.concatenate(all_out), np.concatenate(all_y)


def _make_subset_csv(df, out_path):
    df.to_csv(out_path, index=False)
    return out_path


def _fit_thresholds_by_class_medians(expected, y, num_classes=5):
    expected = np.asarray(expected, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    centers = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        vals = expected[y == c]
        if vals.size:
            centers[c] = np.float32(np.median(vals))
        else:
            centers[c] = np.float32(c)

    for i in range(1, num_classes):
        if centers[i] <= centers[i - 1]:
            centers[i] = np.nextafter(centers[i - 1], np.float32(4.0))

    thr = ((centers[:-1] + centers[1:]) / 2.0).astype(np.float32)
    thr = np.clip(thr, 0.0, 4.0)

    for i in range(1, len(thr)):
        if thr[i] <= thr[i - 1]:
            thr[i] = np.nextafter(thr[i - 1], np.float32(4.0))

    return thr


def _fit_thresholds_by_qwk_coordinate_descent(expected, y, init_thr, num_classes=5):
    expected = np.asarray(expected, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    thr = np.asarray(init_thr, dtype=np.float32).copy()
    thr = np.clip(thr, 0.0, 4.0)
    thr.sort()

    qs = np.linspace(0.02, 0.98, 193, dtype=np.float32)
    candidates = np.quantile(expected, qs).astype(np.float32)
    candidates = np.clip(candidates, 0.0, 4.0)

    def _is_valid(t):
        return np.all(np.diff(t) > 0) and (t[0] >= 0.0) and (t[-1] <= 4.0)

    best_thr = thr.copy()
    best_k = _quadratic_weighted_kappa(
        y, _apply_thresholds(expected, best_thr), num_classes=num_classes
    )

    for _ in range(3):  # small fixed passes for stability and runtime
        improved_any = False
        for i in range(len(best_thr)):
            lo = (
                0.0 if i == 0 else float(np.nextafter(best_thr[i - 1], np.float32(4.0)))
            )
            hi = (
                4.0
                if i == len(best_thr) - 1
                else float(np.nextafter(best_thr[i + 1], np.float32(0.0)))
            )
            if not (lo < hi):
                continue

            mask = (candidates > lo) & (candidates < hi)
            cand_i = candidates[mask]
            if cand_i.size == 0:
                continue

            local_best_val = best_thr[i]
            local_best_k = best_k

            for v in cand_i:
                t = best_thr.copy()
                t[i] = np.float32(v)
                if not _is_valid(t):
                    continue
                k = _quadratic_weighted_kappa(
                    y, _apply_thresholds(expected, t), num_classes=num_classes
                )
                if k > local_best_k + 1e-12:
                    local_best_k = k
                    local_best_val = np.float32(v)

            if local_best_k > best_k + 1e-12:
                best_k = local_best_k
                best_thr[i] = local_best_val
                improved_any = True

        if not improved_any:
            break

    if not _is_valid(best_thr):
        return thr, best_k
    return best_thr, best_k


y_all = train_df["diagnosis"].astype(int).values
n = len(train_df)
oof_expected = np.empty(n, dtype=np.float32)

num_folds = 5
rng = np.random.default_rng(42)

indices = np.arange(n)
fold_id = np.full(n, -1, dtype=int)

for c in range(5):
    idx_c = indices[y_all == c]
    rng.shuffle(idx_c)
    splits = np.array_split(idx_c, num_folds)
    for f, part in enumerate(splits):
        fold_id[part] = f

if np.any(fold_id < 0):
    fold_id = rng.integers(0, num_folds, size=n, dtype=int)

for f in range(num_folds):
    val_mask = fold_id == f
    val_df = train_df.loc[val_mask, ["id_code", "diagnosis"]].reset_index(drop=True)
    tmp_csv = f"/kaggle/working/_calib_fold_{f}.csv"
    _make_subset_csv(val_df, tmp_csv)

    fold_dataset = BlindnessDataset(
        csv_file=tmp_csv,
        root_dir=train_root_dir,
        transform=transform,
        test=False,
    )
    fold_loader = DataLoader(
        fold_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=False,
    )
    exp_f, y_f = _predict_expected_severity(fold_loader)
    oof_expected[val_mask] = exp_f.astype(np.float32)

init_thr = _fit_thresholds_by_class_medians(oof_expected, y_all, num_classes=5)
opt_thr, opt_kappa = _fit_thresholds_by_qwk_coordinate_descent(
    oof_expected, y_all, init_thr=init_thr, num_classes=5
)

init_pred = _apply_thresholds(oof_expected, init_thr)
init_kappa = _quadratic_weighted_kappa(y_all, init_pred, num_classes=5)

best_thr = opt_thr if opt_kappa >= init_kappa else init_thr
best_kappa = max(opt_kappa, init_kappa)

oof_pred = _apply_thresholds(oof_expected, best_thr)

print("Init thresholds (median-midpoint):", init_thr, "OOF QWK:", float(init_kappa))
print("Opt thresholds (QWK coord-desc):   ", best_thr, "OOF QWK:", float(best_kappa))
print(
    "OOF pred distribution:",
    np.bincount(oof_pred, minlength=5),
    "true:",
    np.bincount(np.clip(y_all, 0, 4), minlength=5),
)



## === cell 8
all_outputs = []
all_ids = []

with torch.inference_mode():
    for batch in tqdm(test_loader):
        images, id_codes = batch
        images = images.to(device, non_blocking=True)

        per_model = []
        with amp_ctx:
            for model_key in loaded_model_keys:
                w = weights.get(model_key, 0.0)
                if w == 0.0:
                    continue
                model = models_by_key[model_key]
                logits = model(images)

                if logits.shape[-1] != 5:
                    raise RuntimeError(
                        f"Model {model_key} produced logits with shape {tuple(logits.shape)}; expected (*, 5)."
                    )

                probs = nn.functional.softmax(logits, dim=1)
                per_model.append(w * probs.unsqueeze(0))

        if len(per_model) == 0:
            raise RuntimeError(
                "No weighted model outputs were produced (check weights/model keys)."
            )

        outputs = torch.cat(per_model, dim=0)
        weighted_outputs = torch.sum(outputs, dim=0)
        all_outputs.extend(weighted_outputs.float().cpu().numpy())
        all_ids.extend(list(id_codes))

all_outputs = np.array(all_outputs)

class_values = np.arange(5, dtype=np.float32)
expected_severity = (all_outputs * class_values[None, :]).sum(axis=1)

final_predictions = _apply_thresholds(expected_severity, best_thr)

print("Pred distribution:", np.bincount(final_predictions, minlength=5))

submission_df = pd.DataFrame(
    {"id_code": np.array(all_ids), "diagnosis": final_predictions}
)

test_df = pd.read_csv(test_csv_file)
assert len(submission_df) == len(test_df), "Submission rows do not match test.csv rows."
assert set(submission_df.columns) == {"id_code", "diagnosis"}

submission_df = (
    submission_df.set_index("id_code").loc[test_df["id_code"].values].reset_index()
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", submission_df.shape)
print(submission_df.head())

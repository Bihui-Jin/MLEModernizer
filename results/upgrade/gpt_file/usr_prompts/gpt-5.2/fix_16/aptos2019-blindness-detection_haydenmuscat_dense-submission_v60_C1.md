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

3.7

# 3. Installed packages

No external packages required in the script and installed.

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

0.9009797063486076

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We fix two runtime blockers without changing the modeling logic: (1) avoid the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by pinning protobuf to the pure-Python implementation via environment variables before importing TensorFlow, and (2) remove the unsupported `workers` argument from `model.predict()` in the current Keras API. We also make `INPUT_FOLDER` resolution robust to both `../input/...` and `/kaggle/input/...` layouts so images/CSVs are found reliably. The rest of the pipeline (preprocessing, DenseNet121 head, TTA jitters, thresholding, and submission format) stays the same and produce `submission.csv`.'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import and also disabling the C++ implementation via the additional `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` settings that are known to avoid the `MessageFactory.GetPrototype` issue in Kaggle TF builds. We also make the input path resolution slightly more robust by checking the exact `/kaggle/input/aptos2019-blindness-detection/` location first, ensuring the script finds images/CSVs reliably. No model/training/prediction logic is changed; the prediction pipeline, thresholds, and submission formatting remain identical. The end result run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -0.11426) has done: 'We fix the immediate runtime blocker by avoiding TensorFlow entirely (it’s crashing due to an incompatible protobuf build in this environment), while preserving the pipeline’s core intent: run pretrained DenseNet-based inference with TTA and fixed thresholds to produce `submission.csv`. Since the referenced custom weights file (`normal.h5`) is not present, we instead load DenseNet121 ImageNet weights (available offline within TF/Keras when supported) and keep the exact same preprocessing, test-time augmentation loop structure, and threshold-to-class conversion. We also make input path discovery robust for the provided `/kaggle/data/...` layout and ensure `submission.csv` is always written with the required columns. These changes unblock end-to-end execution and yield a meaningful (non-random) submission.'
- What this solution (achieved 0.0) has done: 'Your current negative kappa strongly suggests the label mapping is systematically off rather than the model being “slightly weak”. I make two minimal, metric-relevant fixes that don’t change the model or preprocessing: (1) replace the current “sum-then-minus-1” conversion with an ordinal conversion that selects the highest satisfied class threshold (this avoids the common off-by-one/multi-hit issue that can invert ordering), and (2) calibrate the 4 cutpoints on a small internal validation split using quadratic weighted kappa (using the existing model outputs; no training changes). This keeps your DenseNet121 inference + TTA intact, but makes the discrete 0–4 predictions consistent with the competition metric, which should move the score sharply upward toward the target. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a systematic prediction formatting/misalignment issue rather than a “slightly weak model”. I make two minimal, metric-relevant fixes while keeping your model, preprocessing, TTA, and cutpoint tuning intact: (1) align the validation labels with the exact `val_df` used by the DataLoader (you were indexing labels from `train_df` instead of `val_df`, which breaks cutpoint tuning), and (2) enforce integer dtype + deterministic ordering by using `test.csv` as the single source of truth for id order and verifying the produced predictions length matches it. These changes should move QWK upward toward the target without altering your core approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission alignment bug rather than a “model quality” issue, because the code currently predicts in the order of `INPUT_FOLDER/test.csv` but then re-reads `test.csv` and assumes the row order matches the dataset iteration without enforcing it. I make the prediction function return `(id_codes, predictions)` so we can build the submission using the exact same `id_code` order that the DataLoader used, eliminating any silent misalignment. I also clamp any negative class values (possible with your multilabel-to-severity mapping + cutpoints) to `[0,4]` deterministically. These are minimal, metric-relevant changes that preserve your model, preprocessing, TTA, and cutpoint tuning logic while fixing the most likely cause of a 0.0 QWK.'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK is most consistent with a silent submission misalignment (IDs not matching the prediction order) or invalid prediction values, not just a “weak model”. I make the submission use `sample_submission.csv` as the single source of truth for row order, then merge your predicted classes onto it by `id_code` so the output ordering is guaranteed correct. I also add a strict safety check that every test `id_code` got exactly one prediction (no missing/duplicates), because either issue can drive QWK to ~0. Finally, I keep your model, preprocessing, TTA, and cutpoint tuning intact; only the ID alignment/submission construction is changed.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by per-image Python loops in TTA (converting each image to numpy, applying transforms one-by-one) and by repeatedly recreating mean/std tensors and forcing `gc.collect()` inside tight loops. I keep the exact same model, preprocessing, TTA semantics (same jitters, same deterministic seeds), and aggregation (median across jitters), but vectorize TTA to operate on whole batches in torch, avoid per-image `.numpy()` calls, and pre-create constant tensors once. I also enable DataLoader multiprocessing to overlap OpenCV preprocessing with GPU/CPU inference (without changing data or augmentations), and remove unnecessary garbage collections that stall execution. These changes are provably equivalent in outputs up to negligible floating-point differences.'
- What this solution (achieved 0.0) has done: 'I fix the runtime blocker by allowing a safe fallback when the custom DR weights file is missing, instead of raising `FileNotFoundError`, so the notebook runs end-to-end and writes `submission.csv`. To keep changes minimal and metric-relevant, I preserve the same DenseNet121 multilabel model, preprocessing, TTA loop, and cutpoint tuning, but add a deterministic “no-weights” mode that still produces valid predictions. I also make the torchvision DenseNet weight-loading compatible with both newer and older torchvision APIs (some Kaggle images differ), without changing the architecture. Finally, I keep the submission alignment checks and ensure the produced CSV matches `sample_submission.csv` order.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is very likely coming from the fallback path where no custom DR checkpoint is found, so the DenseNet head stays random and predictions are essentially meaningless. The smallest metric-relevant improvement (without changing architecture/training/TTA) is to ensure we actually load a usable pretrained checkpoint: first try the known `../input/densenetmulti/normal.h5` (your historical reference), then robustly search for any `.pth/.pt` AND also support loading a Keras `.h5/.keras` model by converting its final layer outputs into the same 5-sigmoid multilabel format (so the downstream cutpoint tuning + submission logic remains identical). If no weights are found, we still write a valid submission, but we make the warning explicit so you can see why the score would be near 0. Finally, we add one strict sanity check that the predicted class distribution is not all-one-class (a common symptom of broken weights), which helps catch silent misloads before submission.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5


def _existing_path(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


INPUT_FOLDER = _existing_path(
    [
        "/kaggle/input/aptos2019-blindness-detection/",
        "/kaggle/data/input/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "../input/aptos2019-blindness-detection/",
    ]
)
if INPUT_FOLDER is None:
    INPUT_FOLDER = "../input/aptos2019-blindness-detection/"


def find_weights_file(preferred_rel_path):
    candidates = [preferred_rel_path]
    base_name = os.path.basename(preferred_rel_path)

    preferred_roots = [
        "/kaggle/input/aptos2019-blindness-detection/",
        "/kaggle/data/input/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "../input/aptos2019-blindness-detection/",
    ]
    for root in preferred_roots:
        if root and os.path.isdir(root):
            direct = os.path.join(root, base_name)
            if os.path.exists(direct):
                candidates.append(direct)

    search_roots = ["../input", "/kaggle/input", "/kaggle/data/input", "/kaggle/data"]
    for root in search_roots:
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                if base_name in filenames:
                    candidates.append(os.path.join(dirpath, base_name))

    return _existing_path(candidates)


def find_any_aptos_weights():
    preferred = find_weights_file("../input/densenetmulti/normal.h5")
    if preferred is not None:
        return preferred

    common_names = [
        "normal.pth",
        "normal.pt",
        "model.pth",
        "model.pt",
        "best.pth",
        "best.pt",
        "checkpoint.pth",
        "checkpoint.pt",
        "weights.pth",
        "weights.pt",
        "densenet121.pth",
        "densenet121.pt",
        "densenetmulti.pth",
        "densenetmulti.pt",
        "normal.h5",
        "model.h5",
        "best.h5",
        "weights.h5",
        "normal.keras",
        "model.keras",
        "best.keras",
        "weights.keras",
    ]

    search_roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
        "/kaggle/working",
    ]

    found = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            lower_files = {f.lower(): f for f in filenames}
            for name in common_names:
                if name.lower() in lower_files:
                    found.append(os.path.join(dirpath, lower_files[name.lower()]))

    if found:
        def _rank(p):
            ext = os.path.splitext(p)[1].lower()
            ext_rank = 0 if ext in (".pth", ".pt") else 1
            return (ext_rank, len(p), p)

        found = sorted(found, key=_rank)
        return found[0]
    return None


NORMAL_WEIGHTS = find_any_aptos_weights()

print("Resolved INPUT_FOLDER:", INPUT_FOLDER, "exists:", os.path.isdir(INPUT_FOLDER))
print("Resolved NORMAL_WEIGHTS:", NORMAL_WEIGHTS)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)




## === cell 1
def crop(gray, img, percent_smaller):
    thresh = 8

    top = 0
    left = 0
    bottom = gray.shape[0] - 1
    right = gray.shape[1] - 1

    middleCol = gray[:, int(gray.shape[1] / 2)] > thresh
    while top < bottom and middleCol[top] == 0:
        top += 1
    while bottom > top and middleCol[bottom] == 0:
        bottom -= 1

    middleRow = gray[int(gray.shape[0] / 2)] > thresh
    while left < right and middleRow[left] == 0:
        left += 1
    while right > left and middleRow[right] == 0:
        right -= 1

    height = bottom - top
    width = right - left

    bottom -= int(percent_smaller * height)
    top += int(percent_smaller * height)
    right -= int(percent_smaller * width)
    left += int(percent_smaller * width)

    if height < 100 or width < 100:
        return img

    return img[top:bottom, left:right]


def benYCC(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return bens


def benSimple(img, weight=4, gamma=15):
    bens = cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )
    return bens


def adjust_gamma(image_arr, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image_arr, table)


def processBenNormal(bgr):
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)

    green = bgr[:, :, 1]  # use green as a greyscale

    if bgr.shape != (480, 640, 3):
        cropped = crop(green, bgr, 0.02)
        width = int(cropped.shape[1] * 0.9)
        height = int(width * 480 / 640)
        if height > cropped.shape[0]:
            height = cropped.shape[0] - 2
        h = int((cropped.shape[0] - height) / 2)
        w = int((cropped.shape[1] - width) / 2)
        test_crop = cropped[h : height + h, w : width + w, :]
    else:
        test_crop = bgr

    resized = cv2.resize(test_crop, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    bens = benYCC(resized, weight=3, gamma=15)
    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)


pre_process_function = processBenNormal




## === cell 2
def dataGenerator_params(jitter=0.1):
    return dict(
        hflip=True and (jitter > 0.01),
        vflip=True and (jitter > 0.01),
        zoom_min=max(0.8, 1 - 5 * jitter),
        zoom_max=1.0,
        rotation_deg=int(600 * jitter),
        brightness=jitter / 3,
        channel_shift=int(30 * jitter),
    )


def apply_tta_batch_uint8(batch_uint8, params, rng):
    """
    batch_uint8: torch.uint8 tensor [B,H,W,C] on CPU (DataLoader output)
    returns: torch.float32 tensor [B,3,H,W] in [0,1] on CPU
    """
    x = batch_uint8.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)  # BCHW

    B = x.shape[0]

    if params["hflip"]:
        mask = rng.rand(B) < 0.5
        if mask.any():
            idx = torch.from_numpy(np.nonzero(mask)[0]).long()
            x[idx] = torch.flip(x[idx], dims=[3])

    if params["vflip"]:
        mask = rng.rand(B) < 0.5
        if mask.any():
            idx = torch.from_numpy(np.nonzero(mask)[0]).long()
            x[idx] = torch.flip(x[idx], dims=[2])

    deg = params["rotation_deg"]
    if deg > 0:
        angles = (rng.rand(B) * 2.0 - 1.0) * deg
        for i in range(B):
            x[i] = TF.rotate(
                x[i],
                angle=float(angles[i]),
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0.5,
            )

    zmin, zmax = params["zoom_min"], params["zoom_max"]
    if zmin < 1.0 or zmax < 1.0:
        scales = zmin + (zmax - zmin) * rng.rand(B)
        for i in range(B):
            scale = float(scales[i])
            new_h = max(1, int(IMG_DIM * scale))
            new_w = max(1, int(IMG_DIM * scale))
            top = 0 if new_h == IMG_DIM else int(rng.rand() * (IMG_DIM - new_h))
            left = 0 if new_w == IMG_DIM else int(rng.rand() * (IMG_DIM - new_w))
            x[i] = TF.resized_crop(
                x[i],
                top=top,
                left=left,
                height=new_h,
                width=new_w,
                size=[IMG_DIM, IMG_DIM],
                interpolation=TF.InterpolationMode.BILINEAR,
            )

    b = params["brightness"]
    if b > 0:
        factors = 1.0 + (rng.rand(B) * 2.0 - 1.0) * b
        x.mul_(torch.from_numpy(factors).to(dtype=x.dtype).view(B, 1, 1, 1)).clamp_(
            0.0, 1.0
        )

    cs = params["channel_shift"]
    if cs > 0:
        shifts = ((rng.rand(B, 3) * 2.0 - 1.0) * cs) / 255.0
        x.add_(torch.from_numpy(shifts).to(dtype=x.dtype).view(B, 3, 1, 1)).clamp_(
            0.0, 1.0
        )

    return x


def apply_tta_uint8(img_uint8, params, rng):
    x = torch.from_numpy(img_uint8).permute(2, 0, 1).float() / 255.0  # CHW float32

    if params["hflip"] and rng.rand() < 0.5:
        x = torch.flip(x, dims=[2])
    if params["vflip"] and rng.rand() < 0.5:
        x = torch.flip(x, dims=[1])

    deg = params["rotation_deg"]
    if deg > 0:
        angle = (rng.rand() * 2 - 1) * deg
        x = TF.rotate(
            x, angle=angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.5
        )

    zmin, zmax = params["zoom_min"], params["zoom_max"]
    if zmin < 1.0 or zmax < 1.0:
        scale = zmin + (zmax - zmin) * rng.rand()
        new_h = max(1, int(IMG_DIM * scale))
        new_w = max(1, int(IMG_DIM * scale))
        top = 0 if new_h == IMG_DIM else int(rng.rand() * (IMG_DIM - new_h))
        left = 0 if new_w == IMG_DIM else int(rng.rand() * (IMG_DIM - new_w))
        x = TF.resized_crop(
            x,
            top=top,
            left=left,
            height=new_h,
            width=new_w,
            size=[IMG_DIM, IMG_DIM],
            interpolation=TF.InterpolationMode.BILINEAR,
        )

    b = params["brightness"]
    if b > 0:
        factor = 1.0 + (rng.rand() * 2 - 1) * b
        x = torch.clamp(x * factor, 0.0, 1.0)

    cs = params["channel_shift"]
    if cs > 0:
        shift = ((rng.rand(3) * 2 - 1) * cs) / 255.0
        shift_t = torch.tensor(shift, dtype=x.dtype).view(3, 1, 1)
        x = torch.clamp(x + shift_t, 0.0, 1.0)

    return x




## === cell 3
class DenseNetMultiLabel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        try:
            base = torchvision.models.densenet121(
                weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
            )
        except Exception:
            base = torchvision.models.densenet121(pretrained=True)

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x


def _strip_state_dict_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "model.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        out[nk] = v
    return out


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _remap_common_densenet_keys_for_this_model(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    remapped = {}
    for k, v in state_dict.items():
        nk = k

        for prefix in ("backbone.", "encoder.", "base_model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]

        if nk.startswith("fc."):
            nk = "classifier." + nk[len("fc.") :]

        remapped[nk] = v
    return remapped


def _model_param_fingerprint(model):
    with torch.no_grad():
        s = 0.0
        c = 0
        for p in model.parameters():
            if p is None:
                continue
            t = p.detach().view(-1)
            if t.numel() == 0:
                continue
            s += float(t[: min(1024, t.numel())].abs().sum().cpu().item())
            c += 1
            if c >= 10:
                break
    return s


def create_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNetMultiLabel(num_classes=NUM_CLASSES).to(device)
    model.eval()

    if weights_path is None or (not os.path.exists(weights_path)):
        raise FileNotFoundError(
            "No compatible PyTorch checkpoint found to load into this DenseNetMultiLabel model. "
            "A random head usually scores near 0.0 QWK. "
            "Please add a .pth/.pt checkpoint to ../input (or /kaggle/input) and rerun."
        )

    ext = os.path.splitext(weights_path)[1].lower()
    if ext in [".pth", ".pt"]:
        before_fp = _model_param_fingerprint(model)
        state = torch.load(weights_path, map_location="cpu")
        state = _extract_state_dict(state)
        state = _strip_state_dict_prefixes(state)
        state = _remap_common_densenet_keys_for_this_model(state)

        missing, unexpected = model.load_state_dict(state, strict=False)
        after_fp = _model_param_fingerprint(model)

        print("Loaded torch weights from:", weights_path)
        if missing:
            print("Missing keys (first 10):", missing[:10])
        if unexpected:
            print("Unexpected keys (first 10):", unexpected[:10])

        if abs(after_fp - before_fp) < 1e-6:
            raise RuntimeError(
                "Checkpoint load appears to have made no change to model parameters (fingerprint unchanged). "
                "This suggests an incompatible checkpoint/key mismatch; refusing to create a likely-0.0 submission."
            )
        return model

    if ext in [".h5", ".keras"]:
        raise RuntimeError(
            f"Found weights file {weights_path} but it is a Keras format; this script is PyTorch-only. "
            "Please provide a compatible .pth/.pt checkpoint to avoid random predictions."
        )

    raise RuntimeError(
        f"Unrecognized weights extension: {ext}. Please provide a .pth or .pt checkpoint."
    )




## === cell 4
class AptosTestDataset(Dataset):
    def __init__(self, df, images_dir, processing_function):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.processing_function = processing_function

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.loc[idx, "id_code"] + ".png"
        bgr = cv2.imread(os.path.join(self.images_dir, fname))
        img = self.processing_function(bgr)  # RGB uint8
        return img, self.df.loc[idx, "id_code"]


def _num_workers():
    return min(4, (os.cpu_count() or 2))


def make_predictions(d_set, processing_function, model, jitters=7):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    device = next(model.parameters()).device
    ds = AptosTestDataset(df, images_dir, processing_function)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=_num_workers(),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_num_workers() > 0),
    )

    total = len(ds)
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)
    id_order = []

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    start_idx = 0
    with torch.no_grad():
        for batch_imgs_uint8, batch_ids in dl:
            id_order.extend(list(batch_ids))
            bs = batch_imgs_uint8.shape[0]

            batch_pred_jitters = torch.empty(
                (bs, jitters, NUM_CLASSES), dtype=torch.float32
            )

            jit = 0.0
            for j in range(jitters):
                params = dataGenerator_params(jit)
                rng = np.random.RandomState(1337 + j)  # deterministic per jitter

                x = apply_tta_batch_uint8(batch_imgs_uint8, params, rng).to(
                    device, non_blocking=True
                )
                x = (x - mean) / std

                pred = model(x).detach().cpu()
                batch_pred_jitters[:, j, :] = pred

                jit += 0.0075

            batch_pred = torch.median(batch_pred_jitters, dim=1).values.numpy()
            predictions[start_idx : start_idx + bs] = batch_pred
            start_idx += bs
            print(f"{start_idx}/{total} done")

    return np.asarray(id_order), predictions




## === cell 5
def prediction_convert_sum(predictions, thresholds):
    thresholded = np.zeros(predictions.shape, dtype=np.int32)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = (predictions[:, i] > thresholds[i]).astype(np.int32)
    y_val = thresholded.sum(axis=1) - 1
    return y_val


def prediction_convert_highest(predictions, thresholds):
    thresholded = np.zeros(predictions.shape, dtype=np.int32)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = (predictions[:, i] > thresholds[i]).astype(np.int32)

    y_val = np.zeros((predictions.shape[0],), dtype=int)
    for i in range(predictions.shape[0]):
        for j in range(4, -1, -1):
            if thresholded[i][j]:
                y_val[i] = j
                break
    return y_val


def label_convert(preds):
    y_val = preds > 0.5
    return y_val.astype(int).sum(axis=1) - 1


def quadratic_weighted_kappa(y_true, y_pred, num_ratings=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    N = num_ratings
    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < N and 0 <= b < N:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=N).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=N).astype(np.float64)
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
    return 1.0 - num / den


def severity_score_from_multilabel(predictions):
    w = np.arange(NUM_CLASSES, dtype=np.float32)
    return (predictions * w[None, :]).sum(axis=1)


def apply_cutpoints(score, cutpoints):
    c1, c2, c3, c4 = cutpoints
    return (
        (score > c1).astype(np.int64)
        + (score > c2).astype(np.int64)
        + (score > c3).astype(np.int64)
        + (score > c4).astype(np.int64)
    )


def tune_cutpoints_on_val(
    scores, y_true, init_cutpoints=(0.5, 1.5, 2.5, 3.5), step=0.05, iters=2
):
    cut = np.array(init_cutpoints, dtype=np.float32)

    def _score(cutpoints):
        pred = apply_cutpoints(scores, cutpoints)
        return quadratic_weighted_kappa(y_true, pred, num_ratings=5)

    best = _score(cut)
    for _ in range(iters):
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for delta in (-step, step):
                    cand = cut.copy()
                    cand[i] += delta
                    if not (cand[0] < cand[1] < cand[2] < cand[3]):
                        continue
                    if cand[0] < -0.5 or cand[3] > 4.5:
                        continue
                    s = _score(cand)
                    if s > best:
                        cut, best = cand, s
                        improved = True
    return cut, best




## === cell 6
torch.manual_seed(2020)
np.random.seed(2020)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = create_model(NORMAL_WEIGHTS)

train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
rng = np.random.RandomState(2020)
val_idx = rng.choice(len(train_df), size=min(256, len(train_df)), replace=False)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

val_images_dir = f"{INPUT_FOLDER}train_images/"
val_ds = AptosTestDataset(val_df[["id_code"]], val_images_dir, processBenNormal)
val_dl = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=_num_workers(),
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers() > 0),
)

device = next(model.parameters()).device

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

val_preds = np.zeros((len(val_ds), NUM_CLASSES), dtype=np.float32)
start_idx = 0
with torch.no_grad():
    for batch_imgs_uint8, _ in val_dl:
        bs = batch_imgs_uint8.shape[0]
        batch_pred_jitters = torch.empty((bs, 7, NUM_CLASSES), dtype=torch.float32)

        jit = 0.0
        for j in range(7):
            params = dataGenerator_params(jit)
            rngj = np.random.RandomState(1337 + j)

            x = apply_tta_batch_uint8(batch_imgs_uint8, params, rngj).to(
                device, non_blocking=True
            )
            x = (x - mean) / std

            pred = model(x).detach().cpu()
            batch_pred_jitters[:, j, :] = pred
            jit += 0.0075

        batch_pred = torch.median(batch_pred_jitters, dim=1).values.numpy()
        val_preds[start_idx : start_idx + bs] = batch_pred
        start_idx += bs

val_scores = severity_score_from_multilabel(val_preds)
y_val_true = val_df["diagnosis"].astype(int).values

init_cutpoints = (0.5, 1.5, 2.5, 3.5)
cutpoints, val_kappa = tune_cutpoints_on_val(
    val_scores, y_val_true, init_cutpoints=init_cutpoints, step=0.05, iters=2
)
print("Tuned cutpoints:", cutpoints, "val QWK:", val_kappa)

test_ids, preds = make_predictions("test", processBenNormal, model, jitters=7)
test_scores = severity_score_from_multilabel(preds)
test_classes = apply_cutpoints(test_scores, cutpoints)
test_classes = np.clip(test_classes, 0, 4).astype(np.int64)

unique, counts = np.unique(test_classes, return_counts=True)
print("Test class distribution:", dict(zip(unique.tolist(), counts.tolist())))
if unique.size == 1:
    print(
        "WARNING: All test predictions are the same class. This usually indicates missing/invalid weights "
        "and will likely score near 0.0."
    )

sample_sub_path = os.path.join(INPUT_FOLDER, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame({"id_code": test_ids, "diagnosis": test_classes})

if pred_df["id_code"].duplicated().any():
    dup = pred_df.loc[pred_df["id_code"].duplicated(), "id_code"].iloc[0]
    raise RuntimeError(f"Duplicate id_code in predictions (example): {dup}")

sub = sample_sub[["id_code"]].merge(pred_df, on="id_code", how="left")

missing = sub["diagnosis"].isna().sum()
if missing != 0:
    missing_ids = sub.loc[sub["diagnosis"].isna(), "id_code"].head(5).tolist()
    raise RuntimeError(
        f"Missing predictions for {missing} test ids. Examples: {missing_ids}"
    )

sub["diagnosis"] = sub["diagnosis"].astype(np.int64)
sub.to_csv("submission.csv", index=False)

print(sub.head(10))
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2416294533.py in <cell line: 0>()
      6 torch.backends.cudnn.benchmark = False
      7 
----> 8 model = create_model(NORMAL_WEIGHTS)
      9 
     10 train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))

/tmp/ipykernel_11/163752565.py in create_model(weights_path)
     92     # Metric-relevant fix: do not silently proceed with random head; that typically yields ~0 QWK.
     93     if weights_path is None or (not os.path.exists(weights_path)):
---> 94         raise FileNotFoundError(
     95             "No compatible PyTorch checkpoint found to load into this DenseNetMultiLabel model. "
     96             "A random head usually scores near 0.0 QWK. "

FileNotFoundError: No compatible PyTorch checkpoint found to load into this DenseNetMultiLabel model. A random head usually scores near 0.0 QWK. Please add a .pth/.pt checkpoint to ../input (or /kaggle/input) and rerun.

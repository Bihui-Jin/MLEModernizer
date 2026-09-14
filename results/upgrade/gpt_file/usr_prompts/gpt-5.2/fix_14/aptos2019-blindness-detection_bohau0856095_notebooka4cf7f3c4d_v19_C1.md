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

0.9053000843210448

# 6. Current score

0.73564

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the failing `pip install` of a missing local timm wheel and instead rely on the already-installed `timm` package. Then I fix the missing weights path by automatically searching common Kaggle input locations for `D5_regre_50epoch.pkl` and, if it truly doesn’t exist, fall back to running the same model architecture with default (untrained) weights so the notebook still completes and writes a valid `submission.csv`. I also make the device selection safe (use CPU if CUDA isn’t available) and make inference robust (no-grad, deterministic ordering, proper dtype/shape) without changing the model logic or prediction thresholds. Finally, I ensure a submission CSV with exactly `id_code,diagnosis` is always produced.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the model is running with random weights (because `D5_regre_50epoch.pkl` is not found), so the minimal path toward the target is to (1) reliably locate the provided weight file inside the competition dataset tree and load it, and (2) ensure the test `id_code` order exactly matches `sample_submission.csv` (some Kaggle scorers can be sensitive to ordering/alignment issues). I keep the exact same model, transforms, and thresholds, only improving weight discovery/loading robustness and aligning output order to the official sample submission. This should move your score upward toward the target without changing the core logic. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with either (a) missing pretrained weights (random model) or (b) weights loading but not actually matching the model, leaving most parameters uninitialized. I make the smallest changes that (1) more reliably find the weight file anywhere under `/kaggle/input` or `/kaggle/data`, (2) load it in a way that handles common checkpoint formats (including nested `model`, `net`, etc.) while preserving the same architecture and thresholds, and (3) add a hard sanity check so we don’t silently submit random-weight predictions if the load mostly failed. This should move the score upward toward your target without changing the model, transforms, or post-processing logic. The script still always produce a valid `submission.csv`.'
- What this solution (achieved 0.64566) has done: 'Your 0.0 score is almost certainly because the script is still running with random weights (the checkpoint isn’t present in this dataset), so the smallest legitimate move toward the 0.9053 target is to train the exact same model architecture on `train.csv` and then run inference on `test.csv`. To keep core logic intact, I only add a minimal training phase (same network, same sigmoid-regression output, same thresholds) and keep your preprocessing and submission ordering aligned to `sample_submission.csv`. This should raise the score substantially from 0.0 while still producing a valid `submission.csv` end-to-end within the time limit by training a small number of epochs and using a basic train/val split for sanity. If your environment later includes `D5_regre_50epoch.pkl`, the code still prefer loading it and skip training.'
- What this solution (achieved -0.00265) has done: 'To move your score up toward the 0.9053 target without changing the model architecture or the regression-to-class thresholds, the smallest reliable gain is to make the training phase stronger and better aligned with the QWK metric while keeping the same loss and prediction logic. I (1) add lightweight data augmentation that’s standard for fundus images (flip/rotate/color jitter) without changing input size or normalization, (2) use a stratified image-level split with a fixed seed as you already do, (3) add a learning-rate scheduler and modest weight decay to stabilize training, and (4) train for a few more epochs (still within the 600s budget on GPU) while keeping inference/submission ordering identical to `sample_submission.csv`. These changes should increase generalization and push QWK upward from 0.64566 toward the target band.'
- What this solution (achieved 0.67834) has done: 'The timeout is dominated by (1) accidentally training EfficientNet-B5 when the weight file isn’t found, and (2) slow input pipelines (PIL decode + transforms) with low DataLoader parallelism and no worker persistence. The optimizations below keep the exact same model, transforms, loss, and training loop semantics, but make execution reliably fit in 600s by (a) avoiding expensive recursive glob searches for weights when they don’t exist, and (b) making DataLoaders faster via persistent workers, prefetching, and more workers on CPU/GPU while keeping determinism. Inference is also sped up by vectorizing submission construction and using an equivalent vectorized `regress2class` implementation (same thresholds, same outputs). No approximations, sampling, early stopping, or reduced precision are introduced.'
- What this solution (achieved 0.71029) has done: 'Your current score (0.67834) is far below the target (0.9053), so we should improve performance with the smallest changes that keep your model/training/prediction semantics intact. The most direct gain for QWK here is to tune the 4 regression→class thresholds on a held-out validation set (same regression model, same MSE loss, same sigmoid*4.5 output), because QWK is very sensitive to discretization boundaries. I keep your exact architecture, transforms, optimizer, scheduler, and training loop, but add a tiny post-training step that searches for better thresholds using the already-created validation split, then uses those thresholds for test inference. If pretrained weights are found and loaded, I still do the same lightweight threshold tuning (no retraining) to push predictions closer to the QWK metric.'
- What this solution (achieved 0.72583) has done: 'You’re currently far below the target (0.710 vs 0.905, higher-is-better), so the smallest reliable improvement is to better match the QWK metric without changing your model, loss, or training loop. I keep your exact EfficientNetB5+GeM+sigmoid*4.5 regression and MSE training, but replace the coarse 7-point delta grid with a deterministic coordinate-descent threshold optimizer on the validation split, which typically yields a meaningful QWK lift because discretization dominates QWK. I also ensure the tuned thresholds are clamped to valid output range and strictly increasing, preventing degenerate solutions that hurt score. Everything else (data paths, transforms, epochs, inference order and submission format) stays the same.'
- What this solution (achieved 0.73564) has done: 'The timeout is dominated by the 5-fold OOF threshold tuning path, which currently precomputes and stores *all* 3295 eval images as a huge CPU tensor and then repeatedly slices/copies it per fold, causing massive CPU RAM pressure and slowdowns. I keep the same OOF evaluation and the same coordinate-descent threshold tuner, but compute OOF predictions by running inference fold-by-fold directly from a lightweight dataset (no giant cached tensor), and cache only the model outputs (a small vector) to disk for reuse across runs. I also make `regress2class` and threshold application avoid redundant CPU transfers (inputs are already on CPU during eval), and ensure DataLoader settings are efficient but deterministic. These changes preserve the exact model, transforms, and tuning semantics while removing the main bottleneck.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


_DL_GENERATOR = torch.Generator()
_DL_GENERATOR.manual_seed(42)



## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]
_THRESHOLD_T = torch.tensor(threshold, dtype=torch.float32)


def set_thresholds(thr_list):
    global threshold, _THRESHOLD_T
    threshold = [float(x) for x in thr_list]
    _THRESHOLD_T = torch.tensor(threshold, dtype=torch.float32)


def regress2class(out):
    if isinstance(out, np.ndarray):
        out = torch.from_numpy(out)
    out = out.detach()
    if out.ndim == 2 and out.size(1) == 1:
        out = out.view(-1)
    else:
        out = out.reshape(-1)
    out_cpu = (
        out
        if out.device.type == "cpu"
        else out.to("cpu", dtype=torch.float32, non_blocking=False)
    )
    out_cpu = out_cpu.to(dtype=torch.float32, copy=False)
    return (out_cpu[:, None] >= _THRESHOLD_T[None, :]).sum(dim=1).to(torch.float32)


_THR_CACHE = {}


def _get_thr_t_cached(thr_list):
    key = tuple(float(x) for x in thr_list)
    t = _THR_CACHE.get(key)
    if t is None:
        t = torch.tensor(key, dtype=torch.float32)
        _THR_CACHE[key] = t
    return t


def _regress2class_with_thresholds(out_1d_cpu_float, thr_list):
    thr_t = _get_thr_t_cached(thr_list)
    out_cpu = out_1d_cpu_float.detach()
    if out_cpu.ndim != 1:
        out_cpu = out_cpu.view(-1)
    if out_cpu.device.type != "cpu":
        out_cpu = out_cpu.to("cpu", dtype=torch.float32, non_blocking=False)
    else:
        out_cpu = out_cpu.to(dtype=torch.float32, copy=False)
    return (out_cpu[:, None] >= thr_t[None, :]).sum(dim=1).to(torch.int64).numpy()




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
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=False)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out




## === cell 3
TRAIN_CSV_CANDIDATES = [
    "../input/aptos2019-blindness-detection/train.csv",
    "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "/kaggle/data/aptos2019-blindness-detection/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/input/train.csv",
]

TEST_CSV_CANDIDATES = [
    "../input/aptos2019-blindness-detection/test.csv",
    "/kaggle/input/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/input/test.csv",
]

SAMPLE_SUB_CANDIDATES = [
    "../input/aptos2019-blindness-detection/sample_submission.csv",
    "/kaggle/input/aptos2019-blindness-detection/sample_submission.csv",
    "/kaggle/data/aptos2019-blindness-detection/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


train_csv_path = _first_existing(TRAIN_CSV_CANDIDATES)
test_csv_path = _first_existing(TEST_CSV_CANDIDATES)
sample_sub_path = _first_existing(SAMPLE_SUB_CANDIDATES)

if train_csv_path is None:
    raise FileNotFoundError(f"Could not find train.csv. Tried: {TRAIN_CSV_CANDIDATES}")
if test_csv_path is None:
    raise FileNotFoundError(f"Could not find test.csv. Tried: {TEST_CSV_CANDIDATES}")
if sample_sub_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv. Tried: {SAMPLE_SUB_CANDIDATES}"
    )

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample_df = pd.read_csv(sample_sub_path)

test_ids = sample_df["id_code"].astype(str).tolist()

input_size = 384

train_transforms = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=15, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ColorJitter(
            brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

eval_transforms = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

TRAIN_IMG_DIR_CANDIDATES = [
    os.path.join(os.path.dirname(train_csv_path), "train_images"),
    os.path.join(os.path.dirname(sample_sub_path), "train_images"),
    "../input/aptos2019-blindness-detection/train_images",
    "/kaggle/input/aptos2019-blindness-detection/train_images",
    "/kaggle/data/aptos2019-blindness-detection/train_images",
    "/kaggle/data/train_images",
    "/kaggle/input/train_images",
]
TEST_IMG_DIR_CANDIDATES = [
    os.path.join(os.path.dirname(test_csv_path), "test_images"),
    os.path.join(os.path.dirname(sample_sub_path), "test_images"),
    "../input/aptos2019-blindness-detection/test_images",
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
    "/kaggle/data/test_images",
    "/kaggle/input/test_images",
]


def _first_existing_dir(paths):
    for d in paths:
        if os.path.isdir(d):
            return d
    return None


train_img_dir = _first_existing_dir(TRAIN_IMG_DIR_CANDIDATES)
test_img_dir = _first_existing_dir(TEST_IMG_DIR_CANDIDATES)

if train_img_dir is None:
    raise FileNotFoundError(
        f"Could not find train_images directory. Tried: {TRAIN_IMG_DIR_CANDIDATES}"
    )
if test_img_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images directory. Tried: {TEST_IMG_DIR_CANDIDATES}"
    )


class AptosDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_label=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def _build_path_cache(self):
        cache = []
        for i in range(len(self.df)):
            id_code = str(self.df.iloc[i]["id_code"])
            png = os.path.join(self.img_dir, f"{id_code}.png")
            if os.path.exists(png):
                cache.append(png)
                continue
            jpg = os.path.join(self.img_dir, f"{id_code}.jpg")
            if os.path.exists(jpg):
                cache.append(jpg)
                continue
            cache.append(png)
        return cache

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_code = str(row["id_code"])
        if hasattr(self, "_path_cache"):
            path = self._path_cache[idx]
        else:
            self._path_cache = self._build_path_cache()
            path = self._path_cache[idx]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing image for id_code={id_code}: {path}")
        with Image.open(path) as im:
            img = im.convert("RGB")
        img = self.transform(img)
        if self.has_label:
            y = float(row["diagnosis"])
            return img, torch.tensor([y], dtype=torch.float32)
        return img, id_code


def _find_weight_file(filename="D5_regre_50epoch.pkl"):
    candidates = [
        f"../input/weights/{filename}",
        f"/kaggle/input/weights/{filename}",
        f"../input/aptos2019-blindness-detection/{filename}",
        f"/kaggle/input/aptos2019-blindness-detection/{filename}",
        f"/kaggle/data/aptos2019-blindness-detection/{filename}",
        f"/kaggle/data/{filename}",
        f"/kaggle/input/{filename}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    limited_patterns = [
        f"../input/*/{filename}",
        f"/kaggle/input/*/{filename}",
        f"/kaggle/data/*/{filename}",
    ]
    hits_all = []
    for pat in limited_patterns:
        hits_all.extend(glob.glob(pat))
    hits_all = [h for h in hits_all if os.path.isfile(h)]
    hits_all.sort(key=lambda x: (len(x), x))
    return hits_all[0] if hits_all else None


def _unwrap_state_dict(ckpt_obj):
    state = ckpt_obj
    if isinstance(state, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    return state


def _load_weights_if_available(net, filename="D5_regre_50epoch.pkl"):
    weight_path = _find_weight_file(filename)
    if weight_path is None:
        return False, None
    print("Loading weights from:", weight_path)
    ckpt = torch.load(weight_path, map_location="cpu")
    state = _unwrap_state_dict(ckpt)

    if not isinstance(state, dict):
        raise RuntimeError("Loaded checkpoint is not a state_dict-like mapping.")

    new_state = {}
    for k, v in state.items():
        nk = k
        if isinstance(nk, str) and nk.startswith("module."):
            nk = nk[len("module.") :]
        new_state[nk] = v
    state = new_state

    missing, unexpected = net.load_state_dict(state, strict=False)
    total_params = len(net.state_dict().keys())
    miss_ratio = len(missing) / max(1, total_params)
    print(
        f"State dict load report: total={total_params}, missing={len(missing)}, "
        f"unexpected={len(unexpected)}, missing_ratio={miss_ratio:.3f}"
    )
    loaded_ok = miss_ratio < 0.20
    if not loaded_ok:
        raise RuntimeError(
            "Checkpoint was found but did not match the model sufficiently "
            f"(missing_ratio={miss_ratio:.3f})."
        )
    return True, weight_path




## === cell 4
net = Regressor()
loaded_ok, weight_path = _load_weights_if_available(net, "D5_regre_50epoch.pkl")
net = net.to(device)

train_split, val_split = train_test_split(
    train_df, test_size=0.15, random_state=42, stratify=train_df["diagnosis"]
)

if not loaded_ok:
    train_ds = AptosDataset(
        train_split, train_img_dir, train_transforms, has_label=True
    )
    val_ds = AptosDataset(val_split, train_img_dir, eval_transforms, has_label=True)

    batch_size = 8 if torch.cuda.is_available() else 4
    if torch.cuda.is_available():
        num_workers = min(8, os.cpu_count() or 2)
    else:
        num_workers = min(4, max(1, (os.cpu_count() or 2) // 2))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=_DL_GENERATOR,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=_DL_GENERATOR,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=6, eta_min=2e-5
    )
    epochs = 6 if torch.cuda.is_available() else 2

    for ep in range(1, epochs + 1):
        net.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * xb.size(0)

        scheduler.step()

        net.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                out = net(xb).detach().cpu().view(-1)
                pred_cls = regress2class(out).numpy().astype(int).tolist()
                val_preds.extend(pred_cls)
                val_true.extend(yb.view(-1).numpy().astype(int).tolist())

        kappa = cohen_kappa_score(val_true, val_preds, weights="quadratic")
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={ep}/{epochs} lr={lr_now:.6f} train_mse={running/len(train_ds):.5f} val_qwk={kappa:.5f}"
        )

net.eval()




## === cell 5
def _project_thresholds(thr, lo=0.0, hi=4.5, min_gap=1e-4):
    t = [float(x) for x in thr]
    t = [min(max(x, lo), hi) for x in t]
    for i in range(1, 4):
        if t[i] <= t[i - 1] + min_gap:
            t[i] = t[i - 1] + min_gap
    for i in range(3, -1, -1):
        if t[i] >= hi - (3 - i) * min_gap:
            t[i] = hi - (3 - i) * min_gap
    for i in range(1, 4):
        if t[i] <= t[i - 1] + min_gap:
            t[i] = t[i - 1] + min_gap
    return t


def _qwk_from_preds(y_true_int, y_pred_int, n_classes=5):
    y_true_int = np.asarray(y_true_int, dtype=np.int64)
    y_pred_int = np.asarray(y_pred_int, dtype=np.int64)

    idx = y_true_int * n_classes + y_pred_int
    O = (
        np.bincount(idx, minlength=n_classes * n_classes)
        .reshape(n_classes, n_classes)
        .astype(np.float64, copy=False)
    )

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = (np.subtract.outer(np.arange(n_classes), np.arange(n_classes)) ** 2).astype(
        np.float64
    )
    W /= (n_classes - 1) ** 2

    denom = (W * E).sum()
    if denom == 0.0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


_KAPPA_THR_MEMO = {}


def _kappa_for_thr(out_1d_cpu_float, y_true_int, thr):
    key = tuple(round(float(x), 6) for x in thr)
    v = _KAPPA_THR_MEMO.get(key)
    if v is not None:
        return float(v)
    pred = _regress2class_with_thresholds(out_1d_cpu_float, thr)
    k = float(_qwk_from_preds(y_true_int, pred, n_classes=5))
    _KAPPA_THR_MEMO[key] = k
    return k


def _tune_thresholds_coordinate_descent(out_1d_cpu_float, y_true_int, base_thr):
    thr = _project_thresholds(base_thr)
    best_k = _kappa_for_thr(out_1d_cpu_float, y_true_int, thr)

    step_schedule = [0.20, 0.10, 0.05, 0.02, 0.01]
    mults = [-3, -2, -1, 0, 1, 2, 3]

    for step in step_schedule:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                best_local_thr = thr
                best_local_k = best_k

                for m in mults:
                    cand = thr.copy()
                    cand[i] = cand[i] + m * step
                    cand = _project_thresholds(cand)
                    k = _kappa_for_thr(out_1d_cpu_float, y_true_int, cand)
                    if k > best_local_k + 1e-12:
                        best_local_k = k
                        best_local_thr = cand

                if best_local_k > best_k + 1e-12:
                    thr = best_local_thr
                    best_k = best_local_k
                    improved = True

    return thr, best_k


def _oof_cache_path(train_csv_path, input_size):
    base = os.path.splitext(os.path.basename(train_csv_path))[0]
    tag = f"{base}_oof_outs_{input_size}_mean384_258_174_std124_089_094_v1.pt"
    return os.path.join("/kaggle/working", tag)


def _compute_oof_outs_foldwise(
    df,
    img_dir,
    transform,
    y_all_np,
    batch_size,
    num_workers,
    cache_path,
):
    if os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        outs = obj.get("oof_outs", None)
        ids = obj.get("id", None)
        if outs is None or ids is None:
            raise RuntimeError(
                "OOF cache missing required keys; delete cache and retry."
            )
        expected_ids = df["id_code"].astype(str).values
        if len(ids) != len(expected_ids) or not np.array_equal(
            np.asarray(ids, dtype=str), expected_ids
        ):
            raise RuntimeError(
                "OOF cache exists but id_code order mismatch; delete cache and retry."
            )
        return outs.to(dtype=torch.float32).contiguous()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_outs = torch.empty(len(df), dtype=torch.float32)

    for fold, (_, va_idx) in enumerate(skf.split(df, y_all_np), 1):
        va_df = df.iloc[va_idx].reset_index(drop=True)
        va_ds = AptosDataset(va_df, img_dir, transform, has_label=True)
        va_loader = DataLoader(
            va_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=_DL_GENERATOR,
        )

        outs_fold = torch.empty(len(va_df), dtype=torch.float32)
        off = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device, non_blocking=True)
                out = net(xb).detach().cpu().view(-1).to(torch.float32)
                bsz = out.numel()
                outs_fold[off : off + bsz] = out
                off += bsz

        if off != len(va_df):
            raise RuntimeError("Fold inference size mismatch.")

        oof_outs[va_idx] = outs_fold

        pred_cls_fold = regress2class(outs_fold).numpy().astype(int)
        kappa_fold = _qwk_from_preds(y_all_np[va_idx], pred_cls_fold, n_classes=5)
        print(f"OOF fold {fold}/5 QWK (base thresholds): {kappa_fold:.5f}")

    torch.save(
        {"oof_outs": oof_outs, "id": df["id_code"].astype(str).values}, cache_path
    )
    return oof_outs.contiguous()


use_oof_threshold_tuning = True

if use_oof_threshold_tuning:
    y_all_np = train_df["diagnosis"].astype(int).values
    batch_size_thr = 32 if torch.cuda.is_available() else 8
    if torch.cuda.is_available():
        num_workers_thr = min(8, os.cpu_count() or 2)
    else:
        num_workers_thr = min(4, max(1, (os.cpu_count() or 2) // 2))

    cache_path = _oof_cache_path(train_csv_path, input_size)
    oof_outs = _compute_oof_outs_foldwise(
        train_df,
        train_img_dir,
        eval_transforms,
        y_all_np,
        batch_size_thr,
        num_workers_thr,
        cache_path,
    )

    y_true_int = y_all_np
    base_thr = threshold
    _KAPPA_THR_MEMO.clear()
    best_thr, best_kappa = _tune_thresholds_coordinate_descent(
        oof_outs, y_true_int, base_thr
    )

    print(
        "Base thresholds:",
        [round(x, 4) for x in base_thr],
        "OOF_qwk:",
        round(_kappa_for_thr(oof_outs, y_true_int, _project_thresholds(base_thr)), 5),
    )
    print(
        "Tuned thresholds:",
        [round(x, 4) for x in best_thr],
        "OOF_qwk:",
        round(best_kappa, 5),
    )
    set_thresholds(best_thr)
else:
    val_ds_for_thr = AptosDataset(
        val_split, train_img_dir, eval_transforms, has_label=True
    )

    batch_size_thr = 16 if torch.cuda.is_available() else 4
    if torch.cuda.is_available():
        num_workers_thr = min(8, os.cpu_count() or 2)
    else:
        num_workers_thr = min(4, max(1, (os.cpu_count() or 2) // 2))

    val_loader_thr = DataLoader(
        val_ds_for_thr,
        batch_size=batch_size_thr,
        shuffle=False,
        num_workers=num_workers_thr,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers_thr > 0),
        prefetch_factor=2 if num_workers_thr > 0 else None,
        worker_init_fn=_seed_worker if num_workers_thr > 0 else None,
        generator=_DL_GENERATOR,
    )

    val_outs = []
    val_y = []
    with torch.no_grad():
        for xb, yb in val_loader_thr:
            xb = xb.to(device, non_blocking=True)
            out = net(xb).detach().cpu().view(-1)
            val_outs.append(out)
            val_y.append(yb.detach().cpu().view(-1))
    val_outs = torch.cat(val_outs, dim=0).to(torch.float32)
    val_y = torch.cat(val_y, dim=0).to(torch.float32)

    y_true_int = val_y.numpy().astype(int)
    base_thr = threshold
    _KAPPA_THR_MEMO.clear()
    best_thr, best_kappa = _tune_thresholds_coordinate_descent(
        val_outs, y_true_int, base_thr
    )
    print(
        "Base thresholds:",
        [round(x, 4) for x in base_thr],
        "val_qwk:",
        round(_kappa_for_thr(val_outs, y_true_int, _project_thresholds(base_thr)), 5),
    )
    print(
        "Tuned thresholds:",
        [round(x, 4) for x in best_thr],
        "val_qwk:",
        round(best_kappa, 5),
    )
    set_thresholds(best_thr)



## === cell 6
test_infer_df = pd.DataFrame({"id_code": test_ids})
test_ds = AptosDataset(test_infer_df, test_img_dir, eval_transforms, has_label=False)

batch_size = 32 if torch.cuda.is_available() else 8
if torch.cuda.is_available():
    num_workers = min(8, os.cpu_count() or 2)
else:
    num_workers = min(4, max(1, (os.cpu_count() or 2) // 2))

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=_DL_GENERATOR,
)

all_ids = []
all_preds = []
with torch.no_grad():
    for xb, id_codes in test_loader:
        xb = xb.to(device, non_blocking=True)
        out = net(xb).detach().cpu().view(-1)
        preds = regress2class(out).to(torch.int64).numpy()
        all_preds.append(preds)
        all_ids.extend([str(x) for x in id_codes])

all_preds = np.concatenate(all_preds, axis=0).astype(int)
submission = pd.DataFrame({"id_code": all_ids, "diagnosis": all_preds})



## === cell 7
df = submission.copy()
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = sample_df[["id_code"]].merge(df, on="id_code", how="left")
if df["diagnosis"].isna().any():
    df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Final thresholds used:", [round(x, 4) for x in threshold])

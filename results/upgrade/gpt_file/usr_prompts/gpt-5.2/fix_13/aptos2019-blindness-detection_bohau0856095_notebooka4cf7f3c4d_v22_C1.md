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

0.9004503090121132

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weights/runtime issues so the notebook runs end-to-end and always writes a non-empty `submission.csv`. Specifically, I (1) remove the failing `pip install` and add robust path checks for the weights file, (2) make device selection safe by falling back to CPU when CUDA isn’t available, and (3) ensure image loading/trim never returns `None` and predictions are clamped to valid classes 0–4. If the external weight file is absent, the code still generate a valid submission using the untrained model (score be low, but it run and produce a correctly formatted CSV).'
- What this solution (achieved 0.69778) has done: 'Your current 0.0 score is because the model is effectively untrained (the weights file isn’t present), so predictions are near-constant and kappa collapses. To move the score upward with minimal core-logic change, I keep your exact model/forward pass and inference loop, but (1) add a small training step on the provided `train.csv` images to learn usable weights, and (2) tune the four class thresholds on a held-out validation split specifically to maximize quadratic weighted kappa (the competition metric) while keeping the same “regression then threshold to classes” semantics. I also fix one bug where your `regress2class()` used `.data` and ignored device/shape, and I ensure the submission ordering matches `test.csv` exactly. This should reliably move the score toward your target without changing the model architecture or loss type (still regression with MSE).'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import time
import hashlib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from PIL import Image, ImageChops, ImageFile

import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    print("Warning: could not enable deterministic algorithms:", repr(e))
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 200_000_000


def trim(im: Image.Image) -> Image.Image:
    """
    Ensure trim always returns an image (never None), otherwise transforms(...) can crash.
    """
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -10)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im




## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out_1d: torch.Tensor, thr=None) -> torch.Tensor:
    """
    Map regression output to ordinal classes via thresholds.
    """
    if thr is None:
        thr = threshold
    thr_t = torch.as_tensor(thr, device=out_1d.device, dtype=out_1d.dtype).view(1, -1)
    return (out_1d.view(-1, 1) >= thr_t).sum(dim=1).to(torch.int64)




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
BASE = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

input_size = 384

train_tfms = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)
test_tfms = train_tfms


CACHE_ROOT = "/kaggle/working/aptos_tensor_cache"
os.makedirs(CACHE_ROOT, exist_ok=True)


def _tfms_fingerprint(tfms) -> str:
    s = repr(tfms) + f"|input_size={input_size}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


_TFMS_FP = _tfms_fingerprint(train_tfms)


def _cache_path_for(img_path: str, tfms_fp: str) -> str:
    key = hashlib.sha1((img_path + "|" + tfms_fp).encode("utf-8")).hexdigest()
    return os.path.join(CACHE_ROOT, f"{key}.npy")


def _load_trim_pil_no_cache(img_path: str) -> Image.Image:
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = trim(img)
        return img.copy()


def _numpy_load_cached_array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def _load_preprocess_tensor_cached(img_path: str, tfms, tfms_fp: str) -> torch.Tensor:
    cpath = _cache_path_for(img_path, tfms_fp)
    if os.path.exists(cpath):
        arr = _numpy_load_cached_array(cpath)
        return torch.from_numpy(np.array(arr, copy=False))
    img = _load_trim_pil_no_cache(img_path)
    x = tfms(img)  # float32 CHW
    arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
    arr = np.ascontiguousarray(arr)
    tmp = cpath + f".tmp{os.getpid()}"
    np.save(tmp, arr)
    os.replace(tmp, cpath)
    return x


def build_tensor_cache(id_codes, img_dir: str, tfms, tfms_fp: str, desc: str):
    total = len(pd.Series(id_codes, dtype=str))
    print(
        f"[cache] {desc}: lazy mode enabled, skipping upfront build for {total} images."
    )


class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, tfms, with_label: bool, tfms_fp: str):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfms = tfms
        self.with_label = with_label
        self.tfms_fp = tfms_fp

        self.id_codes = self.df["id_code"].astype(str).to_numpy()

        img_paths = [os.path.join(self.img_dir, f"{idc}.png") for idc in self.id_codes]
        self.cache_paths = [_cache_path_for(p, self.tfms_fp) for p in img_paths]
        self.img_paths = np.asarray(img_paths, dtype=object)

        self._cached_flags = np.fromiter(
            (os.path.exists(p) for p in self.cache_paths),
            dtype=np.bool_,
            count=len(self.cache_paths),
        )

        self.labels = None
        if with_label:
            self.labels = self.df["diagnosis"].astype(np.float32).to_numpy()

    def __len__(self):
        return self.id_codes.shape[0]

    def __getitem__(self, i):
        cpath = self.cache_paths[i]
        if self._cached_flags[i] and os.path.exists(cpath):
            arr = _numpy_load_cached_array(cpath)
            x = torch.from_numpy(np.array(arr, copy=False))
        else:
            x = _load_preprocess_tensor_cached(
                self.img_paths[i], self.tfms, self.tfms_fp
            )
            self._cached_flags[i] = True

        if self.with_label:
            y = torch.tensor(float(self.labels[i]), dtype=torch.float32)
            return x, y
        return x, self.id_codes[i]


label_counts = train_df["diagnosis"].value_counts().sort_index()
class_weights = (1.0 / label_counts).values
class_weights = class_weights / class_weights.mean()
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print("Label counts:", label_counts.to_dict())
print("Class weights (normalized):", class_weights.tolist())

build_tensor_cache(
    train_df["id_code"].astype(str).values, TRAIN_IMG_DIR, train_tfms, _TFMS_FP, "train"
)
build_tensor_cache(
    test_df["id_code"].astype(str).values, TEST_IMG_DIR, test_tfms, _TFMS_FP, "test"
)




## === cell 4
def evaluate_regression(model, loader):
    model.eval()
    ys = []
    outs = []
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb).squeeze(1)
            ys.append(yb.detach().cpu().numpy())
            outs.append(out.detach().cpu().numpy())
    y = np.concatenate(ys, axis=0)
    o = np.concatenate(outs, axis=0)
    return y, o


def qwk_from_thresholds(y_true_int, y_pred_reg, thr):
    thr = np.asarray(thr, dtype=np.float64)
    pred_cls = (y_pred_reg[:, None] >= thr[None, :]).sum(axis=1).astype(np.int64)
    pred_cls = np.clip(pred_cls, 0, 4)
    return cohen_kappa_score(y_true_int, pred_cls, weights="quadratic")


def tune_thresholds(y_true_int, y_pred_reg, init_thr):
    """
    Coordinate-ascent threshold tuning to optimize QWK.
    """
    thr = np.array(init_thr, dtype=np.float64)
    best = qwk_from_thresholds(y_true_int, y_pred_reg, thr)

    for _ in range(6):
        improved = False
        for k in range(4):
            candidates = np.linspace(max(0.0, thr[k] - 0.7), min(4.5, thr[k] + 0.7), 31)
            local_best_thr = thr[k]
            local_best = best
            for c in candidates:
                thr_try = thr.copy()
                thr_try[k] = c
                thr_try = np.sort(np.clip(thr_try, 0.0, 4.5))
                score = qwk_from_thresholds(y_true_int, y_pred_reg, thr_try)
                if score > local_best:
                    local_best = score
                    local_best_thr = c
            if local_best > best + 1e-6:
                thr[k] = local_best_thr
                thr = np.sort(np.clip(thr, 0.0, 4.5))
                best = local_best
                improved = True
        if not improved:
            break
    return thr.tolist(), best


class WeightedMSELoss(nn.Module):
    def __init__(self, class_weights_5: torch.Tensor):
        super().__init__()
        self.register_buffer("w", class_weights_5.view(1, -1))

    def forward(self, pred, target_float):
        t_int = torch.clamp(target_float.round().long(), 0, 4)
        w = self.w[0, t_int]  # [B]
        loss = (pred - target_float) ** 2
        return (w * loss).mean()




## === cell 5
net = Regressor().to(device)

WEIGHTS_CANDIDATES = [
    "../input/weights/D5_regre_70epoch.pkl",
    "/kaggle/input/weights/D5_regre_70epoch.pkl",
]
weights_path = next((p for p in WEIGHTS_CANDIDATES if os.path.exists(p)), None)

if weights_path is not None:
    print("Loading weights:", weights_path)
    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state, strict=True)
    trained_now = False
else:
    print(
        "Weights not found; training model on provided train set to obtain non-trivial predictions."
    )
    trained_now = True




## === cell 6
def _seed_worker(worker_id):
    s = SEED + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


_GLOBAL_WORKER_TENSOR_CACHE = None
_GLOBAL_WORKER_TENSOR_CACHE_MAX = 4096  # more reuse within worker, reduces disk I/O


def _get_worker_cache():
    global _GLOBAL_WORKER_TENSOR_CACHE
    if _GLOBAL_WORKER_TENSOR_CACHE is None:
        _GLOBAL_WORKER_TENSOR_CACHE = {}
    return _GLOBAL_WORKER_TENSOR_CACHE


def _load_tensor_cached_mem_npy(cpath: str) -> torch.Tensor:
    cache = _get_worker_cache()
    x = cache.get(cpath, None)
    if x is not None:
        return x
    arr = _numpy_load_cached_array(cpath)
    x = torch.from_numpy(np.array(arr, copy=False))
    if len(cache) >= _GLOBAL_WORKER_TENSOR_CACHE_MAX:
        cache.pop(next(iter(cache)))
    cache[cpath] = x
    return x


def _dataset_getitem_fast(self, i):
    cpath = self.cache_paths[i]
    cached = getattr(self, "_cached_flags", None)

    if (cached is not None and cached[i]) and os.path.exists(cpath):
        x = _load_tensor_cached_mem_npy(cpath)
    elif os.path.exists(cpath):
        if cached is not None:
            cached[i] = True
        x = _load_tensor_cached_mem_npy(cpath)
    else:
        x = _load_preprocess_tensor_cached(self.img_paths[i], self.tfms, self.tfms_fp)
        if cached is not None:
            cached[i] = True

    if self.with_label:
        y = torch.tensor(float(self.labels[i]), dtype=torch.float32)
        return x, y
    return x, self.id_codes[i]


APTOSDataset.__getitem__ = _dataset_getitem_fast


def _collate_with_ids(batch):
    xs, ids = zip(*batch)
    return torch.stack(xs, dim=0), list(ids)


def _make_loader(ds, batch_size, shuffle):
    use_cuda = device == "cuda"
    num_workers = min(8, os.cpu_count() or 2) if use_cuda else 0
    g = torch.Generator()
    g.manual_seed(SEED)
    collate_fn = _collate_with_ids if not getattr(ds, "with_label", False) else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
        collate_fn=collate_fn,
    )


def train_one_fold_and_predict_oof(fold, tr_idx, va_idx, epochs=4):
    tr_df = train_df.iloc[tr_idx].copy()
    va_df = train_df.iloc[va_idx].copy()

    train_loader = _make_loader(
        APTOSDataset(
            tr_df, TRAIN_IMG_DIR, train_tfms, with_label=True, tfms_fp=_TFMS_FP
        ),
        batch_size=8,
        shuffle=True,
    )
    val_loader = _make_loader(
        APTOSDataset(
            va_df, TRAIN_IMG_DIR, train_tfms, with_label=True, tfms_fp=_TFMS_FP
        ),
        batch_size=8,
        shuffle=False,
    )

    model = Regressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=2e-5
    )
    criterion = WeightedMSELoss(class_weights.to(device))

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb).squeeze(1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            n += xb.size(0)

        scheduler.step()
        train_loss = running / max(1, n)

        y_val, o_val = evaluate_regression(model, val_loader)
        qwk_round = cohen_kappa_score(
            y_val.astype(int),
            np.clip(np.rint(o_val), 0, 4).astype(int),
            weights="quadratic",
        )
        print(
            f"[fold {fold}] Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} - val_qwk(round)={qwk_round:.4f} - time={time.time()-t0:.1f}s"
        )

    y_val, o_val = evaluate_regression(model, val_loader)
    return va_idx, y_val, o_val


if trained_now:
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(train_df), dtype=np.float32)
    oof_true = train_df["diagnosis"].values.astype(np.int64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, oof_true), start=1):
        va_idx2, yv, ov = train_one_fold_and_predict_oof(fold, tr_idx, va_idx, epochs=4)
        oof_pred[va_idx2] = ov.astype(np.float32)

    full_loader = _make_loader(
        APTOSDataset(
            train_df, TRAIN_IMG_DIR, train_tfms, with_label=True, tfms_fp=_TFMS_FP
        ),
        batch_size=8,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5, eta_min=2e-5
    )
    criterion = WeightedMSELoss(class_weights.to(device))

    EPOCHS_FULL = 5
    for epoch in range(1, EPOCHS_FULL + 1):
        net.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for xb, yb in full_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = net(xb).squeeze(1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            n += xb.size(0)

        scheduler.step()
        train_loss = running / max(1, n)
        print(
            f"[full] Epoch {epoch}/{EPOCHS_FULL} - train_loss={train_loss:.4f} - time={time.time()-t0:.1f}s"
        )



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/176210139.py in <cell line: 0>()
    150 
    151     for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, oof_true), start=1):
--> 152         va_idx2, yv, ov = train_one_fold_and_predict_oof(fold, tr_idx, va_idx, epochs=4)
    153         oof_pred[va_idx2] = ov.astype(np.float32)
    154 

/tmp/ipykernel_55/176210139.py in train_one_fold_and_predict_oof(fold, tr_idx, va_idx, epochs)
    114         running = 0.0
    115         n = 0
--> 116         for xb, yb in train_loader:
    117             xb = xb.to(device, non_blocking=True)
    118             yb = yb.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/176210139.py", line 45, in _dataset_getitem_fast
    x = _load_preprocess_tensor_cached(self.img_paths[i], self.tfms, self.tfms_fp)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/1034987379.py", line 67, in _load_preprocess_tensor_cached
    os.replace(tmp, cpath)
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/aptos_tensor_cache/4379675218ba0a119f02604cec7e4b8610d8922c.npy.tmp103' -> '/kaggle/working/aptos_tensor_cache/4379675218ba0a119f02604cec7e4b8610d8922c.npy'


## === cell 7
_CAN_COMPILE = hasattr(torch, "compile")


def _maybe_compile_for_inference(m: nn.Module) -> nn.Module:
    if not _CAN_COMPILE:
        return m
    try:
        return torch.compile(m, mode="reduce-overhead", fullgraph=False)
    except Exception as e:
        print(
            "Warning: torch.compile unavailable/failed, continuing without compile:",
            repr(e),
        )
        return m


if trained_now:
    y_all = train_df["diagnosis"].values.astype(int)
    o_all = oof_pred.astype(np.float64)
    best_thr, best_qwk = tune_thresholds(y_all, o_all, threshold)
    threshold = best_thr
    print("Tuned thresholds (OOF):", threshold)
    print("OOF QWK after tuning:", best_qwk)
else:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    idx = np.arange(len(train_df))
    y_int = train_df["diagnosis"].values.astype(int)
    tr_idx, va_idx = next(splitter.split(idx, y_int))
    va_df = train_df.iloc[va_idx].copy()
    val_loader = _make_loader(
        APTOSDataset(
            va_df, TRAIN_IMG_DIR, train_tfms, with_label=True, tfms_fp=_TFMS_FP
        ),
        batch_size=8,
        shuffle=False,
    )

    net.eval()
    net_infer = _maybe_compile_for_inference(net)
    y_val, o_val = evaluate_regression(net_infer, val_loader)
    best_thr, best_qwk = tune_thresholds(y_val.astype(int), o_val, threshold)
    threshold = best_thr
    print("Tuned thresholds (holdout):", threshold)
    print("Val QWK after tuning:", best_qwk)

net.eval()



## === cell 8
test_loader = _make_loader(
    APTOSDataset(test_df, TEST_IMG_DIR, test_tfms, with_label=False, tfms_fp=_TFMS_FP),
    batch_size=8,
    shuffle=False,
)

net_infer = _maybe_compile_for_inference(net)

rows = []
with torch.inference_mode():
    for xb, id_codes in test_loader:
        xb = xb.to(device, non_blocking=True)
        out = net_infer(xb).squeeze(1)  # [B]
        pred = regress2class(out, thr=threshold).detach().cpu().numpy().astype(int)
        rows.extend(
            [[str(idc), int(np.clip(pi, 0, 4))] for idc, pi in zip(id_codes, pred)]
        )

sub_df = pd.DataFrame(rows, columns=["id_code", "diagnosis"])
sub_df = (
    sub_df.set_index("id_code").loc[test_df["id_code"].astype(str).values].reset_index()
)

assert len(sub_df) == len(test_df) and sub_df["diagnosis"].notna().all()

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2782933739.py in <cell line: 0>()
      9 rows = []
     10 with torch.inference_mode():
---> 11     for xb, id_codes in test_loader:
     12         xb = xb.to(device, non_blocking=True)
     13         out = net_infer(xb).squeeze(1)  # [B]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/176210139.py", line 45, in _dataset_getitem_fast
    x = _load_preprocess_tensor_cached(self.img_paths[i], self.tfms, self.tfms_fp)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/1034987379.py", line 67, in _load_preprocess_tensor_cached
    os.replace(tmp, cpath)
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/aptos_tensor_cache/0292f26ad76e54e963038f06683152b5e99da4b9.npy.tmp130' -> '/kaggle/working/aptos_tensor_cache/0292f26ad76e54e963038f06683152b5e99da4b9.npy'

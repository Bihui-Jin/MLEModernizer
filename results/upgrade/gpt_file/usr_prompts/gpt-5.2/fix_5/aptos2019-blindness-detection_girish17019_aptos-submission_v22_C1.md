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

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
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

0.8401562603955726

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by removing the missing external weight file dependency and instead use torchvision’s built-in EfficientNet-B0 pretrained weights (same architecture) so inference can run end-to-end. I also fix the dataset tensor creation bug (using `np.resize` was corrupting tensors) and ensure deterministic inference by using non-random test transforms. Finally, I make the submission length always match `test.csv` by building the submission directly from `test_csv` and clipping/rounding predictions into valid classes 0–4 to produce a valid `submission.csv`.'
- What this solution (achieved 0.6868) has done: 'Your current 0.0 score is consistent with a model that was never trained for this task (ImageNet EfficientNet-B0 with a randomly initialized 1-unit head), so predictions are essentially noise after rounding. To move toward the 0.84 target while preserving your core pipeline, I keep the same EfficientNet-B0 architecture and inference loop, but train only the replaced classifier head on `train.csv` using the same image preprocessing and normalization you already use. I also switch the head to the correct 5-class output and use `argmax` at inference (matching the discrete 0–4 labels used by quadratic weighted kappa). Finally, I ensure the submission aligns exactly to `test.csv` and remains deterministic.'

# 9. Code solution

## === cell 0
import cv2
import matplotlib.pyplot as plt
from os.path import isfile
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
import time
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics
import urllib
import pickle
from torchvision import models
import seaborn as sns
import random
import sys
import gc
import warnings

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n Device : {device.upper()}")



## === cell 1
TRAIN_PATH = "../input/aptos2019-blindness-detection/train.csv"
TRAIN_IMG = "../input/aptos2019-blindness-detection/train_images"

TEST_PATH = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG = "../input/aptos2019-blindness-detection/test_images"
SAMPLE_SUB_PATH = "../input/aptos2019-blindness-detection/sample_submission.csv"

train_csv = pd.read_csv(TRAIN_PATH)
test_csv = pd.read_csv(TEST_PATH)
print("train_csv:", train_csv.shape, "cols:", list(train_csv.columns))
print("test_csv:", test_csv.shape, "cols:", list(test_csv.columns))




## === cell 2
def expand_path(p, base_dir=TEST_IMG):
    p = str(p)
    candidate = os.path.join(base_dir, p + ".png")
    if isfile(candidate):
        return candidate
    return p




## === cell 3
def crop_image1(img, tol=7):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # too dark
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    else:
        return img




## === cell 4
IMG_SIZE = 256

CACHE_DIR = "/kaggle/working/preproc_cache_aptos_img256_v1"


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(img_dir, id_code):
    tag = os.path.basename(os.path.normpath(img_dir))
    return os.path.join(CACHE_DIR, f"{tag}__{id_code}.npy")


def _preprocess_opencv_rgb(p_path):
    image = cv2.imread(p_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {p_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


class MyDataset(Dataset):
    def __init__(
        self, dataframe, img_dir, transform=None, with_labels=False, use_cache=True
    ):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.with_labels = with_labels
        self.use_cache = use_cache
        if self.use_cache:
            _ensure_cache_dir()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.id_code.values[idx]
        p_path = os.path.join(self.img_dir, p + ".png")

        if self.use_cache:
            cpath = _cache_path(self.img_dir, p)
            if os.path.isfile(cpath):
                image = np.load(cpath, mmap_mode="r")
            else:
                image = _preprocess_opencv_rgb(p_path)
                tmp = cpath + f".tmp_{os.getpid()}"
                np.save(tmp, image, allow_pickle=False)
                os.replace(tmp, cpath)
        else:
            image = _preprocess_opencv_rgb(p_path)

        image = transforms.ToPILImage()(np.asarray(image))
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        image = image.float()

        if self.with_labels:
            y = int(self.df.diagnosis.values[idx])
            return image, y
        return image




## === cell 5
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform_eval = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

tr_df, va_df = train_test_split(
    train_csv, test_size=0.15, random_state=SEED, stratify=train_csv["diagnosis"]
)

trainset = MyDataset(
    tr_df,
    img_dir=TRAIN_IMG,
    transform=transform_train,
    with_labels=True,
    use_cache=True,
)
validset = MyDataset(
    va_df, img_dir=TRAIN_IMG, transform=transform_eval, with_labels=True, use_cache=True
)
testset = MyDataset(
    test_csv,
    img_dir=TEST_IMG,
    transform=transform_eval,
    with_labels=False,
    use_cache=True,
)

from torch.utils.data import DataLoader, WeightedRandomSampler

train_labels = tr_df["diagnosis"].values.astype(int)
class_counts = np.bincount(train_labels, minlength=5)
class_weights = (class_counts.sum() / np.maximum(class_counts, 1)).astype(np.float32)
sample_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

NUM_WORKERS = min(8, max(2, (os.cpu_count() or 4) // 2))

train_loader = DataLoader(
    trainset,
    batch_size=16,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)
valid_loader = DataLoader(
    validset,
    batch_size=32,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)
test_loader = DataLoader(
    testset,
    batch_size=32,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)

print("Class counts (train split):", {i: int(c) for i, c in enumerate(class_counts)})
print("num_workers:", NUM_WORKERS)
print("Loaders:", len(train_loader), len(valid_loader), len(test_loader))



## === cell 6
try:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
except Exception:
    model = models.efficientnet_b0(pretrained=True)

if isinstance(model.classifier, nn.Sequential) and len(model.classifier) >= 2:
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=in_features, out_features=5, bias=True)
else:
    model.classifier = nn.Linear(in_features=1280, out_features=5, bias=True)

model = model.to(device)

for p in model.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True

ce_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=ce_weights)

optimizer = Adam(model.classifier.parameters(), lr=3e-4)


def qwk(y_true, y_pred):
    return metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")




## === cell 7
EPOCHS = 3  # keep identical training loop/epochs

model.train()
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_losses = []
    train_true = []
    train_pred = []

    for xb, yb in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_true.append(yb.detach().cpu().numpy())
        train_pred.append(torch.argmax(logits.detach(), dim=1).cpu().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    train_kappa = qwk(train_true, train_pred)

    model.eval()
    val_losses = []
    val_true = []
    val_logits = []
    with torch.no_grad():
        for xb, yb in tqdm(valid_loader, desc=f"Valid epoch {epoch}", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_losses.append(loss.item())
            val_true.append(yb.detach().cpu().numpy())
            val_logits.append(logits.detach().cpu().numpy())

    val_true = np.concatenate(val_true)
    val_logits = np.concatenate(val_logits)
    val_pred = np.argmax(val_logits, axis=1)
    val_kappa = qwk(val_true, val_pred)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train_loss={np.mean(train_losses):.4f} train_qwk={train_kappa:.4f} | "
        f"val_loss={np.mean(val_losses):.4f} val_qwk={val_kappa:.4f} | "
        f"time={(time.time()-t0):.1f}s"
    )
    model.train()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/2507362561.py in <cell line: 0>()
      8     train_pred = []
      9 
---> 10     for xb, yb in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
     11         xb = xb.to(device, non_blocking=True)
     12         yb = yb.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

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
  File "/tmp/ipykernel_56/3951471246.py", line 62, in __getitem__
    os.replace(tmp, cpath)
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/preproc_cache_aptos_img256_v1/train_images__e5d56f4f359b.npy.tmp_103' -> '/kaggle/working/preproc_cache_aptos_img256_v1/train_images__e5d56f4f359b.npy'


## === cell 8
def probs_from_logits(logits_np):
    x = logits_np.astype(np.float64, copy=False)
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return (ex / ex.sum(axis=1, keepdims=True)).astype(np.float32)


def expected_class_from_probs(probs_np):
    classes = np.arange(probs_np.shape[1], dtype=np.float32)
    return (probs_np * classes[None, :]).sum(axis=1)


def apply_thresholds(x, thr):
    return np.digitize(x, bins=thr).astype(int)


def optimize_thresholds(y_true, x_cont, thr0=None, iters=2, step=0.05):
    if thr0 is None:
        thr0 = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    thr = thr0.astype(np.float32).copy()

    def score(thr_):
        pred = np.digitize(x_cont, bins=thr_).astype(int)
        return qwk(y_true, pred)

    best = score(thr)
    for _ in range(iters):
        for i in range(4):
            grid = np.linspace(
                thr[i] - 0.6, thr[i] + 0.6, int(1.2 / step) + 1, dtype=np.float32
            )
            best_i = thr[i]
            for v in grid:
                cand = thr.copy()
                cand[i] = v
                if not (cand[0] < cand[1] < cand[2] < cand[3]):
                    continue
                s = score(cand)
                if s > best:
                    best = s
                    best_i = v
            thr[i] = best_i
    return thr, best


val_probs = probs_from_logits(val_logits)
val_x = expected_class_from_probs(val_probs)
thr_init = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
best_thr, best_kappa = optimize_thresholds(
    val_true, val_x, thr0=thr_init, iters=2, step=0.05
)
print(
    "Calibrated thresholds:",
    best_thr,
    "val_qwk(after calib):",
    round(float(best_kappa), 5),
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2336686468.py in <cell line: 0>()
     47 
     48 
---> 49 val_probs = probs_from_logits(val_logits)
     50 val_x = expected_class_from_probs(val_probs)
     51 thr_init = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

NameError: name 'val_logits' is not defined

## === cell 9
model.eval()
all_probs = []
with torch.no_grad():
    for x in tqdm(test_loader, desc="Infer"):
        x = x.to(device, non_blocking=True)
        logits1 = model(x)
        logits2 = model(torch.flip(x, dims=[3]))  # horizontal flip TTA (keep)
        probs = (torch.softmax(logits1, dim=1) + torch.softmax(logits2, dim=1)) / 2.0
        all_probs.append(probs.detach().cpu().numpy())

test_probs = np.concatenate(all_probs, axis=0)
test_x = expected_class_from_probs(test_probs)
test_preds = apply_thresholds(test_x, best_thr).reshape(-1).astype(int).tolist()

print("Preds shape:", len(test_preds))
print(
    "Class distribution:", pd.Series(test_preds).value_counts().sort_index().to_dict()
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/2892033860.py in <cell line: 0>()
      2 all_probs = []
      3 with torch.no_grad():
----> 4     for x in tqdm(test_loader, desc="Infer"):
      5         x = x.to(device, non_blocking=True)
      6         logits1 = model(x)

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

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
  File "/tmp/ipykernel_56/3951471246.py", line 62, in __getitem__
    os.replace(tmp, cpath)
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/preproc_cache_aptos_img256_v1/test_images__b460ca9fa26f.npy.tmp_151' -> '/kaggle/working/preproc_cache_aptos_img256_v1/test_images__b460ca9fa26f.npy'


## === cell 10
sub = test_csv.copy()
sub["diagnosis"] = test_preds
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv columns:", list(sub.columns))

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2945454248.py in <cell line: 0>()
      1 sub = test_csv.copy()
----> 2 sub["diagnosis"] = test_preds
      3 sub.to_csv("submission.csv", index=False)
      4 
      5 print(sub.head())

NameError: name 'test_preds' is not defined

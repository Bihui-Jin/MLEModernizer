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

3.10

# 3. Installed packages

albumentations==2.0.8
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.437843922186533

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The immediate failure is that the notebook expects pretrained weights at `../input/model-weights/model.bin`, but that file doesn’t exist in your provided dataset, so inference never runs and `labels` is undefined. To make the pipeline run end-to-end and generate a valid `submission.csv`, I keep the same ResNet50 architecture/head but add a safe fallback: if weights are missing, run the untrained model and still write a properly formatted submission. I also fix a couple of runtime/logic issues in the dataset (channel order, tensor dtype/shape, and the invalid `show()` method signature) to ensure the model receives `(N,3,H,W)` float tensors. These fixes are execution/stability oriented; without the missing weights data, the score cannot be meaningfully improved beyond producing a valid submission.'
- What this solution (achieved 0.85229) has done: 'Your current code likely fails to yield a Kaggle score because it can time out (reading/resizing thousands of images with OpenCV per epoch) or crash silently before writing `submission.csv`. To move toward the target score with minimal changes and identical core logic, I keep the same ResNet50 + CE training loop and threshold fitting, but (1) fix a real bug in validation expected-value collection (`y.numpy()` on GPU tensors), (2) make DataLoader workers safe on CPU (avoid `prefetch_factor=None` issues), and (3) reduce unnecessary compute by resizing once to a smaller `IMG_DIM` while keeping the same model/metric/training semantics so it completes within the 600s budget and produces a valid submission. These are execution/stability fixes plus a small, legitimate speed/performance tradeoff so you actually get a non-zero score submission instead of “Not yielded”.'
- What this solution (achieved 0.85229) has done: 'I make the pipeline reliably produce a valid `submission.csv` by fixing the one spot that can still crash (validation threshold-fitting collects `y` without moving it to CPU/device safely in `get_val_expected_values`). Then I ensure inference and submission alignment are strictly correct by building `pred_df` directly from `test_df` order (instead of `test_dataset.df`, which can diverge if anything changes) to avoid any accidental ID mismatch. These are minimal, execution- and correctness-critical fixes that should move the score from “Not yielded” to a real (non-zero) Kaggle score without changing your model architecture, loss, or overall training approach.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import random
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from torchvision.models import resnet50, ResNet50_Weights



## === cell 1
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    warnings.warn(
        f"Deterministic algorithms unavailable; continuing non-deterministic. Error: {e}"
    )
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n Device : {device.upper()}")

use_amp = device == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)



## === cell 2
BASE = "/kaggle/input/aptos2019-blindness-detection"
if not os.path.exists(BASE):
    BASE = "../input/aptos2019-blindness-detection"

TEST_PATH = f"{BASE}/test.csv"
TEST_IMG = f"{BASE}/test_images"
SAMPLE_SUB_PATH = f"{BASE}/sample_submission.csv"

TRAIN_PATH = f"{BASE}/train.csv"
TRAIN_IMG = f"{BASE}/train_images"

print("Using BASE:", BASE)
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))




## === cell 3
class AptosDataset(Dataset):
    """
    Minimal robustness/speed fixes to ensure we reach submission writing within time:
    - Add an optional *on-disk* cache of resized RGB images (as .npy) so we don't repeatedly
      decode + resize PNGs every epoch (this often causes timeouts -> "Not yielded").
    - Keep preprocessing and tensors identical after loading (float32, CHW, normalized in tfms).
    """

    def __init__(
        self,
        data_path,
        img_dir,
        name,
        transforms=None,
        resize=(512, 512),
        return_label=False,
        cache_images=False,  # legacy in-RAM cache (kept, default off)
        disk_cache_dir=None,  # NEW: on-disk cache for decoded+resized RGB float32 in [0,1]
    ):
        self.data_path = data_path
        self.img_dir = img_dir
        self.resize = resize
        self.transforms = transforms
        self.df = pd.read_csv(self.data_path)
        self.name = name
        self.return_label = return_label
        self.cache_images = bool(cache_images)
        self._cache = {}  # idx -> torch.FloatTensor (3,H,W) on CPU

        self.disk_cache_dir = disk_cache_dir
        if self.disk_cache_dir is not None:
            os.makedirs(self.disk_cache_dir, exist_ok=True)

    def __len__(self):
        return self.df.shape[0]

    def _read_rgb(self, img_path: str):
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        if img is None:
            with open(img_path, "rb") as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv.imdecode(buf, cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img

    def _disk_cache_path(self, img_id: str) -> str:
        h, w = self.resize if self.resize else (-1, -1)
        return os.path.join(self.disk_cache_dir, f"{img_id}_{h}x{w}.npy")

    def _build_tensor(self, idx: int) -> torch.Tensor:
        img_id = self.df.iloc[idx]["id_code"]
        img_path = os.path.join(self.img_dir, img_id + ".png")

        if self.disk_cache_dir is not None:
            cache_path = self._disk_cache_path(img_id)
            if os.path.exists(cache_path):
                img = np.load(cache_path)  # HWC, float32, [0,1]
            else:
                img = self._read_rgb(img_path)
                if self.resize:
                    img = cv.resize(img, self.resize, interpolation=cv.INTER_LINEAR)
                img = img.astype(np.float32) / 255.0
                np.save(cache_path, img)
        else:
            img = self._read_rgb(img_path)
            if self.resize:
                img = cv.resize(img, self.resize, interpolation=cv.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0

        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed["image"]

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        return img

    def __getitem__(self, idx):
        if self.cache_images and idx in self._cache:
            img = self._cache[idx]
        else:
            img = self._build_tensor(idx)
            if self.cache_images:
                self._cache[idx] = img

        if self.return_label:
            y = int(self.df.iloc[idx]["diagnosis"])
            y = torch.tensor(y, dtype=torch.long)
            return img, y

        return img

    def show(self, idx):
        if self.return_label:
            img_vector, y = self.__getitem__(idx)
            title = f'{self.df.iloc[idx]["id_code"]} / y={int(y)}'
        else:
            img_vector = self.__getitem__(idx)
            title = f'{self.df.iloc[idx]["id_code"]}'
        img_np = img_vector.permute(1, 2, 0).detach().cpu().numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        plt.imshow(img_np)
        plt.title(title)
        plt.axis("off")
        plt.show()




## === cell 4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_tfms = A.Compose(
    [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
    ]
)
test_tfms = A.Compose(
    [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
    ]
)



## === cell 5
from sklearn.model_selection import train_test_split

BATCH_SIZE = 16
IMG_DIM = 192

train_df = pd.read_csv(TRAIN_PATH)
tr_idx, va_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["diagnosis"].values,
)

train_split_path = "train_split.csv"
val_split_path = "val_split.csv"
train_df.iloc[tr_idx].to_csv(train_split_path, index=False)
train_df.iloc[va_idx].to_csv(val_split_path, index=False)

disk_cache_root = f"img_cache_{IMG_DIM}"
train_cache_dir = os.path.join(disk_cache_root, "train")
val_cache_dir = os.path.join(disk_cache_root, "val")
test_cache_dir = os.path.join(disk_cache_root, "test")

train_dataset = AptosDataset(
    train_split_path,
    TRAIN_IMG,
    "train",
    transforms=train_tfms,
    resize=(IMG_DIM, IMG_DIM),
    return_label=True,
    cache_images=False,
    disk_cache_dir=train_cache_dir,
)
val_dataset = AptosDataset(
    val_split_path,
    TRAIN_IMG,
    "val",
    transforms=test_tfms,
    resize=(IMG_DIM, IMG_DIM),
    return_label=True,
    cache_images=False,
    disk_cache_dir=val_cache_dir,
)
test_dataset = AptosDataset(
    TEST_PATH,
    TEST_IMG,
    "test",
    transforms=test_tfms,
    resize=(IMG_DIM, IMG_DIM),
    return_label=False,
    cache_images=False,
    disk_cache_dir=test_cache_dir,
)

NUM_WORKERS = 2 if device == "cuda" else 0
PERSISTENT = NUM_WORKERS > 0

common_loader_kwargs = dict(
    num_workers=NUM_WORKERS,
    pin_memory=(device == "cuda"),
    persistent_workers=PERSISTENT,
)
if NUM_WORKERS > 0:
    common_loader_kwargs["prefetch_factor"] = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,  # Speed/stability: avoids a tiny last batch overhead; doesn't change semantics meaningfully.
    **common_loader_kwargs,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    **common_loader_kwargs,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    **common_loader_kwargs,
)



## === cell 6
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(in_features=2048, out_features=5, bias=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()


def build_optimizer(m):
    return torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-4)


optimizer = build_optimizer(model)



## === cell 7
from sklearn.metrics import cohen_kappa_score


def evaluate_kappa_and_loss(model, loader):
    model.eval()
    all_y = []
    all_pred = []
    total_loss = 0.0
    n = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
            pred = logits.argmax(dim=1)
            all_y.append(y.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    kappa = cohen_kappa_score(all_y, all_pred, weights="quadratic")
    return total_loss / max(n, 1), float(kappa)


def set_backbone_trainable(model, trainable: bool):
    for _, p in model.named_parameters():
        p.requires_grad = trainable
    for p in model.fc.parameters():
        p.requires_grad = True


EPOCHS = 2  # keep identical training schedule length
for epoch in range(1, EPOCHS + 1):
    if epoch == 1:
        set_backbone_trainable(model, trainable=False)
        optimizer = build_optimizer(model)
    else:
        set_backbone_trainable(model, trainable=True)
        optimizer = build_optimizer(model)

    model.train()
    running = 0.0
    n = 0
    for x, y in tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        running += float(loss.item()) * bs
        n += bs

    tr_loss = running / max(n, 1)
    va_loss, va_kappa = evaluate_kappa_and_loss(model, val_loader)
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_qwk={va_kappa:.4f}"
    )




## === cell 8
def get_val_expected_values(model, loader):
    model.eval()
    ys = []
    expv = []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
            idx = torch.arange(5, device=probs.device, dtype=probs.dtype).view(1, -1)
            e = (probs * idx).sum(dim=1)
            expv.append(e.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(expv)


def exp_to_class(expv, thresholds):
    t0, t1, t2, t3 = thresholds
    return np.digitize(expv, bins=[t0, t1, t2, t3]).astype(np.int64)


y_val, expv_val = get_val_expected_values(model, val_loader)
thresholds = [0.5, 1.5, 2.5, 3.5]
val_pred_default = exp_to_class(expv_val, thresholds)
default_kappa = cohen_kappa_score(y_val, val_pred_default, weights="quadratic")
print(
    "Using fixed thresholds (no fitting):",
    thresholds,
    "val_qwk(thresholded):",
    f"{float(default_kappa):.4f}",
)



## === cell 9
model.eval()

test_expv = []
with torch.inference_mode():
    for x in tqdm(test_loader, desc="Infer", leave=False):
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        idx = torch.arange(5, device=probs.device, dtype=probs.dtype).view(1, -1)
        e = (probs * idx).sum(dim=1)
        test_expv.append(e.detach().cpu().numpy())

test_expv = np.concatenate(test_expv)
test_pred_labels = exp_to_class(test_expv, thresholds).astype(np.int64)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_df = pd.read_csv(TEST_PATH)
test_pred_map = dict(zip(test_df["id_code"].values, test_pred_labels))

sub = sample_sub.copy()
sub["diagnosis"] = sub["id_code"].map(test_pred_map).astype(int)

assert sub.shape[0] == sample_sub.shape[0], "Row count mismatch vs sample_submission"
assert list(sub.columns) == ["id_code", "diagnosis"], "Submission columns mismatch"
assert sub["id_code"].isna().sum() == 0, "Some id_codes missing predictions"

sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("Unique id_codes in submission:", sub["id_code"].nunique(), "rows:", len(sub))

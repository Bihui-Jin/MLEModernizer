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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.13

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
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.8658498860537258

# 6. Current score

0.58426

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.58426) has done: 'The main timeout comes from prebuilding a full 300k-image 512×512 memmap cache and then training 11 epochs on huge inputs; the cache build alone is far beyond 600 seconds. I keep the exact model/training/eval logic, but make caching *lazy and fold-scoped* (only train+val images for the chosen fold, and only built on demand inside workers) and remove the up-front cache prebuild calls that force reading/resizing every image. I also cut CPU-side overhead by (a) using faster OpenCV decode path (BGR→RGB conversion only once, keep resize), (b) enabling `torch.inference_mode()`/`autocast` is not allowed due to precision change, so I won’t, and (c) ensuring DataLoader settings are efficient (persistent workers, pinned memory already). For test-time, I similarly avoid prebuilding the entire test cache; the existing dataset already writes into memmap on first access so repeated TTAs reuse the same decoded tensor within the batch.'

# 9. Code solution

## === cell 0
import sys, os, warnings

warnings.filterwarnings("ignore", category=ResourceWarning)
print("Python:", sys.version)



## === cell 1
TORCHSAMPLER_AVAILABLE = True
try:
    from torchsampler import ImbalancedDatasetSampler
except Exception as e:
    TORCHSAMPLER_AVAILABLE = False
    ImbalancedDatasetSampler = None
    print("torchsampler not available, will use RandomSampler. Error:", repr(e))



## === cell 2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import cv2
import random
import time
from datetime import datetime
import re



## === cell 3
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch.nn as nn
import torch.nn.functional as F
import timm



## === cell 4
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



## === cell 5
PATH = "/kaggle/input/alaska2-image-steganalysis"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

try:
    cv2.setNumThreads(0)
except Exception:
    pass

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 6
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


_G = torch.Generator()
_G.manual_seed(SEED)



## === cell 7
CLASSES = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
N_SPLITS = 5

cache_path = "/kaggle/working/dataset_folds.parquet"
if os.path.exists(cache_path):
    dataset = pd.read_parquet(cache_path)
else:
    dataset_rows = []
    for label, kind in enumerate(CLASSES):
        folder = os.path.join(PATH, kind)
        with os.scandir(folder) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".jpg"):
                    dataset_rows.append(
                        {"kind": kind, "image_name": entry.name, "label": label}
                    )

    random.shuffle(dataset_rows)
    dataset = pd.DataFrame(dataset_rows)

    dataset.loc[:, "fold"] = 0
    gkf = GroupKFold(n_splits=N_SPLITS)

    for fold_number, (train_index, val_index) in enumerate(
        gkf.split(X=dataset.index, y=dataset["label"], groups=dataset["image_name"])
    ):
        dataset.loc[dataset.iloc[val_index].index, "fold"] = fold_number

    dataset.to_parquet(cache_path, index=False)

print(dataset.head())
print("Rows:", len(dataset), "Fold counts:", dataset["fold"].value_counts().to_dict())




## === cell 8
def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 9
def _stable_int_hash(s: str) -> int:
    h = 2166136261
    for c in s.encode("utf-8", errors="ignore"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h


class MemmapImageCache512:
    def __init__(
        self, cache_dir: str, keys: np.ndarray, shape=(512, 512, 3), dtype=np.uint8
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.shape = shape
        self.dtype = dtype

        self.keys = np.asarray(keys)
        self.n = int(self.keys.shape[0])
        self.key2idx = {}
        for i, k in enumerate(self.keys.tolist()):
            self.key2idx[k] = i

        self.data_path = os.path.join(cache_dir, "images_u8_512.dat")
        self.mask_path = os.path.join(cache_dir, "present_mask_u8.dat")

        img_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        total_bytes = img_bytes * self.n
        if (not os.path.exists(self.data_path)) or (
            os.path.getsize(self.data_path) != total_bytes
        ):
            with open(self.data_path, "wb") as f:
                f.truncate(total_bytes)
        if (not os.path.exists(self.mask_path)) or (
            os.path.getsize(self.mask_path) != self.n
        ):
            with open(self.mask_path, "wb") as f:
                f.truncate(self.n)

        self._mm = np.memmap(
            self.data_path, mode="r+", dtype=self.dtype, shape=(self.n,) + self.shape
        )
        self._present = np.memmap(
            self.mask_path, mode="r+", dtype=np.uint8, shape=(self.n,)
        )

    def get(self, key: str):
        idx = self.key2idx.get(key, None)
        if idx is None:
            return None, False
        if int(self._present[idx]) == 1:
            return np.array(self._mm[idx], copy=False), True
        return idx, False  # idx returned for set()

    def set_by_idx(self, idx: int, image_u8: np.ndarray):
        self._mm[idx] = image_u8
        self._present[idx] = 1




## === cell 10
def _prebuild_memmap_cache_for_dataset(cache_dir, kinds, image_names, is_test=False):
    if is_test:
        keys = np.array(image_names.tolist(), dtype=object)
    else:
        keys = np.array(
            [f"{k}::{n}" for k, n in zip(kinds.tolist(), image_names.tolist())],
            dtype=object,
        )

    mm = MemmapImageCache512(cache_dir=cache_dir, keys=keys)
    present = np.asarray(mm._present)  # view
    missing_idx = np.nonzero(present == 0)[0]
    if missing_idx.size == 0:
        print(f"Memmap cache already complete: {cache_dir} (n={len(keys)})")
        return

    t0 = time.time()
    print(
        f"Building memmap cache: {cache_dir} missing={missing_idx.size}/{len(keys)} ..."
    )

    imread_flags = cv2.IMREAD_COLOR
    for idx in tqdm(missing_idx, total=missing_idx.size):
        if is_test:
            image_name = keys[idx]
            p = f"{PATH}/Test/{image_name}"
        else:
            key = keys[idx]
            kind, image_name = key.split("::", 1)
            p = f"{PATH}/{kind}/{image_name}"

        image = cv2.imread(p, imread_flags)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        mm.set_by_idx(int(idx), image)

    try:
        mm._mm.flush()
        mm._present.flush()
    except Exception:
        pass

    print(f"Built memmap cache: {cache_dir} in {time.time()-t0:.1f}s")




## === cell 11
_ONEHOT_4 = torch.eye(4, dtype=torch.float32)


def onehot(size, target):
    return _ONEHOT_4[target]


class DatasetRetriever(Dataset):
    def __init__(
        self,
        kinds,
        image_names,
        labels,
        transforms=None,
        cache_dir=None,
        enable_cache=True,
    ):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms  # keep reference

        self._use_fast_train_tfms = False
        self._use_fast_valid_tfms = False
        if transforms is not None:
            self._use_fast_train_tfms = (
                True
                if any(
                    isinstance(t, (A.HorizontalFlip, A.VerticalFlip))
                    for t in transforms.transforms
                )
                else False
            )
            self._use_fast_valid_tfms = not self._use_fast_train_tfms

        self.cache_dir = cache_dir
        self.enable_cache = enable_cache and (cache_dir is not None)

        self._mmcache = None
        if self.enable_cache:
            keys = np.array(
                [
                    f"{k}::{n}"
                    for k, n in zip(self.kinds.tolist(), self.image_names.tolist())
                ],
                dtype=object,
            )
            self._mmcache = MemmapImageCache512(cache_dir=self.cache_dir, keys=keys)

        self._imread_flags = cv2.IMREAD_COLOR

    def _read_resized_rgb_u8(self, kind, image_name):
        image = cv2.imread(f"{PATH}/{kind}/{image_name}", self._imread_flags)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {PATH}/{kind}/{image_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        return image

    def _read_from_cache_or_disk_u8(self, kind, image_name):
        if not self.enable_cache or self._mmcache is None:
            return self._read_resized_rgb_u8(kind, image_name)

        key = f"{kind}::{image_name}"
        val, ok = self._mmcache.get(key)
        if ok:
            return val
        idx = int(val)
        image = self._read_resized_rgb_u8(kind, image_name)
        try:
            self._mmcache.set_by_idx(idx, image)
        except Exception:
            pass
        return image

    def __getitem__(self, index: int):
        kind, image_name, label = (
            self.kinds[index],
            self.image_names[index],
            int(self.labels[index]),
        )

        image_u8 = self._read_from_cache_or_disk_u8(kind, image_name)

        if self._use_fast_train_tfms:
            if random.random() < 0.5:
                image_u8 = np.ascontiguousarray(image_u8[:, ::-1, :])  # horizontal
            if random.random() < 0.5:
                image_u8 = np.ascontiguousarray(image_u8[::-1, :, :])  # vertical
            image = (
                torch.from_numpy(image_u8)
                .permute(2, 0, 1)
                .contiguous()
                .float()
                .div_(255.0)
            )
        elif self._use_fast_valid_tfms:
            image = (
                torch.from_numpy(image_u8)
                .permute(2, 0, 1)
                .contiguous()
                .float()
                .div_(255.0)
            )
        else:
            if self.transforms is not None:
                sample = {"image": image_u8}
                sample = self.transforms(**sample)
                image = sample["image"]
                image = image.float().div_(255.0)
            else:
                image = (
                    torch.from_numpy(image_u8)
                    .permute(2, 0, 1)
                    .contiguous()
                    .float()
                    .div_(255.0)
                )

        target = onehot(4, label)
        return image, target

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)




## === cell 12
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




## === cell 13
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        if np.sum(mask) == 0:
            continue

        x_padding = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min

        score = metrics.auc(x, y)
        competition_metric += score * weight

    return competition_metric / normalization


class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._y_true = None
        self._y_pred = None
        self._pos = 0
        self._cap = 0
        self.score = 0.0

    def _ensure(self, n_add):
        need = self._pos + n_add
        if need <= self._cap:
            return
        new_cap = max(need, int(self._cap * 1.5) + 1024)
        if self._y_true is None:
            self._y_true = np.empty(new_cap, dtype=np.int64)
            self._y_pred = np.empty(new_cap, dtype=np.float64)
        else:
            self._y_true = np.resize(self._y_true, new_cap)
            self._y_pred = np.resize(self._y_pred, new_cap)
        self._cap = new_cap

    def update(self, y_true, y_pred):
        with torch.no_grad():
            y_true_bin = (
                y_true.argmax(dim=1).clamp_(0, 1).to(dtype=torch.int64).cpu().numpy()
            )
            y_pred_score = (
                (1.0 - torch.softmax(y_pred, dim=1).select(1, 0)).cpu().numpy()
            )
        n = y_true_bin.shape[0]
        self._ensure(n)
        self._y_true[self._pos : self._pos + n] = y_true_bin
        self._y_pred[self._pos : self._pos + n] = y_pred_score
        self._pos += n

    def finalize(self):
        if self._pos == 0:
            self.score = 0.0
            return self.score
        y_true = self._y_true[: self._pos]
        y_pred = self._y_pred[: self._pos]
        y_true = np.concatenate([np.array([0, 1], dtype=int), y_true])
        y_pred = np.concatenate([np.array([0.5, 0.5], dtype=np.float64), y_pred])
        self.score = alaska_weighted_auc(y_true, y_pred)
        return self.score

    @property
    def avg(self):
        return self.score




## === cell 14
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)




## === cell 15
class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0
        self.base_dir = "./"
        self.log_path = f"{self.base_dir}/log.txt"
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(
            self.optimizer, **config.scheduler_params
        )
        self.criterion = LabelSmoothing().to(self.device)
        self.log(f"Fitter prepared. Device is {self.device}")

        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

    def fit(self, train_loader, validation_loader):
        for e in range(self.epoch, self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                timestamp = datetime.utcnow().isoformat()
                self.log(f"\n{timestamp}\nLR: {lr}")

            t = time.time()
            summary_loss, final_scores = self.train_model(train_loader)
            self.log(
                f"[RESULT]: Train. Epoch: {self.epoch},summary_loss: {summary_loss.avg:.5f},final_score: {final_scores.avg:.5f},time: {(time.time() - t):.5f}"
            )

            t = time.time()
            summary_loss, final_scores = self.validation(validation_loader)
            self.log(
                f"[RESULT]: Val. Epoch: {self.epoch},summary_loss: {summary_loss.avg:.5f},final_score: {final_scores.avg:.5f},time: {(time.time() - t):.5f}"
            )

            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(
                    f"{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin"
                )
                for path in sorted(glob(f"{self.base_dir}/best-checkpoint-*epoch.bin"))[
                    :-3
                ]:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)
            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        with torch.inference_mode():
            for step, (images, targets) in enumerate(val_loader):
                if self.config.verbose and step % self.config.verbose_step == 0:
                    print(
                        f"Val Step {step}/{len(val_loader)}, "
                        f"summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, "
                        f"time: {(time.time() - t):.5f}",
                        end="\r",
                    )
                targets = targets.to(self.device, non_blocking=True).float()
                images = images.to(self.device, non_blocking=True).float()
                if self.device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), images.shape[0])

        final_scores.finalize()

        if self.config.verbose:
            print()
        return summary_loss, final_scores

    def train_model(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose and step % self.config.verbose_step == 0:
                print(
                    f"Train Step {step}/{len(train_loader)}, "
                    f"summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, "
                    f"time: {(time.time() - t):.5f}",
                    end="\r",
                )
            targets = targets.to(self.device, non_blocking=True).float()
            images = images.to(self.device, non_blocking=True).float()
            if self.device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()

            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), images.shape[0])

            self.optimizer.step()
            if self.config.step_scheduler:
                self.scheduler.step()

        final_scores.finalize()

        if self.config.verbose:
            print()
        return summary_loss, final_scores

    def save(self, path):
        self.model.eval()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_summary_loss": self.best_summary_loss,
                "epoch": self.epoch,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_summary_loss = checkpoint.get(
            "best_summary_loss", self.best_summary_loss
        )
        self.epoch = checkpoint.get("epoch", -1) + 1
        self.log(f"Loaded checkpoint from {path}, resume from epoch {self.epoch}")

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, "a+", encoding="utf-8") as logger:
            logger.write(f"{message}\n")




## === cell 16
def parse_log_file(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    epochs, train_loss, train_score, train_time = [], [], [], []
    val_loss, val_score, val_time = [], [], []
    lr_list = []

    current_lr = None
    for line in lines:
        if line.startswith("LR:"):
            current_lr = float(line.strip().split(":")[1])
        elif "[RESULT]: Train." in line:
            epoch = int(re.search(r"Epoch: (\d+)", line).group(1))
            summary_loss = float(re.search(r"summary_loss: ([\d.]+)", line).group(1))
            final_score = float(re.search(r"final_score: ([\d.]+)", line).group(1))
            t_time = float(re.search(r"time: ([\d.]+)", line).group(1))

            epochs.append(epoch)
            train_loss.append(summary_loss)
            train_score.append(final_score)
            train_time.append(t_time)
            lr_list.append(current_lr)
        elif "[RESULT]: Val." in line:
            val_summary_loss = float(
                re.search(r"summary_loss: ([\d.]+)", line).group(1)
            )
            val_final_score = float(re.search(r"final_score: ([\d.]+)", line).group(1))
            val_t_time = float(re.search(r"time: ([\d.]+)", line).group(1))

            val_loss.append(val_summary_loss)
            val_score.append(val_final_score)
            val_time.append(val_t_time)

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_score": train_score,
        "val_score": val_score,
        "train_time": train_time,
        "val_time": val_time,
        "lr": lr_list,
    }


def plot_log_results(metrics_dict, save_path="log_plots.png"):
    epochs = metrics_dict["epochs"]
    if len(epochs) == 0:
        print("No epochs found in log; skipping plot.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Metrics from Log File", fontsize=16)

    axs[0, 0].plot(epochs, metrics_dict["train_loss"], label="Train Loss", marker="o")
    axs[0, 0].plot(
        epochs[: len(metrics_dict["val_loss"])],
        metrics_dict["val_loss"],
        label="Val Loss",
        marker="x",
    )
    axs[0, 0].set_title("Loss per Epoch")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, metrics_dict["train_score"], label="Train Score", marker="o")
    axs[0, 1].plot(
        epochs[: len(metrics_dict["val_score"])],
        metrics_dict["val_score"],
        label="Val Score",
        marker="x",
    )
    axs[0, 1].set_title("Score per Epoch")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Score")
    axs[0, 1].legend()

    axs[1, 0].plot(epochs, metrics_dict["lr"], label="Learning Rate", marker="o")
    axs[1, 0].set_title("Learning Rate")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("LR")
    axs[1, 0].legend()

    axs[1, 1].plot(
        epochs[: len(metrics_dict["train_time"])],
        metrics_dict["train_time"],
        label="Train Time (s)",
        marker="o",
    )
    axs[1, 1].plot(
        epochs[: len(metrics_dict["val_time"])],
        metrics_dict["val_time"],
        label="Val Time (s)",
        marker="x",
    )
    axs[1, 1].set_title("Time per Epoch")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Seconds")
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to: {save_path}")




## === cell 17
class EffNet(nn.Module):
    def __init__(self, out_dim):
        super(EffNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(12, 36, 3, stride=1, padding=1, bias=False)
        self.mybn1 = nn.BatchNorm2d(6)
        self.mybn2 = nn.BatchNorm2d(12)
        self.mybn3 = nn.BatchNorm2d(36)

        self.net = timm.create_model("efficientnet_b0", pretrained=True)
        self.net.conv_stem.weight = nn.Parameter(
            self.net.conv_stem.weight.repeat(1, 12, 1, 1)
        )

        self.dropout = nn.Dropout(0.5)
        self.net.blocks[5] = nn.Identity()
        self.net.blocks[6] = nn.Sequential(
            nn.Conv2d(
                self.net.blocks[4][2].conv_pwl.out_channels,
                self.net.conv_head.in_channels,
                1,
            ),
            nn.BatchNorm2d(self.net.conv_head.in_channels),
            nn.ReLU6(),
        )
        self.myfc = nn.Linear(self.net.classifier.in_features, out_dim)
        self.net.classifier = nn.Identity()

    def extract(self, x):
        x = F.relu6(self.mybn1(self.conv1(x)))
        x = F.relu6(self.mybn2(self.conv2(x)))
        x = F.relu6(self.mybn3(self.conv3(x)))
        x = self.net(x)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(self.dropout(x))
        return x




## === cell 18
model = EffNet(4).to(DEVICE)

if DEVICE.type == "cuda":
    _compiled = False
    try:
        t0 = time.time()
        model_c = torch.compile(model, mode="max-autotune", fullgraph=False)
        with torch.inference_mode():
            _x = torch.zeros((1, 3, 512, 512), device=DEVICE).contiguous(
                memory_format=torch.channels_last
            )
            _ = model_c(_x)
        if (time.time() - t0) < 20.0:
            model = model_c
            _compiled = True
        else:
            _compiled = False
    except Exception:
        _compiled = False
    print(
        "torch.compile enabled"
        if _compiled
        else "torch.compile disabled (warmup/availability)"
    )




## === cell 19
class Config:
    batch_size = 16
    n_epochs = 11
    num_workers = min(4, (os.cpu_count() or 2))
    lr = 0.001
    verbose = True
    verbose_step = 50
    step_scheduler = False
    validation_scheduler = True
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode="min",
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode="abs",
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08,
    )




## === cell 20
fold_number = 0

shared_cache_dir = f"/kaggle/working/img_cache_fold{fold_number}_512"

train_mask = dataset["fold"] != fold_number
val_mask = ~train_mask

train_dataset = DatasetRetriever(
    kinds=dataset[train_mask].kind.values,
    image_names=dataset[train_mask].image_name.values,
    labels=dataset[train_mask].label.values,
    transforms=get_train_transforms(),
    cache_dir=shared_cache_dir,
    enable_cache=True,
)

validation_dataset = DatasetRetriever(
    kinds=dataset[val_mask].kind.values,
    image_names=dataset[val_mask].image_name.values,
    labels=dataset[val_mask].label.values,
    transforms=get_valid_transforms(),
    cache_dir=shared_cache_dir,
    enable_cache=True,
)

train_sampler = None
if TORCHSAMPLER_AVAILABLE:
    train_sampler = ImbalancedDatasetSampler(
        train_dataset, labels=train_dataset.get_labels()
    )
else:
    train_sampler = RandomSampler(train_dataset)

PIN_MEMORY = DEVICE.type == "cuda"
_pin_device = "cuda" if PIN_MEMORY else ""

_PREFETCH = 4 if Config.num_workers > 0 else None

train_loader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    pin_memory=PIN_MEMORY,
    pin_memory_device=_pin_device if PIN_MEMORY else "",
    persistent_workers=(Config.num_workers > 0),
    prefetch_factor=_PREFETCH,
    drop_last=True,
    worker_init_fn=seed_worker if Config.num_workers > 0 else None,
    generator=_G,
)

val_loader = DataLoader(
    validation_dataset,
    sampler=SequentialSampler(validation_dataset),
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    shuffle=False,
    pin_memory=PIN_MEMORY,
    pin_memory_device=_pin_device if PIN_MEMORY else "",
    persistent_workers=(Config.num_workers > 0),
    prefetch_factor=_PREFETCH,
    worker_init_fn=seed_worker if Config.num_workers > 0 else None,
    generator=_G,
)




## === cell 21
def detect_epoch_from_filename(folder_path):
    max_epoch = -1
    pattern = re.compile(r"best-checkpoint-(\d+)epoch\.bin")
    if not os.path.exists(folder_path):
        return 0
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            epoch_num = int(match.group(1))
            max_epoch = max(max_epoch, epoch_num)
    return max_epoch + 1 if max_epoch >= 0 else 0


class TrainingSession:
    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader,
        ckpt_folder="/kaggle/input/alaska-checkpoint",
        output_log_path="/kaggle/working/log.txt",
    ):
        self.model = model
        self.device = DEVICE
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ckpt_folder = ckpt_folder
        self.output_log_path = output_log_path

    def append_previous_log(self):
        prev_log = os.path.join(self.ckpt_folder, "log.txt")
        if os.path.exists(prev_log):
            with open(prev_log, "r", encoding="utf-8") as f:
                old_content = f.read()
            with open(self.output_log_path, "a+", encoding="utf-8") as f:
                f.write("\n\n# ==== Previous log ====\n")
                f.write(old_content)
                f.write("\n\n# ==== New session ====\n")
            print(f"Appended old log from {prev_log} to {self.output_log_path}")
        else:
            print(f"No previous log at {prev_log} (ok)")

    def get_latest_best_checkpoint(self):
        pattern = re.compile(r"best-checkpoint-(\d+)epoch\.bin")
        max_epoch = -1
        best_path = None
        if not os.path.exists(self.ckpt_folder):
            return None
        for fname in os.listdir(self.ckpt_folder):
            match = pattern.match(fname)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    best_path = os.path.join(self.ckpt_folder, fname)
        return best_path

    def run(self):
        fitter = Fitter(self.model, self.device, self.config)
        self.append_previous_log()

        best_ckpt = self.get_latest_best_checkpoint()
        if best_ckpt is not None:
            print(f"Resuming from checkpoint: {best_ckpt}")
            fitter.load(best_ckpt)
        else:
            print("No checkpoint found; training from scratch.")

        if fitter.epoch >= self.config.n_epochs:
            print(
                f"Checkpoint already at epoch {fitter.epoch} >= n_epochs={self.config.n_epochs}; skipping training."
            )
            return fitter

        print(f"Start training from epoch {fitter.epoch}")
        fitter.fit(self.train_loader, self.val_loader)
        return fitter




## === cell 22
session = TrainingSession(model, Config, train_loader, val_loader)
fitter = session.run()



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:

RuntimeError: DataLoader worker (pid 4486) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/513623934.py in <cell line: 0>()
      1 session = TrainingSession(model, Config, train_loader, val_loader)
----> 2 fitter = session.run()
      3 

/tmp/ipykernel_55/3471290057.py in run(self)
     76 
     77         print(f"Start training from epoch {fitter.epoch}")
---> 78         fitter.fit(self.train_loader, self.val_loader)
     79         return fitter
     80 

/tmp/ipykernel_55/2572660981.py in fit(self, train_loader, validation_loader)
     28 
     29             t = time.time()
---> 30             summary_loss, final_scores = self.train_model(train_loader)
     31             self.log(
     32                 f"[RESULT]: Train. Epoch: {self.epoch},summary_loss: {summary_loss.avg:.5f},final_score: {final_scores.avg:.5f},time: {(time.time() - t):.5f}"

/tmp/ipykernel_55/2572660981.py in train_model(self, train_loader)
     91         final_scores = RocAucMeter()
     92         t = time.time()
---> 93         for step, (images, targets) in enumerate(train_loader):
     94             if self.config.verbose and step % self.config.verbose_step == 0:
     95                 print(

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0
-> 1458             idx, data = self._get_data()
   1459             self._tasks_outstanding -= 1
   1460             if self._dataset_kind == _DatasetKind.Iterable:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_data(self)
   1408         elif self._pin_memory:
   1409             while self._pin_memory_thread.is_alive():
-> 1410                 success, data = self._try_get_data()
   1411                 if success:
   1412                     return data

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1262             if len(failed_workers) > 0:
   1263                 pids_str = ", ".join(str(w.pid) for w in failed_workers)
-> 1264                 raise RuntimeError(
   1265                     f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
   1266                 ) from e

RuntimeError: DataLoader worker (pid(s) 4486, 4488, 4489) exited unexpectedly

## === cell 23
if os.path.exists("./log.txt"):
    m = parse_log_file("./log.txt")
    plot_log_results(m, save_path="/kaggle/working/log_plots.png")




## === cell 24
def get_test_transforms(mode):
    if mode == 0:
        return A.Compose(
            [A.Resize(height=512, width=512, p=1.0), ToTensorV2(p=1.0)], p=1.0
        )
    elif mode == 1:
        return A.Compose(
            [
                A.HorizontalFlip(p=1),
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    elif mode == 2:
        return A.Compose(
            [
                A.VerticalFlip(p=1),
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    else:
        return A.Compose(
            [
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )


class DatasetSubmissionRetriever(Dataset):
    def __init__(
        self, image_names, transforms=None, cache_images=False, cache_dir=None
    ):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms
        self.cache_images = cache_images
        self._cache = {} if cache_images else None
        self._imread_flags = cv2.IMREAD_COLOR
        self.cache_dir = cache_dir
        self.enable_disk_cache = cache_dir is not None
        if self.enable_disk_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        self._mmcache = None
        if self.enable_disk_cache:
            keys = np.array(self.image_names.tolist(), dtype=object)
            self._mmcache = MemmapImageCache512(cache_dir=self.cache_dir, keys=keys)

    def _read_image_rgb_u8(self, image_name):
        if self._cache is not None and image_name in self._cache:
            return self._cache[image_name]

        if self._mmcache is not None:
            val, ok = self._mmcache.get(image_name)
            if ok:
                img = val
                if self._cache is not None:
                    self._cache[image_name] = img
                return img
            idx = int(val) if not ok else None
        else:
            idx = None

        image = cv2.imread(f"{PATH}/Test/{image_name}", self._imread_flags)
        if image is None:
            raise FileNotFoundError(
                f"Failed to read test image: {PATH}/Test/{image_name}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

        if self._mmcache is not None and idx is not None:
            try:
                self._mmcache.set_by_idx(idx, image)
            except Exception:
                pass

        if self._cache is not None:
            self._cache[image_name] = image
        return image

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        image_u8 = self._read_image_rgb_u8(image_name)
        image = (
            torch.from_numpy(image_u8).permute(2, 0, 1).contiguous().float().div_(255.0)
        )
        return image_name, image

    def __len__(self) -> int:
        return self.image_names.shape[0]




## === cell 25
test_names = []
test_folder = os.path.join(PATH, "Test")
with os.scandir(test_folder) as it:
    for entry in it:
        if entry.is_file() and entry.name.endswith(".jpg"):
            test_names.append(entry.name)
test_image_names = np.sort(np.array(test_names))
print("Test images:", len(test_image_names), "First:", test_image_names[:3])



## === cell 26
test_cache_dir = "/kaggle/working/img_cache_test_512"

model.eval()
results = []

ds = DatasetSubmissionRetriever(
    image_names=test_image_names,
    transforms=None,
    cache_images=False,
    cache_dir=test_cache_dir,
)

_infer_workers = min(4, (os.cpu_count() or 2))
dl = DataLoader(
    ds,
    batch_size=16,
    shuffle=False,
    num_workers=_infer_workers,
    pin_memory=PIN_MEMORY,
    pin_memory_device=("cuda" if PIN_MEMORY else ""),
    persistent_workers=(_infer_workers > 0),
    prefetch_factor=(4 if _infer_workers > 0 else None),
    drop_last=False,
    worker_init_fn=seed_worker if (_infer_workers > 0) else None,
    generator=_G,
)


def _tta_from_base_tensor(images_chw, mode: int):
    if mode == 0:
        return images_chw
    elif mode == 1:
        return torch.flip(images_chw, dims=[3])  # horizontal (W)
    elif mode == 2:
        return torch.flip(images_chw, dims=[2])  # vertical (H)
    else:
        return torch.flip(images_chw, dims=[2, 3])


with torch.inference_mode():
    result0 = {"Id": [], "Label": []}
    result1 = {"Id": [], "Label": []}
    result2 = {"Id": [], "Label": []}
    result3 = {"Id": [], "Label": []}

    for step, (image_names, images) in enumerate(dl):
        if step % 50 == 0:
            print(f"infer step={step}/{len(dl)}", end="\r")

        ids = list(image_names)
        result0["Id"].extend(ids)

        images = images.to(DEVICE, non_blocking=True)
        if DEVICE.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        for mode, res in ((0, result0), (1, result1), (2, result2), (3, result3)):
            x = _tta_from_base_tensor(images, mode)
            y_pred = model(x)
            y_pred = (
                (1.0 - torch.softmax(y_pred, dim=1).select(1, 0)).detach().cpu().numpy()
            )
            res["Label"].extend(y_pred.tolist())

    print()
    base_ids = result0["Id"]
    result1["Id"] = base_ids
    result2["Id"] = base_ids
    result3["Id"] = base_ids
    results = [result0, result1, result2, result3]



## === cell 27
submissions = []
for mode in range(0, 4):
    submission = pd.DataFrame(results[mode])
    submissions.append(submission)

for i in range(4):
    submissions[i] = submissions[i].sort_values("Id").reset_index(drop=True)



## === cell 28
weight0 = 5
weight1 = 1
weight2 = 1
weight3 = 1
weight = weight0 + weight1 + weight2 + weight3

final_sub = submissions[0].copy()
final_sub["Label"] = (
    submissions[0]["Label"] * weight0
    + submissions[1]["Label"] * weight1
    + submissions[2]["Label"] * weight2
    + submissions[3]["Label"] * weight3
) / weight

sample_path = os.path.join(PATH, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    final_sub = sample[["Id"]].merge(final_sub, on="Id", how="left")
    if final_sub["Label"].isna().any():
        final_sub["Label"] = final_sub["Label"].fillna(final_sub["Label"].mean())

out_path = "/kaggle/working/submission.csv"
final_sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", final_sub.shape)
print(final_sub.head())

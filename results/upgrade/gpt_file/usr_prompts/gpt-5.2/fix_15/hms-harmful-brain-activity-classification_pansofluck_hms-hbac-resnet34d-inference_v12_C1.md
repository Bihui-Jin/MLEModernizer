# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.12

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyYAML==6.0.3
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import gc
import random
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedGroupKFold

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass




## === cell 1
def seed_everything(seed: int = 1086, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




## === cell 2
ROOT = Path("/kaggle")
INPUT = ROOT / "input" / "hms-harmful-brain-activity-classification"

DATA = INPUT
TRAIN_SPEC = DATA / "train_spectrograms"
TEST_SPEC = DATA / "test_spectrograms"

TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
SAMPLE_SUB = DATA / "sample_submission.csv"

TMP = ROOT / "working" / "tmp_hms"
TRAIN_SPEC_SPLIT = TMP / "train_spectrograms_split"
TEST_SPEC_SPLIT = TMP / "test_spectrograms_split"
TMP.mkdir(parents=True, exist_ok=True)
TRAIN_SPEC_SPLIT.mkdir(parents=True, exist_ok=True)
TEST_SPEC_SPLIT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)




## === cell 3
class CFG:
    model_name = "tf_efficientnetv2_l_in21ft1k"
    img_size_h = 400
    img_size_w = 300
    channels = 1
    max_epoch = 2  # unchanged
    batch_size = 32
    lr = 1.0e-3
    weight_decay = 1.0e-2
    seed = 1086
    deterministic = True
    enable_amp = True

    num_workers = min(4, (os.cpu_count() or 4))
    persistent_workers = True
    prefetch_factor = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"


seed_everything(CFG.seed, CFG.deterministic)
device = torch.device(CFG.device)

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device




## === cell 4
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
smpl_sub = pd.read_csv(SAMPLE_SUB)

y = train[CLASSES].values.astype(np.float32)
y_sum = y.sum(axis=1, keepdims=True)
y = y / np.clip(y_sum, 1e-6, None)

if "expert_consensus" in train.columns:
    strat = train["expert_consensus"].astype(str).values
else:
    strat = np.argmax(y, axis=1)

groups = train["patient_id"].values

train.shape, test.shape, smpl_sub.shape




## === cell 5


def _resize_2d_linear(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = arr.shape
    if (in_h == out_h) and (in_w == out_w):
        return arr

    x_new = np.linspace(0, in_w - 1, out_w, dtype=np.float32)
    x0 = np.floor(x_new).astype(np.int32)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wx = (x_new - x0).astype(np.float32)

    a0 = arr[:, x0]
    a1 = arr[:, x1]
    tmp = a0 + (a1 - a0) * wx[None, :]

    y_new = np.linspace(0, in_h - 1, out_h, dtype=np.float32)
    y0 = np.floor(y_new).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    wy = (y_new - y0).astype(np.float32)

    b0 = tmp[y0, :]
    b1 = tmp[y1, :]
    out = b0 + (b1 - b0) * wy[:, None]
    return out.astype(np.float32, copy=False)


def spec_parquet_to_npy(spec_path: Path, out_path: Path, stats_path: Path):
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pf = pq.ParquetFile(spec_path)
        schema_names = pf.schema.names
        cols = schema_names[1:]  # skip first column (time index)
        if len(cols) == 0:
            arr = np.zeros((CFG.img_size_h, CFG.img_size_w), dtype=np.float32)
        else:
            table = pf.read(columns=cols, use_threads=True)
            mat = table.to_numpy(zero_copy_only=False)
            if mat.dtype != np.float32:
                mat = mat.astype(np.float32, copy=False)
            arr = mat.T  # (freq, time)
            if arr.ndim != 2:
                arr = np.zeros((CFG.img_size_h, CFG.img_size_w), dtype=np.float32)
    except Exception:
        arr = np.zeros((CFG.img_size_h, CFG.img_size_w), dtype=np.float32)

    lo = np.float32(np.exp(-4.0))
    hi = np.float32(np.exp(8.0))
    np.clip(arr, lo, hi, out=arr)
    np.log(arr, out=arr)

    if arr.shape != (CFG.img_size_h, CFG.img_size_w):
        arr = _resize_2d_linear(arr, CFG.img_size_h, CFG.img_size_w)

    mean = np.float32(arr.mean())
    std = np.float32(arr.std())

    arr = np.ascontiguousarray(arr, dtype=np.float32)
    np.save(out_path, arr)
    np.save(stats_path, np.array([mean, std], dtype=np.float32))




## === cell 6
from collections import OrderedDict
from threading import Lock

_TRAIN_NPY_CACHE: dict[int, Path] = {}
_TRAIN_STATS_CACHE: dict[int, Path] = {}
_TRAIN_STATS_ARR_CACHE: dict[int, np.ndarray] = {}

_NPY_ARRAY_LRU: "OrderedDict[str, np.ndarray]" = OrderedDict()
_NPY_ARRAY_LRU_LOCK = Lock()
_NPY_ARRAY_LRU_MAX_ITEMS = 384  # Speed/memory: reduce RAM pressure for huge tensors.


def _lru_get_npy(path: tp.Union[str, Path]) -> np.ndarray:
    key = str(path)
    with _NPY_ARRAY_LRU_LOCK:
        arr = _NPY_ARRAY_LRU.get(key)
        if arr is not None:
            _NPY_ARRAY_LRU.move_to_end(key)
            return arr
    arr = np.load(
        key, mmap_mode="r"
    )  # Speed: mmap avoids repeated full reads; correctness unchanged.
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    with _NPY_ARRAY_LRU_LOCK:
        _NPY_ARRAY_LRU[key] = arr
        _NPY_ARRAY_LRU.move_to_end(key)
        while len(_NPY_ARRAY_LRU) > _NPY_ARRAY_LRU_MAX_ITEMS:
            _NPY_ARRAY_LRU.popitem(last=False)
    return arr


def ensure_train_npy(spec_id: int):
    spec_id = int(spec_id)
    p = _TRAIN_NPY_CACHE.get(spec_id)
    if p is not None:
        return p
    out = TRAIN_SPEC_SPLIT / f"{spec_id}.npy"
    stats = TRAIN_SPEC_SPLIT / f"{spec_id}_stats.npy"
    if out.exists() and stats.exists():
        _TRAIN_NPY_CACHE[spec_id] = out
        _TRAIN_STATS_CACHE[spec_id] = stats
        return out
    in_path = TRAIN_SPEC / f"{spec_id}.parquet"
    spec_parquet_to_npy(in_path, out, stats)
    _TRAIN_NPY_CACHE[spec_id] = out
    _TRAIN_STATS_CACHE[spec_id] = stats
    return out


def ensure_train_stats(spec_id: int):
    spec_id = int(spec_id)
    p = _TRAIN_STATS_CACHE.get(spec_id)
    if p is not None:
        return p
    ensure_train_npy(spec_id)
    return _TRAIN_STATS_CACHE[spec_id]


def load_train_stats_arr(spec_id: int) -> np.ndarray:
    spec_id = int(spec_id)
    arr = _TRAIN_STATS_ARR_CACHE.get(spec_id)
    if arr is not None:
        return arr
    p = ensure_train_stats(spec_id)
    ms = np.load(p)
    ms = np.asarray(ms, dtype=np.float32)
    _TRAIN_STATS_ARR_CACHE[spec_id] = ms
    return ms




## === cell 7
FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]


class HMSHBACSpecDataset(Dataset):
    def __init__(
        self,
        image_paths: tp.Sequence[FilePath],
        labels: tp.Sequence[Label],
        transform: A.Compose,
        stats_paths: tp.Optional[tp.Sequence[FilePath]] = None,
        stats_arr: tp.Optional[np.ndarray] = None,
    ):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

        self.stats_paths = list(stats_paths) if stats_paths is not None else None
        self.stats_arr = stats_arr  # float32 array shape (N,2) or None

        self._split_indices = np.array_split(np.arange(CFG.img_size_h), CFG.channels)
        self._eps = 1e-6

        self._last_path = None
        self._last_img = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]

        if img_path == self._last_path and self._last_img is not None:
            img2d = self._last_img
        else:
            img2d = _lru_get_npy(img_path)
            self._last_path = img_path
            self._last_img = img2d

        if CFG.channels == 1:
            img = img2d[..., None]  # (H, W, 1)
        else:
            chunks = [img2d[idx, :] for idx in self._split_indices]
            img = np.stack(chunks, axis=-1)  # (H, W, C)

        img = self.transform(image=img)["image"]  # torch tensor, shape (C,H,W)

        if self.stats_arr is not None:
            mean = float(self.stats_arr[index, 0])
            std = float(self.stats_arr[index, 1])
            img = img - mean
            img = img / (std + self._eps)
        elif self.stats_paths is not None:
            ms = np.load(self.stats_paths[index])
            mean = float(ms[0])
            std = float(ms[1])
            img = img - mean
            img = img / (std + self._eps)
        else:
            img_mean = img.mean(dim=(1, 2), keepdim=True)
            img = img - img_mean
            img_std = img.std(dim=(1, 2), keepdim=True)
            img = img / (img_std + self._eps)

        return {"data": img, "target": torch.tensor(label, dtype=torch.float32)}




## === cell 8
class HMSHBACSpecModel(nn.Module):
    def __init__(
        self, model_name: str, pretrained: bool, in_channels: int, num_classes: int
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )

    def forward(self, x):
        return self.model(x)




## === cell 9
_TRANSFORMS_TRAIN = A.Compose(
    [
        ToTensorV2(p=1.0),
    ]
)
_TRANSFORMS_VALID = A.Compose(
    [
        ToTensorV2(p=1.0),
    ]
)


def get_transforms(CFG, train: bool):
    return _TRANSFORMS_TRAIN if train else _TRANSFORMS_VALID


def kl_div_loss(p_pred: torch.Tensor, p_true: torch.Tensor, eps: float = 1e-7):
    p = torch.softmax(p_pred, dim=1).clamp(eps, 1.0)
    q = p_true.clamp(eps, 1.0)
    return (q * (q.log() - p.log())).sum(dim=1).mean()




## === cell 10
FOLDS = [0, 1]  # unchanged
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.seed)

fold_indices = []
for fold, (tr_idx, va_idx) in enumerate(sgkf.split(train, strat, groups)):
    if fold in FOLDS:
        fold_indices.append((fold, tr_idx, va_idx))

len(fold_indices), [f for f, _, _ in fold_indices]




## === cell 11
from concurrent.futures import ThreadPoolExecutor, as_completed

train_spec_ids_all = train["spectrogram_id"].to_numpy(dtype=np.int64, copy=False)
_needed_mask = np.zeros(len(train), dtype=bool)
for _, tr_idx, va_idx in fold_indices:
    _needed_mask[tr_idx] = True
    _needed_mask[va_idx] = True
_needed_train_spec_ids = np.unique(train_spec_ids_all[_needed_mask]).astype(np.int64)

_to_make = []
for sid in _needed_train_spec_ids:
    sid = int(sid)
    out = TRAIN_SPEC_SPLIT / f"{sid}.npy"
    stats = TRAIN_SPEC_SPLIT / f"{sid}_stats.npy"
    if out.exists() and stats.exists():
        _TRAIN_NPY_CACHE[sid] = out
        _TRAIN_STATS_CACHE[sid] = stats
        continue
    _to_make.append(sid)


def _make_train_one(sid: int) -> int:
    sid = int(sid)
    out = TRAIN_SPEC_SPLIT / f"{sid}.npy"
    stats = TRAIN_SPEC_SPLIT / f"{sid}_stats.npy"
    spec_parquet_to_npy(TRAIN_SPEC / f"{sid}.parquet", out, stats)
    _TRAIN_NPY_CACHE[sid] = out
    _TRAIN_STATS_CACHE[sid] = stats
    return sid


max_workers = min(12, (os.cpu_count() or 12))
if len(_to_make) > 0:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_make_train_one, sid) for sid in _to_make]
        for _ in tqdm(
            as_completed(futs),
            total=len(futs),
            desc="Preparing train npy (needed folds)",
        ):
            pass

gc.collect()




## === cell 12
def _seed_worker(worker_id: int):
    worker_seed = (CFG.seed + worker_id) % (2**32 - 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _fast_collate(batch):
    x = torch.stack([b["data"] for b in batch], dim=0)
    t = torch.stack([b["target"] for b in batch], dim=0)
    return {"data": x, "target": t}


def make_loader(
    df: pd.DataFrame,
    labels_arr: np.ndarray,
    indices: np.ndarray,
    transform,
    shuffle: bool,
):
    spec_ids_full = df["spectrogram_id"].to_numpy(dtype=np.int64, copy=False)
    spec_ids = spec_ids_full[indices]

    img_paths = [ensure_train_npy(int(sid)) for sid in spec_ids]

    uniq, inv = np.unique(spec_ids, return_inverse=True)
    uniq_stats = np.empty((len(uniq), 2), dtype=np.float32)
    for i, sid in enumerate(uniq):
        uniq_stats[i] = load_train_stats_arr(int(sid))
    stats_arr = uniq_stats[inv]

    labels = labels_arr[indices]
    ds = HMSHBACSpecDataset(
        img_paths, labels, transform=transform, stats_paths=None, stats_arr=stats_arr
    )

    num_workers = int(CFG.num_workers)
    g = torch.Generator()
    g.manual_seed(CFG.seed)

    kwargs = dict(
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
        collate_fn=_fast_collate,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(CFG.persistent_workers)
        kwargs["prefetch_factor"] = int(CFG.prefetch_factor)
    return DataLoader(ds, **kwargs)




## === cell 13
def train_one_fold(fold: int, tr_idx: np.ndarray, va_idx: np.ndarray):
    model = HMSHBACSpecModel(
        model_name=CFG.model_name,
        pretrained=True,
        in_channels=CFG.channels,
        num_classes=N_CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(CFG.enable_amp and device.type == "cuda")
    )

    train_loader = make_loader(
        train, y, tr_idx, get_transforms(CFG, train=True), shuffle=True
    )
    valid_loader = make_loader(
        train, y, va_idx, get_transforms(CFG, train=False), shuffle=False
    )

    for epoch in range(CFG.max_epoch):
        model.train()
        tr_loss = 0.0
        n = 0
        for batch in tqdm(
            train_loader,
            desc=f"Fold {fold} epoch {epoch+1}/{CFG.max_epoch} train",
            leave=False,
        ):
            x = batch["data"].to(device, non_blocking=True)
            t = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=(CFG.enable_amp and device.type == "cuda")
            ):
                logits = model(x)
                loss = kl_div_loss(logits, t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            tr_loss += loss.item() * bs
            n += bs

        model.eval()
        va_loss = 0.0
        m = 0
        with torch.no_grad():
            for batch in tqdm(
                valid_loader,
                desc=f"Fold {fold} epoch {epoch+1}/{CFG.max_epoch} valid",
                leave=False,
            ):
                x = batch["data"].to(device, non_blocking=True)
                t = batch["target"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(
                    enabled=(CFG.enable_amp and device.type == "cuda")
                ):
                    logits = model(x)
                    loss = kl_div_loss(logits, t)
                bs = x.size(0)
                va_loss += loss.item() * bs
                m += bs

        print(
            f"Fold {fold} epoch {epoch+1}: train_loss={tr_loss/max(n,1):.5f} valid_loss={va_loss/max(m,1):.5f}"
        )

    return model




## === cell 14
models = []
for fold, tr_idx, va_idx in fold_indices:
    torch.cuda.empty_cache()
    gc.collect()
    mdl = train_one_fold(fold, tr_idx, va_idx)
    models.append(mdl)

len(models)




## === cell 15
from concurrent.futures import ThreadPoolExecutor, as_completed

test_spec_ids = test["spectrogram_id"].to_numpy(dtype=np.int64, copy=False)
test_img_paths = [TEST_SPEC_SPLIT / f"{int(sid)}.npy" for sid in test_spec_ids]
test_stats_paths = [TEST_SPEC_SPLIT / f"{int(sid)}_stats.npy" for sid in test_spec_ids]

to_make = []
for sid, p_img, p_stats in zip(test_spec_ids, test_img_paths, test_stats_paths):
    if p_img.exists() and p_stats.exists():
        continue
    to_make.append(int(sid))


def _make_test_one(sid: int):
    sid = int(sid)
    p_img = TEST_SPEC_SPLIT / f"{sid}.npy"
    p_stats = TEST_SPEC_SPLIT / f"{sid}_stats.npy"
    spec_parquet_to_npy(TEST_SPEC / f"{sid}.parquet", p_img, p_stats)
    return sid


max_workers = min(12, (os.cpu_count() or 12))
if len(to_make) > 0:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_make_test_one, int(sid)) for sid in to_make]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="Preparing test npy"):
            pass

missing = [i for i, p in enumerate(test_stats_paths) if not p.exists()]
for i in missing:
    sid = int(test_spec_ids[i])
    spec_parquet_to_npy(
        TEST_SPEC / f"{sid}.parquet",
        TEST_SPEC_SPLIT / f"{sid}.npy",
        test_stats_paths[i],
    )

test_stats_arr = np.stack(
    [np.asarray(np.load(p), dtype=np.float32) for p in test_stats_paths], axis=0
)

dummy_labels = np.full((len(test_img_paths), N_CLASSES), -1, dtype="float32")
test_ds = HMSHBACSpecDataset(
    test_img_paths,
    [l for l in dummy_labels],
    transform=get_transforms(CFG, train=False),
    stats_paths=None,
    stats_arr=test_stats_arr,
)

num_workers = int(CFG.num_workers)
test_loader_kwargs = dict(
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    collate_fn=_fast_collate,
)
if num_workers > 0:
    test_loader_kwargs["persistent_workers"] = bool(CFG.persistent_workers)
    test_loader_kwargs["prefetch_factor"] = int(CFG.prefetch_factor)

test_loader = DataLoader(test_ds, **test_loader_kwargs)




## === cell 16
def run_inference_loop(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer", leave=False):
            x = batch["data"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(
                enabled=(CFG.enable_amp and device.type == "cuda")
            ):
                y = model(x)
                p = torch.softmax(y, dim=1)
            preds.append(p.float().cpu().numpy())
    return np.concatenate(preds, axis=0)




## === cell 17
test_pred = np.zeros((len(test), N_CLASSES), dtype=np.float32)
for mdl in models:
    test_pred += run_inference_loop(mdl, test_loader, device)
test_pred /= max(len(models), 1)

test_pred = np.clip(test_pred, 1e-7, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

test_pred.shape




## === cell 18
test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df["eeg_id"] = test["eeg_id"].values

test_pred_eeg = test_pred_df.groupby("eeg_id", as_index=False)[CLASSES].mean()

sub = smpl_sub[["eeg_id"]].merge(test_pred_eeg, on="eeg_id", how="left")

fill_uniform = np.full((1, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float32)
for c in CLASSES:
    sub[c] = sub[c].astype(np.float32)
missing_rows = sub[CLASSES].isna().any(axis=1)
if missing_rows.any():
    sub.loc[missing_rows, CLASSES] = fill_uniform

vals = sub[CLASSES].values.astype(np.float64)
vals = np.clip(vals, 1e-7, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[CLASSES] = vals.astype(np.float32)

sub.to_csv("submission.csv", index=False)

print("submission.csv written:", sub.shape)
sub.head()




## === cell 19
assert sub.shape[0] == smpl_sub.shape[0], (sub.shape, smpl_sub.shape)
assert list(sub.columns) == ["eeg_id"] + CLASSES
row_sums = sub[CLASSES].sum(axis=1).values
assert np.all(np.isfinite(row_sums))
assert np.max(np.abs(row_sums - 1.0)) < 1e-4
print("All checks passed.")

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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

# 5. Target score

1.0598384737455746

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import StratifiedGroupKFold

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass



## === cell 2
base_path = "/kaggle/input/hms-harmful-brain-activity-classification"
train_csv_path = os.path.join(base_path, "train.csv")
test_csv_path = os.path.join(base_path, "test.csv")
sample_submission_csv_path = os.path.join(base_path, "sample_submission.csv")

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_raw = pd.read_csv(train_csv_path)
test = pd.read_csv(test_csv_path)
sample_sub = pd.read_csv(sample_submission_csv_path)

grp_cols = [
    "label_id",
    "eeg_id",
    "spectrogram_id",
    "patient_id",
    "expert_consensus",
    "eeg_label_offset_seconds",
]
vote_agg = train_raw.groupby(grp_cols, as_index=False)[TARGET_COLS].sum()
df = vote_agg.copy()

df["eeg_path"] = base_path + "/train_eegs/" + df["eeg_id"].astype(str) + ".parquet"
df["spec_path"] = (
    base_path + "/train_spectrograms/" + df["spectrogram_id"].astype(str) + ".parquet"
)
df["class_name"] = df["expert_consensus"].copy()

class_name_to_label = {
    "Seizure": 0,
    "LPD": 1,
    "GPD": 2,
    "LRDA": 3,
    "GRDA": 4,
    "Other": 5,
}
df["class_label"] = df["class_name"].map(class_name_to_label)

df["spectrogram_label_offset_seconds"] = df["eeg_label_offset_seconds"].astype(
    np.float32
)

test["eeg_path"] = base_path + "/test_eegs/" + test["eeg_id"].astype(str) + ".parquet"
test["spec_path"] = (
    base_path + "/test_spectrograms/" + test["spectrogram_id"].astype(str) + ".parquet"
)
if "spectrogram_label_offset_seconds" not in test.columns:
    test["spectrogram_label_offset_seconds"] = 0.0
if "class_label" not in test.columns:
    test["class_label"] = 0

print("Train/val pool rows:", len(df), "Kaggle test rows:", len(test))



## === cell 3
df.head()



## === cell 4
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
num_classes = 6
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)
model = model.to(device)

ENABLE_TORCH_COMPILE = False
if ENABLE_TORCH_COMPILE and torch.cuda.is_available():
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print("torch.compile unavailable/failed, continuing without compile:", repr(e))



## === cell 5
_PARQUET_ENGINE = None
_HAVE_PYARROW = False
_HAVE_FASTPARQUET = False
try:
    import pyarrow.parquet as pq  # noqa: F401

    _HAVE_PYARROW = True
    _PARQUET_ENGINE = "pyarrow"
except Exception:
    _HAVE_PYARROW = False

if not _HAVE_PYARROW:
    try:
        import fastparquet  # noqa: F401

        _HAVE_FASTPARQUET = True
        _PARQUET_ENGINE = "fastparquet"
    except Exception:
        _HAVE_FASTPARQUET = False
        _PARQUET_ENGINE = None

_PF_CACHE = {}
_PF_CACHE_ORDER = []
_PF_CACHE_MAX_ITEMS = 1024  # a bit larger to reduce repeated ParquetFile construction

_SPEC_COLS_CACHE = {}


def _pf_cache_get(path):
    return _PF_CACHE.get(path, None)


def _pf_cache_put(path, pf):
    if path in _PF_CACHE:
        return
    _PF_CACHE[path] = pf
    _PF_CACHE_ORDER.append(path)
    if len(_PF_CACHE_ORDER) > _PF_CACHE_MAX_ITEMS:
        old = _PF_CACHE_ORDER.pop(0)
        _PF_CACHE.pop(old, None)


def _get_spec_cols(pf, path):
    cols = _SPEC_COLS_CACHE.get(path)
    if cols is not None:
        return cols
    cols = [c for c in pf.schema.names if c != "time"]
    _SPEC_COLS_CACHE[path] = cols
    return cols


def read_parquet_subset(parquet_file_path, offset_seconds, length, is_eeg=True):
    offset_seconds = float(offset_seconds) if offset_seconds is not None else 0.0
    start_row = int(offset_seconds * 200) if is_eeg else int(offset_seconds / 2)
    end_row = start_row + (10000 if is_eeg else 300)

    if _HAVE_PYARROW:
        pf = _pf_cache_get(parquet_file_path)
        if pf is None:
            pf = pq.ParquetFile(parquet_file_path)
            _pf_cache_put(parquet_file_path, pf)

        cols = None
        if not is_eeg:
            cols = _get_spec_cols(pf, parquet_file_path)

        needed = []
        acc = 0
        md = pf.metadata
        for rg in range(pf.num_row_groups):
            rg_n = md.row_group(rg).num_rows
            rg_start = acc
            rg_end = rg_start + rg_n
            acc = rg_end
            if rg_end <= start_row:
                continue
            if rg_start >= end_row:
                break
            needed.append((rg, rg_start, rg_n))

        if not needed:
            if cols is None:
                cols = pf.schema.names
            return pd.DataFrame({c: [] for c in cols})

        frames = []
        for rg, rg_start, rg_n in needed:
            t = pf.read_row_group(rg, columns=cols)
            dfp = t.to_pandas(types_mapper=None)
            local_start = max(0, start_row - rg_start)
            local_end = min(rg_n, end_row - rg_start)
            frames.append(dfp.iloc[local_start:local_end])

        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, axis=0, ignore_index=True)

    if _PARQUET_ENGINE is None:
        dfp = pd.read_parquet(parquet_file_path)
    else:
        dfp = pd.read_parquet(parquet_file_path, engine=_PARQUET_ENGINE)

    if (not is_eeg) and ("time" in dfp.columns):
        dfp = dfp.drop(columns=["time"])
    return dfp.iloc[start_row:end_row]


def convert_2d_to_3d(data_2d: pd.DataFrame):
    x = data_2d.to_numpy(copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0, 255).astype(np.uint8, copy=False)
    x = np.repeat(x[:, :, None], 3, axis=2)
    return x


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_CACHE_DIR = "/kaggle/working/spec_tensor_cache_v1"
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_key(spec_path: str, offset_seconds: float):
    s = f"{spec_path}|{float(offset_seconds):.6f}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _cache_path(spec_path: str, offset_seconds: float):
    return os.path.join(_CACHE_DIR, _cache_key(spec_path, offset_seconds) + ".pt")


def load_or_build_spec_tensor(spec_path: str, offset_seconds: float):
    cp = _cache_path(spec_path, offset_seconds)
    if os.path.exists(cp):
        return torch.load(cp, map_location="cpu", weights_only=False)

    spec_data = read_parquet_subset(
        spec_path,
        offset_seconds,
        300,
        is_eeg=False,
    )
    spec_data_3d = convert_2d_to_3d(spec_data)
    data_img = Image.fromarray(spec_data_3d, mode="RGB")
    features = transform(data_img)

    tmp = f"{cp}.{os.getpid()}.tmp"
    try:
        torch.save(features, tmp)
        try:
            os.replace(tmp, cp)  # atomic if same filesystem
        except FileNotFoundError:
            if os.path.exists(cp):
                return torch.load(cp, map_location="cpu", weights_only=False)
            raise
        except OSError:
            if os.path.exists(cp):
                return torch.load(cp, map_location="cpu", weights_only=False)
            raise
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

    return features


class SPECTROGRAM_Dataset(Dataset):
    """
    Return soft target distribution from vote columns when available.
    This aligns KLDivLoss with the competition's KL metric while keeping your model and loop structure.
    """

    def __init__(
        self, dataframe: pd.DataFrame, transform, return_soft_targets: bool = True
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_soft_targets = return_soft_targets
        self.has_votes = all(c in self.dataframe.columns for c in TARGET_COLS)

        self._spec_path = self.dataframe["spec_path"].to_numpy()
        if "spectrogram_label_offset_seconds" in self.dataframe.columns:
            self._offset = self.dataframe["spectrogram_label_offset_seconds"].to_numpy(
                dtype=np.float32, copy=False
            )
        else:
            self._offset = np.zeros(len(self.dataframe), dtype=np.float32)
        if "class_label" in self.dataframe.columns:
            self._class_label = (
                self.dataframe["class_label"]
                .fillna(0)
                .to_numpy(dtype=np.int64, copy=False)
            )
        else:
            self._class_label = np.zeros(len(self.dataframe), dtype=np.int64)

        if self.has_votes and self.return_soft_targets:
            self._votes = self.dataframe[TARGET_COLS].to_numpy(
                dtype=np.float32, copy=False
            )
        else:
            self._votes = None

    def __len__(self):
        return len(self._spec_path)

    def __getitem__(self, idx):
        offset = float(self._offset[idx])
        features = load_or_build_spec_tensor(self._spec_path[idx], offset)

        if self.return_soft_targets and (self._votes is not None):
            votes = self._votes[idx]
            s = float(votes.sum())
            if s <= 0:
                target = torch.full((6,), 1.0 / 6.0, dtype=torch.float32)
            else:
                target = torch.from_numpy(votes / s).to(dtype=torch.float32)
            label = int(self._class_label[idx])
            return features, target, label
        else:
            label = int(self._class_label[idx])
            return features, label




## === cell 6
lr = 0.001
num_epochs = 100
batch_size = 32
factor = 0.8

cpu_count = os.cpu_count() or 2
if torch.cuda.is_available():
    num_workers = min(8, max(4, cpu_count // 2))
else:
    num_workers = min(4, max(2, cpu_count // 2))



## === cell 7
labels = df["class_label"].values
groups = df["eeg_id"].values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

train_idx, val_idx = next(sgkf.split(X=df, y=labels, groups=groups))
train_df, val_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[
    val_idx
].reset_index(drop=True)

train_dataset = SPECTROGRAM_Dataset(train_df, transform, return_soft_targets=True)
val_dataset = SPECTROGRAM_Dataset(val_df, transform, return_soft_targets=True)
test2_dataset = SPECTROGRAM_Dataset(test, transform, return_soft_targets=False)

loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    prefetch_factor=4 if num_workers > 0 else None,
)

g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=g,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)
test2_loader = DataLoader(
    test2_dataset,
    batch_size=batch_size,
    shuffle=False,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)

print(f"Size of train: {len(train_dataset)}")
print(f"Size of val: {len(val_dataset)}")
print(f"Size of test2 (Kaggle test): {len(test2_dataset)}")
print(f"num_workers: {num_workers}, pin_memory: {torch.cuda.is_available()}")



## === cell 8
criterion = torch.nn.KLDivLoss(reduction="batchmean")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=factor, patience=1)



## === cell 9
best_val_loss = float("inf")
training_losses = []
validation_losses = []
accumulation_steps = 4

use_amp = torch.cuda.is_available()
scaler = GradScaler(enabled=use_amp)

try:
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        batch_count = 0
        n_train_batches = len(train_loader)

        for i, batch in enumerate(train_loader):
            features, target_dist, _ = batch
            features = features.to(device, non_blocking=True)
            target_dist = target_dist.to(device, non_blocking=True)
            bsz = features.shape[0]

            with autocast(enabled=use_amp):
                outputs = model(features)
                log_probs = F.log_softmax(outputs, dim=1)
                loss = criterion(log_probs, target_dist) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == n_train_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * bsz
            batch_count += 1

            if batch_count % 500 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}")

        train_loss /= len(train_loader.dataset)
        training_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features, target_dist, _ = batch
                features = features.to(device, non_blocking=True)
                target_dist = target_dist.to(device, non_blocking=True)
                bsz = features.shape[0]
                with autocast(enabled=use_amp):
                    outputs = model(features)
                    log_probs = F.log_softmax(outputs, dim=1)
                    loss = criterion(log_probs, target_dist)
                val_loss += loss.item() * bsz

        val_loss /= len(val_loader.dataset)
        validation_losses.append(val_loss)
        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved with Validation Loss: {val_loss:.4f}")

except Exception as e:
    print("TRAINING ERROR (will fall back to current weights for inference):", repr(e))
    if not os.path.exists("best_model.pth"):
        torch.save(model.state_dict(), "best_model.pth")
        print(
            "Saved fallback best_model.pth from current model state due to training error."
        )

if not os.path.exists("best_model.pth"):
    torch.save(model.state_dict(), "best_model.pth")
    print("WARNING: best_model.pth was missing; saved current model state.")



## === cell 10
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

all_probs = []
with torch.no_grad():
    for batch in test2_loader:
        features = batch[0]
        features = features.to(device, non_blocking=True)
        outputs = model(features)
        probs = F.softmax(outputs, dim=1)
        all_probs.append(probs.detach().cpu().numpy())

all_probs = np.concatenate(all_probs, axis=0)

eps = 1e-7
all_probs = np.clip(all_probs, eps, 1.0)
all_probs = all_probs / all_probs.sum(axis=1, keepdims=True)

print(all_probs.shape)



## === cell 11
pred_df = pd.DataFrame(all_probs, columns=TARGET_COLS)
pred_df["eeg_id"] = test["eeg_id"].values

submission = sample_sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")

missing_mask = submission[TARGET_COLS].isna().any(axis=1)
if missing_mask.any():
    submission.loc[missing_mask, TARGET_COLS] = 1.0 / 6.0

vals = submission[TARGET_COLS].to_numpy(dtype=np.float64)
vals = np.clip(vals, eps, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
submission[TARGET_COLS] = vals

submission = submission[["eeg_id"] + TARGET_COLS]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(
    "Row sums (min/max):",
    submission[TARGET_COLS].sum(axis=1).min(),
    submission[TARGET_COLS].sum(axis=1).max(),
)
print("Saved submission.csv")

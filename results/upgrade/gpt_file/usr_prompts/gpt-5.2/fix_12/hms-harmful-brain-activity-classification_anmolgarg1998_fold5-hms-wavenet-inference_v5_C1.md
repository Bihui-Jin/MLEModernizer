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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.7099183199280108

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'The timeout is dominated by (1) reading/parsing thousands of Parquet files into pandas inside `_table_to_numpy_float32`, (2) expensive repeated `np.roll` smoothing allocations in `denoise_filter`, and (3) redundant Parquet row-group concatenation work for every EEG. The refactor keeps the exact same signal construction and model/training logic, but speeds up I/O by converting Parquet directly to NumPy via Arrow (`table.to_numpy`) without pandas, and by reading only the needed row-groups/rows more efficiently. It also makes `denoise_filter` allocation-free (still mathematically identical) and reduces Python overhead in `build_signal_from_parquet` by preallocating/reusing arrays. These changes are equivalent in results (same samples, same filtering, same downsampling), but substantially reduce wall time so the pipeline can complete within 600 seconds given the cache.'

# 9. Code solution

## === cell 0
import os, gc, math
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from scipy.signal import butter, sosfiltfilt


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"
TRAIN_EEG_DIR = f"{DATA_DIR}/train_eegs"
TEST_EEG_DIR = f"{DATA_DIR}/test_eegs"

CACHE_DIR = "/kaggle/working/eeg_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

_NUM_CPU = os.cpu_count() or 1
try:
    torch.set_num_threads(min(8, _NUM_CPU))
except Exception:
    pass

NUM_WORKERS = min(2, _NUM_CPU)
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False
DL_PREFETCH = 2 if NUM_WORKERS > 0 else None
DL_IN_ORDER = True

CACHE_WORKERS = max(1, min(4, _NUM_CPU))

ENABLE_TORCH_COMPILE = False



## === cell 1
df = pd.read_csv(TRAIN_CSV)
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))
df.head()



## === cell 2
train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds"]
].agg({"spectrogram_id": "first", "spectrogram_label_offset_seconds": "min"})
train.columns = ["spec_id", "min"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

cnt = (
    df.groupby(["eeg_id", "expert_consensus"], sort=True)
    .size()
    .rename("cnt")
    .reset_index()
)
cnt = cnt.sort_values(
    ["eeg_id", "cnt", "expert_consensus"], ascending=[True, False, True]
)
mode_df = cnt.drop_duplicates("eeg_id", keep="first")[["eeg_id", "expert_consensus"]]

tmp2 = (
    df.groupby(["eeg_id", "expert_consensus"], sort=False)[["eeg_sub_id"]]
    .agg(min)
    .reset_index()
)
mode_df = pd.merge(mode_df, tmp2, on=["eeg_id", "expert_consensus"], how="left")
train["target"] = mode_df["expert_consensus"].values
train["eeg_sub_id"] = mode_df["eeg_sub_id"].values

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)
train.head()



## === cell 3
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

_FS = 200.0
_LOWCUT = 1.0
_HIGHCUT = 25.0
_ORDER = 6
_SOS = butter(_ORDER, [_LOWCUT, _HIGHCUT], fs=_FS, btype="band", output="sos")


_tmp_smooth = np.empty((10_000,), dtype=np.float32)


def denoise_filter(x: np.ndarray) -> np.ndarray:
    y = sosfiltfilt(_SOS, x).astype(np.float32, copy=False)

    out = _tmp_smooth
    out[:-3] = y[:-3] + y[1:-2] + y[2:-1] + y[3:]
    out[-3] = y[-3] + y[-2] + y[-1] + y[0]
    out[-2] = y[-2] + y[-1] + y[0] + y[1]
    out[-1] = y[-1] + y[0] + y[1] + y[2]
    out *= 0.25

    y = out[0:-1:4]
    return y.astype(np.float32, copy=False)




## === cell 4
IS_TRAINING = True

_ALL_EEG_COLS = sorted({c for group in FEATS for c in group})
_COL2IDX = {c: i for i, c in enumerate(_ALL_EEG_COLS)}
_FEAT_IDXS = [[_COL2IDX[c] for c in group] for group in FEATS]

import pyarrow.parquet as pq
import pyarrow as pa


def _table_to_numpy_float32(table: pa.Table) -> np.ndarray:
    arr = table.to_numpy(zero_copy_only=False)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


_READ_CACHE = {}


def _read_parquet_cols_centered(parq_path: str, cols, n: int = 10_000) -> np.ndarray:
    key = (parq_path, tuple(cols), int(n))
    cached = _READ_CACHE.get(key)
    if cached is not None:
        return cached

    pf = pq.ParquetFile(parq_path)
    try:
        rows = pf.metadata.num_rows
        offset = (rows - n) // 2
        if offset < 0:
            offset = 0
        end = min(rows, offset + n)
        need = end - offset

        rg_starts = []
        cur = 0
        for rg in range(pf.num_row_groups):
            rg_n = pf.metadata.row_group(rg).num_rows
            rg_start = cur
            rg_end = cur + rg_n
            if rg_end > offset and rg_start < end:
                rg_starts.append((rg, rg_start, rg_end))
            cur = rg_end

        if not rg_starts:
            table = pf.read(columns=cols)
            table = table.slice(0, need) if table.num_rows > need else table
        else:
            tables = [pf.read_row_group(rg, columns=cols) for rg, _, _ in rg_starts]
            table = pa.concat_tables(tables, promote_options="none")

            base = rg_starts[0][1]
            local_off = offset - base
            if local_off != 0 or table.num_rows != need:
                table = table.slice(local_off, need)

        arr = _table_to_numpy_float32(table)

        if arr.shape[0] < n:
            out = np.empty((n, arr.shape[1]), dtype=np.float32)
            out[: arr.shape[0]] = arr
            out[arr.shape[0] :] = arr[arr.shape[0] - 1 : arr.shape[0]]
            arr = out
        elif arr.shape[0] > n:
            arr = arr[:n]

        _READ_CACHE[key] = arr
        return arr
    finally:
        try:
            pf.close()
        except Exception:
            pass


def build_signal_from_parquet(eeg_id: int, eeg_dir: str):
    parq_path = f"{eeg_dir}/{eeg_id}.parquet"
    arr_all = _read_parquet_cols_centered(parq_path, _ALL_EEG_COLS, n=10_000)

    signals = np.empty((16, 2500), dtype=np.float32)
    out_i = 0
    for idxs in _FEAT_IDXS:
        arr = arr_all[:, idxs]  # (10000,5)
        diffs0 = arr[:, 0] - arr[:, 1]
        diffs1 = arr[:, 1] - arr[:, 2]
        diffs2 = arr[:, 2] - arr[:, 3]
        diffs3 = arr[:, 3] - arr[:, 4]

        signals[out_i] = denoise_filter(diffs0)
        out_i += 1
        signals[out_i] = denoise_filter(diffs1)
        out_i += 1
        signals[out_i] = denoise_filter(diffs2)
        out_i += 1
        signals[out_i] = denoise_filter(diffs3)
        out_i += 1

    return signals  # (16, 2500) float32


def _cache_paths(prefix: str):
    mm_path = os.path.join(CACHE_DIR, f"{prefix}_signals.f32.mmap")
    ids_path = os.path.join(CACHE_DIR, f"{prefix}_eeg_ids.npy")
    done_path = os.path.join(CACHE_DIR, f"{prefix}_done_mask.npy")
    return mm_path, ids_path, done_path


def cache_eeg_signals(eeg_ids, eeg_dir: str, prefix: str):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    eeg_ids = np.asarray(eeg_ids, dtype=np.int64)
    eeg_ids = np.unique(eeg_ids)

    mm_path, ids_path, done_path = _cache_paths(prefix)

    def _compute_one(i: int, eid: int):
        return i, build_signal_from_parquet(int(eid), eeg_dir)

    if os.path.exists(mm_path) and os.path.exists(ids_path):
        cached_ids = np.load(ids_path)
        if np.array_equal(cached_ids, eeg_ids):
            mm = np.memmap(
                mm_path, mode="r+", dtype="float32", shape=(len(eeg_ids), 16, 2500)
            )
            if os.path.exists(done_path):
                done = np.load(done_path)
                if done.shape[0] == len(eeg_ids) and done.dtype == np.bool_:
                    missing = np.flatnonzero(~done)
                    if missing.size:
                        with ThreadPoolExecutor(max_workers=CACHE_WORKERS) as ex:
                            futures = [
                                ex.submit(_compute_one, int(i), int(eeg_ids[int(i)]))
                                for i in missing
                            ]
                            for fut in tqdm(
                                as_completed(futures),
                                total=len(futures),
                                desc=f"Resuming cache {prefix}",
                            ):
                                i, sig = fut.result()
                                mm[int(i)] = sig
                                done[int(i)] = True
                        mm.flush()
                        np.save(done_path, done)
            return (
                np.memmap(
                    mm_path, mode="r", dtype="float32", shape=(len(eeg_ids), 16, 2500)
                ),
                eeg_ids,
            )

    mm = np.memmap(mm_path, mode="w+", dtype="float32", shape=(len(eeg_ids), 16, 2500))
    done = np.zeros((len(eeg_ids),), dtype=np.bool_)

    with ThreadPoolExecutor(max_workers=CACHE_WORKERS) as ex:
        futures = [
            ex.submit(_compute_one, int(i), int(eid)) for i, eid in enumerate(eeg_ids)
        ]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Caching {prefix} EEG signals",
        ):
            i, sig = fut.result()
            mm[int(i)] = sig
            done[int(i)] = True

    mm.flush()
    np.save(ids_path, eeg_ids)
    np.save(done_path, done)
    mm = np.memmap(mm_path, mode="r", dtype="float32", shape=(len(eeg_ids), 16, 2500))
    return mm, eeg_ids


class CustomDataset(Dataset):
    def __init__(self, dataframe, eegs_data=None, mode="Train", transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.mode = mode
        self.eegs_data = eegs_data

        self._cache_map = None
        if (
            isinstance(self.eegs_data, dict)
            and ("_mm" in self.eegs_data)
            and ("_id2idx" in self.eegs_data)
        ):
            self._cache_map = self.eegs_data["_id2idx"]

        self._eeg_id = self.dataframe["eeg_id"].to_numpy(dtype=np.int64, copy=False)
        if self.mode != "Test":
            self._labels = self.dataframe[list(TARGETS)].to_numpy(
                dtype=np.float32, copy=True
            )
            self._eeg_sub_id = self.dataframe["eeg_sub_id"].to_numpy(copy=False)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        eeg_id = int(self._eeg_id[idx])

        if self.mode == "Test":
            if self.eegs_data is not None and self._cache_map is not None:
                mm = self.eegs_data["_mm"]
                signal_eeg = mm[self._cache_map[eeg_id]]
            else:
                signal_eeg = build_signal_from_parquet(eeg_id, TEST_EEG_DIR)
            return torch.from_numpy(signal_eeg)

        if self.eegs_data is not None:
            if self._cache_map is not None:
                mm = self.eegs_data["_mm"]
                signal_eeg = mm[self._cache_map[eeg_id]]
            else:
                eeg_sub_id = self._eeg_sub_id[idx]
                eeg_key = f"{eeg_id}_{eeg_sub_id}"
                signal_eeg = self.eegs_data[eeg_key]
        else:
            signal_eeg = build_signal_from_parquet(eeg_id, TRAIN_EEG_DIR)

        labels = self._labels[idx]
        s = float(np.sum(labels))
        if s > 0:
            labels = labels / s
        else:
            labels = np.ones(6, dtype=np.float32) / 6.0
        return torch.from_numpy(signal_eeg), torch.from_numpy(
            labels.astype(np.float32, copy=False)
        )




## === cell 5
class wave_residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num):
        super(wave_residual_block, self).__init__()
        dilatn = 2 ** (layer_num - 1)
        self.dilatn = dilatn
        self.filter_conv = nn.Conv1d(
            in_channels,
            out_channels,
            2,
            stride=1,
            padding=dilatn,
            dilation=dilatn,
            bias=False,
        )
        self.gate_conv = nn.Conv1d(
            in_channels,
            out_channels,
            2,
            stride=1,
            padding=dilatn,
            dilation=dilatn,
            bias=False,
        )
        self.conv_skip = nn.Conv1d(out_channels, out_channels, 1, 1)
        self.conv_res = nn.Conv1d(out_channels, out_channels, 1, 1)

    def forward(self, x):
        y = F.tanh(self.filter_conv(x)) * F.sigmoid(self.gate_conv(x))
        y = y[:, :, : -self.dilatn]
        y_skip = self.conv_skip(y)
        y_res = self.conv_res(y)
        x = x + y_res
        return x, y_skip


class WaveBlock(nn.Module):
    def __init__(self, in_channels=4, num_layers=6):
        super(WaveBlock, self).__init__()
        self.waveblocks = nn.ModuleList(
            [wave_residual_block(16, 16, i) for i in range(1, num_layers + 1)]
        )
        self.conv0 = nn.Conv1d(in_channels, 16, 1, 1)
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(16, 4, 1)

    def forward(self, x):
        x = self.conv0(x)
        skip_connections = []
        for i in range(self.num_layers):
            x, y = self.waveblocks[i](x)
            skip_connections.append(y)
        y_list = torch.stack(skip_connections)
        x = torch.sum(y_list, dim=0, keepdim=True)
        x = torch.squeeze(x, dim=0)
        x = self.conv1(x)
        x = F.relu(x)
        return x


class WaveClassifier(nn.Module):
    def __init__(self, in_channels=16, num_layers=6):
        super(WaveClassifier, self).__init__()
        self.waveblock = WaveBlock()
        self.conv1 = nn.Conv1d(16, 48, 20, 10)
        self.conv2 = nn.Conv1d(48, 32, 10, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1536, 6)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, inp):
        x = []
        for i in range(4):
            x.append(self.waveblock(inp[:, i : i + 4]))
        x = torch.concat(x, dim=1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x




## === cell 6
criterion = nn.KLDivLoss(reduction="batchmean").to(device)


def validatn_loss(data_loader, model):
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.inference_mode():
        for x, y in data_loader:
            bs = x.size(0)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            outputs = model(x)
            loss = criterion(outputs.log(), y)
            total_loss += float(loss.item()) * bs
            total_n += bs
    return total_loss / max(1, total_n)




## === cell 7
if not os.path.exists("wavenet_model"):
    os.makedirs("wavenet_model")




## === cell 8
def have_local_fold_models(n_folds=5):
    for i in range(n_folds):
        if not os.path.exists(f"wavenet_model/model_best_fold_{i}.pt"):
            return False
    return True


unique_train_eeg_ids = np.unique(train["eeg_id"].astype(np.int64).values)
train_mm, train_ids = cache_eeg_signals(
    unique_train_eeg_ids, TRAIN_EEG_DIR, prefix="train"
)
train_id2idx = {int(eid): int(i) for i, eid in enumerate(train_ids)}
train_cache = {"_mm": train_mm, "_id2idx": train_id2idx}


def _maybe_compile(model: nn.Module) -> nn.Module:
    if not ENABLE_TORCH_COMPILE:
        return model
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode="max-autotune", fullgraph=False)
        except Exception:
            return model
    return model


if IS_TRAINING and (not have_local_fold_models(5)):
    gkf = GroupKFold(n_splits=5)
    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.target, train.patient_id)
    ):
        dataset_train = CustomDataset(
            dataframe=train.iloc[train_index], eegs_data=train_cache, mode="Train"
        )
        dataset_valid = CustomDataset(
            dataframe=train.iloc[valid_index], eegs_data=train_cache, mode="Train"
        )

        train_dl_kwargs = dict(
            batch_size=32,
            shuffle=True,
            drop_last=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
        )
        if NUM_WORKERS > 0:
            train_dl_kwargs["prefetch_factor"] = DL_PREFETCH

        val_dl_kwargs = dict(
            batch_size=16,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            in_order=DL_IN_ORDER,
        )
        if NUM_WORKERS > 0:
            val_dl_kwargs["prefetch_factor"] = DL_PREFETCH

        train_dataloader = DataLoader(dataset_train, **train_dl_kwargs)
        val_loader = DataLoader(dataset_valid, **val_dl_kwargs)

        min_val_loss = 99.0
        epochs = 6

        our_model = WaveClassifier().to(device)
        our_model = _maybe_compile(our_model)
        optimizer = optim.AdamW(our_model.parameters(), lr=0.001, weight_decay=0.01)

        for epoch in range(epochs):
            pbar = tqdm(train_dataloader, desc=f"Fold {i} Epoch {epoch}")
            running_loss = 0.0
            cnt = 0
            our_model.train()

            for inp1, label in pbar:
                cnt += 1
                inp1 = inp1.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                pred = our_model(inp1)
                loss = criterion(pred.log(), label)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += (
                    loss.detach() * inp1.size(0) / len(train_dataloader.dataset)
                )
                pbar.set_description(
                    f"Fold {i} Ep {epoch} loss {loss.item():.4f} run {running_loss.item():.4f}"
                )

                if cnt == len(train_dataloader) // 2:
                    val_loss = validatn_loss(val_loader, our_model)
                    if min_val_loss > val_loss:
                        min_val_loss = val_loss
                        torch.save(
                            our_model.state_dict(),
                            f"wavenet_model/model_best_fold_{i}.pt",
                        )
                        print("half-check fold,epoch,val loss : ", i, epoch, val_loss)
                    our_model.train()

            val_loss = validatn_loss(val_loader, our_model)
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                torch.save(
                    our_model.state_dict(), f"wavenet_model/model_best_fold_{i}.pt"
                )
                print("end-epoch fold,epoch,val loss : ", i, epoch, val_loss)

        del our_model, dataset_train, dataset_valid, train_dataloader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("Local fold models ready:", have_local_fold_models(5))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1762177077.py in <cell line: 0>()
      7 
      8 unique_train_eeg_ids = np.unique(train["eeg_id"].astype(np.int64).values)
----> 9 train_mm, train_ids = cache_eeg_signals(
     10     unique_train_eeg_ids, TRAIN_EEG_DIR, prefix="train"
     11 )

/tmp/ipykernel_55/917407192.py in cache_eeg_signals(eeg_ids, eeg_dir, prefix)
    175             desc=f"Caching {prefix} EEG signals",
    176         ):
--> 177             i, sig = fut.result()
    178             mm[int(i)] = sig
    179             done[int(i)] = True

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_55/917407192.py in _compute_one(i, eid)
    128 
    129     def _compute_one(i: int, eid: int):
--> 130         return i, build_signal_from_parquet(int(eid), eeg_dir)
    131 
    132     if os.path.exists(mm_path) and os.path.exists(ids_path):

/tmp/ipykernel_55/917407192.py in build_signal_from_parquet(eeg_id, eeg_dir)
     88 def build_signal_from_parquet(eeg_id: int, eeg_dir: str):
     89     parq_path = f"{eeg_dir}/{eeg_id}.parquet"
---> 90     arr_all = _read_parquet_cols_centered(parq_path, _ALL_EEG_COLS, n=10_000)
     91 
     92     signals = np.empty((16, 2500), dtype=np.float32)

/tmp/ipykernel_55/917407192.py in _read_parquet_cols_centered(parq_path, cols, n)
     65                 table = table.slice(local_off, need)
     66 
---> 67         arr = _table_to_numpy_float32(table)
     68 
     69         if arr.shape[0] < n:

/tmp/ipykernel_55/917407192.py in _table_to_numpy_float32(table)
     13 def _table_to_numpy_float32(table: pa.Table) -> np.ndarray:
     14     # to_numpy gives (rows, cols) when table is "rows x columns"; ensure contiguous float32.
---> 15     arr = table.to_numpy(zero_copy_only=False)
     16     if arr.dtype != np.float32:
     17         arr = arr.astype(np.float32, copy=False)

AttributeError: 'pyarrow.lib.Table' object has no attribute 'to_numpy'

## === cell 9
test = pd.read_csv(TEST_CSV)
print("Test shape:", test.shape)
test.head()



## === cell 10
test_eeg_ids = test["eeg_id"].astype(np.int64).values
test_mm, test_ids = cache_eeg_signals(test_eeg_ids, TEST_EEG_DIR, prefix="test")
test_id2idx = {int(eid): int(i) for i, eid in enumerate(test_ids)}
test_cache = {"_mm": test_mm, "_id2idx": test_id2idx}

dataset_test = CustomDataset(dataframe=test, mode="Test", eegs_data=test_cache)

test_dl_kwargs = dict(
    batch_size=64,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    in_order=DL_IN_ORDER,
)
if NUM_WORKERS > 0:
    test_dl_kwargs["prefetch_factor"] = DL_PREFETCH

test_loader = DataLoader(dataset_test, **test_dl_kwargs)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3300961904.py in <cell line: 0>()
      1 test_eeg_ids = test["eeg_id"].astype(np.int64).values
----> 2 test_mm, test_ids = cache_eeg_signals(test_eeg_ids, TEST_EEG_DIR, prefix="test")
      3 test_id2idx = {int(eid): int(i) for i, eid in enumerate(test_ids)}
      4 test_cache = {"_mm": test_mm, "_id2idx": test_id2idx}
      5 

/tmp/ipykernel_55/917407192.py in cache_eeg_signals(eeg_ids, eeg_dir, prefix)
    175             desc=f"Caching {prefix} EEG signals",
    176         ):
--> 177             i, sig = fut.result()
    178             mm[int(i)] = sig
    179             done[int(i)] = True

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_55/917407192.py in _compute_one(i, eid)
    128 
    129     def _compute_one(i: int, eid: int):
--> 130         return i, build_signal_from_parquet(int(eid), eeg_dir)
    131 
    132     if os.path.exists(mm_path) and os.path.exists(ids_path):

/tmp/ipykernel_55/917407192.py in build_signal_from_parquet(eeg_id, eeg_dir)
     88 def build_signal_from_parquet(eeg_id: int, eeg_dir: str):
     89     parq_path = f"{eeg_dir}/{eeg_id}.parquet"
---> 90     arr_all = _read_parquet_cols_centered(parq_path, _ALL_EEG_COLS, n=10_000)
     91 
     92     signals = np.empty((16, 2500), dtype=np.float32)

/tmp/ipykernel_55/917407192.py in _read_parquet_cols_centered(parq_path, cols, n)
     65                 table = table.slice(local_off, need)
     66 
---> 67         arr = _table_to_numpy_float32(table)
     68 
     69         if arr.shape[0] < n:

/tmp/ipykernel_55/917407192.py in _table_to_numpy_float32(table)
     13 def _table_to_numpy_float32(table: pa.Table) -> np.ndarray:
     14     # to_numpy gives (rows, cols) when table is "rows x columns"; ensure contiguous float32.
---> 15     arr = table.to_numpy(zero_copy_only=False)
     16     if arr.dtype != np.float32:
     17         arr = arr.astype(np.float32, copy=False)

AttributeError: 'pyarrow.lib.Table' object has no attribute 'to_numpy'

## === cell 11
n_folds = 5
preds_all_fold = []

if have_local_fold_models(n_folds):
    our_model = WaveClassifier().float()
    for i in range(n_folds):
        ckpt_path = f"wavenet_model/model_best_fold_{i}.pt"
        state = torch.load(ckpt_path, map_location="cpu")
        our_model.load_state_dict(state)
        our_model.to(device)
        our_model.eval()

        inf_model = _maybe_compile(our_model)

        preds_list = []
        with torch.inference_mode():
            for batch in tqdm(test_loader, desc=f"Infer fold {i}"):
                inp1 = batch.to(device, non_blocking=True)
                pred = inf_model(inp1).detach().cpu().numpy()
                preds_list.append(pred)
        preds = np.vstack(preds_list)
        preds_all_fold.append(preds)

    prediction_all_fold = np.mean(preds_all_fold, axis=0)
else:
    print(
        "WARNING: Fold checkpoints not found. Writing uniform-probability submission."
    )
    prediction_all_fold = np.ones((len(test), 6), dtype=np.float64) / 6.0

print("Pred shape:", prediction_all_fold.shape)



## === cell 12
sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
probs = prediction_all_fold.astype(np.float64)
probs = np.clip(probs, 1e-12, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

sub[list(TARGETS)] = probs
sub.to_csv("submission.csv", index=False)

print("Submission shape", sub.shape)
print("Row 0 sums to:", sub.iloc[0, -6:].sum())
sub.head()

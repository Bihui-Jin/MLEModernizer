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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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
import sys
from pathlib import Path

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

print("Python:", sys.version)
print("DATA_PATH exists:", os.path.exists(DATA_PATH))
print("OUT_PATH exists:", os.path.exists(OUT_PATH))

train_csv = f"{DATA_PATH}/train.csv"
test_csv = f"{DATA_PATH}/test.csv"
sample_csv = f"{DATA_PATH}/sample_submission.csv"

print("train.csv exists:", os.path.exists(train_csv))
print("test.csv exists:", os.path.exists(test_csv))
print("sample_submission.csv exists:", os.path.exists(sample_csv))




## === cell 1
import numpy as np
import pandas as pd

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
sample_sub = pd.read_csv(sample_csv)

assert (
    list(sample_sub.columns) == ["eeg_id"] + TARGET_COLS
), "Unexpected sample_submission columns"
print("Train rows:", len(train_df), "Test rows:", len(test_df))

y_agg = train_df.groupby("eeg_id")[TARGET_COLS].sum().astype(np.float64)
y_prob = y_agg.div(y_agg.sum(axis=1), axis=0)

print("Unique train eeg_id:", y_prob.shape[0])
print("Target row-sum stats:", y_prob.sum(axis=1).describe())




## === cell 2
import warnings
import pyarrow as pa
import pyarrow.parquet as pq

_NUMERIC_COLS_CACHE = {}  # key: directory path -> list[str]


def _read_parquet_numeric_to_numpy(path: Path, numeric_names: list[str]) -> np.ndarray:
    """
    Bugfix: pyarrow.Table.to_pandas in this environment doesn't accept copy=.
    Keep conversion semantics the same (to pandas -> numpy -> float32).
    """
    if not numeric_names:
        return np.empty((0, 0), dtype=np.float32)

    table = pq.read_table(str(path), columns=numeric_names, use_threads=True)

    df = table.to_pandas(split_blocks=True, self_destruct=True)
    out = df.to_numpy(copy=False)
    if out.dtype != np.float32:
        out = out.astype(np.float32, copy=False)
    return out


def _get_numeric_columns_for_dir(eeg_dir: str, example_eeg_id: int) -> list[str]:
    key = str(eeg_dir)
    cols = _NUMERIC_COLS_CACHE.get(key)
    if cols is not None:
        return cols

    path = Path(eeg_dir) / f"{int(example_eeg_id)}.parquet"
    if not path.exists():
        _NUMERIC_COLS_CACHE[key] = []
        return []

    schema = pq.read_schema(str(path))
    cols = [
        f.name
        for f in schema
        if pa.types.is_integer(f.type) or pa.types.is_floating(f.type)
    ]
    _NUMERIC_COLS_CACHE[key] = cols
    return cols


def _safe_stats_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if x.size == 0:
        return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(x)
        s = np.nanstd(x)
        q10, q50, q90 = np.nanquantile(x, [0.10, 0.50, 0.90], method="linear")
        mad = np.nanmedian(np.abs(x - q50))
    stats = np.array([m, s, q10, q50, q90, mad], dtype=np.float32)
    stats[~np.isfinite(stats)] = 0.0
    return stats


def _safe_stats_per_channel(arr: np.ndarray) -> np.ndarray:
    """Return (C,6) stats for arr (T,C), matching per-channel _safe_stats_1d."""
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(arr, axis=0)
        s = np.nanstd(arr, axis=0)
        q = np.nanquantile(arr, [0.10, 0.50, 0.90], axis=0, method="linear")
        q10, q50, q90 = q[0], q[1], q[2]
        mad = np.nanmedian(np.abs(arr - q50[None, :]), axis=0)

    stats = np.stack([m, s, q10, q50, q90, mad], axis=1).astype(np.float32, copy=False)
    stats[~np.isfinite(stats)] = 0.0
    return stats


def extract_eeg_features(
    eeg_id: int, eeg_dir: str, numeric_names: list[str]
) -> np.ndarray:
    path = Path(eeg_dir) / f"{int(eeg_id)}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing EEG parquet: {path}")

    arr = _read_parquet_numeric_to_numpy(path, numeric_names)  # (T, C)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return np.zeros(14, dtype=np.float32)

    g = _safe_stats_1d(arr.ravel())

    feats = _safe_stats_per_channel(arr)  # (C, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cmean = np.nanmean(feats, axis=0).astype(np.float32, copy=False)
    cmean[~np.isfinite(cmean)] = 0.0

    nan_frac = np.float32(np.isnan(arr).mean())

    if arr.shape[0] >= 2:
        d = arr[1:, :] - arr[:-1, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            madiff = np.float32(np.nanmean(np.abs(d)))
        if not np.isfinite(madiff):
            madiff = np.float32(0.0)
    else:
        madiff = np.float32(0.0)

    return np.concatenate(
        [g, cmean, np.array([nan_frac, madiff], dtype=np.float32)], axis=0
    )


train_eeg_dir = f"{DATA_PATH}/train_eegs"
test_eeg_dir = f"{DATA_PATH}/test_eegs"

some_id = int(y_prob.index[0])
print("Example eeg_id:", some_id)

train_numeric_cols = _get_numeric_columns_for_dir(train_eeg_dir, some_id)
feat = extract_eeg_features(some_id, train_eeg_dir, train_numeric_cols)
print("Feature dim:", feat.shape, "First 5:", feat[:5])




## === cell 3
from time import time
import multiprocessing as mp

_EEG_DIR = None
_NUMERIC_NAMES = None


def _init_pool(eeg_dir: str, numeric_names: list[str]):
    global _EEG_DIR, _NUMERIC_NAMES
    _EEG_DIR = eeg_dir
    _NUMERIC_NAMES = numeric_names


def _featurize_one(eid: int):
    return extract_eeg_features(int(eid), _EEG_DIR, _NUMERIC_NAMES)


def build_feature_matrix(
    eeg_ids: np.ndarray,
    eeg_dir: str,
    n_features: int = 14,
    log_every: int = 1000,
    numeric_names: list[str] | None = None,
):
    eeg_ids = np.asarray(eeg_ids, dtype=np.int64)
    n = len(eeg_ids)
    X = np.zeros((n, n_features), dtype=np.float32)
    t0 = time()

    if numeric_names is None:
        numeric_names = _get_numeric_columns_for_dir(eeg_dir, int(eeg_ids[0]))

    cpu = os.cpu_count() or 2
    n_proc = max(1, min(4, cpu - 1))

    if n >= 20000:
        chunksize = 1024
    elif n >= 5000:
        chunksize = 512
    else:
        chunksize = 256

    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)

    with ctx.Pool(
        processes=n_proc,
        initializer=_init_pool,
        initargs=(eeg_dir, numeric_names),
    ) as pool:
        for i, feat in enumerate(
            pool.imap(_featurize_one, eeg_ids, chunksize=chunksize), start=0
        ):
            X[i] = feat
            k = i + 1
            if log_every and k % log_every == 0:
                print(f"Features: {k}/{n} ({time()-t0:.1f}s)")
    return X, time() - t0


train_ids = y_prob.index.to_numpy()
test_ids = test_df["eeg_id"].to_numpy()

train_numeric_cols = _get_numeric_columns_for_dir(train_eeg_dir, int(train_ids[0]))
test_numeric_cols = _get_numeric_columns_for_dir(test_eeg_dir, int(test_ids[0]))

X_train, dt = build_feature_matrix(
    train_ids,
    train_eeg_dir,
    n_features=14,
    log_every=1000,
    numeric_names=train_numeric_cols,
)
print("Done train features in", f"{dt:.1f}s")

X_test, dt = build_feature_matrix(
    test_ids,
    test_eeg_dir,
    n_features=14,
    log_every=1000,
    numeric_names=test_numeric_cols,
)
print("Done test features in", f"{dt:.1f}s")

print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)




## === cell 4
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

y_class = y_prob.to_numpy().argmax(axis=1)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=300,
                n_jobs=None,
                C=1.0,
                verbose=0,
                random_state=42,
            ),
        ),
    ]
)

clf.fit(X_train, y_class)

proba = clf.predict_proba(X_test).astype(np.float64)
if proba.shape[1] != len(TARGET_COLS):
    raise RuntimeError(f"Unexpected proba shape {proba.shape}")

eps = 1e-4
proba = np.clip(proba, eps, 1.0)
proba = proba / proba.sum(axis=1, keepdims=True)

print("Proba row-sum stats:", pd.Series(proba.sum(axis=1)).describe())




## === cell 5
sub = sample_sub[["eeg_id"]].copy()
sub[TARGET_COLS] = proba

sub[TARGET_COLS] = sub[TARGET_COLS].clip(lower=0.0)
sub[TARGET_COLS] = sub[TARGET_COLS].div(sub[TARGET_COLS].sum(axis=1), axis=0)

print(sub.head())
print(
    "Row-sum min/max:",
    sub[TARGET_COLS].sum(axis=1).min(),
    sub[TARGET_COLS].sum(axis=1).max(),
)

sub_path = f"{OUT_PATH}/submission.csv"
sub.to_csv(sub_path, index=False)
print("Saved submission to:", sub_path)
print("Submission shape:", sub.shape)
print("Columns:", list(sub.columns))

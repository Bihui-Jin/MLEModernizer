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

# 5. Target score

0.3090887487737234

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
import pyarrow.parquet as pq
import pyarrow.compute as pc


def _safe_stats_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if x.size == 0:
        return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    m = np.nanmean(x)
    s = np.nanstd(x)
    q10 = np.nanpercentile(x, 10)
    q50 = np.nanpercentile(x, 50)
    q90 = np.nanpercentile(x, 90)
    mad = np.nanmedian(np.abs(x - q50))
    if not np.isfinite(m):
        m = 0.0
    if not np.isfinite(s):
        s = 0.0
    if not np.isfinite(q10):
        q10 = 0.0
    if not np.isfinite(q50):
        q50 = 0.0
    if not np.isfinite(q90):
        q90 = 0.0
    if not np.isfinite(mad):
        mad = 0.0
    return np.array([m, s, q10, q50, q90, mad], dtype=np.float32)


def _safe_stats_per_channel(arr: np.ndarray) -> np.ndarray:
    """Return (C,6) stats for arr (T,C), matching per-channel _safe_stats_1d."""
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(arr, axis=0)
        s = np.nanstd(arr, axis=0)
        q10 = np.nanpercentile(arr, 10, axis=0)
        q50 = np.nanpercentile(arr, 50, axis=0)
        q90 = np.nanpercentile(arr, 90, axis=0)
        mad = np.nanmedian(np.abs(arr - q50[None, :]), axis=0)

    stats = np.stack([m, s, q10, q50, q90, mad], axis=1).astype(np.float32, copy=False)
    stats[~np.isfinite(stats)] = 0.0
    return stats


def _read_parquet_numeric_to_numpy(path: Path) -> np.ndarray:
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow

    numeric_names = []
    for i in range(schema.num_fields):
        t = schema.field(i).type
        if t in (  # common fast paths
            getattr(__import__("pyarrow"), "int8")(),
            getattr(__import__("pyarrow"), "int16")(),
            getattr(__import__("pyarrow"), "int32")(),
            getattr(__import__("pyarrow"), "int64")(),
            getattr(__import__("pyarrow"), "uint8")(),
            getattr(__import__("pyarrow"), "uint16")(),
            getattr(__import__("pyarrow"), "uint32")(),
            getattr(__import__("pyarrow"), "uint64")(),
            getattr(__import__("pyarrow"), "float16")(),
            getattr(__import__("pyarrow"), "float32")(),
            getattr(__import__("pyarrow"), "float64")(),
        ):
            numeric_names.append(schema.names[i])
        else:
            if schema.field(i).type.__class__.__name__ in (
                "Int8Type",
                "Int16Type",
                "Int32Type",
                "Int64Type",
                "UInt8Type",
                "UInt16Type",
                "UInt32Type",
                "UInt64Type",
                "FloatType",
                "DoubleType",
                "HalfFloatType",
            ):
                numeric_names.append(schema.names[i])

    if not numeric_names:
        return np.empty((0, 0), dtype=np.float32)

    table = pf.read(columns=numeric_names, use_threads=True)

    cols = []
    for name in numeric_names:
        col = table[name]
        if not str(col.type).startswith("float"):
            col = pc.cast(col, target_type=__import__("pyarrow").float32())
        else:
            if str(col.type) != "float32":
                col = pc.cast(col, target_type=__import__("pyarrow").float32())
        cols.append(col)

    arrays = [c.combine_chunks().to_numpy(zero_copy_only=False) for c in cols]
    arr = np.stack(arrays, axis=1).astype(np.float32, copy=False)
    return arr


def extract_eeg_features(eeg_id: int, eeg_dir: str) -> np.ndarray:
    path = Path(eeg_dir) / f"{int(eeg_id)}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing EEG parquet: {path}")

    arr = _read_parquet_numeric_to_numpy(path)  # (T, C)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return np.zeros(14, dtype=np.float32)

    flat = arr.reshape(-1)
    g = _safe_stats_1d(flat)

    feats = _safe_stats_per_channel(arr)  # (C, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cmean = np.nanmean(feats, axis=0).astype(np.float32, copy=False)
    cmean[~np.isfinite(cmean)] = 0.0

    nan_frac = np.float32(np.isnan(arr).mean())
    diff = np.diff(arr, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        madiff = np.float32(np.nanmean(np.abs(diff))) if diff.size else np.float32(0.0)
    if not np.isfinite(madiff):
        madiff = np.float32(0.0)

    return np.concatenate(
        [g, cmean, np.array([nan_frac, madiff], dtype=np.float32)], axis=0
    )


train_eeg_dir = f"{DATA_PATH}/train_eegs"
test_eeg_dir = f"{DATA_PATH}/test_eegs"

some_id = int(y_prob.index[0])
print("Example eeg_id:", some_id)
feat = extract_eeg_features(some_id, train_eeg_dir)
print("Feature dim:", feat.shape, "First 5:", feat[:5])



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3840295560.py in <cell line: 0>()
    153 some_id = int(y_prob.index[0])
    154 print("Example eeg_id:", some_id)
--> 155 feat = extract_eeg_features(some_id, train_eeg_dir)
    156 print("Feature dim:", feat.shape, "First 5:", feat[:5])
    157 

/tmp/ipykernel_11/3840295560.py in extract_eeg_features(eeg_id, eeg_dir)
    122         raise FileNotFoundError(f"Missing EEG parquet: {path}")
    123 
--> 124     arr = _read_parquet_numeric_to_numpy(path)  # (T, C)
    125     if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
    126         return np.zeros(14, dtype=np.float32)

/tmp/ipykernel_11/3840295560.py in _read_parquet_numeric_to_numpy(path)
     57     # Select only numeric columns (float/int); ignore non-numeric.
     58     numeric_names = []
---> 59     for i in range(schema.num_fields):
     60         t = schema.field(i).type
     61         if t in (  # common fast paths

AttributeError: 'pyarrow.lib.Schema' object has no attribute 'num_fields'

## === cell 3
from time import time
import multiprocessing as mp


_EEG_DIR = None


def _init_pool(eeg_dir: str):
    global _EEG_DIR
    _EEG_DIR = eeg_dir


def _featurize_one(eid: int):
    return extract_eeg_features(int(eid), _EEG_DIR)


def build_feature_matrix(
    eeg_ids: np.ndarray, eeg_dir: str, n_features: int = 14, log_every: int = 1000
):
    eeg_ids = np.asarray(eeg_ids)
    n = len(eeg_ids)
    X = np.zeros((n, n_features), dtype=np.float32)
    t0 = time()

    cpu = os.cpu_count() or 2
    n_proc = max(1, min(8, cpu - 1))

    indexed_ids = list(enumerate(eeg_ids.tolist()))

    chunksize = 64

    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=n_proc,
        initializer=_init_pool,
        initargs=(eeg_dir,),
        maxtasksperchild=1000,  # reduce process respawn overhead vs 200 while keeping memory bounded
    ) as pool:
        for k, (i, feat) in enumerate(
            pool.imap_unordered(
                lambda p: (p[0], _featurize_one(p[1])),
                indexed_ids,
                chunksize=chunksize,
            ),
            start=1,
        ):
            X[i] = feat
            if log_every and k % log_every == 0:
                print(f"Features: {k}/{n} ({time()-t0:.1f}s)")
    return X, time() - t0


train_ids = y_prob.index.to_numpy()
test_ids = test_df["eeg_id"].to_numpy()

X_train, dt = build_feature_matrix(
    train_ids, train_eeg_dir, n_features=14, log_every=1000
)
print("Done train features in", f"{dt:.1f}s")

X_test, dt = build_feature_matrix(test_ids, test_eeg_dir, n_features=14, log_every=1000)
print("Done test features in", f"{dt:.1f}s")

print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3130546112.py in <cell line: 0>()
     59 test_ids = test_df["eeg_id"].to_numpy()
     60 
---> 61 X_train, dt = build_feature_matrix(
     62     train_ids, train_eeg_dir, n_features=14, log_every=1000
     63 )

/tmp/ipykernel_11/3130546112.py in build_feature_matrix(eeg_ids, eeg_dir, n_features, log_every)
     42         maxtasksperchild=1000,  # reduce process respawn overhead vs 200 while keeping memory bounded
     43     ) as pool:
---> 44         for k, (i, feat) in enumerate(
     45             pool.imap_unordered(
     46                 lambda p: (p[0], _featurize_one(p[1])),

/usr/lib/python3.11/multiprocessing/pool.py in <genexpr>(.0)
    449                     result._set_length
    450                 ))
--> 451             return (item for chunk in result for item in chunk)
    452 
    453     def apply_async(self, func, args=(), kwds={}, callback=None,

/usr/lib/python3.11/multiprocessing/pool.py in next(self, timeout)
    871         if success:
    872             return value
--> 873         raise value
    874 
    875     __next__ = next                    # XXX

/usr/lib/python3.11/multiprocessing/pool.py in _handle_tasks(taskqueue, put, outqueue, pool, cache)
    538                         break
    539                     try:
--> 540                         put(task)
    541                     except Exception as e:
    542                         job, idx = task[:2]

/usr/lib/python3.11/multiprocessing/connection.py in send(self, obj)
    204         self._check_closed()
    205         self._check_writable()
--> 206         self._send_bytes(_ForkingPickler.dumps(obj))
    207 
    208     def recv_bytes(self, maxlength=None):

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object 'build_feature_matrix.<locals>.<lambda>'

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



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3244745273.py in <cell line: 0>()
     23 )
     24 
---> 25 clf.fit(X_train, y_class)
     26 
     27 proba = clf.predict_proba(X_test).astype(np.float64)

NameError: name 'X_train' is not defined

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

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1794533711.py in <cell line: 0>()
      1 sub = sample_sub[["eeg_id"]].copy()
----> 2 sub[TARGET_COLS] = proba
      3 
      4 sub[TARGET_COLS] = sub[TARGET_COLS].clip(lower=0.0)
      5 sub[TARGET_COLS] = sub[TARGET_COLS].div(sub[TARGET_COLS].sum(axis=1), axis=0)

NameError: name 'proba' is not defined

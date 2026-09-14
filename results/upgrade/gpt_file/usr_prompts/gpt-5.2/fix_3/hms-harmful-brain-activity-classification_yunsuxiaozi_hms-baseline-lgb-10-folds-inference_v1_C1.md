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

dill==0.4.0
geopandas==0.14.4
lightgbm==4.6.0
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

0.8327273980133865

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import warnings
import random
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")



## === cell 1
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(TARGETS)

seed = 2024
num_folds = 10

np.random.seed(seed)
random.seed(seed)

DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TEST_CSV = f"{DATA_ROOT}/test.csv"
TRAIN_SPEC_DIR = f"{DATA_ROOT}/train_spectrograms"
TEST_SPEC_DIR = f"{DATA_ROOT}/test_spectrograms"



## === cell 2
import pyarrow.parquet as pq

sample_spec_path = f"{TRAIN_SPEC_DIR}/1000086677.parquet"
schema_names = pq.read_schema(sample_spec_path).names
SPEC_COLS = pd.Index(
    schema_names[1:]
)  # drop time column by position (same as original logic)

FEATURES = [f"{c}_mean_10m" for c in SPEC_COLS]
FEATURES += [f"{c}_min_10m" for c in SPEC_COLS]
FEATURES += [f"{c}_mean_20s" for c in SPEC_COLS]
FEATURES += [f"{c}_min_20s" for c in SPEC_COLS]

assert len(FEATURES) == 4 * len(SPEC_COLS)



## === cell 3
from concurrent.futures import ThreadPoolExecutor

_SPEC_COLS_LIST = [str(c) for c in SPEC_COLS]


def make_spec_features_from_parquet(spec_path: str, r: int = 10) -> np.ndarray:
    """
    Replicates the original feature extraction logic:
    - mean over 10-minute-ish window: rows r:r+300
    - min over same
    - mean over central 20s-ish window: rows r+145:r+155
    - min over central window: same
    """
    table = pq.read_table(spec_path, columns=_SPEC_COLS_LIST)
    X = table.to_numpy(
        zero_copy_only=False
    )  # ensures a writable/contiguous ndarray if needed

    n = X.shape[0]
    r0 = max(0, min(r, n - 1))
    a = r0
    b = min(r0 + 300, n)

    c = min(r0 + 145, n)
    d = min(r0 + 155, n)
    if d <= c:
        c = max(0, n - 10)
        d = n

    f1 = np.nanmean(X[a:b, :], axis=0)
    f2 = np.nanmin(X[a:b, :], axis=0)
    f3 = np.nanmean(X[c:d, :], axis=0)
    f4 = np.nanmin(X[c:d, :], axis=0)

    return np.concatenate([f1, f2, f3, f4], axis=0).astype(np.float32)


def safe_softmax(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def _build_features_matrix(
    spec_ids: np.ndarray, spec_dir: str, r: int = 10, max_workers: int | None = None
) -> np.ndarray:
    spec_ids = np.asarray(spec_ids, dtype=np.int64)
    out = np.zeros((len(spec_ids), len(FEATURES)), dtype=np.float32)

    def _one(i_s):
        i, s = i_s
        path = f"{spec_dir}/{int(s)}.parquet"
        return i, make_spec_features_from_parquet(path, r=r)

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in ex.map(_one, enumerate(spec_ids), chunksize=32):
            out[i] = feats
    return out




## === cell 4
test = pd.read_csv(TEST_CSV)
test_spec_ids = test["spectrogram_id"].to_numpy(dtype=np.int64)

data_test = _build_features_matrix(test_spec_ids, TEST_SPEC_DIR, r=10)
test_features = pd.DataFrame(data_test, columns=FEATURES)
test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

test = pd.concat([test, test_features], axis=1)
print("test shape", test.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3102902124.py in <cell line: 0>()
      3 test_spec_ids = test["spectrogram_id"].to_numpy(dtype=np.int64)
      4 
----> 5 data_test = _build_features_matrix(test_spec_ids, TEST_SPEC_DIR, r=10)
      6 test_features = pd.DataFrame(data_test, columns=FEATURES)
      7 test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

/tmp/ipykernel_11/806851465.py in _build_features_matrix(spec_ids, spec_dir, r, max_workers)
     61         max_workers = min(32, (os.cpu_count() or 4))
     62     with ThreadPoolExecutor(max_workers=max_workers) as ex:
---> 63         for i, feats in ex.map(_one, enumerate(spec_ids), chunksize=32):
     64             out[i] = feats
     65     return out

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

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

/tmp/ipykernel_11/806851465.py in _one(i_s)
     55         i, s = i_s
     56         path = f"{spec_dir}/{int(s)}.parquet"
---> 57         return i, make_spec_features_from_parquet(path, r=r)
     58 
     59     # Use threads: parquet reading is I/O heavy and pyarrow releases the GIL in many ops.

/tmp/ipykernel_11/806851465.py in make_spec_features_from_parquet(spec_path, r)
     16     # Read only required columns; avoids loading the time column and any metadata overhead.
     17     table = pq.read_table(spec_path, columns=_SPEC_COLS_LIST)
---> 18     X = table.to_numpy(
     19         zero_copy_only=False
     20     )  # ensures a writable/contiguous ndarray if needed

AttributeError: 'pyarrow.lib.Table' object has no attribute 'to_numpy'

## === cell 5
train = pd.read_csv(TRAIN_CSV)

votes = train[TARGETS].values.astype(np.float64)
votes_sum = votes.sum(axis=1, keepdims=True)
votes_sum[votes_sum == 0] = 1.0
y_prob = votes / votes_sum
y_class = np.argmax(y_prob, axis=1).astype(int)

train = train.copy()
train["y_class"] = y_class
train = (
    train.sort_values(["spectrogram_id", "eeg_id", "eeg_sub_id"])
    .drop_duplicates("spectrogram_id")
    .reset_index(drop=True)
)
print("deduped train rows:", len(train))



## === cell 6
train_spec_ids = train["spectrogram_id"].to_numpy(dtype=np.int64)
data_train = _build_features_matrix(train_spec_ids, TRAIN_SPEC_DIR, r=10)

X_train = pd.DataFrame(data_train, columns=FEATURES)
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
y_train = train["y_class"].to_numpy()

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2283455264.py in <cell line: 0>()
      1 # Speed fix: parallelize train feature extraction like test; keep identical features.
      2 train_spec_ids = train["spectrogram_id"].to_numpy(dtype=np.int64)
----> 3 data_train = _build_features_matrix(train_spec_ids, TRAIN_SPEC_DIR, r=10)
      4 
      5 X_train = pd.DataFrame(data_train, columns=FEATURES)

/tmp/ipykernel_11/806851465.py in _build_features_matrix(spec_ids, spec_dir, r, max_workers)
     61         max_workers = min(32, (os.cpu_count() or 4))
     62     with ThreadPoolExecutor(max_workers=max_workers) as ex:
---> 63         for i, feats in ex.map(_one, enumerate(spec_ids), chunksize=32):
     64             out[i] = feats
     65     return out

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

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

/tmp/ipykernel_11/806851465.py in _one(i_s)
     55         i, s = i_s
     56         path = f"{spec_dir}/{int(s)}.parquet"
---> 57         return i, make_spec_features_from_parquet(path, r=r)
     58 
     59     # Use threads: parquet reading is I/O heavy and pyarrow releases the GIL in many ops.

/tmp/ipykernel_11/806851465.py in make_spec_features_from_parquet(spec_path, r)
     16     # Read only required columns; avoids loading the time column and any metadata overhead.
     17     table = pq.read_table(spec_path, columns=_SPEC_COLS_LIST)
---> 18     X = table.to_numpy(
     19         zero_copy_only=False
     20     )  # ensures a writable/contiguous ndarray if needed

AttributeError: 'pyarrow.lib.Table' object has no attribute 'to_numpy'

## === cell 7
def make_stratified_folds(y: np.ndarray, n_splits: int, seed: int = 2024):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    folds = np.full(len(y), -1, dtype=int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[j] = i % n_splits
    return folds


fold_ids = make_stratified_folds(y_train, num_folds, seed=seed)

X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
X_test_np = test[FEATURES].to_numpy(dtype=np.float32, copy=False)

preds_test_folds = []
for fold in range(num_folds):
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]  # kept for semantic parity (even if unused)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed + fold,
        n_jobs=-1,
    )

    model.fit(
        X_train_np[tr_idx],
        y_train[tr_idx],
    )

    pred_test = model.predict_proba(X_test_np)
    preds_test_folds.append(pred_test)

pred = np.mean(preds_test_folds, axis=0)
pred = safe_softmax(pred)
print("Test preds shape", pred.shape, "row-sum mean", pred.sum(axis=1).mean())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2334121285.py in <cell line: 0>()
     11 
     12 
---> 13 fold_ids = make_stratified_folds(y_train, num_folds, seed=seed)
     14 
     15 # Speed fix: use numpy arrays for indexing and reuse the already-built test feature matrix for prediction.

NameError: name 'y_train' is not defined

## === cell 8
submission = pd.DataFrame({"eeg_id": test.eeg_id.values})
submission[TARGETS] = pred.astype(np.float64)

submission[TARGETS] = submission[TARGETS].clip(1e-15, 1.0)
submission[TARGETS] = submission[TARGETS].div(submission[TARGETS].sum(axis=1), axis=0)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Saved submission.csv with shape:", submission.shape)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2460232373.py in <cell line: 0>()
      1 submission = pd.DataFrame({"eeg_id": test.eeg_id.values})
----> 2 submission[TARGETS] = pred.astype(np.float64)
      3 
      4 submission[TARGETS] = submission[TARGETS].clip(1e-15, 1.0)
      5 submission[TARGETS] = submission[TARGETS].div(submission[TARGETS].sum(axis=1), axis=0)

NameError: name 'pred' is not defined

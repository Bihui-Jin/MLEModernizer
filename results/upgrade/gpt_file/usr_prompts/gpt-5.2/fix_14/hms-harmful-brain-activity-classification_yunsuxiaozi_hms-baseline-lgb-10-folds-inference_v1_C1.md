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

1.48657

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.49395) has done: 'The timeout is dominated by repeated parquet decoding and Python-level per-column conversion inside `_table_to_2d_float_fast`, plus some avoidable overhead in parallel feature building. I keep the exact same features and LightGBM training loop, but make parquet reads cheaper by converting the entire Arrow table to a single NumPy array in one shot (instead of per-column loops) and by using Arrow’s memory mapping consistently. I also make feature extraction more cache-friendly and reduce thread scheduling overhead by tuning chunk sizes and avoiding unnecessary intermediate allocations, while keeping determinism and identical feature semantics (nanmean/nanmin on the same row windows). The model, targets, folds, and prediction averaging remain unchanged.'
- What this solution (achieved 1.48657) has done: 'Your current pipeline trains on hard argmax classes, but the competition metric is KL divergence against soft vote distributions, so it’s leaving a lot of score on the table. I keep the same feature extraction, folds, LightGBM model family, and averaging, but switch the training target to the normalized vote probabilities and train one LightGBM model per class with an objective aligned to probabilistic targets. Then I renormalize the 6 outputs to ensure each row sums to 1 (as required) and write the same `submission.csv`. This is a minimal semantic change focused directly on reducing KL by predicting distributions rather than classes.'

# 9. Code solution

## === cell 0
import os
import warnings
import random
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore")

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

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/hms-harmful-brain-activity-classification",
    "/kaggle/data/hms-harmful-brain-activity-classification",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = DATA_ROOT_CANDIDATES[0]

TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TEST_CSV = f"{DATA_ROOT}/test.csv"
TRAIN_SPEC_DIR = f"{DATA_ROOT}/train_spectrograms"
TEST_SPEC_DIR = f"{DATA_ROOT}/test_spectrograms"

for fp in [TRAIN_CSV, TEST_CSV]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing required file: {fp}")



## === cell 1
import pyarrow.parquet as pq

sample_spec_path = f"{TRAIN_SPEC_DIR}/1000086677.parquet"
if not os.path.exists(sample_spec_path):
    import glob

    cand = glob.glob(os.path.join(TRAIN_SPEC_DIR, "*.parquet"))
    if len(cand) == 0:
        raise FileNotFoundError(
            f"No spectrogram parquet files found in {TRAIN_SPEC_DIR}"
        )
    sample_spec_path = cand[0]

schema_names = pq.read_schema(sample_spec_path).names
SPEC_COLS = pd.Index(schema_names[1:])  # drop time column by position

FEATURES = [f"{c}_mean_10m" for c in SPEC_COLS]
FEATURES += [f"{c}_min_10m" for c in SPEC_COLS]
FEATURES += [f"{c}_mean_20s" for c in SPEC_COLS]
FEATURES += [f"{c}_min_20s" for c in SPEC_COLS]

assert len(FEATURES) == 4 * len(SPEC_COLS)



## === cell 2
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa

_SPEC_COLS_LIST = [str(c) for c in SPEC_COLS]

_SPEC_FEAT_CACHE_BY_KEY: dict[tuple[str, int], np.ndarray] = {}

_N_SPEC = len(_SPEC_COLS_LIST)
_FEAT_DIM = 4 * _N_SPEC


def _table_to_2d_float_fast(table: pa.Table) -> np.ndarray:
    table = table.combine_chunks()
    n_rows = table.num_rows
    n_cols = table.num_columns
    if n_rows == 0 or n_cols == 0:
        return np.empty((0, n_cols), dtype=np.float32)

    X = table.to_numpy(zero_copy_only=False)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X


def _compute_feats_numpy_inplace(X: np.ndarray, r: int, out: np.ndarray) -> None:
    n = X.shape[0]
    if n == 0:
        out[:] = 0.0
        return

    r0 = r
    if r0 < 0:
        r0 = 0
    elif r0 >= n:
        r0 = n - 1

    a = r0
    b = r0 + 300
    if b > n:
        b = n

    c = r0 + 145
    if c > n:
        c = n
    d = r0 + 155
    if d > n:
        d = n
    if d <= c:
        c = n - 10
        if c < 0:
            c = 0
        d = n

    X1 = X[a:b, :]
    X2 = X[c:d, :]

    out[0:_N_SPEC] = np.nanmean(X1, axis=0)
    out[_N_SPEC : 2 * _N_SPEC] = np.nanmin(X1, axis=0)
    out[2 * _N_SPEC : 3 * _N_SPEC] = np.nanmean(X2, axis=0)
    out[3 * _N_SPEC : 4 * _N_SPEC] = np.nanmin(X2, axis=0)


def make_spec_features_from_parquet(
    spec_id: int, spec_dir: str, r: int = 10
) -> np.ndarray:
    key = (spec_dir, int(spec_id))
    cached = _SPEC_FEAT_CACHE_BY_KEY.get(key)
    if cached is not None:
        return cached

    spec_path = f"{spec_dir}/{spec_id}.parquet"
    z = np.zeros((_FEAT_DIM,), dtype=np.float32)

    if not os.path.exists(spec_path):
        _SPEC_FEAT_CACHE_BY_KEY[key] = z
        return z

    try:
        table = pq.read_table(spec_path, columns=_SPEC_COLS_LIST, memory_map=True)
        if table.num_rows == 0:
            _SPEC_FEAT_CACHE_BY_KEY[key] = z
            return z
        X = _table_to_2d_float_fast(table)
        out = np.empty((_FEAT_DIM,), dtype=np.float32)
        _compute_feats_numpy_inplace(X, r=r, out=out)
    except Exception:
        out = z

    _SPEC_FEAT_CACHE_BY_KEY[key] = out
    return out


def safe_softmax(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def _build_features_matrix(
    spec_ids: np.ndarray, spec_dir: str, r: int = 10, max_workers: int | None = None
) -> np.ndarray:
    spec_ids = np.asarray(spec_ids, dtype=np.int64)
    n = spec_ids.size
    out = np.zeros((n, _FEAT_DIM), dtype=np.float32)
    if n == 0:
        return out

    uniq_ids, inv = np.unique(spec_ids, return_inverse=True)
    m = uniq_ids.size
    uniq_feats = np.empty((m, _FEAT_DIM), dtype=np.float32)

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(16, cpu)

    chunksize = 1024 if m >= 4096 else 256

    def _feat_for_sid(sid: int) -> np.ndarray:
        return make_spec_features_from_parquet(int(sid), spec_dir=spec_dir, r=r)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for j, feats in enumerate(ex.map(_feat_for_sid, uniq_ids, chunksize=chunksize)):
            uniq_feats[j] = feats

    out[:] = uniq_feats[inv]
    return out




## === cell 3
test = pd.read_csv(TEST_CSV)
test_spec_ids = test["spectrogram_id"].to_numpy(dtype=np.int64)

X_test_np = _build_features_matrix(test_spec_ids, TEST_SPEC_DIR, r=10)
X_test_np = X_test_np.astype(np.float32, copy=False)
np.nan_to_num(X_test_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

print("test shape", (len(test), test.shape[1] + _FEAT_DIM))



## === cell 4
train = pd.read_csv(TRAIN_CSV)

votes = train[TARGETS].values.astype(np.float64)
votes_sum = votes.sum(axis=1, keepdims=True)
votes_sum[votes_sum == 0] = 1.0
y_prob_full = votes / votes_sum
y_class = np.argmax(y_prob_full, axis=1).astype(int)

train = train.copy()
train["y_class"] = y_class
train = (
    train.sort_values(["spectrogram_id", "eeg_id", "eeg_sub_id"])
    .drop_duplicates("spectrogram_id")
    .reset_index(drop=True)
)
print("deduped train rows:", len(train))

votes_d = train[TARGETS].values.astype(np.float64)
votes_d_sum = votes_d.sum(axis=1, keepdims=True)
votes_d_sum[votes_d_sum == 0] = 1.0
y_train_prob = votes_d / votes_d_sum



## === cell 5
train_spec_ids = train["spectrogram_id"].to_numpy(dtype=np.int64)

X_train_np = _build_features_matrix(train_spec_ids, TRAIN_SPEC_DIR, r=10)
X_train_np = X_train_np.astype(np.float32, copy=False)
np.nan_to_num(X_train_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

y_train_class = train["y_class"].to_numpy()

print(
    "X_train shape:",
    X_train_np.shape,
    "y_train_class shape:",
    y_train_class.shape,
    "y_train_prob shape:",
    y_train_prob.shape,
)




## === cell 6
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


fold_ids = make_stratified_folds(y_train_class, num_folds, seed=seed)

fold_tr_indices = [np.where(fold_ids != fold)[0] for fold in range(num_folds)]
fold_va_indices = [
    np.where(fold_ids == fold)[0] for fold in range(num_folds)
]  # semantic parity

pred_sum = np.zeros((X_test_np.shape[0], N_CLASSES), dtype=np.float64)

_LGB_NJOBS = min(8, (os.cpu_count() or 4))

for fold in range(num_folds):
    tr_idx = fold_tr_indices[fold]
    va_idx = fold_va_indices[fold]  # kept for semantic parity (even if unused)

    fold_pred = np.zeros((X_test_np.shape[0], N_CLASSES), dtype=np.float64)

    for k in range(N_CLASSES):
        model = LGBMRegressor(
            objective="regression",
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed + 1000 * fold + k,
            n_jobs=_LGB_NJOBS,
            verbose=-1,
        )
        model.fit(X_train_np[tr_idx], y_train_prob[tr_idx, k])
        fold_pred[:, k] = model.predict(X_test_np)

    pred_sum += fold_pred

pred = pred_sum / float(num_folds)

pred = np.clip(pred, 1e-15, None)
pred = pred / pred.sum(axis=1, keepdims=True)

print("Test preds shape", pred.shape, "row-sum mean", pred.sum(axis=1).mean())



## === cell 7
submission = pd.DataFrame({"eeg_id": test.eeg_id.values})
submission[TARGETS] = pred.astype(np.float64)

submission[TARGETS] = submission[TARGETS].clip(1e-15, 1.0)
submission[TARGETS] = submission[TARGETS].div(submission[TARGETS].sum(axis=1), axis=0)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Saved submission.csv with shape:", submission.shape)
print(
    "Row-sum stats:",
    submission[TARGETS].sum(axis=1).min(),
    submission[TARGETS].sum(axis=1).max(),
)

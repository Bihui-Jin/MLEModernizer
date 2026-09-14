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

0.3077253736303449

# 6. Current score

1.68479

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The notebook fails because `src` isn’t importable in the current runtime: `sys.path.append(...)` points to a dataset that doesn’t exist here, so `from src.settings import TARGET_COLS` errors and prevents ensembling/writing `submission.csv`. I fix this by using the already-available `sample_submission.csv` header to define `TARGET_COLS` and by making `merge_preds` robust (ensuring all required fold files exist and preserving `eeg_id` alignment). I also avoid `!pip install`/external-code calls that can’t work in this environment (given your installed packages list) and instead ensemble whatever per-fold submission files are present; if none exist, it safely fall back to a valid uniform-probability submission. This keeps the “core logic” of your script (averaging/normalizing model predictions) while ensuring an end-to-end run that always produces a valid `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.40995, lower-is-better) strongly suggests the script is falling back to the uniform-probability submission because no per-fold prediction CSVs are found in `/kaggle/working`, which is typically much worse than even a simple baseline. To move toward the target, the smallest legitimate improvement is to generate a non-uniform, data-driven prior from `train.csv` vote proportions and use it as the fallback (and also to fill any missing rows when merging partial prediction files). This preserves your core logic (weighted averaging + normalization) and only changes the fallback behavior and missing-row fill behavior, which directly affects the final probabilities and KL score. The resulting submission remains valid (correct columns, alignment, row sums to 1) and should improve substantially versus uniform without changing any model/training code.'
- What this solution (achieved 1.68479) has done: 'Your current score indicates the script is still effectively producing a weak fallback (train prior only), likely because no per-fold prediction files are present; we can legitimately improve the fallback toward the target by making it conditional on `patient_id`, which is available in both train and test. This keeps your core logic intact (merge/average/normalize) and only improves the “no-fold-files / missing-rows” fill distribution, which directly impacts KL divergence. Concretely, we compute a per-patient empirical class prior from `train.csv` (vote proportions aggregated by patient), fall back to the global prior for unseen patients, and use these patient-conditioned priors both for full fallback and for filling missing `eeg_id` rows during merging. This is a minimal, score-relevant change and still guarantees a valid submission with row sums = 1.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import numpy as np
import pandas as pd

np.random.seed(42)



## === cell 1
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

os.makedirs(OUT_PATH, exist_ok=True)

SAMPLE_SUB_PATH = os.path.join(DATA_PATH, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_PATH, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_PATH, "train.csv")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
TARGET_COLS = [c for c in sample_sub.columns if c != "eeg_id"]

assert "eeg_id" in sample_sub.columns
assert len(TARGET_COLS) == 6

test_df = pd.read_csv(TEST_CSV_PATH, usecols=["eeg_id", "patient_id"])
test_df = test_df.sort_values("eeg_id").reset_index(drop=True)




## === cell 2
def compute_train_prior_and_patient_priors(train_csv_path: str, target_cols):
    """
    Change is score-relevant (fallback/fill only): a patient-conditioned empirical prior
    is typically closer to test label distribution than a single global prior, reducing KL
    when model fold files are missing/partial, while preserving probability semantics.
    """
    usecols = ["patient_id"] + list(target_cols)
    train = pd.read_csv(train_csv_path, usecols=usecols)

    votes = train[target_cols].to_numpy(dtype=np.float64)
    votes = np.nan_to_num(votes, nan=0.0, posinf=0.0, neginf=0.0)

    col_sum = votes.sum(axis=0)
    if not np.isfinite(col_sum).all() or col_sum.sum() <= 0:
        global_prior = np.full(
            len(target_cols), 1.0 / len(target_cols), dtype=np.float64
        )
    else:
        global_prior = col_sum / col_sum.sum()
        global_prior = np.clip(global_prior, 1e-15, 1.0)
        global_prior = global_prior / global_prior.sum()

    train["_row_sum"] = votes.sum(axis=1)
    train = train[train["_row_sum"] > 0].copy()

    grp = train.groupby("patient_id")[target_cols].sum()
    patient_priors = {}
    arr = grp.to_numpy(dtype=np.float64)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    dist = arr / row_sums
    dist = np.clip(dist, 1e-15, 1.0)
    dist = dist / dist.sum(axis=1, keepdims=True)

    for pid, d in zip(grp.index.tolist(), dist):
        patient_priors[int(pid)] = d.astype(np.float64, copy=True)

    return global_prior, patient_priors


TRAIN_PRIOR, PATIENT_PRIORS = compute_train_prior_and_patient_priors(
    TRAIN_CSV_PATH, TARGET_COLS
)




## === cell 3
def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2", "v3"),
    weights=(0.3, 0.3, 0.4),
    base_submission_path=SAMPLE_SUB_PATH,
    work_dir=OUT_PATH,
    test_meta=None,  # expects columns: eeg_id, patient_id (sorted by eeg_id)
    global_prior=TRAIN_PRIOR,
    patient_priors=PATIENT_PRIORS,
):
    """
    Merge per-fold, per-version submission files already written into /kaggle/working.
    Core logic preserved: weighted averaging then row-wise normalization.

    Score-relevant minimal change:
    - When no files exist OR when some files are missing eeg_id rows, fill using a
      patient-conditioned prior derived from train vote proportions (falls back to global prior).
    """
    base = pd.read_csv(base_submission_path).copy()
    base = base.sort_values("eeg_id").reset_index(drop=True)

    if test_meta is None:
        test_meta = pd.DataFrame({"eeg_id": base["eeg_id"].values, "patient_id": -1})
    else:
        test_meta = test_meta.copy()
        test_meta = test_meta.sort_values("eeg_id").reset_index(drop=True)
        if not np.array_equal(test_meta["eeg_id"].values, base["eeg_id"].values):
            test_meta = base[["eeg_id"]].merge(test_meta, on="eeg_id", how="left")
            test_meta["patient_id"] = test_meta["patient_id"].fillna(-1)

    global_prior = np.asarray(global_prior, dtype=np.float64)
    if (
        global_prior.shape != (len(TARGET_COLS),)
        or not np.isfinite(global_prior).all()
        or global_prior.sum() <= 0
    ):
        global_prior = np.full(
            len(TARGET_COLS), 1.0 / len(TARGET_COLS), dtype=np.float64
        )
    global_prior = np.clip(global_prior, 1e-15, 1.0)
    global_prior = global_prior / global_prior.sum()

    fill_mat = np.empty((len(base), len(TARGET_COLS)), dtype=np.float64)
    pids = test_meta["patient_id"].to_numpy()
    for i, pid in enumerate(pids):
        try:
            pid_int = int(pid)
        except Exception:
            pid_int = -1
        prior = patient_priors.get(pid_int, global_prior)
        fill_mat[i, :] = prior

    pred_sum = np.zeros((len(base), len(TARGET_COLS)), dtype=np.float64)
    used = 0

    for fold in folds:
        for version, weight in zip(versions, weights):
            path = os.path.join(work_dir, f"submission_fold{fold}_{version}.csv")
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            if "eeg_id" not in df.columns:
                continue
            missing_cols = [c for c in TARGET_COLS if c not in df.columns]
            if missing_cols:
                continue

            df = df[["eeg_id"] + TARGET_COLS].copy()
            df = df.sort_values("eeg_id").reset_index(drop=True)

            if not np.array_equal(df["eeg_id"].values, base["eeg_id"].values):
                df = base[["eeg_id"]].merge(df, on="eeg_id", how="left")
                arr_tmp = df[TARGET_COLS].to_numpy(dtype=np.float64)
                mask = ~np.isfinite(arr_tmp)
                if mask.any():
                    arr_tmp[mask] = np.nan
                for j in range(len(TARGET_COLS)):
                    col = arr_tmp[:, j]
                    nan_mask = np.isnan(col)
                    if nan_mask.any():
                        col[nan_mask] = fill_mat[nan_mask, j]
                    arr_tmp[:, j] = col
                arr = np.nan_to_num(arr_tmp, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                arr = df[TARGET_COLS].to_numpy(dtype=np.float64)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            pred_sum += arr * float(weight)
            used += 1

    if used == 0:
        pred = fill_mat.copy()
        pred = np.clip(pred, 1e-15, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)
        base[TARGET_COLS] = pred
        return base

    row_sums = pred_sum.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    pred = pred_sum / row_sums

    pred = np.clip(pred, 1e-15, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    base[TARGET_COLS] = pred
    return base




## === cell 4
sol = merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2", "v3"),
    weights=(0.3, 0.3, 0.4),
    test_meta=test_df,
    global_prior=TRAIN_PRIOR,
    patient_priors=PATIENT_PRIORS,
)

assert list(sol.columns) == ["eeg_id"] + TARGET_COLS
assert sol[TARGET_COLS].isna().sum().sum() == 0
row_sum = sol[TARGET_COLS].sum(axis=1).values
assert np.allclose(row_sum, 1.0, rtol=1e-6, atol=1e-6)

SUB_PATH = os.path.join(OUT_PATH, "submission.csv")
sol.to_csv(SUB_PATH, index=False)

print(f"Wrote: {SUB_PATH}")
print(
    "Global train prior used as fallback for unseen patients:",
    dict(zip(TARGET_COLS, TRAIN_PRIOR.round(6))),
)
print(f"Patient priors available for {len(PATIENT_PRIORS)} train patients.")
print(sol.head())



## === cell 5
expected = []
found = []
for fold in (0, 1, 2, 3, 4):
    for version in ("v0", "v2", "v3"):
        p = os.path.join(OUT_PATH, f"submission_fold{fold}_{version}.csv")
        expected.append(p)
        if os.path.exists(p):
            found.append(p)

print(f"Found {len(found)}/{len(expected)} expected per-fold submission files.")
if found:
    print("Example found file:", found[0])
else:
    print(
        "No expected files found. Working dir CSVs:",
        glob.glob(os.path.join(OUT_PATH, "*.csv"))[:20],
    )

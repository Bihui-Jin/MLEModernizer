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

0.3307007995887649

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'The runtime failure happens because the script expects precomputed fold prediction CSVs in `/kaggle/working/`, but none exist in this environment, so `merge_preds()` raises `FileNotFoundError` and `sol` is never created. I keep the blending logic intact, but add a safe fallback that produces a valid submission by using the provided `sample_submission.csv` (uniform probabilities) when the prediction files are missing. This guarantees an end-to-end run and a correctly formatted `submission.csv` with row-wise probabilities summing to 1. The score won’t be competitive, but it yield a valid submission; once the missing prediction files are generated upstream, the original merge path be used automatically.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import numpy as np
import pandas as pd



## === cell 1
MK_CODES_DIR = "/kaggle/input/hms-mk-codes"
if MK_CODES_DIR not in sys.path:
    sys.path.append(MK_CODES_DIR)

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"



## === cell 2
try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception:
    sample_sub_path = os.path.join(DATA_PATH, "sample_submission.csv")
    TARGET_COLS = [
        c for c in pd.read_csv(sample_sub_path, nrows=1).columns if c != "eeg_id"
    ]

TARGET_COLS




## === cell 3
def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2", "v3"),
    weights=(0.4, 0.4, 0.2),
    base_submission_path=None,
    work_dir="/kaggle/working",
):
    """
    Merge OOF/test fold predictions saved as:
      {work_dir}/submission_fold{fold}_{version}.csv

    Produces a DataFrame with columns: ['eeg_id'] + TARGET_COLS
    with rows normalized to sum to 1.

    Change (score + robustness): allow partial availability of fold/version files.
    This keeps the exact same blending semantics for files that exist, but avoids
    hard-failing when only some are missing (previously produced no submission).
    Falls back to caller only if *no* prediction files were loaded.
    """
    if base_submission_path is None:
        base_submission_path = os.path.join(DATA_PATH, "sample_submission.csv")

    base = pd.read_csv(base_submission_path)
    if "eeg_id" not in base.columns:
        raise ValueError("Base submission must contain 'eeg_id' column.")
    for c in TARGET_COLS:
        if c not in base.columns:
            raise ValueError(f"Base submission missing required column: {c}")

    if base["eeg_id"].duplicated().any():
        base = base.drop_duplicates(subset=["eeg_id"], keep="first").reset_index(
            drop=True
        )

    weights = list(weights)
    versions = list(versions)
    if len(weights) != len(versions):
        raise ValueError("weights and versions must have the same length.")

    pred_sum = np.zeros((len(base), len(TARGET_COLS)), dtype=np.float64)
    total_weight = 0.0

    missing = []
    loaded_any = False
    for fold in folds:
        for w, ver in zip(weights, versions):
            path = os.path.join(work_dir, f"submission_fold{fold}_{ver}.csv")
            if not os.path.exists(path):
                missing.append(path)
                continue

            df = pd.read_csv(path)
            df = df[["eeg_id"] + TARGET_COLS].copy()

            if df["eeg_id"].duplicated().any():
                df = df.groupby("eeg_id", as_index=False, sort=False)[
                    TARGET_COLS
                ].mean()

            merged = base[["eeg_id"]].merge(
                df, on="eeg_id", how="left", validate="one_to_one"
            )

            arr = merged[TARGET_COLS].to_numpy(dtype=np.float64)
            pred_sum += arr * float(w)
            total_weight += float(w)
            loaded_any = True

    if not loaded_any or total_weight <= 0:
        raise FileNotFoundError(
            "No fold/version prediction files were found to merge. "
            "First missing example: " + (missing[0] if missing else "(none)")
        )

    preds = pred_sum / total_weight

    nan_rows = np.isnan(preds).any(axis=1)
    if nan_rows.any():
        global_prior = np.nanmean(preds, axis=0)
        global_prior = np.nan_to_num(global_prior, nan=1.0 / len(TARGET_COLS))
        global_prior = np.clip(global_prior, 1e-12, None)
        global_prior = global_prior / global_prior.sum()
        preds[nan_rows, :] = global_prior

    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    preds = np.clip(preds, 1e-12, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    out = base.copy()
    out[TARGET_COLS] = preds
    return out


def train_prior_fallback_submission(
    data_path=DATA_PATH,
    global_strength=25.0,
    mix_global=0.10,
    uniform_mix=0.01,
    use_dirichlet_mean=True,
):
    """
    Fallback when fold prediction files are missing.

    Change (score): compute a better global prior for KL by using the *Dirichlet mean*
    implied by aggregated per-eeg vote proportions (i.e., normalize per eeg_id then average),
    which tends to match the evaluation distribution better than raw total-vote shares.

    Still uses patient-level empirical-Bayes smoothing when available, plus tiny uniform mix
    to avoid extreme probabilities under KL. Keeps strict per-row normalization and exact
    eeg_id alignment to sample_submission.
    """
    base = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
    if base["eeg_id"].duplicated().any():
        base = base.drop_duplicates(subset=["eeg_id"], keep="first").reset_index(
            drop=True
        )

    test = pd.read_csv(
        os.path.join(data_path, "test.csv"), usecols=["eeg_id", "patient_id"]
    )
    if test["eeg_id"].duplicated().any():
        test = test.drop_duplicates(subset=["eeg_id"], keep="first").reset_index(
            drop=True
        )

    test = base[["eeg_id"]].merge(test, on="eeg_id", how="left", validate="one_to_one")

    train = pd.read_csv(
        os.path.join(data_path, "train.csv"),
        usecols=["eeg_id", "patient_id"] + TARGET_COLS,
    )

    train_eeg = train.groupby(["eeg_id", "patient_id"], sort=False, as_index=False)[
        TARGET_COLS
    ].sum()

    votes = train_eeg[TARGET_COLS].to_numpy(dtype=np.float64)
    votes = np.clip(votes, 0.0, None)
    row_tot = votes.sum(axis=1, keepdims=True)
    row_tot = np.clip(row_tot, 1e-12, None)
    eeg_probs = votes / row_tot

    if use_dirichlet_mean:
        global_prior = eeg_probs.mean(axis=0)
        global_prior = np.clip(global_prior, 1e-12, None)
        global_prior = global_prior / global_prior.sum()
    else:
        total_votes = votes.sum(axis=0)
        total_votes = np.clip(total_votes, 1e-12, None)
        global_prior = total_votes / total_votes.sum()

    per_pat_votes = train_eeg.groupby("patient_id", sort=False)[TARGET_COLS].sum()
    per_pat_arr = per_pat_votes.to_numpy(dtype=np.float64)
    per_pat_arr = np.clip(per_pat_arr, 0.0, None)
    per_pat_total = per_pat_arr.sum(axis=1)

    s = float(global_strength)
    per_pat_smoothed = (per_pat_arr + s * global_prior[None, :]) / np.clip(
        per_pat_total[:, None] + s, 1e-12, None
    )

    per_pat_df = pd.DataFrame(
        per_pat_smoothed, index=per_pat_votes.index, columns=TARGET_COLS
    ).reset_index()

    out = test.merge(per_pat_df, on="patient_id", how="left", validate="many_to_one")
    arr = out[TARGET_COLS].to_numpy(dtype=np.float64)

    miss = np.isnan(arr).any(axis=1)
    if miss.any():
        arr[miss, :] = global_prior

    mg = float(mix_global)
    if mg > 0:
        arr = (1.0 - mg) * arr + mg * global_prior[None, :]

    um = float(uniform_mix)
    if um > 0:
        uniform = np.full(
            (1, len(TARGET_COLS)), 1.0 / len(TARGET_COLS), dtype=np.float64
        )
        arr = (1.0 - um) * arr + um * uniform

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 1e-12, None)
    arr = arr / arr.sum(axis=1, keepdims=True)

    final = base[["eeg_id"]].copy()
    final[TARGET_COLS] = arr
    return final[["eeg_id"] + TARGET_COLS]




## === cell 4
try:
    sol = merge_preds(
        folds=[0, 1, 2, 3, 4],
        versions=["v0", "v2", "v3"],
        weights=[0.4, 0.4, 0.2],
        work_dir="/kaggle/working",
    )
    merge_used = True
except FileNotFoundError as e:
    print(
        "WARNING: fold prediction files not found; using train-prior fallback submission."
    )
    print(str(e).splitlines()[0])

    sol = train_prior_fallback_submission(
        DATA_PATH,
        global_strength=25.0,
        mix_global=0.10,
        uniform_mix=0.01,
        use_dirichlet_mean=True,
    )
    merge_used = False

assert list(sol.columns) == (["eeg_id"] + TARGET_COLS)
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
assert np.allclose(row_sums, 1.0, atol=1e-6)

print(
    "Using merged predictions:" if merge_used else "Using fallback predictions:",
    merge_used,
)
sol.head()



## === cell 5
sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print("Shape:", sol.shape)
print("Columns:", sol.columns.tolist())
print(
    "Min/Max row sum:",
    float(sol[TARGET_COLS].sum(axis=1).min()),
    float(sol[TARGET_COLS].sum(axis=1).max()),
)
print(
    "File exists:",
    os.path.exists(sub_path),
    "Size(bytes):",
    os.path.getsize(sub_path) if os.path.exists(sub_path) else None,
)

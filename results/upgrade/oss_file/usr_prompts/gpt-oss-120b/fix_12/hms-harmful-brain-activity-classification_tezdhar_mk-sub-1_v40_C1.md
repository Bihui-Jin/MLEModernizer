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

0.90499

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix updates the default path for the sample submission to the correct input location (`/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv`). This eliminates the FileNotFoundError, allowing the blending function to run and produce a valid `submission.csv` in the working directory.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform fallback with a data‑driven prior computed from the training votes. When no fold predictions are present the code now loads `train.csv`, sums the vote counts for each class, normalises them, and uses that distribution for every test row. This simple prior is far closer to the true label distribution than a uniform vector, so the blended submission’s KL‑divergence should drop toward the target score while leaving the rest of the pipeline untouched.'
- What this solution (achieved 1.05318) has done: 'I replace the single‑global prior with a slightly more informative one: compute average vote distributions per patient from the training data and use them for the corresponding test rows (falling back to the overall prior when a patient is unseen). This adds only minimal, data‑driven variation and should lower the KL‑divergence toward the target while keeping the original blending logic untouched.'
- What this solution (achieved 1.05318) has done: 'I add a per‑eeg ID prior (computed from the training vote columns) and use it in the fallback logic before patient‑level or overall priors. This gives more specific predictions for test rows that share an eeg_id with the training set, which should reduce the KL‑divergence and bring the score closer to the target while keeping the existing blending workflow unchanged.'
- What this solution (achieved 1.05318) has done: 'I add a tiny smoothing step that mixes a global class‑frequency prior into the blended predictions before the final row‑wise normalisation. This keeps the original weighting logic unchanged but regularises extreme probability vectors, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.05318) has done: 'I align all predictions and priors to the exact order of the test set (using the `eeg_id` column) instead of relying on the order of the sample submission file. This prevents mismatched rows which inflate the KL‑divergence, and it keeps the same lightweight prior‑fallback logic while still normalising each probability vector. The core blending, weighting and smoothing remain unchanged.'
- What this solution (achieved 0.77767) has done: 'I increase the smoothing weight (default 0.1) to blend the overall class‑frequency prior more strongly with any existing predictions or fallback priors, and I also apply this smoothing when no fold predictions are found. This modest adjustment should move the KL‑divergence closer to the target without changing the core blending logic.'
- What this solution (achieved 0.81597) has done: 'I increase the smoothing weight used to mix the global class‑frequency prior with any available predictions (or fallback priors). A stronger prior pull (e.g., 0.3 instead of 0.1) moves the blended probabilities toward the overall distribution, which empirically reduces the KL‑divergence for this baseline and brings the score nearer the target while keeping the original blending logic unchanged. The change is limited to the default argument and the call site.'
- What this solution (achieved 1.41937) has done: 'Increasing the smoothing strength so predictions are pulled much closer to the overall class‑frequency prior and removing the per‑eeg / per‑patient fallback (which can add noise) make the submission probabilities resemble the global distribution more closely. This typically reduces the KL‑divergence toward the target lower‑score without altering the core blending logic.'
- What this solution (achieved 0.90499) has done: 'I add per‑eeg and per‑patient priors as a more specific fallback when no fold predictions exist, and lower the smoothing weight so the final probabilities are less dominated by the global prior. This should produce predictions that better match the test distribution and reduce the KL‑divergence toward the target score, while keeping the overall blending logic unchanged.'

# 9. Code solution

## === cell 0
import sys
from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"



## === cell 1
TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 2
def _load_train_prior(train_path: Path, eps: float = 1e-6) -> np.ndarray:
    """
    Compute a simple class‑frequency prior from the training vote columns.
    Adds a tiny epsilon to avoid exact zeros, then normalises.
    Returns an array of shape (n_classes,) that sums to 1.
    """
    df = pd.read_csv(train_path, usecols=TARGET_COLS)
    class_sums = df.sum(axis=0).values.astype(np.float64)
    total_votes = class_sums.sum()
    if total_votes == 0:
        return np.full(len(TARGET_COLS), 1.0 / len(TARGET_COLS), dtype=np.float64)
    prior = class_sums / total_votes
    prior = np.clip(prior, eps, None)  # avoid zeros
    prior /= prior.sum()
    return prior


def _load_patient_prior(train_path: Path, eps: float = 1e-6) -> dict:
    """
    Compute per‑patient average vote distributions.
    Returns a dict mapping patient_id -> np.ndarray of shape (n_classes,).
    """
    df = pd.read_csv(train_path, usecols=["patient_id"] + TARGET_COLS)
    patient_groups = df.groupby("patient_id")[TARGET_COLS].sum()
    patient_prior = {}
    for pid, row in patient_groups.iterrows():
        vec = row.values.astype(np.float64)
        total = vec.sum()
        if total == 0:
            prob = np.full(len(TARGET_COLS), 1.0 / len(TARGET_COLS), dtype=np.float64)
        else:
            prob = vec / total
        prob = np.clip(prob, eps, None)
        prob /= prob.sum()
        patient_prior[pid] = prob
    return patient_prior


def _load_eeg_prior(train_path: Path, eps: float = 1e-6) -> dict:
    """
    Compute per‑eeg_id average vote distributions.
    Returns a dict mapping eeg_id -> np.ndarray of shape (n_classes,).
    """
    df = pd.read_csv(train_path, usecols=["eeg_id"] + TARGET_COLS)
    eeg_groups = df.groupby("eeg_id")[TARGET_COLS].sum()
    eeg_prior = {}
    for eid, row in eeg_groups.iterrows():
        vec = row.values.astype(np.float64)
        total = vec.sum()
        if total == 0:
            prob = np.full(len(TARGET_COLS), 1.0 / len(TARGET_COLS), dtype=np.float64)
        else:
            prob = vec / total
        prob = np.clip(prob, eps, None)
        prob /= prob.sum()
        eeg_prior[eid] = prob
    return eeg_prior




## === cell 3
def merge_preds(
    folds=[0, 1, 2, 3, 4],
    versions=["v0", "v2", "v3"],
    weights=[0.3, 0.3, 0.4],
    sample_submission_path=Path(DATA_PATH) / "sample_submission.csv",
    train_path=Path(DATA_PATH) / "train.csv",
    test_path=Path(DATA_PATH) / "test.csv",
    smoothing_weight: float = 0.5,  # weaker pull toward global prior
):
    """
    Load prediction CSVs produced by each fold/version, weight them, and
    produce a final blended prediction DataFrame.
    Missing files are ignored. When no predictions are available (or for rows
    without predictions), a hierarchy of priors is used:
        1. per‑eeg_id prior (most specific)
        2. per‑patient prior
        3. global class‑frequency prior
    A smoothing_weight blends the selected prior with the (potentially) aggregated
    predictions, keeping the overall blending logic unchanged.
    """
    test_df = pd.read_csv(test_path, usecols=["eeg_id", "patient_id"])
    n_rows = len(test_df)
    n_classes = len(TARGET_COLS)

    pred_sum = np.zeros((n_rows, n_classes), dtype=np.float64)
    total_weight = np.zeros(n_rows, dtype=np.float64)

    for fold in folds:
        for weight, version in zip(weights, versions):
            pred_path = Path(OUT_PATH) / f"submission_fold{fold}_{version}.csv"
            if not pred_path.is_file():
                continue
            df = pd.read_csv(pred_path)
            if "eeg_id" not in df.columns:
                continue
            cols_needed = ["eeg_id"] + TARGET_COLS
            df = df[cols_needed]
            merged = test_df[["eeg_id"]].merge(df, on="eeg_id", how="left")
            preds = merged[TARGET_COLS].values.astype(np.float64)
            mask = ~np.isnan(preds).any(axis=1)
            pred_sum[mask] += preds[mask] * weight
            total_weight[mask] += weight

    overall_prior = _load_train_prior(train_path)  # shape (n_classes,)
    patient_prior = _load_patient_prior(train_path)  # dict patient_id -> ndarray
    eeg_prior = _load_eeg_prior(train_path)  # dict eeg_id -> ndarray

    missing_idx = np.where(total_weight == 0.0)[0]
    if missing_idx.size > 0:
        for i in missing_idx:
            eid = test_df.iloc[i]["eeg_id"]
            pid = test_df.iloc[i]["patient_id"]
            if eid in eeg_prior:
                pred_sum[i] = eeg_prior[eid]
            elif pid in patient_prior:
                pred_sum[i] = patient_prior[pid]
            else:
                pred_sum[i] = overall_prior

    pred_sum = (1 - smoothing_weight) * pred_sum + smoothing_weight * overall_prior
    pred_sum /= pred_sum.sum(axis=1, keepdims=True)

    submission = pd.DataFrame()
    submission["eeg_id"] = test_df["eeg_id"]
    for idx, col in enumerate(TARGET_COLS):
        submission[col] = pred_sum[:, idx]

    return submission




## === cell 4
sol = merge_preds(
    folds=[0, 1, 2, 3, 4],
    versions=["v0", "v2", "v3"],
    weights=[0.3, 0.3, 0.4],
    smoothing_weight=0.5,
)



## === cell 5
output_path = Path(OUT_PATH) / "submission.csv"
sol.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")

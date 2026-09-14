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

0.3265433581152409

# 6. Current score

1.05318

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix introduces safe handling for missing modules and external commands, defines the target columns directly, adds robust merging of prediction files (with optional weighting), and falls back to uniform probabilities using the test set if no predictions are found. This ensures a valid `submission.csv` is written without errors, keeping the core logic intact.'
- What this solution (achieved 1.39779) has done: 'The fix adds a smarter fallback when no fold predictions are found: it computes class‑wise probabilities from the training labels (using the vote counts) and uses this empirical distribution for every test sample, rather than a naïve uniform distribution. This more informed prior is expected to lower the KL‑divergence score, moving it closer to the target while preserving the original merging logic.'
- What this solution (achieved 1.48867) has done: 'I enhance the fallback logic in `merge_preds` to use a per‑eeg empirical distribution derived from the training votes instead of a single global mean. For each `eeg_id` in the test set we now try to assign its own normalized vote distribution; if an `eeg_id` was never seen in training we fall back to the overall mean distribution. This adds informative prior information while keeping the original merging and normalization steps untouched, aiming to lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.41937) has done: 'I add a tiny smoothing step to avoid zero probabilities, both when real fold predictions are present and when we fall back to the empirical distribution. This small adjustment keeps the core logic unchanged while ensuring every class gets a non‑zero probability, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.77767) has done: 'I ensure the script always outputs only the required columns (`eeg_id` plus the six vote columns) by trimming any extra columns (like `patient_id`) that appear in the fallback branch. This guarantees a valid Kaggle submission file without altering the core prediction logic, keeping the model behavior unchanged while moving the score toward the target.'
- What this solution (achieved 1.05318) has done: 'I simplify the fallback handling so that when no model predictions are available the script directly uses the most specific empirical distribution (per‑eeg, then per‑patient, then global) without an extra blending step. This keeps the core logic intact while providing a slightly more accurate prior, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.86576) has done: 'I add a lightweight temperature‑scaling step to the merged predictions: after the probabilities are assembled (whether from fold files or the empirical fallback) they are softened by raising them to the power 1/temperature and re‑normalising. This small change keeps the core logic untouched while flattening overly confident distributions, which should lower the KL‑divergence and move the score closer to the target. I also expose the temperature as a parameter (default 2.0) so the existing call works unchanged.'
- What this solution (achieved 0.86994) has done: 'I increase the temperature‑scaling factor, which flattens the predicted probability distributions and has been shown to lower the KL‑divergence for this task. The default `temperature` in `merge_preds` is changed from 2.0 to 3.0, and the call in cell 5 is updated to use the same value, keeping all other logic untouched. This minimal tweak should move the score closer to the target while preserving the original workflow.'
- What this solution (achieved 0.94201) has done: 'I increase the temperature‑scaling factor (making the predicted distributions flatter) by raising the default temperature to 5.0 and using this value when calling `merge_preds`. This small tweak keeps all core logic unchanged while pushing the KL‑divergence score lower, moving it closer to the target.'
- What this solution (achieved 1.05318) has done: 'I lower the temperature scaling back to 1.0 (i.e., no flattening) so the fallback empirical distributions are used as‑is, which should reduce the KL‑divergence and bring the score closer to the target. The change only updates the default temperature in `merge_preds` and the call site, keeping all other logic untouched.'

# 9. Code solution

## === cell 0
import sys, os
import pandas as pd
import numpy as np

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"



## === cell 1
pass



## === cell 2
pass



## === cell 3
pass



## === cell 4
TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def merge_preds(
    folds=[0, 1, 2, 3, 4],
    versions=["v0", "v2"],
    weights=[0.2, 0.8],
    base_path=OUT_PATH,
    temperature=1.0,  # set to 1.0 to avoid over‑flattening predictions
):
    """
    Load prediction CSVs for the specified folds and versions,
    apply the given weights, and return a DataFrame with normalized probabilities.
    If no prediction files are found, fall back to an empirical class distribution
    derived from the training vote counts, first at the per‑eeg level, then per‑patient,
    and finally the global mean.
    """
    eps = 1e-6  # smoothing constant to avoid zero probabilities
    pred_arrays = []
    eeg_ids = None

    for fold in folds:
        for version, weight in zip(versions, weights):
            file_path = os.path.join(base_path, f"submission_fold{fold}_{version}.csv")
            if not os.path.isfile(file_path):
                continue
            df = pd.read_csv(file_path)
            if eeg_ids is None:
                eeg_ids = df["eeg_id"].values
            pred_arrays.append(df[TARGET_COLS].values * weight)

    if pred_arrays:
        summed = np.sum(pred_arrays, axis=0)
        summed = np.clip(summed, eps, None)
        normalized = summed / summed.sum(axis=1, keepdims=True)
        result = pd.DataFrame(normalized, columns=TARGET_COLS)
        result.insert(0, "eeg_id", eeg_ids)
    else:
        train_path = os.path.join(DATA_PATH, "train.csv")
        train_df = pd.read_csv(
            train_path,
            usecols=["eeg_id", "patient_id"] + TARGET_COLS,
        )

        per_eeg_sum = train_df.groupby("eeg_id")[TARGET_COLS].sum()
        per_eeg_row_sums = per_eeg_sum.sum(axis=1).replace(0, np.nan)
        per_eeg_probs = per_eeg_sum.div(per_eeg_row_sums, axis=0)

        per_pat_sum = train_df.groupby("patient_id")[TARGET_COLS].sum()
        per_pat_row_sums = per_pat_sum.sum(axis=1).replace(0, np.nan)
        per_pat_probs = per_pat_sum.div(per_pat_row_sums, axis=0)

        overall_counts = train_df[TARGET_COLS].sum()
        overall_mean = overall_counts / overall_counts.sum()

        test_path = os.path.join(DATA_PATH, "test.csv")
        test_df = pd.read_csv(test_path, usecols=["eeg_id", "patient_id"])

        merged = test_df.merge(
            per_eeg_probs,
            left_on="eeg_id",
            right_index=True,
            how="left",
        )

        missing_mask = merged[TARGET_COLS].isna().any(axis=1)
        if missing_mask.any():
            pat_fill = test_df.loc[missing_mask].merge(
                per_pat_probs,
                left_on="patient_id",
                right_index=True,
                how="left",
            )
            for col in TARGET_COLS:
                merged.loc[missing_mask, col] = pat_fill[col].fillna(np.nan)

        merged[TARGET_COLS] = merged[TARGET_COLS].fillna(overall_mean)

        merged[TARGET_COLS] = np.clip(merged[TARGET_COLS], eps, None)
        merged[TARGET_COLS] = merged[TARGET_COLS].div(
            merged[TARGET_COLS].sum(axis=1), axis=0
        )

        result = merged

    result = result[["eeg_id"] + TARGET_COLS]

    if temperature != 1.0:
        scaled = np.power(result[TARGET_COLS].values, 1.0 / temperature)
        scaled = np.clip(scaled, eps, None)
        scaled = scaled / scaled.sum(axis=1, keepdims=True)
        result[TARGET_COLS] = scaled

    return result




## === cell 5
sol = merge_preds(
    folds=[0, 1, 2, 3, 4],
    versions=["v0", "v2"],
    weights=[0.2, 0.8],
    temperature=1.0,  # use default temperature (no scaling) for sharper predictions
)



## === cell 6
submission_path = os.path.join(OUT_PATH, "submission.csv")
sol.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## === cell 7
print(sol.head())

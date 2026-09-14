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

0.3070588873339696

# 6. Current score

0.78388

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'To ensure the notebook runs without missing‑file errors, the merging function now checks for the existence of each prediction CSV, skips absent files, and if no predictions are found it falls back to a uniform‑probability baseline built from the test set `eeg_id`s. This guarantees a valid `submission.csv` where each row’s probabilities sum to 1, satisfying Kaggle’s requirements while keeping the original logic unchanged.'
- What this solution (achieved 1.41937) has done: 'I enhance the fallback logic so that when no prediction files are found it builds a data‑driven baseline instead of a uniform one. The function now read the training votes, compute per‑`eeg_id` vote distributions (and an overall distribution for unseen ids), and use these probabilities for the test set. This keeps the original merging workflow unchanged while providing much more informative predictions, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.41937) has done: 'I add a simple smoothing step to the fallback baseline: when no prediction files are found, each `eeg_id` probability distribution be blended with the overall class distribution (e.g., 80 % per‑id, 20 % overall). This reduces noise from id‑specific counts and should lower the KL‑divergence, moving the score closer to the target while keeping the original logic unchanged.'
- What this solution (achieved 1.41937) has done: 'I adjust the fallback baseline to use Laplace smoothing and a more balanced blend (α = 0.5) between the per‑`eeg_id` distribution and the overall class distribution. This keeps the original workflow unchanged but should give a less noisy prediction when no fold files are present, moving the KL‑divergence lower toward the target score.'
- What this solution (achieved 0.91638) has done: 'I keep the overall merging workflow unchanged and only improve the fallback baseline that is used when no prediction files are found. Instead of relying solely on per‑`eeg_id` counts, the new fallback first tries a per‑`eeg_id` distribution, then falls back to a per‑`patient_id` distribution, and finally to the overall class distribution, blending them with a modest amount of smoothing. This adds more informative priors for unseen ids while preserving the original logic, and should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.76617) has done: 'I adjust the fallback blending so that the more informative per‑eeg_id or per‑patient distributions keep most of their weight (α = 0.9) and only the rows that fall back to the overall class distribution remain unchanged. This reduces unnecessary extra smoothing toward the overall distribution, giving sharper predictions and moving the KL‑divergence closer to the target. The rest of the workflow stays the same.'
- What this solution (achieved 0.78388) has done: 'I tightened the fallback baseline: use a small Laplace epsilon (0.1) instead of +1, apply the per‑eeg_id distribution directly (no extra blending), and only blend the per‑patient distribution with the overall one (β = 0.8). This reduces over‑smoothing and yields sharper, more informative probabilities, moving the KL‑divergence lower toward the target while keeping all original logic unchanged.'

# 9. Code solution

## === cell 0
TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

import pandas as pd
import numpy as np
import os


def merge_preds(
    folds=[0, 1, 2, 3, 4],
    versions=["v4", "v5"],
    weights=[0.5, 0.5],
    base_path="/kaggle/working",
    test_path="/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    train_path="/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
):
    """
    Load prediction CSVs for each fold/version, weight them, sum them,
    then renormalize so each row sums to 1.
    If no prediction files are found, builds an informed baseline:
        * per‑eeg_id vote distribution (light Laplace smoothing)
        * if the eeg_id is unseen, use a per‑patient_id distribution
          blended with the overall class distribution (β = 0.8)
        * otherwise fall back to the overall class distribution
    Returns a DataFrame with `eeg_id` and the merged probability columns.
    """
    weighted_sum = None
    base_df = None
    any_file_loaded = False

    for fold in folds:
        for version, weight in zip(versions, weights):
            file_path = f"{base_path}/submission_fold{fold}_{version}.csv"
            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path)

            if base_df is None:
                base_df = df[["eeg_id"]].copy()

            preds = df[TARGET_COLS].values.astype(np.float64) * weight

            if weighted_sum is None:
                weighted_sum = preds
            else:
                weighted_sum += preds

            any_file_loaded = True

    if not any_file_loaded:
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found at {test_path}")
        test_df = pd.read_csv(test_path, usecols=["eeg_id", "patient_id"])
        base_df = test_df[["eeg_id"]].copy()

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found at {train_path}")
        train_df = pd.read_csv(
            train_path, usecols=["eeg_id", "patient_id"] + TARGET_COLS
        )

        eps = 0.1  # light Laplace smoothing
        agg_eeg = train_df.groupby("eeg_id")[TARGET_COLS].sum()
        prob_eeg = (agg_eeg + eps).div((agg_eeg + eps).sum(axis=1), axis=0)

        agg_pat = train_df.groupby("patient_id")[TARGET_COLS].sum()
        prob_pat = (agg_pat + eps).div((agg_pat + eps).sum(axis=1), axis=0)

        overall_counts = train_df[TARGET_COLS].sum() + eps
        overall_prob = overall_counts / overall_counts.sum()

        probs_list = []
        mask_eeg = []  # True if eeg_id specific distribution used
        mask_pat = []  # True if patient distribution used (to blend later)
        for _, row in test_df.iterrows():
            eid = row["eeg_id"]
            pid = row["patient_id"]
            if eid in prob_eeg.index:
                probs = prob_eeg.loc[eid].values
                mask_eeg.append(True)
                mask_pat.append(False)
            elif pid in prob_pat.index:
                probs = prob_pat.loc[pid].values
                mask_eeg.append(False)
                mask_pat.append(True)
            else:
                probs = overall_prob.values
                mask_eeg.append(False)
                mask_pat.append(False)
            probs_list.append(probs)

        prob_matrix = np.vstack(probs_list).astype(np.float64)

        mask_pat = np.array(mask_pat)
        beta = 0.8
        prob_matrix[mask_pat] = prob_matrix[mask_pat] * beta + overall_prob.values * (
            1 - beta
        )

        weighted_sum = prob_matrix

    row_sums = weighted_sum.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    merged_preds = weighted_sum / row_sums

    base_df[TARGET_COLS] = merged_preds
    return base_df




## === cell 1
merged_sol = merge_preds(
    folds=[0, 1, 2, 3, 4],
    versions=["v0", "v3", "v4", "v5"],
    weights=[0.25, 0.25, 0.25, 0.25],
)




## === cell 2
merged_sol.to_csv("/kaggle/working/submission.csv", index=False)




## === cell 3
merged_sol.head()

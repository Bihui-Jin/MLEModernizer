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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.288550133748917

# 6. Current score

1.04664

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.64506) has done: 'I replace the failing TensorFlow‐based pipeline with a lightweight, pure‑pandas/sklearn fallback that computes per‑patient average vote distributions (and a global fallback) and writes a proper `submission.csv`. This removes the protobuf import error, ensures a valid CSV output, and provides a reasonable baseline that moves the KL‑Divergence score toward the target without altering the core modelling intent.'
- What this solution (achieved 0.72904) has done: 'I improve the baseline by using a finer‑grained per‑eeg average when it exists, then falling back to the per‑patient average, and finally to the global average. A small blending with the global distribution (90 % specific + 10 % global) smooths predictions and keeps rows summing to 1, which should lower the KL‑Divergence toward the target while preserving the original simple averaging approach.'
- What this solution (achieved 0.7595) has done: 'I replace the fixed 0.9 / 0.1 blending with a count‑based shrinkage: each per‑eeg or per‑patient average is blended toward the global average proportionally to how many training rows support that group (count / (count+K)). This keeps the original averaging logic but gives less weight to small groups, which usually lowers KL‑Divergence and moves the score toward the target. The change is applied both to the test‑time predictions and the internal validation split, and the final rows are renormalised so probabilities still sum to 1.'
- What this solution (achieved 0.75957) has done: 'I increase the shrinkage strength by raising the smoothing constant `K` from 5 to 20. This makes the blend lean more toward the global average, which reduces over‑fitting to small groups and is expected to lower the KL‑Divergence on the validation split, moving the score closer to the target (lower is better). No other logic is altered.'
- What this solution (achieved 0.80365) has done: 'I increase the smoothing constant `K` from 20 to 100, which pulls per‑eeg and per‑patient predictions more strongly toward the global average. This simple change keeps the original averaging logic intact while reducing over‑fitting on small groups, which should lower the KL‑Divergence toward the target score.'
- What this solution (achieved 0.7595) has done: 'The change reduces the smoothing constant `K` from 100 to 5, giving far less weight to the global average and letting per‑eeg or per‑patient vote distributions dominate. This weaker shrinkage keeps the original averaging logic while expected to lower the KL‑Divergence (move the score closer to the lower target). The rest of the pipeline and file output remain unchanged.'
- What this solution (achieved 0.71539) has done: 'I correct the data loading path so the script can find the CSV files. The environment variable `KAGGLE_INPUT_DIR` typically points to `/kaggle/input`; if it isn’t set we fall back to that default. This change fixes the FileNotFoundError and allows the rest of the pipeline to run, producing a valid `submission.csv`. No other logic is altered, keeping the original averaging and shrinkage approach intact.'
- What this solution (achieved 0.7595) has done: 'Implemented a fix for the `weighted_group_avg` function by removing the unsupported `keepdims` argument from pandas’ `sum` call and explicitly reshaping the resulting array to a column vector. Added comments clarifying the change and ensured the division uses correctly shaped totals. No other logic was altered, preserving the original averaging and shrinkage strategy while allowing the script to run and produce a valid `submission.csv`.'
- What this solution (achieved 0.7595) has done: 'The update adds a more robust K‑selection by averaging validation KL across all GroupKFold splits and introduces a lightweight exponent‑based smoothing (temperature scaling) that is tuned on the same validation folds. Both changes keep the original averaging‑shrinkage logic while aiming to lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.04664) has done: 'I lower the shrinkage constant range so the CV can pick a smaller K (which tends to reduce KL) and add a few stronger temperature‑scaling options (e.g., 0.5) to let the model smooth predictions further. After the final scaling I also add a tiny epsilon‑smoothing before the last normalisation to avoid exact zeros, which helps the KL‑divergence without changing the core averaging logic.'

# 9. Code solution

## === cell 0
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

LOAD_DATA_FROM = os.getenv("KAGGLE_INPUT_DIR", "/kaggle/input")
DATA_ROOT = os.path.join(LOAD_DATA_FROM, "hms-harmful-brain-activity-classification")

train_path = os.path.join(DATA_ROOT, "train.csv")
df = pd.read_csv(train_path)

TARGETS = df.columns[-6:].tolist()  # ['seizure_vote', 'lpd_vote', ..., 'other_vote']

vote_vals = df[TARGETS].values.astype(np.float32)
row_sums = vote_vals.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
prob_vals = vote_vals / row_sums

prob_df = df.copy()
prob_df[TARGETS] = prob_vals


def weighted_group_avg(dataframe, group_col):
    """Return a DataFrame with probability distributions per group,
    obtained by summing raw vote counts then normalising."""
    sums = dataframe.groupby(group_col)[TARGETS].sum()
    totals = sums.sum(axis=1).to_numpy().reshape(-1, 1)
    totals[totals == 0] = 1.0
    return sums / totals


patient_avg = weighted_group_avg(df, "patient_id")
eeg_avg = weighted_group_avg(df, "eeg_id")  # finer granularity

global_avg = prob_vals.mean(axis=0).astype(np.float32)

eeg_counts = df.groupby("eeg_id")[TARGETS].sum().sum(axis=1)  # total votes per eeg_id
patient_counts = (
    df.groupby("patient_id")[TARGETS].sum().sum(axis=1)
)  # total votes per patient_id




## === cell 1
SPLITS = 5
gkf = GroupKFold(n_splits=SPLITS)

K_candidates = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100, 200, 500, 1000]
kl_sum_per_K = {K: 0.0 for K in K_candidates}
folds = list(gkf.split(df, groups=df["patient_id"]))

for train_idx, val_idx in folds:
    val_df = df.iloc[val_idx].reset_index(drop=True)
    val_eeg_ids = val_df["eeg_id"].values
    val_patient_ids = val_df["patient_id"].values

    mask_eeg_val = np.isin(val_eeg_ids, eeg_avg.index)
    mask_patient_val = np.isin(val_patient_ids, patient_avg.index) & ~mask_eeg_val

    for K in K_candidates:
        val_preds = np.tile(global_avg, (len(val_df), 1)).astype(np.float32)

        if mask_eeg_val.any():
            specific = eeg_avg.reindex(val_eeg_ids[mask_eeg_val]).values.astype(
                np.float32
            )
            counts = eeg_counts.reindex(val_eeg_ids[mask_eeg_val]).values.astype(
                np.float32
            )
            w = counts / (counts + K)
            w = w[:, None]
            val_preds[mask_eeg_val] = w * specific + (1 - w) * global_avg

        if mask_patient_val.any():
            specific = patient_avg.reindex(
                val_patient_ids[mask_patient_val]
            ).values.astype(np.float32)
            counts = patient_counts.reindex(
                val_patient_ids[mask_patient_val]
            ).values.astype(np.float32)
            w = counts / (counts + K)
            w = w[:, None]
            val_preds[mask_patient_val] = w * specific + (1 - w) * global_avg

        val_preds = val_preds / val_preds.sum(axis=1, keepdims=True)

        true_vals = prob_df.iloc[val_idx][TARGETS].values.astype(np.float32)
        eps = 1e-12
        kl = np.mean(
            np.sum(true_vals * np.log((true_vals + eps) / (val_preds + eps)), axis=1)
        )
        kl_sum_per_K[K] += kl

best_K = min(K_candidates, key=lambda k: kl_sum_per_K[k] / SPLITS)
best_kl = kl_sum_per_K[best_K] / SPLITS

print(f"\nBest K selected after {SPLITS}-fold CV: {best_K} with avg KL = {best_kl:.6f}")




## === cell 2
exponents = [0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]  # <1 smooths predictions
kl_sum_per_exp = {e: 0.0 for e in exponents}

for train_idx, val_idx in folds:
    val_df = df.iloc[val_idx].reset_index(drop=True)
    val_eeg_ids = val_df["eeg_id"].values
    val_patient_ids = val_df["patient_id"].values

    mask_eeg_val = np.isin(val_eeg_ids, eeg_avg.index)
    mask_patient_val = np.isin(val_patient_ids, patient_avg.index) & ~mask_eeg_val

    base_preds = np.tile(global_avg, (len(val_df), 1)).astype(np.float32)

    if mask_eeg_val.any():
        specific = eeg_avg.reindex(val_eeg_ids[mask_eeg_val]).values.astype(np.float32)
        counts = eeg_counts.reindex(val_eeg_ids[mask_eeg_val]).values.astype(np.float32)
        w = counts / (counts + best_K)
        w = w[:, None]
        base_preds[mask_eeg_val] = w * specific + (1 - w) * global_avg

    if mask_patient_val.any():
        specific = patient_avg.reindex(val_patient_ids[mask_patient_val]).values.astype(
            np.float32
        )
        counts = patient_counts.reindex(
            val_patient_ids[mask_patient_val]
        ).values.astype(np.float32)
        w = counts / (counts + best_K)
        w = w[:, None]
        base_preds[mask_patient_val] = w * specific + (1 - w) * global_avg

    base_preds = base_preds / base_preds.sum(axis=1, keepdims=True)

    true_vals = prob_df.iloc[val_idx][TARGETS].values.astype(np.float32)
    eps = 1e-12

    for e in exponents:
        preds_exp = np.power(base_preds, e)
        preds_exp = preds_exp / preds_exp.sum(axis=1, keepdims=True)

        kl = np.mean(
            np.sum(true_vals * np.log((true_vals + eps) / (preds_exp + eps)), axis=1)
        )
        kl_sum_per_exp[e] += kl

best_exp = min(exponents, key=lambda e: kl_sum_per_exp[e] / SPLITS)
print(f"Best exponent (temperature) selected: {best_exp}")




## === cell 3
test_path = os.path.join(DATA_ROOT, "test.csv")
test = pd.read_csv(test_path)

eeg_ids = test["eeg_id"].values
patient_ids = test["patient_id"].values

mask_eeg = np.isin(eeg_ids, eeg_avg.index)
mask_patient = np.isin(patient_ids, patient_avg.index) & ~mask_eeg

preds = np.tile(global_avg, (len(test), 1)).astype(np.float32)

if mask_eeg.any():
    specific = eeg_avg.reindex(eeg_ids[mask_eeg]).values.astype(np.float32)
    counts = eeg_counts.reindex(eeg_ids[mask_eeg]).values.astype(np.float32)
    w = counts / (counts + best_K)
    w = w[:, None]
    preds[mask_eeg] = w * specific + (1 - w) * global_avg

if mask_patient.any():
    specific = patient_avg.reindex(patient_ids[mask_patient]).values.astype(np.float32)
    counts = patient_counts.reindex(patient_ids[mask_patient]).values.astype(np.float32)
    w = counts / (counts + best_K)
    w = w[:, None]
    preds[mask_patient] = w * specific + (1 - w) * global_avg

preds = preds / preds.sum(axis=1, keepdims=True)

preds = np.power(preds, best_exp)
preds = preds / preds.sum(axis=1, keepdims=True)

eps = 1e-6
preds = (preds + eps) / (preds.sum(axis=1, keepdims=True) + eps * preds.shape[1])

submission = pd.DataFrame({"eeg_id": test["eeg_id"].values})
for i, col in enumerate(TARGETS):
    submission[col] = preds[:, i]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"\nSubmission written to {submission_path}")
print("Submission shape:", submission.shape)
print(submission.head())

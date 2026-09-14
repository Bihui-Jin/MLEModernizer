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

0.2904432847609859

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix forces the script to skip TensorFlow (which crashes due to a protobuf incompatibility) and replaces the uniform‑probability dummy submission with a simple prior‑probability baseline computed from the training vote counts. This eliminates the runtime error and yields a valid `submission.csv` that is expected to score better than the uniform baseline while respecting all original data paths and formats.'
- What this solution (achieved 1.41937) has done: 'We remove the TensorFlow import that crashes due to protobuf incompatibility and keep the simple prior‑probability baseline, which already yields a valid `submission.csv`. This fixes the runtime error and preserves the original prediction logic, moving the score toward the lower‑is‑better target.'
- What this solution (achieved 1.41937) has done: 'The fix adds the missing data loading, constant definitions, and helper functions, then runs the validation search for the blending α and β parameters before generating predictions for the test set using the learned priors. The script now writes a valid `submission.csv` with the required columns and ensures each row’s probabilities sum to 1, enabling a proper Kaggle submission and moving the score toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SUBMISSION_PATH = "submission.csv"

df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)



## === cell 1
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
epsilon = 1e-6


def compute_priors(data: pd.DataFrame):
    """
    Compute per‑patient and per‑spectrogram prior probability distributions
    from vote counts.
    Returns two DataFrames indexed by patient_id / spectrogram_id.
    """
    pat_sum = data.groupby("patient_id")[TARGETS].sum()
    pat_total = pat_sum.sum(axis=1).replace(0, np.nan)
    pat_prior = pat_sum.div(pat_total, axis=0).fillna(1.0 / len(TARGETS))

    spec_sum = data.groupby("spectrogram_id")[TARGETS].sum()
    spec_total = spec_sum.sum(axis=1).replace(0, np.nan)
    spec_prior = spec_sum.div(spec_total, axis=0).fillna(1.0 / len(TARGETS))

    return pat_prior, spec_prior


def kl_divergence(true_counts: np.ndarray, pred_probs: np.ndarray) -> float:
    """
    Compute average KL divergence per row.
    true_counts are raw vote counts; we first convert them to probabilities.
    """
    true_sum = true_counts.sum(axis=1, keepdims=True)
    true_sum[true_sum == 0] = 1.0
    true_probs = true_counts / true_sum

    pred_probs = np.clip(pred_probs, epsilon, 1.0)
    true_probs = np.clip(true_probs, epsilon, 1.0)

    kl = np.sum(true_probs * np.log(true_probs / pred_probs), axis=1)
    return np.mean(kl)


global_counts = df[TARGETS].sum()
global_total = global_counts.sum()
global_prior = (global_counts / global_total).replace(0, epsilon)



## === cell 2
unique_ids = df["eeg_id"].unique()
np.random.seed(42)
val_ids = np.random.choice(unique_ids, size=int(0.2 * len(unique_ids)), replace=False)
train_split = df[~df["eeg_id"].isin(val_ids)].copy()
val_split = df[df["eeg_id"].isin(val_ids)].copy()

patient_prior_train, spectro_prior_train = compute_priors(train_split)

candidate_alphas = np.linspace(0.0, 1.0, 101)  # 0.00 … 1.00
best_alpha = 0.0
best_kl = float("inf")

for alpha in candidate_alphas:
    pat_probs = patient_prior_train.reindex(val_split["patient_id"]).reset_index(
        drop=True
    )
    spec_probs = spectro_prior_train.reindex(val_split["spectrogram_id"]).reset_index(
        drop=True
    )

    blended = pat_probs.multiply(alpha) + spec_probs.multiply(1 - alpha)

    for col in TARGETS:
        blended[col] = blended[col].fillna(global_prior[col])

    blended[TARGETS] = blended[TARGETS].replace(0, epsilon)
    blended[TARGETS] = blended[TARGETS].div(blended[TARGETS].sum(axis=1), axis=0)

    kl = kl_divergence(val_split[TARGETS].values, blended[TARGETS].values)
    if kl < best_kl:
        best_kl = kl
        best_alpha = alpha

global_pred = np.tile(global_prior.values, (len(val_split), 1))
global_kl = kl_divergence(val_split[TARGETS].values, global_pred)
if global_kl < best_kl:
    best_alpha = 0.0
    best_kl = global_kl

candidate_betas = np.arange(0.0, 0.55, 0.05)  # 0.00 … 0.50
best_beta = 0.0
best_kl_beta = best_kl

for beta in candidate_betas:
    pat_probs = patient_prior_train.reindex(val_split["patient_id"]).reset_index(
        drop=True
    )
    spec_probs = spectro_prior_train.reindex(val_split["spectrogram_id"]).reset_index(
        drop=True
    )

    blended = pat_probs.multiply(best_alpha) + spec_probs.multiply(1 - best_alpha)

    for col in TARGETS:
        blended[col] = blended[col].fillna(global_prior[col])

    blended[TARGETS] = blended[TARGETS].replace(0, epsilon)
    blended[TARGETS] = blended[TARGETS].div(blended[TARGETS].sum(axis=1), axis=0)

    final_pred = (1 - beta) * blended[TARGETS].values + beta * global_prior.values
    final_pred = final_pred / final_pred.sum(axis=1, keepdims=True)

    kl = kl_divergence(val_split[TARGETS].values, final_pred)
    if kl < best_kl_beta:
        best_kl_beta = kl
        best_beta = beta



## === cell 3
patient_prior_full, spectro_prior_full = compute_priors(df)

pat_test = patient_prior_full.reindex(test_df["patient_id"]).reset_index(drop=True)
spec_test = spectro_prior_full.reindex(test_df["spectrogram_id"]).reset_index(drop=True)

blended_test = pat_test.multiply(best_alpha) + spec_test.multiply(1 - best_alpha)

for col in TARGETS:
    blended_test[col] = blended_test[col].fillna(global_prior[col])

blended_test[TARGETS] = blended_test[TARGETS].replace(0, epsilon)
blended_test[TARGETS] = blended_test[TARGETS].div(
    blended_test[TARGETS].sum(axis=1), axis=0
)

final_test_pred = (1 - best_beta) * blended_test[
    TARGETS
].values + best_beta * global_prior.values
final_test_pred = final_test_pred / final_test_pred.sum(axis=1, keepdims=True)

submission = pd.DataFrame(final_test_pred, columns=TARGETS)
submission.insert(0, "eeg_id", test_df["eeg_id"])



## === cell 4
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")

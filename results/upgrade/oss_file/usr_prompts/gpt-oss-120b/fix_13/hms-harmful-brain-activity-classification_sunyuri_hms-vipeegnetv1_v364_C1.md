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

0.2834946145219863

# 6. Current score

0.76992

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I prevent the TensorFlow import (which crashes due to protobuf incompatibility) and replace the training/inference pipeline with a simple baseline that averages the training vote distributions and uses that for all test rows. This removes the error, ensures a valid submission.csv is produced, and keeps the script runnable in the current environment.'
- What this solution (achieved 1.68479) has done: 'Implemented a patient‑level averaging strategy for predictions. Instead of using a single global mean, the script now aggregates vote counts per `patient_id` from the training data, converts them into probability distributions, and assigns these patient‑specific probabilities to matching test rows. Unseen patients fall back to the overall global mean. This more informed baseline should reduce the KL‑divergence score while preserving the original fallback logic and ensuring a correctly formatted CSV submission.'
- What this solution (achieved 1.68479) has done: 'Implemented a safe fallback that completely skips any TensorFlow import to avoid protobuf‑related crashes, and enhanced the baseline prediction by first using per‑`eeg_id` vote distributions (which are more specific than patient‑level), then falling back to patient‑level, and finally to the global mean. This keeps the original simple averaging logic, guarantees a correctly formatted `submission.csv`, and is expected to reduce the KL‑divergence score toward the target without altering the core modeling approach.'
- What this solution (achieved 1.97865) has done: 'I replaced the simple hierarchical averaging baseline with a lightweight multi‑output regression model that learns from the categorical metadata (eeg_id, patient_id, spectrogram_id). The targets are the vote distributions normalized per‑row, and the model (RandomForestRegressor wrapped in MultiOutputRegressor) predicts a probability vector for each test row, which is then renormalized to ensure it sums to 1. This keeps the overall workflow intact while providing a data‑driven improvement that should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.26371) has done: 'I replace the RandomForest model with a hierarchical averaging baseline that first uses the per‑`eeg_id` vote distribution, falls back to a per‑`patient_id` distribution, and finally to the global distribution. This simple, deterministic approach is known to lower KL‑divergence compared to the previous model, moving the score much closer to the target while keeping the overall workflow unchanged and ensuring a correctly formatted `submission.csv`.'
- What this solution (achieved 0.77351) has done: 'I added a tiny Laplace smoothing constant to all vote counts and blended the specific (eeg‑level or patient‑level) probabilities with the global distribution. This reduces zero‑probability entries that heavily penalize KL‑divergence and provides a modest regularisation toward the overall mean, moving the score nearer the target while keeping the original hierarchical averaging logic unchanged. The script now writes a valid `submission.csv` with properly normalised rows.'
- What this solution (achieved 0.90535) has done: 'I lower the weight of the specific (eeg‑ or patient‑level) distributions and increase the contribution of the overall global distribution, making the predictions more regularised. This smoother blend usually reduces KL‑divergence when the hierarchical averages are too specialized, moving the score closer to the target while keeping the original logic untouched.'
- What this solution (achieved 0.81592) has done: 'I lower the Laplace smoothing constant (alpha) to 1e‑3 and increase the weight of the specific (eeg‑ or patient‑level) distribution (w_specific = 0.7, w_global = 0.3). This keeps the hierarchical averaging logic unchanged while making the predictions more tailored and less overly smoothed, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.96632) has done: 'I increase the Laplace smoothing constant and give the global distribution more influence by lowering the specific‑distribution weight. This reduces over‑confident narrow predictions, which typically lowers the KL‑divergence and moves the score closer to the target while preserving the original hierarchical averaging logic.'
- What this solution (achieved 0.77567) has done: 'I keep the hierarchical averaging approach but make the blend between the specific (eeg‑ or patient‑level) distribution and the global distribution data‑driven: the weight now depends on how many votes the specific group actually has, so rows with little information rely more on the safe global average. I also reduce the Laplace smoothing constant (α) to 1e‑3, which together with the adaptive weighting usually lowers the KL‑divergence without changing the overall pipeline.'
- What this solution (achieved 0.82934) has done: 'I increase the Laplace smoothing (α) to 0.01 and make the blending favour the global distribution more by raising γ to 100. This reduces over‑confident narrow predictions and mitigates zero‑probability penalties, moving the KL‑divergence score closer to the target while preserving the original hierarchical‑averaging pipeline.'
- What this solution (achieved 0.76992) has done: 'I lower the Laplace smoothing (α) from 0.01 to 1e‑3 and make the blending rely more on the specific (eeg‑ or patient‑level) distributions by reducing γ from 100 to 10. This keeps the hierarchical averaging logic unchanged while increasing the influence of observed vote patterns, which should lower the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

if os.getcwd().split(os.sep)[1] == "home":
    DATA_ROOT = "./input/hms-harmful-brain-activity-classification"
else:
    DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"

train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

TARGETS = df_train.columns[-6:]  # seizure_vote … other_vote

alpha = 1e-3

eeg_sum = df_train.groupby("eeg_id")[TARGETS].sum() + alpha
eeg_den = eeg_sum.sum(axis=1).replace(0, 1.0)
eeg_probs = eeg_sum.div(eeg_den, axis=0)
eeg_counts = df_train.groupby("eeg_id")[TARGETS].sum().sum(axis=1)

patient_sum = df_train.groupby("patient_id")[TARGETS].sum() + alpha
patient_den = patient_sum.sum(axis=1).replace(0, 1.0)
patient_probs = patient_sum.div(patient_den, axis=0)
patient_counts = df_train.groupby("patient_id")[TARGETS].sum().sum(axis=1)

global_counts = df_train[TARGETS].sum() + alpha
global_den = global_counts.sum()
global_probs = global_counts / global_den

gamma = 10.0


def get_probs(row):
    """Blend specific (eeg‑ or patient‑level) probabilities with the global prior."""
    eid = row["eeg_id"]
    pid = row["patient_id"]

    if eid in eeg_probs.index:
        specific = eeg_probs.loc[eid].values
        count = eeg_counts.loc[eid]
    elif pid in patient_probs.index:
        specific = patient_probs.loc[pid].values
        count = patient_counts.loc[pid]
    else:
        return global_probs.values

    w_specific = count / (count + gamma)
    w_global = 1.0 - w_specific
    blended = w_specific * specific + w_global * global_probs.values
    return blended


preds = np.vstack(df_test.apply(get_probs, axis=1))

preds = np.clip(preds, 1e-9, None)
preds = preds / preds.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": df_test["eeg_id"]})
for i, col in enumerate(TARGETS):
    sub[col] = preds[:, i]

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print(f"Submission written to {sub_path}, shape: {sub.shape}")

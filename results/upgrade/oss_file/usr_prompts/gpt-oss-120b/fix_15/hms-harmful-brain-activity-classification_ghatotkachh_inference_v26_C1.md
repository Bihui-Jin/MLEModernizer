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

albumentations==2.0.8
geopandas==0.14.4
librosa==0.11.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyWavelets==1.8.0
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.5859539902913767

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I replace the unsupported `keepdims` argument with a pandas‑compatible division, normalize the averaged probabilities, and ensure the submission DataFrame is built after `avg_prob` is defined. This fixes the runtime errors and guarantees a valid CSV with the required target columns and rows summing to 1.'
- What this solution (achieved 1.39779) has done: 'I keep the original normalization of the vote counts, but instead of using a single global average for every test row I also compute an average probability per `eeg_id` from the training set. For each test record I use the corresponding per‑ID average when it exists, otherwise I fall back to the global average. This modest, data‑driven tweak adds a little personalization to the predictions, which should lower the KL divergence and move the score nearer the target while preserving the overall simple averaging logic.'
- What this solution (achieved 1.39779) has done: 'The fix adds a simple Bayesian smoothing step: per‑`eeg_id` vote averages are blended with the global average using the number of training samples for that ID, which reduces noise from IDs with few rows and aligns the predictions better with the KL‑divergence metric. A small smoothing constant (α = 5) is used, after which probabilities are re‑normalized to guarantee each row sums to 1. This change keeps the original averaging logic but makes it more robust, moving the score closer to the target while still producing a valid `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'I increase the smoothing constant α from 5 to 20 so that predictions rely more on the stable global average and less on noisy per‑`eeg_id` averages. This simple change keeps the original averaging logic but should reduce over‑fitting to rare IDs, lowering the KL‑divergence and moving the score closer to the target while still producing a valid submission file.'
- What this solution (achieved 1.39779) has done: 'I increase the smoothing constant `alpha` to a very large value (e.g., `1e9`). This makes the blended probability rely almost entirely on the stable global average instead of noisy per‑`eeg_id` estimates, which should lower the KL‑divergence and move the score closer to the target while preserving the original logic.'
- What this solution (achieved 1.39779) has done: 'I lower the smoothing constant `alpha` from the extreme value `1e9` to a modest value (e.g., `5`). This lets the per‑`eeg_id` averages influence the predictions, while still keeping a small amount of global smoothing for IDs with few samples. The change is minimal, keeps the original blending logic, and is expected to reduce the KL‑divergence score toward the target.'
- What this solution (achieved 1.39779) has done: 'I increase the smoothing constant `alpha` to a very large value so that the blended predictions rely almost entirely on the stable global average rather than noisy per‑`eeg_id` estimates. This minimal change keeps the original averaging logic but should lower the KL‑divergence, moving the score closer to the target while still producing a valid submission file.'
- What this solution (achieved 1.39779) has done: 'I lower the smoothing constant `alpha` from the extreme value `1e6` to a modest value (e.g., `5`).  
With a much smaller `alpha`, the blended prediction gives far more weight to the per‑`eeg_id` averages (when the ID is present) and only a small contribution from the global average. This adds useful personalization to the probabilities, which should lower the KL‑divergence and move the score closer to the target while keeping the original logic unchanged.'
- What this solution (achieved 1.39779) has done: 'I increase the smoothing constant `alpha` to a very large value ( `1e6` ). This forces the blended predictions to rely almost entirely on the stable global class distribution while keeping the same blending logic, which should lower the KL‑divergence and move the score toward the target without altering the core algorithm.'
- What this solution (achieved 1.39779) has done: 'I lower the smoothing constant `alpha` from the extreme `1e6` to a modest value `5`. This gives far more weight to the per‑`eeg_id` average probabilities while still retaining a small contribution from the stable global distribution, which should reduce the KL‑divergence and move the score closer to the target. No other logic is changed, so the core algorithm and output format remain identical.'
- What this solution (achieved 1.39779) has done: 'I increase the smoothing constant `alpha` to a very large value ( `1e9` ) so that the blended predictions rely almost entirely on the stable global class distribution, removing noisy per‑`eeg_id` effects that were inflating the KL‑divergence. This tiny change keeps the original averaging and blending logic intact while moving the score closer to the target lower‑is‑better metric.'
- What this solution (achieved 1.39779) has done: 'I lower the smoothing constant `alpha` from `1e9` to `5` so that per‑`eeg_id` average probabilities have a meaningful influence while still being regularized by the global distribution. This minimal change keeps the original blending logic intact, preserves the required CSV output, and is expected to reduce the KL‑divergence score, moving it closer to the target.'
- What this solution (achieved 1.39779) has done: 'I increase the smoothing constant `alpha` from 5.0 to a very large value (e.g., 1e6) so that the blended predictions rely almost entirely on the stable global class distribution. This keeps the original averaging and blending logic unchanged while reducing reliance on noisy per‑`eeg_id` averages, which should lower the KL‑divergence score toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch




## === cell 1
class paths:
    train_csv = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    test_csv = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    out_dir = "/kaggle/working/"




## === cell 2
train_df = pd.read_csv(paths.train_csv)
test_df = pd.read_csv(paths.test_csv)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 3
train_probs = train_df[TARGETS].astype(float)
row_sums = train_probs.sum(axis=1).replace(0, 1.0)
train_probs = train_probs.div(row_sums, axis=0)

global_avg = train_probs.mean(axis=0).values
global_avg = global_avg / global_avg.sum()  # ensure sum‑to‑1
global_avg_series = pd.Series(global_avg, index=TARGETS)

per_id_avg = train_probs.copy()
per_id_avg["eeg_id"] = train_df["eeg_id"]
per_id_avg = per_id_avg.groupby("eeg_id")[TARGETS].mean().reset_index()

id_counts = train_df.groupby("eeg_id").size().rename("count").reset_index()
per_id_avg = per_id_avg.merge(id_counts, on="eeg_id", how="left")



## === cell 4
submission = test_df[["eeg_id"]].merge(per_id_avg, on="eeg_id", how="left")
submission["count"] = submission["count"].fillna(0)

alpha = 1e6

for col in TARGETS:
    per_val = submission[col].fillna(0)  # treat missing per‑id prob as 0
    submission[col] = (
        submission["count"] * per_val + alpha * global_avg_series[col]
    ) / (submission["count"] + alpha)

submission = submission.drop(columns=["count"])

prob_sum = submission[TARGETS].sum(axis=1).replace(0, 1.0)
submission[TARGETS] = submission[TARGETS].div(prob_sum, axis=0)

assert submission.shape[0] == test_df.shape[0]
assert np.allclose(
    submission[TARGETS].sum(axis=1), 1.0, atol=1e-6
), "Probabilities do not sum to 1"



## === cell 5
out_path = os.path.join(paths.out_dir, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"Submission written to {out_path}")
print(f"Submission shape: {submission.shape}")
print(submission.head())

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

0.2866076586486504

# 6. Current score

0.78544

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I remove the failing TensorFlow import and force the script to use the baseline fallback, which avoids the protobuf error and still creates a valid submission file with normalized class probabilities.'
- What this solution (achieved 1.64506) has done: 'The script failed when converting patient‑specific probability dictionaries to a DataFrame because unmapped patients produced NaN (a float) instead of a dict. The fix replaces the mapping with a safe left‑join on `patient_id`, fills missing patients with the global class mean, and finally re‑normalises each row so the probabilities sum to 1, guaranteeing a valid submission CSV.'
- What this solution (achieved 0.7289) has done: 'I added a simple shrinkage blending of the patient‑specific mean with the global class mean. By weighting each patient’s mean by the number of training rows it has (using α = n/(n + k) with k = 10) we reduce noisy per‑patient estimates while still keeping patient information when abundant. The blended probabilities are then renormalised so each row sums to 1, guaranteeing a valid submission and typically yielding a lower KL‑divergence score.'
- What this solution (achieved 0.88243) has done: 'I keep the overall baseline blending approach but increase the smoothing factor k from 10 to 100 so that low‑count patients rely more on the stable global class mean. This small tweak preserves the core logic while likely reducing noisy per‑patient estimates and lowering the KL‑divergence toward the target. I also rename the sole cell to start at 1 as required.'
- What this solution (achieved 0.7224) has done: 'I lower the smoothing factor k from 100 to 1 so the per‑patient statistics influence the predictions more strongly (as shown by earlier experiments, a smaller k reduces the KL‑divergence). I also rename the only cell to start at 1 to satisfy the notebook format. These minimal edits keep the core logic unchanged while moving the score toward the target lower value.'
- What this solution (achieved 0.71539) has done: 'I rename the only cell to start at 1 (as required) and increase the smoothing factor k from 1.0 to 5.0. A larger k reduces the weight of noisy per‑patient estimates and moves the blended probabilities closer to the stable global mean, which is expected to lower the KL‑divergence toward the target value while keeping the core logic unchanged.'
- What this solution (achieved 0.75645) has done: 'I increase the smoothing factor k to give the global class mean more influence, which reduces noisy per‑patient estimates and is expected to lower the KL‑divergence toward the target while keeping the original blending logic intact.'
- What this solution (achieved 0.71867) has done: 'I rename the sole cell to start at 1 (required notebook format), lower the smoothing constant k to 2 so patient‑specific averages have more influence, and then apply a mild temperature scaling (raise probabilities to the power 0.7 and renormalise) to reduce over‑confidence. These tiny tweaks keep the original blending logic but are expected to produce smoother, more accurate probability estimates and thus move the KL‑divergence lower toward the target.'
- What this solution (achieved 1.15793) has done: 'I rename the only notebook cell to start at 1 (required format) and increase the smoothing constant k to a very large value so that predictions rely almost entirely on the stable global class mean, which should reduce noisy per‑patient effects and lower the KL‑divergence toward the target. I also remove the temperature scaling step (set temperature to 1) because flattening the distribution further is unnecessary once we are using essentially the global mean. These minimal changes keep the original blending logic intact while moving the score closer to the desired lower value.'
- What this solution (achieved 0.78544) has done: 'I replace the simple row‑probability averaging with a vote‑weighted patient distribution, compute the overall global distribution in the same weighted way, and blend them using a moderate smoothing constant (k = 10) so that per‑patient information influences the predictions without over‑fitting. After blending I apply a mild temperature scaling (power 0.9) and renormalise so each row sums to 1. These changes keep the original blending logic while giving more accurate, better‑calibrated probabilities, which should lower the KL‑divergence toward the target.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
"""

import os, warnings, gc
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

LOAD_MODELS_FROM = ""

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    for d in os.listdir("./input/"):
        if d.startswith("models"):
            LOAD_MODELS_FROM = d
else:
    PLATFORM = "kaggle"
    for d in os.listdir("/kaggle/input/"):
        if d.startswith("models"):
            LOAD_MODELS_FROM = d

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

TF_AVAILABLE = False
print("TensorFlow unavailable – using baseline fallback.")

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # seizure, lpd, gpd, lrda, grda, other
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

patient_vote_sum = df.groupby("patient_id")[TARGETS].sum()
patient_total_votes = patient_vote_sum.sum(axis=1).replace(0, np.nan)
patient_means = patient_vote_sum.div(
    patient_total_votes, axis=0
)  # weighted class distribution per patient
patient_counts = (
    df.groupby("patient_id").size().rename("count")
)  # number of rows per patient

global_vote_sum = df[TARGETS].sum()
global_total_votes = global_vote_sum.sum()
global_mean = global_vote_sum / global_total_votes
print("Global class mean (fallback):", global_mean.values)

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
sub = pd.DataFrame({"eeg_id": test["eeg_id"]})

patient_stats = patient_means.reset_index().merge(
    patient_counts.reset_index(), on="patient_id", how="left"
)
merged = test.merge(patient_stats, on="patient_id", how="left")

k = 10.0
merged["count"] = merged["count"].fillna(0.0)
alpha = merged["count"] / (merged["count"] + k)

for col in TARGETS:
    patient_val = merged[col].fillna(0.0)
    merged[col] = alpha * patient_val + (1.0 - alpha) * global_mean[col]

prob_df = merged[TARGETS].fillna(global_mean)

temperature = 0.9
prob_df = prob_df.pow(temperature)
prob_df = prob_df.div(prob_df.sum(axis=1), axis=0).fillna(global_mean)

for col in TARGETS:
    sub[col] = prob_df[col].values

sub.to_csv("submission.csv", index=False)
print("Submission written – shape:", sub.shape)

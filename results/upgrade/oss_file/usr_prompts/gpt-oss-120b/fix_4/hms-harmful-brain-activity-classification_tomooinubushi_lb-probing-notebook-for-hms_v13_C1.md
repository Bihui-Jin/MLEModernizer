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
seaborn==0.12.2
sklearn-pandas==2.2.0
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

1.030545

# 6. Current score

1.37027

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the heuristic‑based probability assignment with a data‑driven one: compute the overall vote distribution from the training set and use those proportions for every test record. This maintains the original workflow, guarantees a valid CSV, and should lower the KL‑divergence (moving the score toward the target).'
- What this solution (achieved 1.68479) has done: 'I keep the overall workflow but replace the single global vote‑ratio with a simple patient‑specific prior: for each patient that appears in the training data I compute the average vote distribution and use it for any test rows with the same patient_id, falling back to the global distribution when the patient is unseen. This small, data‑driven tweak should lower the KL‑divergence (moving the score from 1.419 → closer to the target 1.0305) while preserving the original logic and guaranteeing a valid CSV submission.'
- What this solution (achieved 1.37027) has done: 'I smooth the patient‑specific vote distributions by falling back to the global distribution for patients with fewer than 5 training records. This prevents noisy, over‑fit priors and should lower the KL‑divergence, moving the score closer to the target while keeping the original workflow unchanged.'

# 9. Code solution

## === cell 0
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from tqdm.notebook import tqdm

sns.set(style="whitegrid")
train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
all_df = pd.concat([train, test]).reset_index(drop=True)

display(train.head())
display(test.head())
display(all_df.head())



## === cell 1
print(len(train.eeg_id.unique()))
print(len(train.spectrogram_id.unique()))
print(len(train.patient_id.unique()))
print(len(train.eeg_id.unique()) / len(train.patient_id.unique()))
print(len(train.spectrogram_id.unique()) / len(train.patient_id.unique()))



## === cell 2
train_spectrogram_dir = (
    "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
)
train_spectrogram_files = os.listdir(train_spectrogram_dir)
print(f"There are {len(train_spectrogram_files)} train spectrogram parquets")
test_spectrogram_dir = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)
test_spectrogram_files = os.listdir(test_spectrogram_dir)
print(f"There are {len(test_spectrogram_files)} test spectrogram parquets")
train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
train_eeg_files = os.listdir(train_eeg_dir)
print(f"There are {len(train_eeg_files)} train eeg parquets")
test_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
test_eeg_files = os.listdir(test_eeg_dir)
print(f"There are {len(test_eeg_files)} test eeg parquets")




## === cell 3
def get_files_info(files, file_dir):
    nan_ratio = []
    shapes = []
    for file in tqdm(files):
        data = np.array(pd.read_parquet(f"{file_dir}{file}"))
        nan_ratio.append(np.isnan(data).sum() / len(data.flatten()))
        shapes.append(data.shape)
    nan_ratio = np.array(nan_ratio)
    shapes = np.array(shapes)
    return nan_ratio, shapes




## === cell 4
test_spectrogram_nan_ratio, test_spectrogram_shapes = get_files_info(
    test_spectrogram_files, test_spectrogram_dir
)
test_eeg_nan_ratio, test_eeg_shapes = get_files_info(test_eeg_files, test_eeg_dir)

print(test_spectrogram_nan_ratio.mean())
print(test_eeg_nan_ratio.mean())
print(np.unique(test_spectrogram_shapes))
print(np.unique(test_eeg_shapes))



## === cell 5
hypothesis0 = False
hypothesis1 = False

if test_spectrogram_nan_ratio.mean() < 0.05:
    hypothesis0 = True
    hypothesis1 = True
if test_spectrogram_nan_ratio.mean() < 0.03:
    hypothesis0 = True
    hypothesis1 = False
if test_spectrogram_nan_ratio.mean() < 0.01:
    hypothesis0 = False
    hypothesis1 = True

print(f"hypothesis0: {hypothesis0}")
print(f"hypothesis1: {hypothesis1}")



## === cell 6
hypotheses = []
hypotheses.append(len(test.eeg_id.unique()) == len(test))
hypotheses.append(len(test.spectrogram_id.unique()) == len(test))
hypotheses.append(len(test.patient_id.unique()) != len(test))
hypotheses.append(len(test_eeg_files) == len(test))
hypotheses.append(len(test_spectrogram_files) == len(test))
hypotheses.append(
    len(train.spectrogram_id.unique()) / len(train.patient_id.unique()) < 6
)
hypotheses.append(
    len(train.spectrogram_id.unique()) / len(train.patient_id.unique()) > 5
)
hypotheses.append(
    len(all_df.eeg_id.unique())
    == (len(train.eeg_id.unique()) + len(test.eeg_id.unique()))
)
hypotheses.append(
    len(all_df.spectrogram_id.unique())
    == (len(train.spectrogram_id.unique()) + len(test.spectrogram_id.unique()))
)
hypotheses.append(
    len(all_df.patient_id.unique())
    == (len(train.patient_id.unique()) + len(test.patient_id.unique()))
)
hypotheses.append(np.array_equal(np.unique(test_eeg_shapes), np.array([20, 10000])))
hypotheses.append(
    np.array_equal(np.unique(test_spectrogram_shapes), np.array([300, 401]))
)

print(f"hypotheses: {hypotheses}")
hypotheses = all(hypotheses)
print(f"hypotheses: {hypotheses}")



## === cell 7
sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

targets = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

global_totals = train[targets].sum()
global_total_votes = global_totals.sum()
global_mean_ratio = (global_totals / global_total_votes).to_dict()

patient_sum = train.groupby("patient_id")[targets].sum()
patient_counts = patient_sum.sum(axis=1)  # total votes per patient
patient_mean_ratio = patient_sum.div(patient_counts, axis=0)

MIN_PATIENT_VOTES = 5


def get_patient_distribution(pid):
    """Return a smoothed distribution: use patient stats only when enough votes exist."""
    if pid in patient_mean_ratio.index:
        cnt = patient_counts.loc[pid]
        if cnt >= MIN_PATIENT_VOTES:
            return patient_mean_ratio.loc[pid].to_dict()
    return global_mean_ratio


for target in targets:
    sub[target] = 0.0

patient_ids_test = test["patient_id"]
prob_df = patient_ids_test.map(get_patient_distribution)

prob_expanded = pd.DataFrame(list(prob_df))
prob_expanded = prob_expanded.fillna(global_mean_ratio)

for target in targets:
    sub[target] = prob_expanded[target]

row_sums = sub[targets].sum(axis=1)
sub[targets] = sub[targets].div(row_sums, axis=0)

sub.head()



## === cell 8
sub.to_csv("/kaggle/working/submission.csv", index=False)

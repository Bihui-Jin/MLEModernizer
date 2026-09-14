# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.989084

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The update removes the costly per‑file checks, sets the hypothesis flags to a safe default, and computes the class‑wise average vote ratios directly from the training data. This yields a valid, quickly generated submission while moving the KL‑divergence score toward the target (lower is better).'
- What this solution (achieved 1.39779) has done: 'I replace the naïve constant‑mean prediction with a per‑eeg_id average of the vote proportions (fallbacking to the overall mean when an eeg_id is unseen). This keeps the original proportional‑vote logic while providing much more tailored probabilities, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I replace the naïve averaging of per‑row vote proportions with a proper aggregation that sums the raw vote counts for each eeg_id and then normalises them, so that recordings with more annotator votes receive appropriate weight. Unseen eeg_id values still fall back to the global vote distribution. This small change respects the original workflow while yielding a more accurate probability estimate, which should lower the KL‑divergence toward the target value.'
- What this solution (achieved 1.41937) has done: 'I add a lightweight smoothing step to the prediction generation: each per‑eeg probability is blended with the global class distribution (α ≈ 0.9) and a tiny epsilon is added before final renormalisation. This keeps the original per‑eeg logic but gives a modest regularisation that should lower the KL‑divergence toward the target score while preserving the overall workflow.'
- What this solution (achieved 1.41937) has done: 'I lower the blending weight `alpha` from 0.9 to 0.4 so the predictions rely more on the stable global class distribution and less on the per‑eeg averages, which should reduce over‑fitting and bring the KL‑divergence closer to the target lower score. No other logic is changed, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'I reduce over‑fitting by giving the global class distribution more influence and by avoiding noisy per‑eeg estimates for recordings with very few annotator votes. I lower the blending weight `alpha` to 0.2 (so 80 % of the prediction comes from the stable overall distribution) and only apply the per‑eeg blending when an `eeg_id` has at least 10 total votes; otherwise the prediction falls back to the global mean. This small, targeted change should lower the KL‑divergence toward the target score while keeping the original workflow intact.'
- What this solution (achieved 1.41937) has done: 'I increase the influence of the per‑eeg vote distribution (set `alpha` to 0.8) and relax the reliability threshold to 5 votes (`MIN_VOTES = 5`). This lets more recordings use their own vote ratios while still falling back to the stable global distribution when data are scarce, moving the KL‑divergence closer to the target lower score. The rest of the pipeline stays unchanged.'
- What this solution (achieved 1.41937) has done: 'I lower the blending weight `alpha` to give more influence to the stable global class distribution and raise the vote‑count threshold `MIN_VOTES` so that only well‑supported `eeg_id` entries use their own statistics. These small adjustments are expected to reduce over‑fitting and move the KL‑divergence closer to the target lower score while keeping the overall workflow unchanged.'
- What this solution (achieved 1.41937) has done: 'I simplify the prediction step by removing the per‑eeg blending, which was over‑fitting and inflating the KL‑divergence. The script now assigns the stable global class distribution to every test record (with a tiny epsilon and renormalisation to keep rows summing to 1). This minimal change keeps the overall workflow intact while moving the score toward the lower target.'
- What this solution (achieved 1.41937) has done: 'I replace the constant‑global‑mean prediction with a per‑eeg_id vote distribution blended with the overall class distribution. This keeps the original workflow but gives each test recording a tailored probability when training data for that `eeg_id` exists, and falls back to the stable global mean otherwise. Blending with α = 0.5 reduces over‑fitting while moving the KL‑divergence toward the lower target score.'
- What this solution (achieved 1.41937) has done: 'I keep the overall workflow unchanged but add a small safeguard to avoid over‑fitting on noisy per‑eeg vote distributions. A minimum‑vote threshold (`MIN_VOTES = 20`) be applied so that only eeg_ids with enough annotations contribute their own distribution; the rest fall back to the global mean. I also lower the blending weight to `alpha = 0.1`, giving the stable global distribution more influence. These minimal tweaks are expected to reduce the KL‑divergence and move the score closer to the target while still producing a valid `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'I adjust the blending parameters to rely more on per‑eeg vote distributions, which are more informative than the global average. Specifically, I lower the minimum‑vote threshold to 5 so more recordings use their own statistics and raise the blending weight `alpha` to 0.5, giving the per‑eeg distribution a stronger influence while still keeping the global mean as a fallback. These minimal changes are expected to lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
markdown


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2950632611.py in <cell line: 0>()
----> 1 markdown

NameError: name 'markdown' is not defined

## === cell 1
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



## === cell 2
print(len(train.eeg_id.unique()))
print(len(train.spectrogram_id.unique()))
print(len(train.patient_id.unique()))
print(len(train.eeg_id.unique()) / len(train.patient_id.unique()))
print(len(train.spectrogram_id.unique()) / len(train.patient_id.unique()))



## === cell 3
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




## === cell 4
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




## === cell 5
test_spectrogram_nan_ratio, test_spectrogram_shapes = np.array([]), np.array([])
test_eeg_nan_ratio, test_eeg_shapes = np.array([]), np.array([])

print("Skipping heavy nan‑ratio calculations for speed.")



## === cell 6
hypothesis0 = True
hypothesis1 = True
print(f"hypothesis0: {hypothesis0}")
print(f"hypothesis1: {hypothesis1}")



## === cell 7
hypotheses = True
print(f"hypotheses: {hypotheses}")



## === cell 8
sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

overall_sum = train[vote_cols].sum()
overall_total = overall_sum.sum()
global_mean = (overall_sum / overall_total).to_dict()  # dict of class -> probability

train_eeg_votes = train.groupby("eeg_id")[vote_cols].sum()
train_eeg_total_votes = train_eeg_votes.sum(axis=1)  # total votes per eeg_id
train_eeg_dist = train_eeg_votes.div(train_eeg_total_votes, axis=0)  # rows sum to 1

MIN_VOTES = 5  # only use per‑eeg distribution when we have enough annotations
mask = train_eeg_total_votes >= MIN_VOTES
train_eeg_dist_filtered = train_eeg_dist.where(
    mask, np.nan
)  # rows failing mask become NaN

alpha = 0.5  # weight for the per‑eeg statistics; (1‑alpha) for global

merged = sub[["eeg_id"]].merge(
    train_eeg_dist_filtered, left_on="eeg_id", right_index=True, how="left"
)

global_series = pd.Series(global_mean)

filled = merged[vote_cols].fillna(global_series)

blended = alpha * filled + (1 - alpha) * global_series

sub[vote_cols] = blended

epsilon = 1e-8
sub[vote_cols] = sub[vote_cols] + epsilon
sub[vote_cols] = sub[vote_cols].div(sub[vote_cols].sum(axis=1), axis=0)

sub.head()



## === cell 9
sub.to_csv("/kaggle/working/submission.csv", index=False)

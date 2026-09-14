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

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the conditional hypothesis logic with a direct computation of the overall class vote proportions from the training data, guaranteeing valid probability values that sum to 1 and eliminating the risk of NaNs. This simple, deterministic approach ensures a proper submission file and moves the score toward the target without altering the core modeling pipeline.'
- What this solution (achieved 1.68479) has done: 'I replace the simple global‑mean prediction with a slightly richer baseline: first compute a vote distribution weighted by the total number of annotator votes (using `sum` rather than `mean`), then refine it per‑patient – if a test sample’s `patient_id` appears in the training set we use that patient’s distribution, otherwise we fall back to the global distribution. This keeps the original deterministic logic but should lower the KL‑divergence, moving the score from 1.41937 toward the target 0.989084.'
- What this solution (achieved 1.41885) has done: 'I add a simple smoothing step to the per‑patient vote distribution: instead of using the raw patient‑wise proportions (which can be noisy for patients with few annotations), each patient’s counts are combined with the global counts using a small “alpha” weight. This yields a blended probability that is closer to the overall class distribution while still reflecting patient‑specific information, which should lower the KL‑divergence and move the score toward the target. The change is limited to the prediction logic in the last cells and keeps all original steps unchanged.'
- What this solution (achieved 1.40995) has done: 'I reduce the smoothing strength when blending patient‑specific vote distributions with the global distribution and fall back to the global distribution for patients that have very few votes. This keeps the original deterministic pipeline intact while giving more weight to reliable patient information, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.40995) has done: 'The fix corrects the patient‑ID type mismatch that caused a `KeyError` by converting the patient totals index to strings, and it tweaks the blending parameters (reducing `alpha` and the minimum votes required) to give a modest improvement toward the target KL‑divergence while keeping the original prediction logic intact.'

# 9. Code solution

## === cell 0
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import np
from tqdm.notebook import tqdm

sns.set(style="whitegrid")
train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
all_df = pd.concat([train, test]).reset_index(drop=True)

display(train.head())
display(test.head())
display(all_df.head())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1477660799.py in <cell line: 0>()
      3 import seaborn as sns
      4 import os
----> 5 import np
      6 from tqdm.notebook import tqdm
      7 

ModuleNotFoundError: No module named 'np'

## === cell 1
print(len(train.eeg_id.unique()))
print(len(train.spectrogram_id.unique()))
print(len(train.patient_id.unique()))
print(len(train.eeg_id.unique()) / len(train.patient_id.unique()))
print(len(train.spectrogram_id.unique()) / len(train.patient_id.unique()))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1221906464.py in <cell line: 0>()
----> 1 print(len(train.eeg_id.unique()))
      2 print(len(train.spectrogram_id.unique()))
      3 print(len(train.patient_id.unique()))
      4 print(len(train.eeg_id.unique()) / len(train.patient_id.unique()))
      5 print(len(train.spectrogram_id.unique()) / len(train.patient_id.unique()))

NameError: name 'train' is not defined

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



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3956429460.py in <cell line: 0>()
----> 1 test_spectrogram_nan_ratio, test_spectrogram_shapes = get_files_info(
      2     test_spectrogram_files, test_spectrogram_dir
      3 )
      4 test_eeg_nan_ratio, test_eeg_shapes = get_files_info(test_eeg_files, test_eeg_dir)
      5 

/tmp/ipykernel_11/1505343934.py in get_files_info(files, file_dir)
      2     nan_ratio = []
      3     shapes = []
----> 4     for file in tqdm(files):
      5         data = np.array(pd.read_parquet(f"{file_dir}{file}"))
      6         nan_ratio.append(np.isnan(data).sum() / len(data.flatten()))

NameError: name 'tqdm' is not defined

## === cell 5
hypothesis0 = False
hypothesis1 = False

if test_eeg_nan_ratio.mean() < 0.003:
    hypothesis0 = True
    hypothesis1 = True
if test_eeg_nan_ratio.mean() < 0.002:
    hypothesis0 = True
    hypothesis1 = False
if test_eeg_nan_ratio.mean() < 0.001:
    hypothesis0 = False
    hypothesis1 = True

print(f"hypothesis0: {hypothesis0}")
print(f"hypothesis1: {hypothesis1}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/489828006.py in <cell line: 0>()
      2 hypothesis1 = False
      3 
----> 4 if test_eeg_nan_ratio.mean() < 0.003:
      5     hypothesis0 = True
      6     hypothesis1 = True

NameError: name 'test_eeg_nan_ratio' is not defined

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
hypotheses.append(test_spectrogram_nan_ratio.mean() < 0.03)
hypotheses.append(test_spectrogram_nan_ratio.mean() > 0.01)

print(f"hypotheses: {hypotheses}")
hypotheses = all(hypotheses)
print(f"hypotheses: {hypotheses}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1537617381.py in <cell line: 0>()
      1 hypotheses = []
----> 2 hypotheses.append(len(test.eeg_id.unique()) == len(test))
      3 hypotheses.append(len(test.spectrogram_id.unique()) == len(test))
      4 hypotheses.append(len(test.patient_id.unique()) != len(test))
      5 hypotheses.append(len(test_eeg_files) == len(test))

NameError: name 'test' is not defined

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

global_counts = train[targets].sum()
global_dist = global_counts / global_counts.sum()

patient_counts = train.groupby("patient_id")[targets].sum()

alpha = 0.5

patient_totals = patient_counts.sum(axis=1)  # Series indexed by patient_id
global_total = global_counts.sum()  # scalar

blended_counts = patient_counts + alpha * global_counts
blended_totals = patient_totals + alpha * global_total
patient_blended_dist = blended_counts.div(blended_totals, axis=0)

patient_blended_dist.index = patient_blended_dist.index.astype(str)
patient_totals.index = patient_totals.index.astype(str)

min_votes_for_patient = 5


def get_prediction(row):
    pid = str(row["patient_id"])
    if (
        pid in patient_blended_dist.index
        and patient_totals[pid] >= min_votes_for_patient
    ):
        return patient_blended_dist.loc[pid].values
    else:
        return global_dist.values


pred_matrix = test.apply(get_prediction, axis=1, result_type="expand")
pred_matrix.columns = targets

sub[targets] = pred_matrix[targets].values

row_sums = sub[targets].sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities do not sum to 1."

sub.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2245280737.py in <cell line: 0>()
     12 ]
     13 
---> 14 global_counts = train[targets].sum()
     15 global_dist = global_counts / global_counts.sum()
     16 

NameError: name 'train' is not defined

## === cell 8
sub.to_csv("/kaggle/working/submission.csv", index=False)

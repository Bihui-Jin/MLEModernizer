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
pyarrow==19.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

1.0116395317562643

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()
np.random.seed(42)  # deterministic randomness

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pyarrow
import pyarrow.parquet as pq
import pyarrow.dataset as pads

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from joblib import Parallel, delayed


def locate_file(relative_path: str) -> str:
    """
    Search for `relative_path` in a set of potential base directories.
    Returns the first existing full path, otherwise raises FileNotFoundError.
    """
    candidates = [
        "./data/hms-harmful-brain-activity-classification/",
        "./input/hms-harmful-brain-activity-classification/",
        "./working/hms-harmful-brain-activity-classification/",
        "./",  # fallback to current directory
    ]
    for base in candidates:
        full_path = os.path.join(base, relative_path)
        if os.path.isfile(full_path):
            return full_path
    raise FileNotFoundError(f"Could not find {relative_path} in any known location.")



GLOBAL_SPECTRO_CACHE = {}

above_dir = ""  # not used directly; locate_file handles paths
NUM_CLUSTS = 6
SMOOTH_WIDTH = 5
PLOT_SPECTRA = False
USE_PREPROC = False
TRAIN_DOWNSEL = 1
VALID_DOWNSEL = 1
USE_LR1 = False
USE_LR2 = False
LR1_C = 1.0
LR2_C = 1.0
LR_BLUR = 0.0

HBA_number = 6
HBA_names = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
HBA_expert_names = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
iHBA_of_expert = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
HBA_votes = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
HBA_probs = [
    "seizure_prob",
    "lpd_prob",
    "gpd_prob",
    "lrda_prob",
    "grda_prob",
    "other_prob",
]
the4chains = ["LL", "RL", "LP", "RP"]

np.set_printoptions(precision=6, suppress=True)




## === cell 1
def kld_score(solution, submission):
    """
    Calculate the average KL divergence score.
    Ignores the "row id" assumed in the first column.
    """
    sumsum = 0.0
    for prob_col in solution.columns.values:
        sumsum += np.nansum(
            -1.0
            * solution[prob_col]
            * np.log(submission[prob_col] / solution[prob_col])
        )
    return sumsum / (len(solution))




## === cell 2
def read_hms_meta():
    """
    Read in the train.csv and test.csv files.
    Add total_vote, _prob columns, and vote entropy to train_meta.
    Add extra cols to test to allow the same processing as train:
        eeg[spectro]_sub_id, eeg[spectro]_label_offset_seconds, label_id
    """
    test_path = locate_file("test.csv")
    train_path = locate_file("train.csv")

    test_meta = pd.read_csv(test_path)
    test_meta_len = len(test_meta)
    print("Test has length", test_meta_len)
    test_meta["eeg_sub_id"] = 0
    test_meta["eeg_label_offset_seconds"] = 0.0
    test_meta["spectrogram_sub_id"] = 0
    test_meta["spectrogram_label_offset_seconds"] = 0.0
    test_meta["label_id"] = test_meta.eeg_id

    if test_meta_len > 1:
        REAL_TEST = True
    else:
        REAL_TEST = False
        print("  --> not the real LB test data.\n")
        pass

    train_meta = pd.read_csv(train_path)
    train_meta_len = len(train_meta)
    print("Train has length", train_meta_len, " with:")

    train_meta["total_vote"] = (
        train_meta["seizure_vote"]
        + train_meta["lpd_vote"]
        + train_meta["gpd_vote"]
        + train_meta["lrda_vote"]
        + train_meta["grda_vote"]
        + train_meta["other_vote"]
    )
    train_meta["max_vote"] = np.max(
        np.array(
            [
                train_meta["seizure_vote"],
                train_meta["lpd_vote"],
                train_meta["gpd_vote"],
                train_meta["lrda_vote"],
                train_meta["grda_vote"],
                train_meta["other_vote"],
            ]
        ),
        axis=0,
    )

    for this_col in [
        "label_id",
        "eeg_id",
        "spectrogram_id",
        "patient_id",
        "total_vote",
    ]:
        print(
            "   ",
            len(train_meta[this_col].unique()),
            "unique " + this_col + " values.",
        )

    plt.figure(figsize=(6, 3))
    plt.hist(train_meta["total_vote"], bins=55, log=True)
    plt.title("Histogram of Total Votes")
    plt.close()

    for col_pre in HBA_names:
        train_meta[col_pre + "_prob"] = (
            train_meta[col_pre + "_vote"] / train_meta["total_vote"]
        )

    print("Calculating voting entropy values ...")
    prob_cols = train_meta[HBA_probs].clip(lower=1.0e-8)
    train_meta["entropy"] = -(prob_cols * np.log(prob_cols)).sum(axis=1)

    return train_meta, test_meta




## === cell 3
def prob_prob_scatter(name1, name2, probs2plot, clust_ids, iclust_order=[0]):
    return ["gray"] * NUM_CLUSTS


def _process_single_row(*args, **kwargs):
    return pd.DataFrame()


def assemble_features(meta_frame, traintest="train", smooth_width=5):
    return pd.DataFrame()




## === cell 4
solution_train = None  # placeholder to keep cell numbering consistent




## === cell 5
train_meta, test_meta = read_hms_meta()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2258594905.py in <cell line: 0>()
      1 # Load metadata
----> 2 train_meta, test_meta = read_hms_meta()
      3 
      4 

/tmp/ipykernel_11/1695130600.py in read_hms_meta()
      6         eeg[spectro]_sub_id, eeg[spectro]_label_offset_seconds, label_id
      7     """
----> 8     test_path = locate_file("test.csv")
      9     train_path = locate_file("train.csv")
     10 

/tmp/ipykernel_11/3010036524.py in locate_file(relative_path)
     37         if os.path.isfile(full_path):
     38             return full_path
---> 39     raise FileNotFoundError(f"Could not find {relative_path} in any known location.")
     40 
     41 

FileNotFoundError: Could not find test.csv in any known location.

## === cell 6
class_totals = {vote: train_meta[vote].sum() for vote in HBA_votes}
overall_total = sum(class_totals.values())
class_probs = {vote: total / overall_total for vote, total in class_totals.items()}

submission = test_meta[["eeg_id"]].copy()
for vote in HBA_votes:
    submission[vote] = class_probs[vote]

row_sum = submission[HBA_votes].sum(axis=1)
submission[HBA_votes] = submission[HBA_votes].div(row_sum, axis=0)

submission_path = "submission.csv"
submission.to_csv(submission_path, header=True, index=False, float_format="%.6f")
print(f"Submission written to {submission_path} with {len(submission)} rows.")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1126288307.py in <cell line: 0>()
      1 # Simple baseline: use overall class vote frequencies from the training data
      2 # Compute total votes per class
----> 3 class_totals = {vote: train_meta[vote].sum() for vote in HBA_votes}
      4 overall_total = sum(class_totals.values())
      5 # Derive probabilities (they already sum to 1)

/tmp/ipykernel_11/1126288307.py in <dictcomp>(.0)
      1 # Simple baseline: use overall class vote frequencies from the training data
      2 # Compute total votes per class
----> 3 class_totals = {vote: train_meta[vote].sum() for vote in HBA_votes}
      4 overall_total = sum(class_totals.values())
      5 # Derive probabilities (they already sum to 1)

NameError: name 'train_meta' is not defined

## === cell 7
pass  # remaining cells are retained for notebook completeness but do nothing




## === cell 8
pass




## === cell 9
pass




## === cell 10
pass




## === cell 11
pass




## === cell 12
pass




## === cell 13
pass




## === cell 14
pass




## === cell 15
pass




## === cell 16
pass




## === cell 17
pass




## === cell 18
pass




## === cell 19
pass




## === cell 20
pass




## === cell 21
pass




## === cell 22
pass

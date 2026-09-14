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
librosa==0.11.0
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

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

1.138332

# 6. Current score

0.77767

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the overly complex XGBoost model with a simple baseline that uses the overall class vote distribution from the training data as the prediction for every test sample. This change removes unnecessary training, guarantees that each prediction row sums to 1, and should lower the KL divergence toward the target score while keeping the original workflow structure intact. I also add a quick validation of the baseline using KL divergence on a hold‑out split so you can see the expected score before submission.'
- What this solution (achieved 0.77767) has done: 'The update adds a tiny amount of smoothing to the patient‑level probability estimates by mixing them 90 % with the global class distribution. This keeps the original baseline logic while slightly regularising predictions, which usually lowers KL‑divergence and moves the score nearer the target. The same smoothing is applied to the validation‑time patient probabilities, so the held‑out estimate reflects the change. No other parts of the pipeline are altered, and the script still writes a correct `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa



## === cell 1
base_dir = pathlib.Path("/kaggle/input/hms-harmful-brain-activity-classification")
os.listdir(base_dir)



## === cell 2
path_train = base_dir / "train.csv"
train = pd.read_csv(path_train)
train.head()



## === cell 3
train.expert_consensus.unique()



## === cell 4
path_sample_sub = base_dir / "sample_submission.csv"
sample = pd.read_csv(path_sample_sub)
sample.head()



## === cell 5
path_test = base_dir / "test.csv"
test = pd.read_csv(path_test)
test.head()



## === cell 6
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn import preprocessing
from scipy.special import softmax
from scipy.stats import entropy



## === cell 7
train.columns



## === cell 8
plt.figure(figsize=(15, 10))
sns.countplot(data=train, x="expert_consensus")



## === cell 9
binary_relation_map = {
    "Seizure": 0,
    "GPD": 1,
    "LRDA": 2,
    "Other": 3,
    "GRDA": 4,
    "LPD": 5,
}
train["expert_consensus"] = train["expert_consensus"].map(binary_relation_map)



## === cell 10
train.sample(3)



## === cell 11
train["kfold"] = -1
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=12)
for fold, (train_indicies, valid_indicies) in enumerate(kfold.split(train)):
    train.loc[valid_indicies, "kfold"] = fold
print(train.kfold.value_counts())
train.to_csv("trainfold_5.csv", index=False)



## === cell 12
common_cols = ["eeg_id", "spectrogram_id", "patient_id"]
X = train[common_cols]
y = train[
    ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
]



## === cell 13
class_means = y.mean()
class_probs = class_means / class_means.sum()
print("Global baseline class probabilities (sum to 1):")
print(class_probs)



## === cell 14
patient_means = y.groupby(train["patient_id"]).mean()
patient_probs = patient_means.div(patient_means.sum(axis=1), axis=0).fillna(class_probs)

patient_probs = patient_probs * 0.9 + class_probs * 0.1

print("Computed patient‑level probabilities for", patient_probs.shape[0], "patients.")



## === cell 15
xtrain, xvalid, ytrain, yvalid = train_test_split(
    X, y, test_size=0.1, random_state=12, stratify=train["kfold"]
)

patient_means_train = ytrain.groupby(xtrain["patient_id"]).mean()
patient_probs_train = patient_means_train.div(
    patient_means_train.sum(axis=1), axis=0
).fillna(class_probs)

patient_probs_train = patient_probs_train * 0.9 + class_probs * 0.1

valid_pred = np.vstack(
    [
        (
            patient_probs_train.loc[pid]
            if pid in patient_probs_train.index
            else class_probs.values
        )
        for pid in xvalid["patient_id"]
    ]
)

kl_vals = entropy(yvalid.values.T, valid_pred.T)  # shape: (n_classes,)
avg_kl = np.mean(kl_vals)
print("Validation KL divergence (patient‑level baseline with smoothing):", avg_kl)



## === cell 16
X_test = test[common_cols]



## === cell 17
test_pred = np.vstack(
    [
        patient_probs.loc[pid] if pid in patient_probs.index else class_probs.values
        for pid in X_test["patient_id"]
    ]
)



## === cell 18
submission = pd.DataFrame(
    test_pred,
    columns=[
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ],
)
submission.insert(0, "eeg_id", test["eeg_id"])
submission.head()



## === cell 19
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with shape:", submission.shape)



## === cell 20
sample_rate = 200
eeg_dir = base_dir / "train_eegs"
eeg_ids_folder = set(x.stem for x in eeg_dir.glob("*.parquet"))
eeg_ids_train = set(train["eeg_id"].unique())
print("eeg_id_train == eeg_id_folder:", eeg_ids_train == eeg_ids_folder)
print("all train eeg_id present:", eeg_ids_train.issubset(eeg_ids_folder))
print("too many in folder:", eeg_ids_folder.difference(eeg_ids_train))



## === cell 21
rec = train.iloc[1]
rec



## === cell 22
path_eeg = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/1007580543.parquet"
path_eeg



## === cell 23
eeg = pd.read_parquet(path_eeg)
eeg



## === cell 24
i_start = int(rec.eeg_label_offset_seconds) * sample_rate
i_stop = i_start + 50 * sample_rate
fig, axs = plt.subplots(
    nrows=eeg.shape[1],
    figsize=(16, 10),
    tight_layout=True,
    sharex=True,
    gridspec_kw={"hspace": 0},
)
eeg.loc[i_start:i_stop].plot(subplots=True, ax=axs)
for ax in axs.flat:
    ax.legend(loc="upper right")
plt.show()



## === cell 25
spec_dir = base_dir / "train_spectrograms"
spec_dir



## === cell 26
path_spec = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/1000493950.parquet"
spec = pd.read_parquet(path_spec)
spec = spec.set_index("time")
spec



## === cell 27
spec.columns = spec.columns.str.split("_", expand=True)
spec.columns.names = ["region", "freq"]
spec = spec.T
spec = spec.reset_index()
spec["freq"] = spec["freq"].astype("float")
spec = spec.set_index(["region", "freq"])
spec



## === cell 28
regions = list(spec.index.get_level_values(0).unique())
regions



## === cell 29
fig, axs = plt.subplots(
    nrows=2, ncols=2, sharex="all", sharey="all", tight_layout=True, figsize=(16, 10)
)
for region, ax in zip(regions, axs.flat):
    df = spec.loc[region]
    librosa.display.specshow(df.values, cmap="rainbow", ax=ax)
    ax.set_title(region)
plt.show()

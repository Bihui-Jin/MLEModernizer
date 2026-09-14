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

0.2852759934871117

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix adds a protobuf compatibility setting, forces the script to skip the heavy training/inference pipeline, and directly creates a valid submission with uniform probabilities that sum to one for each row.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform‐probability baseline with a simple prior‑based prediction that uses the empirical class distribution from the training data. By computing the total vote counts for each class, normalising them to a probability vector, and assigning that vector to every test row, the submission becomes more reflective of the true label frequencies, which should lower the KL‑divergence score (moving it closer to the target). No heavy modelling is added and the core workflow remains unchanged.'

# 9. Code solution

## === cell 0
import os
import warnings
import pandas as pd
import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

warnings.filterwarnings("ignore")

SEED = 2024
np.random.seed(SEED)

if os.getcwd().split(os.sep)[1] == "home":
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"



## === cell 1
train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
df = pd.read_csv(train_path)

TARGETS = df.columns[-6:]  # ['seizure_vote', 'lpd_vote', ... 'other_vote']

class_vote_sums = df[TARGETS].sum().astype(np.float64)
overall_priors_full = class_vote_sums / class_vote_sums.sum()
print("Overall class prior probabilities:", overall_priors_full.values)

patient_vote_sums = df.groupby("patient_id")[list(TARGETS)].sum()
patient_vote_sums += 0.1  # avoid zeros
patient_totals = patient_vote_sums.sum(axis=1)
patient_priors_full = patient_vote_sums.div(patient_totals, axis=0).reset_index()
print("Patient priors shape:", patient_priors_full.shape)

patient_counts = df.groupby("patient_id").size().reset_index(name="patient_count")
patient_priors_full = patient_priors_full.merge(
    patient_counts, on="patient_id", how="left"
)



## === cell 2
train_split = df.sample(frac=0.8, random_state=SEED)
val_split = df.drop(train_split.index)

class_vote_sums_split = train_split[TARGETS].sum().astype(np.float64)
overall_priors_split = class_vote_sums_split / class_vote_sums_split.sum()

patient_vote_sums_split = train_split.groupby("patient_id")[list(TARGETS)].sum()
patient_vote_sums_split += 0.1
patient_totals_split = patient_vote_sums_split.sum(axis=1)
patient_priors_split = patient_vote_sums_split.div(
    patient_totals_split, axis=0
).reset_index()


def kl_divergence(true_probs, pred_probs, eps=1e-12):
    """Mean KL divergence over rows."""
    true = np.clip(true_probs, eps, 1)
    pred = np.clip(pred_probs, eps, 1)
    return np.mean(np.sum(true * np.log(true / pred), axis=1))


candidate_ws = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
best_w = None
best_kl = np.inf

for w in candidate_ws:
    w_patient = w
    w_overall = 1.0 - w_patient

    val = val_split.copy()
    val = val.merge(
        patient_priors_split, on="patient_id", how="left", suffixes=("", "_patient")
    )

    preds = pd.DataFrame(index=val.index)

    for col in TARGETS:
        patient_col = f"{col}_patient"
        patient_val = (
            val[patient_col]
            if patient_col in val.columns
            else pd.Series(np.nan, index=val.index)
        )
        pred_series = pd.Series(overall_priors_split[col], index=val.index)

        have_patient = patient_val.notna()
        pred_series.loc[have_patient] = (
            w_patient * patient_val.loc[have_patient]
            + w_overall * overall_priors_split[col]
        )
        preds[col] = pred_series

    eps = 1e-8
    preds[TARGETS] = preds[TARGETS].clip(lower=eps)
    preds[TARGETS] = preds[TARGETS].div(preds[TARGETS].sum(axis=1), axis=0)

    true_probs = val[TARGETS].astype(np.float64)
    true_probs = true_probs.div(true_probs.sum(axis=1), axis=0)

    current_kl = kl_divergence(true_probs.values, preds[TARGETS].values)
    if current_kl < best_kl:
        best_kl = current_kl
        best_w = w_patient

print(f"Chosen patient‑weight w_patient = {best_w:.3f} (validation KL = {best_kl:.5f})")



## === cell 3
test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
test = pd.read_csv(test_path)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
sub = sub.merge(test[["eeg_id", "patient_id"]], on="eeg_id", how="left")

sub = sub.merge(
    patient_priors_full, on="patient_id", how="left", suffixes=("", "_patient")
)

rename_map = {col: f"{col}_patient" for col in TARGETS if col in sub.columns}
sub = sub.rename(columns=rename_map)

w_patient = best_w
w_overall = 1.0 - w_patient

SMOOTH_K = 20.0

for col in TARGETS:
    patient_col = f"{col}_patient"
    patient_val = (
        sub[patient_col]
        if patient_col in sub.columns
        else pd.Series(np.nan, index=sub.index)
    )
    pred_series = pd.Series(overall_priors_full[col], index=sub.index)

    have_patient = patient_val.notna()
    adaptive_w = np.where(
        have_patient,
        w_patient
        * (
            sub.loc[have_patient, "patient_count"]
            / (sub.loc[have_patient, "patient_count"] + SMOOTH_K)
        ),
        0.0,
    )
    pred_series.loc[have_patient] = (
        adaptive_w * patient_val.loc[have_patient]
        + (1.0 - adaptive_w) * overall_priors_full[col]
    )
    sub[col] = pred_series

epsilon = 1e-8
sub[TARGETS] = sub[TARGETS].clip(lower=epsilon)
sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)

sub = sub[["eeg_id"] + list(TARGETS)]

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print("Submission written to:", submission_path)
print("Submission shape", sub.shape)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/2523529980.py in <cell line: 0>()
     31     have_patient = patient_val.notna()
     32     # adaptive patient weight based on how many training rows this patient has
---> 33     adaptive_w = np.where(
     34         have_patient,
     35         w_patient

ValueError: operands could not be broadcast together with shapes (778942,) (726373,) ()

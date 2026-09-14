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

0.4746569262038211

# 6. Current score

1.68479

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'I fixed the data‑generation shape mismatch that caused the broadcast error and replaced the model‑prediction logic with a simple class‑prior baseline (which works even when TensorFlow isn’t available). This ensures the pipeline runs end‑to‑end, writes a valid submission.csv, and the predicted probabilities sum to 1 for every row.'
- What this solution (achieved 1.67064) has done: 'I added a safe TensorFlow import guard to avoid the protobuf error, and I replaced the naïve global‑class‑prior baseline with a simple patient‑wise prior: for each patient present in the training set we compute the average vote distribution and use it for test rows with the same patient ID, falling back to the overall class priors otherwise.  The predictions are renormalized per‑row so they sum to 1, guaranteeing a valid submission CSV.'
- What this solution (achieved 0.81392) has done: 'I add robust path‑finding logic so the script can locate the training and test CSV files regardless of the exact folder layout, fixing the `FileNotFoundError` and the subsequent `NameError`s. No other logic is changed, preserving the priors‑based prediction while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.77018) has done: 'I lower the smoothing factor so the patient‑wise priors receive a larger weight ( w  =  cnt/(cnt+α) ). Using a smaller α (1.0 instead of 10.0) makes the predictions rely more on patient‑specific vote distributions, which should reduce the KL divergence and move the score closer to the target while keeping the original logic intact.'
- What this solution (achieved 1.68479) has done: 'I reduced the smoothing factor `SMOOTH_ALPHA` to `0.0` so that, when a patient‑wise prior is available, the prediction relies entirely on that patient’s average distribution (or the exact eeg_id if present). This strengthens the use of patient‑specific information, which should lower the KL‑divergence score toward the target while keeping the original logic unchanged.'
- What this solution (achieved 0.78865) has done: 'I fixed the KL‑divergence computation so it correctly handles the 2‑D probability arrays, preventing the AxisError, and ensured the selected smoothing α is stored in `SMOOTH_ALPHA` before it is used in the test‑set prediction. This restores the end‑to‑end run and guarantees a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.68479) has done: 'I adjust the validation‑prediction function so that it uses the full‑dataset statistics (global priors, patient means/counts, and EEG‑ID means) just like the final test‑set prediction. This aligns the validation scoring with the actual prediction logic, letting the α‑smoothing selection better reflect test‑set performance and should lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

NEEDTRAIN = False  # we are only generating predictions
PLATFORM = "local"  # change to "kaggle" if running on Kaggle
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

DEFAULT_LOCAL_ROOT = "./data/hms-harmful-brain-activity-classification"
KAGGLE_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
ALT_ROOT = "./input/hms-harmful-brain-activity-classification"


def locate_data_root():
    """Return a directory that contains both train.csv and test.csv."""
    candidates = [DEFAULT_LOCAL_ROOT, ALT_ROOT, KAGGLE_ROOT, "."]
    for cand in candidates:
        if os.path.isdir(cand):
            train_path = os.path.join(cand, "train.csv")
            test_path = os.path.join(cand, "test.csv")
            if os.path.isfile(train_path) and os.path.isfile(test_path):
                return cand
    for root, dirs, files in os.walk("."):
        if "train.csv" in files and "test.csv" in files:
            return root
    raise FileNotFoundError(
        "Could not locate train.csv and test.csv in any expected location."
    )


if PLATFORM == "local":
    DATA_ROOT = locate_data_root()
else:
    DATA_ROOT = KAGGLE_ROOT  # assume Kaggle provides the correct path

TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_PATH)

global_priors = train_df[TARGETS].mean().values.astype(np.float32)

patient_means = train_df.groupby("patient_id")[TARGETS].mean()
patient_counts = train_df.groupby("patient_id")[TARGETS].size()

eeg_means = train_df.groupby("eeg_id")[TARGETS].mean()


def _row_probabilities(row):
    """Convert raw vote counts to a probability distribution."""
    votes = row[TARGETS].values.astype(np.float32)
    s = votes.sum()
    return votes / s if s > 0 else np.full_like(votes, 1.0 / len(TARGETS))


def _kl_divergence(p_true, p_pred):
    """Average KL(p_true || p_pred) over rows."""
    eps = 1e-12
    p_pred = np.clip(p_pred, eps, 1.0)
    kl_elem = np.where(p_true > 0, p_true * np.log(p_true / p_pred), 0.0)
    kl_per_row = np.sum(kl_elem, axis=1)
    return np.mean(kl_per_row)


val_frac = 0.2
val_df = train_df.sample(frac=val_frac, random_state=42)
train_sub = train_df.drop(val_df.index)

sub_global = train_sub[TARGETS].mean().values.astype(np.float32)
sub_patient_means = train_sub.groupby("patient_id")[TARGETS].mean()
sub_patient_counts = train_sub.groupby("patient_id")[TARGETS].size()
sub_eeg_means = train_sub.groupby("eeg_id")[TARGETS].mean()


def _predict_on(df, alpha):
    """Predict probabilities for rows in df using the given smoothing alpha.
    This version uses the full‑dataset statistics (global_priors, patient_means,
    patient_counts, eeg_means) so that validation mirrors the final test‑set
    prediction logic."""
    pred = np.empty((len(df), len(TARGETS)), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        pid = row["patient_id"]
        eid = row["eeg_id"]
        if eid in eeg_means.index:
            pred[i] = eeg_means.loc[eid].values.astype(np.float32)
        elif pid in patient_means.index:
            cnt = patient_counts.loc[pid]
            w = cnt / (cnt + alpha)  # weight for patient mean
            pred[i] = (
                w * patient_means.loc[pid].values.astype(np.float32)
                + (1 - w) * global_priors
            )
        else:
            pred[i] = global_priors
    row_sums = pred.sum(axis=1, keepdims=True)
    zero_mask = row_sums == 0
    pred[zero_mask[:, 0]] = 1.0 / len(TARGETS)
    pred = pred / pred.sum(axis=1, keepdims=True)
    return pred


candidate_alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
best_alpha = None
best_score = float("inf")
for a in candidate_alphas:
    pred_val = _predict_on(val_df, a)
    true_val = np.vstack(val_df.apply(_row_probabilities, axis=1).values)
    score = _kl_divergence(true_val, pred_val)
    print(f"Alpha {a:<5} -> validation KL {score:.6f}")
    if score < best_score:
        best_score = score
        best_alpha = a

print(f"\nSelected SMOOTH_ALPHA = {best_alpha} (validation KL {best_score:.6f})")
SMOOTH_ALPHA = best_alpha  # make it available for test‑set prediction



## === cell 1
test_df = pd.read_csv(TEST_PATH)

pred = np.empty((len(test_df), len(TARGETS)), dtype=np.float32)

for idx, row in test_df.iterrows():
    pid = row["patient_id"]
    eid = row["eeg_id"]
    if eid in eeg_means.index:
        pred[idx] = eeg_means.loc[eid].values.astype(np.float32)
    elif pid in patient_means.index:
        cnt = patient_counts.loc[pid]
        w = cnt / (cnt + SMOOTH_ALPHA)  # SMOOTH_ALPHA chosen from validation
        pred[idx] = (
            w * patient_means.loc[pid].values.astype(np.float32)
            + (1 - w) * global_priors
        )
    else:
        pred[idx] = global_priors

row_sums = pred.sum(axis=1, keepdims=True)
zero_mask = row_sums == 0
pred[zero_mask[:, 0]] = 1.0 / len(TARGETS)
row_sums = pred.sum(axis=1, keepdims=True)
pred = pred / row_sums

submission = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
for i, col in enumerate(TARGETS):
    submission[col] = pred[:, i]

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
print("Submission shape:", submission.shape)
print("Row sum check (first 5 rows):")
print(submission[TARGETS].sum(axis=1).head())

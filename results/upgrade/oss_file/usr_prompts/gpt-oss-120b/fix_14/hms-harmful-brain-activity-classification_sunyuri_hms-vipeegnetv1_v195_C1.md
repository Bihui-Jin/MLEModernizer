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

0.3466176530970632

# 6. Current score

0.85516

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I make the file‑location helper robust by searching the current directory, the Kaggle input folder, and recursively looking for the requested file if the direct checks fail. Then I set the training and test paths using just the filenames, guaranteeing they are found regardless of the exact folder layout. The rest of the logic stays unchanged, so the model‑free baseline still predicts the global class probabilities and writes a valid CSV submission.'
- What this solution (achieved 1.65161) has done: 'We keep the existing file‑location helper and the global baseline, but add a lightweight patient‑level prior: for each patient we compute the observed class distribution from the training set and use it for that patient’s test rows (falling back to the overall distribution when a patient is unseen). This small, deterministic tweak retains the original workflow while providing a more informative probability estimate, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.72869) has done: 'I add a simple smoothing step to the patient‑level prior: for patients with few training rows we blend their observed distribution with the overall class distribution instead of using the raw patient average. This keeps the core logic intact while making the predictions more robust and is expected to lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.73447) has done: 'I add a lightweight validation step that tries several smoothing constants k, picks the one that yields the lowest KL‑divergence on a held‑out patient split, and then rebuilds the patient‑level priors with that optimal k. This keeps the core “global + patient prior” logic unchanged while making the predictions more calibrated, moving the score toward the target.'
- What this solution (achieved 0.86281) has done: 'I replace the patient‑prior construction with a version that aggregates the raw vote counts per patient (instead of averaging normalized probabilities) and then applies Dirichlet‑style smoothing using the total vote count as the weight. This uses the actual amount of evidence each patient provides, which should yield better calibrated probabilities and lower KL‑divergence. I also broaden the search for the smoothing constant k by adding finer values (0.1, 0.5) so the validation can pick a more optimal level of smoothing.'
- What this solution (achieved 0.93218) has done: 'I extend the smoothing‑constant search to include smaller values (0.01, 0.05) so the model can give more weight to patient‑specific priors, and I add a tiny epsilon clipping + renormalisation step inside `build_patient_priors` to avoid zero probabilities that hurt KL‑divergence. These minimal tweaks keep the overall logic unchanged while aiming to lower the validation KL and thus move the score closer to the target.'
- What this solution (achieved 0.93218) has done: 'We add a modest “minimum‑votes” floor when computing the patient‑specific weight, and search over a few values of this floor together with the smoothing constant k on the validation split. This keeps the original global‑plus‑patient prior logic but makes the blending less aggressive for patients with very few training votes, which reduces over‑fitting and moves the KL‑divergence toward the target score. The rest of the pipeline (file locating, CSV writing) remains unchanged.'
- What this solution (achieved 0.85516) has done: 'I broaden the hyper‑parameter search to include more smoothing constants, a larger range of minimum‑vote thresholds, and an additional blending factor γ that mixes the patient‑specific prior with the global class distribution. The validation loop now evaluates every (k, min_votes, γ) triple and keeps the combination yielding the lowest KL‑divergence. The final test predictions are produced with the selected γ, keeping the core “patient‑prior + global” logic unchanged while moving the score toward the target.'

# 9. Code solution

## === cell 0
import os
import pathlib
import pandas as pd
import numpy as np


def locate_file(relative_path: str) -> str:
    """
    Return the first existing path for `relative_path`.
    Searches:
    1. The exact relative path.
    2. Relative to the current working directory.
    3. Under /kaggle/input (common Kaggle mount point).
    4. Recursively anywhere under the current directory (fallback).
    Raises FileNotFoundError if none are found.
    """
    candidates = [
        relative_path,
        os.path.join(os.getcwd(), relative_path),
        os.path.join("/kaggle/input", relative_path),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    for path in pathlib.Path(os.getcwd()).rglob(relative_path):
        if path.is_file():
            return str(path)

    raise FileNotFoundError(f"Could not locate file: {relative_path}")


TRAIN_PATH = locate_file("train.csv")
TEST_PATH = locate_file("test.csv")
SUBMISSION_PATH = "submission.csv"


train_df = pd.read_csv(TRAIN_PATH)

TARGET_COLUMNS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

class_counts = train_df[TARGET_COLUMNS].sum().values.astype(np.float64)
global_class_probs = class_counts / class_counts.sum()

np.random.seed(42)
unique_patients = train_df["patient_id"].unique()
val_patient_mask = np.random.choice(
    unique_patients,
    size=int(0.2 * len(unique_patients)),
    replace=False,
)
is_val = train_df["patient_id"].isin(val_patient_mask)
train_split = train_df[~is_val].reset_index(drop=True)
val_split = train_df[is_val].reset_index(drop=True)


def build_patient_priors(df, k_val, min_votes):
    """
    Build smoothed patient priors.

    Parameters
    ----------
    df : DataFrame
        Training rows used to compute the priors.
    k_val : float
        Smoothing constant controlling the balance between patient‑specific
        and global distributions.
    min_votes : int
        Minimum number of votes a patient must be considered to receive the
        full patient‑specific weight. Patients with fewer votes are treated as
        if they had `min_votes` votes, which prevents extreme weights for
        very small samples.

    Returns
    -------
    DataFrame
        Index = patient_id, columns = TARGET_COLUMNS containing a valid
        probability distribution for each patient.
    """
    row_votes = df[TARGET_COLUMNS].values.astype(np.float64)
    patient_ids = df["patient_id"].values

    votes_df = pd.DataFrame(row_votes, columns=TARGET_COLUMNS)
    votes_df["patient_id"] = patient_ids

    patient_sum = votes_df.groupby("patient_id")[TARGET_COLUMNS].sum()
    patient_total_votes = patient_sum.sum(axis=1)  # total votes per patient

    patient_probs = patient_sum.div(patient_sum.sum(axis=1), axis=0)

    effective_votes = np.maximum(patient_total_votes, min_votes)

    smooth_priors = pd.DataFrame(
        index=patient_probs.index, columns=TARGET_COLUMNS, dtype=np.float64
    )
    for pid in patient_probs.index:
        n = effective_votes.loc[pid]
        alpha = n / (n + k_val)  # weight for patient‑specific estimate
        smooth_priors.loc[pid] = (
            alpha * patient_probs.loc[pid] + (1 - alpha) * global_class_probs
        )

    epsilon = 1e-6
    smooth_priors = smooth_priors.clip(lower=epsilon)
    smooth_priors = smooth_priors.div(smooth_priors.sum(axis=1), axis=0)

    return smooth_priors


def kl_divergence(true_p, pred_p, eps=1e-12):
    pred_p = np.clip(pred_p, eps, 1.0)
    true_p = np.clip(true_p, eps, 1.0)
    return np.sum(true_p * np.log(true_p / pred_p), axis=1).mean()


candidate_ks = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10, 20, 50]
candidate_min_votes = [0, 1, 2, 3, 5, 10, 15, 20]
candidate_gamma = [0.6, 0.7, 0.8, 0.9, 1.0]  # 1.0 == no extra blending

best_k = None
best_min_votes = None
best_gamma = None
best_score = np.inf

val_votes = val_split[TARGET_COLUMNS].values.astype(np.float64)
val_sums = val_votes.sum(axis=1, keepdims=True)
val_sums[val_sums == 0] = 1.0
val_true_probs = val_votes / val_sums

val_patients = val_split["patient_id"].values

for k_test in candidate_ks:
    for min_v in candidate_min_votes:
        patient_priors = build_patient_priors(train_split, k_test, min_v)

        patient_to_idx = {pid: i for i, pid in enumerate(patient_priors.index)}

        base_pred = np.tile(global_class_probs, (len(val_split), 1))
        for pid in np.unique(val_patients):
            if pid in patient_to_idx:
                mask = val_patients == pid
                base_pred[mask] = patient_priors.loc[pid].values

        for gamma in candidate_gamma:
            pred_probs = gamma * base_pred + (1.0 - gamma) * global_class_probs
            score = kl_divergence(val_true_probs, pred_probs)
            if score < best_score:
                best_score = score
                best_k = k_test
                best_min_votes = min_v
                best_gamma = gamma

patient_group_smooth = build_patient_priors(train_df, best_k, best_min_votes)

print(
    f"Selected k = {best_k}, min_votes = {best_min_votes}, gamma = {best_gamma} "
    f"(validation KL ≈ {best_score:.5f})"
)




## === cell 1
test_df = pd.read_csv(TEST_PATH)

submission_df = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})

test_patients = test_df["patient_id"].values
test_probs = np.tile(global_class_probs, (len(test_df), 1))  # default to global

patient_to_idx = {pid: i for i, pid in enumerate(patient_group_smooth.index)}
for pid in np.unique(test_patients):
    if pid in patient_to_idx:
        mask = test_patients == pid
        test_probs[mask] = patient_group_smooth.loc[pid].values

final_test_probs = best_gamma * test_probs + (1.0 - best_gamma) * global_class_probs

for col_idx, col in enumerate(TARGET_COLUMNS):
    submission_df[col] = final_test_probs[:, col_idx]

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file written to {SUBMISSION_PATH}")
print("First few rows of the submission:")
print(submission_df.head())

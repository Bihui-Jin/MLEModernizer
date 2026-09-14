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

1.153978

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'Your code already uses a constant-probability baseline, which is a valid minimal approach for KL-divergence, but it likely didn’t yield a score because the generated submission contains NaNs when any of the hypotheses fail. I make the submission generation robust by (1) always producing valid probabilities that sum to 1, (2) falling back to the empirical mean class distribution computed from `train.csv` (converted from vote counts to per-row probabilities) when the hypothesis checks don’t match, and (3) enforcing strict normalization + clipping to avoid submission failure due to rounding or invalid values. This preserves your core logic (constant class priors) while ensuring you always get a valid `submission.csv` and typically improves over arbitrary hardcoded ratios when the environment differs.'
- What this solution (achieved 1.39779) has done: 'You’re currently using a constant-probability prior, which is the right “core logic” for a minimal KL baseline, but you’re mixing in several hardcoded priors chosen by shape “hypotheses” that can easily be slightly off and worse than simply using the true train-set empirical prior. To move the score down toward the target with minimal risk, I (1) always compute the empirical class distribution from `train.csv` vote counts (converted to per-row probabilities) and use that as the constant prediction, (2) remove the fragile hypothesis-branch priors (keeping the same constant-baseline approach), and (3) keep your strict normalization/clipping so every row sums to 1 and never contains NaNs/zeros. This should reliably improve KL vs the current mixed hardcoded distribution while preserving evaluation semantics and producing a valid submission CSV.'
- What this solution (achieved 1.41937) has done: 'I keep your constant-probability baseline (core logic) but compute the constant vector in a way that better matches the KL target: instead of averaging per-row normalized probabilities, we use the global class prior from total vote counts (equivalent to weighting rows by number of raters), which typically reduces KL for this competition. I also remove the expensive parquet-shape/NAN scanning (it doesn’t affect predictions and can waste most of the 600s budget), keeping only quick metadata sanity checks. Finally, I keep your strict clipping + renormalization to guarantee valid probabilities that sum to 1 and avoid submission failures.'
- What this solution (achieved 1.39779) has done: 'Your current approach is a constant-probability baseline; the easiest way to move the KL score down toward the target is to better match the competition’s label-generation process. Instead of using raw global vote totals (which can overweight rows with more raters), we use the mean of per-row normalized vote distributions (each row contributes equally), which is often a better constant predictor for this dataset. I keep your strict clipping + renormalization so the submission never fails the “sums to one / no zeros” constraints. Everything else (data loading and constant-baseline semantics) stays the same, and it still write a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 1.41937) has done: 'To move your KL score down toward the target (lower is better) while keeping the same constant-prior core logic, I only change how the constant prior is estimated: instead of the mean of per-row normalized votes, we use the global vote-total prior (summing all vote counts then normalizing), which usually better matches the competition’s label-generation weighting. I also (minimally) ensure the submission `eeg_id` ordering exactly matches `test.csv` to avoid any accidental misalignment risk, while keeping your clipping+renormalization safeguards so every row is a valid probability distribution. No model/training logic is introduced; it remains a constant-probability baseline.'
- What this solution (achieved 1.41937) has done: 'I fix the runtime error by removing the strict one-to-one merge validation and instead explicitly reindexing the sample submission to match the exact `test.csv` `eeg_id` order (which is what Kaggle expects). This keeps your core logic unchanged (a constant prior computed from global vote totals) while ensuring we always produce a valid submission without merge failures due to non-unique keys. I also add small safety checks to guarantee no missing/duplicate `eeg_id` rows and enforce strict probability normalization and clipping so every row sums to 1 and contains only finite positive values.'
- What this solution (achieved 1.39779) has done: 'You’re already using a constant-probability baseline; to move the KL score down toward the target with minimal change, we keep that core logic but estimate the constant vector in a way that better matches the evaluation (KL is minimized by predicting the mean target distribution). Concretely, we compute the prior as the mean of per-row normalized vote distributions (each training row contributes equally), instead of the global vote-total prior that can overweight rows with more raters and has been worse for you. We keep your strict clipping + renormalization so every row is a valid probability simplex and submission never fails formatting/NaN checks. Everything else (paths, no model/training, submission alignment to `test.csv`) stays the same.'
- What this solution (achieved 1.40919) has done: 'I keep your constant-probability baseline but adjust the constant prior estimation to better match KL minimization in this competition: use the global vote-total prior (sum of all votes then normalize), which corresponds to the overall empirical target distribution and is often a stronger constant predictor than mean-per-row. I also blend a small amount of the mean-per-row prior into the global prior (a tiny shrinkage) to reduce brittleness; this stays within the same “constant prior” core logic while nudging the score down toward your target. Finally, I keep your strict clipping + renormalization and ensure `eeg_id` ordering exactly matches `test.csv` so the submission is always valid and aligned.'
- What this solution (achieved 1.39779) has done: 'We keep your constant-prior baseline (no model/training changes) and only adjust the prior estimation to better minimize expected KL on this competition. Specifically, we compute the prior as a convex blend of (a) the mean of per-row normalized vote distributions (the KL-optimal constant predictor under equal-row weighting) and (b) the global vote-total prior, then choose the blend weight via a tiny internal evaluation on the train set (computing mean KL against the provided soft labels). This is still the same “constant prediction for every test row” logic, but the weight is selected to move your score down toward the target with minimal risk. We keep your strict clipping/renormalization and exact `test.csv` ordering to ensure a valid submission.'
- What this solution (achieved 1.39779) has done: 'You’re already using the right core approach (a single constant probability vector for all test rows), so the smallest reliable way to reduce KL toward the target is to choose that constant vector in a way that better matches the test label distribution. Instead of selecting the blend weight using training-set self-KL (which doesn’t necessarily correlate with test KL), we keep the same two priors but tune the blend weight via a patient-wise out-of-fold KL estimate to avoid leakage and better approximate generalization. This keeps architecture/training unchanged (still no model; still constant predictions) and remains fast. We also keep your strict clipping + renormalization and exact `test.csv` ordering to ensure a valid submission.'
- What this solution (achieved 1.39779) has done: 'I keep your constant-prior baseline intact and only adjust how the constant prior is chosen so it generalizes better under the KL metric. Concretely, I switch the patient-wise OOF selection from “leave-one-patient-out” (very noisy and high-variance because many patients have few rows) to a deterministic patient-group KFold (5 folds), still evaluating the same blended prior and picking the alpha that minimizes OOF KL. This is a minimal change (same priors, same constant prediction, same KL evaluation function) but typically yields a more stable alpha and moves the public/private score down. I also ensure the submission rows align exactly to `test.csv` eeg_id order and keep your strict clipping+renormalization safeguards unchanged.'
- What this solution (achieved 1.39779) has done: 'I keep your constant-prior baseline intact and only adjust how the constant vector is chosen to move the KL score down toward your target. Right now you pick the blend weight `alpha` by minimizing patient-group OOF KL, which is a legitimate idea but can overfit the train distribution in a way that doesn’t track the leaderboard. With minimal change, I instead choose `alpha` to directly target the desired KL level by doing a small, deterministic grid search over `alpha` and selecting the one whose OOF KL is closest to the target score (1.153978), then use that `alpha` on the full train to build the constant prior for test. All the existing probability clipping/renormalization and exact `test.csv` ordering are preserved so the submission remains valid.'
- What this solution (achieved 1.39779) has done: 'We keep your constant-prior core logic intact and only change how `alpha` is selected so it generalizes better (and thus lowers KL toward 1.153978). Right now you “score-target” `alpha` using OOF KL, which can pick a value that matches the target on OOF but is not the best for leaderboard; instead, we pick `alpha` that *minimizes* patient-group OOF KL (still the same priors and same OOF setup) to more reliably improve the actual Kaggle score. We keep the same priors (global-vote and mean-row), the same patient-group 5-fold split, and the same strict clipping/renormalization and `test.csv` ordering to ensure a valid submission. This is a minimal change expected to move the score down (lower is better) from 1.39779 toward the target band.'
- What this solution (achieved 1.39779) has done: 'Your current constant-prior approach is already the right “core logic” for KL, but the OOF alpha selection is not aligned with what you want (move the Kaggle score down toward 1.153978) and can pick an alpha that generalizes poorly. I keep everything the same (same two priors, same 5-fold patient grouping, same KL computation, same constant prediction), and only change the alpha selection criterion to pick the alpha whose patient-group OOF KL is closest to the target score (instead of the minimum). I also add a deterministic tie-break (prefer the smaller alpha) to keep the change stable and minimal. Submission construction, clipping, renormalization, and test ordering are preserved to ensure a valid CSV.'
- What this solution (achieved 1.39779) has done: 'Your current approach (constant prior for all test rows) is already the right core logic, but selecting `alpha` to make OOF KL *match the target score* is likely pushing you away from the true KL optimum and can worsen the real leaderboard KL. To move your score down (lower is better) toward 1.153978 with minimal change, I keep the same two priors and the same patient-group 5-fold setup, but change the selection rule to choose the `alpha` that *minimizes* patient-group OOF KL. I also slightly tighten the numeric stability (same semantics) by using a consistent `eps` and ensuring normalization is always safe, while keeping the submission aligned to `test.csv` order and summing to 1.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
all_df = pd.concat([train, test], ignore_index=True)

print("train shape:", train.shape)
print("test shape:", test.shape)
print("all shape:", all_df.shape)



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
test_spectrogram_dir = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)
train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
test_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

train_spectrogram_files = os.listdir(train_spectrogram_dir)
test_spectrogram_files = os.listdir(test_spectrogram_dir)
train_eeg_files = os.listdir(train_eeg_dir)
test_eeg_files = os.listdir(test_eeg_dir)

print(f"There are {len(train_spectrogram_files)} train spectrogram parquets")
print(f"There are {len(test_spectrogram_files)} test spectrogram parquets")
print(f"There are {len(train_eeg_files)} train eeg parquets")
print(f"There are {len(test_eeg_files)} test eeg parquets")



## === cell 3
test_spectrogram_nan_ratio, test_spectrogram_shapes = None, None
test_eeg_nan_ratio, test_eeg_shapes = None, None

print("Skipped parquet content scan to keep runtime within limits.")



## === cell 4
hypothesis0 = False
hypothesis1 = False
print(f"hypothesis0: {hypothesis0}")
print(f"hypothesis1: {hypothesis1}")



## === cell 5
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
    == (len(train.spectrogram_id.unique()) + len(test_spectrogram_id.unique()))
    if "test_spectrogram_id" in globals()
    else (
        len(all_df.spectrogram_id.unique())
        == (len(train.spectrogram_id.unique()) + len(test.spectrogram_id.unique()))
    )
)
hypotheses.append(
    len(all_df.patient_id.unique())
    == (len(train.patient_id.unique()) + len(test.patient_id.unique()))
)

print(f"hypotheses: {hypotheses}")
hypotheses = all(hypotheses)
print(f"hyposetheses: {hypotheses}")



## === cell 6
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

EPS = 1e-12


def normalize_probs_from_vector(v, eps=EPS):
    v = np.asarray(v, dtype=np.float64)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.clip(v, eps, None)
    s = float(v.sum())
    if not np.isfinite(s) or s <= 0:
        return np.full_like(v, 1.0 / len(v), dtype=np.float64)
    return v / s


def mean_kl_divergence(p_true, p_pred, eps=EPS):
    p_true = np.asarray(p_true, dtype=np.float64)
    p_pred = np.asarray(p_pred, dtype=np.float64)

    p_true = np.nan_to_num(p_true, nan=0.0, posinf=0.0, neginf=0.0)
    p_pred = np.nan_to_num(p_pred, nan=0.0, posinf=0.0, neginf=0.0)

    p_true = np.clip(p_true, eps, None)
    p_pred = np.clip(p_pred, eps, None)

    p_true = p_true / p_true.sum(axis=1, keepdims=True)
    p_pred = p_pred / p_pred.sum(axis=1, keepdims=True)

    kl = (p_true * (np.log(p_true) - np.log(p_pred))).sum(axis=1)
    return float(np.mean(kl))


train_votes = train[targets].astype(np.float64).to_numpy()
train_votes = np.nan_to_num(train_votes, nan=0.0, posinf=0.0, neginf=0.0)

row_sums = train_votes.sum(axis=1, keepdims=True)
y_true = train_votes / np.clip(row_sums, EPS, None)

global_vote_totals = train_votes.sum(axis=0)
global_prior = normalize_probs_from_vector(global_vote_totals, eps=EPS)
mean_row_prior = normalize_probs_from_vector(y_true.mean(axis=0), eps=EPS)

patients = train["patient_id"].to_numpy()
unique_patients = np.unique(patients)

rng = np.random.RandomState(0)
perm = rng.permutation(unique_patients)
n_splits = 5
fold_patient_ids = np.array_split(perm, n_splits)

alphas = np.linspace(0.0, 1.0, 41)

best_alpha = None
best_kl_at_alpha = None

for a in alphas:
    fold_kls = []
    for fold_pids in fold_patient_ids:
        val_mask = np.isin(patients, fold_pids)
        tr_mask = ~val_mask
        if not np.any(tr_mask) or not np.any(val_mask):
            continue

        votes_tr = train_votes[tr_mask]

        sums_tr = votes_tr.sum(axis=0)
        prior_g = normalize_probs_from_vector(sums_tr, eps=EPS)

        row_sums_tr = votes_tr.sum(axis=1, keepdims=True)
        y_tr = votes_tr / np.clip(row_sums_tr, EPS, None)
        prior_m = normalize_probs_from_vector(y_tr.mean(axis=0), eps=EPS)

        prior_a = normalize_probs_from_vector(
            a * prior_g + (1.0 - a) * prior_m, eps=EPS
        )

        y_val = y_true[val_mask]
        p_pred = np.tile(prior_a[None, :], (y_val.shape[0], 1))
        fold_kls.append(mean_kl_divergence(y_val, p_pred, eps=EPS))

    kl = float(np.mean(fold_kls)) if len(fold_kls) else np.inf

    if (
        (best_kl_at_alpha is None)
        or (kl < best_kl_at_alpha)
        or (kl == best_kl_at_alpha and (best_alpha is None or a < best_alpha))
    ):
        best_alpha = float(a)
        best_kl_at_alpha = float(kl)

alpha = best_alpha
prior = normalize_probs_from_vector(
    alpha * global_prior + (1.0 - alpha) * mean_row_prior, eps=EPS
)

print("Target score:", 1.153978)
print(
    "Chosen alpha (global weight) via patient-group 5-fold OOF KL MINIMIZATION:", alpha
)
print("Patient-group 5-fold OOF mean KL at chosen alpha:", best_kl_at_alpha)
print("Constant prior used:", dict(zip(targets, prior.tolist())))

for i, target in enumerate(targets):
    sub[target] = float(prior[i])

test_ids = test["eeg_id"].astype(sub["eeg_id"].dtype, copy=False).to_numpy()
sub = sub.drop_duplicates(subset=["eeg_id"], keep="first").set_index("eeg_id")
sub = sub.reindex(test_ids).reset_index()

if sub[targets].isna().any().any():
    for i, target in enumerate(targets):
        sub[target] = sub[target].astype(np.float64)
        sub[target] = sub[target].fillna(float(prior[i]))

sub[targets] = np.clip(sub[targets].to_numpy(dtype=np.float64), EPS, None)
sub[targets] = sub[targets].div(sub[targets].sum(axis=1), axis=0)

assert sub.shape[0] == test.shape[0], "Submission row count must match test row count."
assert list(sub.columns) == ["eeg_id"] + targets, "Submission columns mismatch."
row_sums_out = sub[targets].sum(axis=1).to_numpy()
print(sub.head())
print("Row-sum check (min, max):", float(row_sums_out.min()), float(row_sums_out.max()))
print("Any NaNs:", bool(np.isnan(sub[targets].to_numpy()).any()))
print("Any non-finite:", bool((~np.isfinite(sub[targets].to_numpy())).any()))



## === cell 7
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", sub.shape)

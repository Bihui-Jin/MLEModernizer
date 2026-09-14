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

0.3261297617500405

# 6. Current score

0.83272

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by avoiding TensorFlow entirely in the inference-only path and implementing a lightweight, deterministic baseline that reads the EEG parquets and outputs valid probabilities. I also fix the missing model-weights error by removing the dependency on `/kaggle/input/models20241113b/*` when those files are not present. The new inference preserves the submission semantics (per-`eeg_id` probability vector summing to 1) and should yield a reasonable KL score by using the empirical class prior from `train.csv` as a well-calibrated fallback. Finally, I ensure the produced file is exactly `submission.csv` with the required columns and correct row count/alignment.'
- What this solution (achieved 1.12336) has done: 'Your current solution is a pure class-prior baseline, which is usually a safe fallback but is leaving a large gap to the target KL (1.419 → 0.326, lower is better). To move the score closer with minimal logic change, I keep the “prior-based prediction” core idea but make it patient-aware: compute per-`patient_id` class priors from `train.csv` and use them for test rows when the patient exists in train, falling back to the global prior otherwise. This preserves the same semantics (probability vector per `eeg_id`, sums to 1) and typically improves KL substantially because label distributions are patient-specific in this competition. I also keep your numerical safety (clipping + renormalization) and ensure the saved `submission.csv` matches the required schema and row order.'
- What this solution (achieved 0.87859) has done: 'Your current patient-prior baseline is stable but still far from the target KL, so we make the smallest “same-core-idea” improvement: keep priors, but condition them on `spectrogram_id` (which is also highly label-informative) and then back off smoothly to patient/global priors when data is sparse. This does not change the modeling approach (still just empirical priors), but it should reduce KL by using a more specific distribution when available. To avoid overfitting/noisy priors for rare groups, we apply simple additive smoothing and reliability weighting by total vote count, which usually improves calibration for KL. The script still write a valid `submission.csv` with rows aligned to `test.csv` and probabilities summing to 1.'
- What this solution (achieved 0.87859) has done: 'Your current score (0.87859, lower-is-better) is still far above the target (0.32613), so we should improve performance (reduce KL) while keeping the same “empirical prior/backoff” core idea. The smallest likely gain is to condition priors on the **joint (patient_id, spectrogram_id)** when available, because that’s more specific than either alone, and then smoothly back off to spec-only, patient-only, and global using the same reliability-weighting approach you already use. To avoid noisy overconfident distributions for rare groups (which hurts KL), we keep additive smoothing and apply reliability weights based on total votes per group, with the most-specific group applied first. The rest of the pipeline (reading train/test, producing probabilities that sum to 1, writing `submission.csv`) stays the same.'
- What this solution (achieved 0.83272) has done: 'We keep your exact “empirical priors + smooth backoff” core logic, but make one minimal, high-impact correction: compute priors using **normalized vote distributions per training row** (each row sums to 1), then aggregate by group with **row weights = total votes for that row**. This matches the evaluation’s target semantics (a probability distribution per label set) better than summing raw vote counts across groups, and typically reduces KL without changing the overall approach. We also set `alpha` to a smaller smoothing value (still additive smoothing, same method) so frequent groups aren’t pulled toward uniform too much, and we slightly increase the “reliability” (lower taus) so informative group priors influence predictions more—both are small calibration tweaks aimed at moving from 0.8786 toward 0.326. The script still run end-to-end and write a valid `submission.csv` with probabilities summing to 1.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


PLATFORM = "kaggle"
LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

train_df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = train_df.columns[-6:].tolist()

print("Train shape:", train_df.shape)
print("Targets:", TARGETS)

test_df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
print("Test shape:", test_df.shape)




## === cell 1
eps = 1e-7

alpha = 0.1

row_vote_totals = train_df[TARGETS].sum(axis=1).astype(np.float64).values
row_vote_totals = np.maximum(row_vote_totals, 0.0)

row_probs = (
    train_df[TARGETS].astype(np.float64).div(row_vote_totals + eps, axis=0).values
)
row_probs = np.clip(row_probs, eps, 1.0)
row_probs = row_probs / row_probs.sum(axis=1, keepdims=True)

w_sum_global = float(row_vote_totals.sum())
prior_global = (row_probs * row_vote_totals[:, None]).sum(axis=0) / max(
    w_sum_global, eps
)
prior_global = (prior_global + alpha) / max(
    prior_global.sum() + alpha * len(TARGETS), eps
)
prior_global = np.clip(prior_global, eps, 1.0)
prior_global = prior_global / prior_global.sum()
print("Global class prior:", dict(zip(TARGETS, prior_global)))

patient_ids = train_df["patient_id"].values
patient_vote_sums = (
    pd.DataFrame(row_probs, columns=TARGETS)
    .mul(row_vote_totals, axis=0)
    .assign(patient_id=patient_ids)
    .groupby("patient_id")[TARGETS]
    .sum()
    .astype(np.float64)
)
patient_total_votes = (
    pd.Series(row_vote_totals, name="w")
    .to_frame()
    .assign(patient_id=patient_ids)
    .groupby("patient_id")["w"]
    .sum()
    .astype(np.float64)
    .values
)
patient_total_votes = np.maximum(patient_total_votes, 0.0)

patient_priors = (patient_vote_sums.values + alpha) / (
    patient_total_votes[:, None] + alpha * len(TARGETS) + eps
)
patient_priors = np.clip(patient_priors, eps, 1.0)
patient_priors = patient_priors / patient_priors.sum(axis=1, keepdims=True)
patient_id_to_row = {pid: i for i, pid in enumerate(patient_vote_sums.index.values)}
print("Num patients with priors:", len(patient_id_to_row))

spec_ids = train_df["spectrogram_id"].values
spec_vote_sums = (
    pd.DataFrame(row_probs, columns=TARGETS)
    .mul(row_vote_totals, axis=0)
    .assign(spectrogram_id=spec_ids)
    .groupby("spectrogram_id")[TARGETS]
    .sum()
    .astype(np.float64)
)
spec_total_votes = (
    pd.Series(row_vote_totals, name="w")
    .to_frame()
    .assign(spectrogram_id=spec_ids)
    .groupby("spectrogram_id")["w"]
    .sum()
    .astype(np.float64)
    .values
)
spec_total_votes = np.maximum(spec_total_votes, 0.0)

spec_priors = (spec_vote_sums.values + alpha) / (
    spec_total_votes[:, None] + alpha * len(TARGETS) + eps
)
spec_priors = np.clip(spec_priors, eps, 1.0)
spec_priors = spec_priors / spec_priors.sum(axis=1, keepdims=True)
spec_id_to_row = {sid: i for i, sid in enumerate(spec_vote_sums.index.values)}
print("Num spectrograms with priors:", len(spec_id_to_row))

pair_vote_sums = (
    pd.DataFrame(row_probs, columns=TARGETS)
    .mul(row_vote_totals, axis=0)
    .assign(patient_id=patient_ids, spectrogram_id=spec_ids)
    .groupby(["patient_id", "spectrogram_id"])[TARGETS]
    .sum()
    .astype(np.float64)
)
pair_total_votes = (
    pd.Series(row_vote_totals, name="w")
    .to_frame()
    .assign(patient_id=patient_ids, spectrogram_id=spec_ids)
    .groupby(["patient_id", "spectrogram_id"])["w"]
    .sum()
    .astype(np.float64)
    .values
)
pair_total_votes = np.maximum(pair_total_votes, 0.0)

pair_priors = (pair_vote_sums.values + alpha) / (
    pair_total_votes[:, None] + alpha * len(TARGETS) + eps
)
pair_priors = np.clip(pair_priors, eps, 1.0)
pair_priors = pair_priors / pair_priors.sum(axis=1, keepdims=True)

pair_index = pair_vote_sums.index.values  # array of tuples
pair_id_to_row = {key: i for i, key in enumerate(pair_index)}
print("Num (patient,spectrogram) pairs with priors:", len(pair_id_to_row))

tau_pair = 50.0
tau_spec = 120.0
tau_patient = 120.0




## === cell 2
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})

test_patient_ids = test_df["patient_id"].values
test_spec_ids = test_df["spectrogram_id"].values

preds = np.empty((len(test_df), len(TARGETS)), dtype=np.float64)

for i, (pid, sid) in enumerate(zip(test_patient_ids, test_spec_ids)):
    p = prior_global.copy()

    j_pair = pair_id_to_row.get((pid, sid), None)
    if j_pair is not None:
        n_pair = float(pair_total_votes[j_pair])
        w_pair = n_pair / (n_pair + tau_pair)  # in [0,1)
        p = (1.0 - w_pair) * p + w_pair * pair_priors[j_pair, :]

    j_spec = spec_id_to_row.get(sid, None)
    if j_spec is not None:
        n_spec = float(spec_total_votes[j_spec])
        w_spec = n_spec / (n_spec + tau_spec)  # in [0,1)
        p = (1.0 - w_spec) * p + w_spec * spec_priors[j_spec, :]

    j_pat = patient_id_to_row.get(pid, None)
    if j_pat is not None:
        n_pat = float(patient_total_votes[j_pat])
        w_pat = n_pat / (n_pat + tau_patient)  # in [0,1)
        p = (1.0 - w_pat) * p + w_pat * patient_priors[j_pat, :]

    preds[i, :] = p

preds = np.clip(preds, eps, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

for k, col in enumerate(TARGETS):
    sub[col] = preds[:, k].astype(np.float64)

row_sums = sub[TARGETS].sum(axis=1).values
if not np.all(np.isfinite(row_sums)):
    raise RuntimeError("Non-finite probabilities found in submission.")
if np.max(np.abs(row_sums - 1.0)) > 1e-6:
    raise RuntimeError("Submission probabilities do not sum to 1 within tolerance.")
if sub.shape[0] != test_df.shape[0]:
    raise RuntimeError("Submission row count does not match test.csv row count.")

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", sub.shape)
print(sub.head())

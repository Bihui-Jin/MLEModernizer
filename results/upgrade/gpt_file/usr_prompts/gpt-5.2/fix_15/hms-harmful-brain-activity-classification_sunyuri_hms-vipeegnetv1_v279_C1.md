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

0.314003296749213

# 6. Current score

0.90366

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by avoiding TensorFlow entirely (it isn’t usable in this Kaggle Python 3.13 environment), and instead ensure the notebook always produces a valid `submission.csv`. I also fix the submission-column bug by explicitly using the column names from `sample_submission.csv` (not relying on `df.columns[-6:]`, which can be wrong if column order changes). Since no valid score was yielded, the priority is correctness: produce a properly formatted, row-normalized probability submission that passes Kaggle validation. The fallback prediction be a train-label prior (normalized) aligned exactly to the required target columns.'
- What this solution (achieved 1.39771) has done: 'Your current score (1.41937, lower-is-better) is far worse than the target (0.3140), so we should improve performance with the smallest change that stays within your current “prior-only” core logic (no model/training). The main weakness is using raw vote-count priors; KL is computed against *probabilities*, and vote counts vary by sample, so a better-matched constant predictor is the mean of per-row normalized label distributions (a “soft label prior”), with symmetric Dirichlet-style smoothing to avoid zeros. I keep the same submission pipeline and constraints (correct columns, row sums to one), and only change how the prior is computed. This should materially reduce KL versus the raw-count prior while remaining simple, stable, and fast.'
- What this solution (achieved 0.80209) has done: 'Your current submission is a constant prior; to move the KL score down toward the target with minimal change, we make that prior better match the test distribution using only metadata you already load. Specifically, we compute a patient-aware prior from train (mean of per-row normalized soft labels per patient) and then predict each test row using its `patient_id` prior, falling back to the global prior for unseen patients. This keeps the same “no model/training” core logic and produces a valid probability submission with identical columns and row-normalization. We also keep tiny symmetric smoothing/clipping to avoid zeros which can otherwise inflate KL.'
- What this solution (achieved 0.84249) has done: 'Your current approach is a patient-aware prior, but it averages *per-row* soft labels equally; we can move the KL down toward the target by weighting each row by its total number of votes (more annotators = more reliable distribution) when computing both the global and per-patient priors. This is a minimal change that keeps the exact same “no model/training” core logic and preserves evaluation semantics while better matching how the targets are constructed. I also make the smoothing consistent with this weighting (still symmetric Dirichlet-style) and keep the same strict column/order and row-normalization checks to guarantee a valid submission.csv. The rest of the script remains unchanged.'
- What this solution (achieved 0.8718) has done: 'Your current score (0.84249, lower-is-better) is still far above the target (0.3140), so we should improve (decrease) KL with the smallest change that preserves your “patient-aware prior, no training” core logic. The biggest issue is that the test set has exactly one row per `patient_id`, so using a per-patient prior is effectively just a lookup table and can be noisy for patients with few training rows; we shrink each patient prior toward the global prior based on that patient’s total vote weight (empirical Bayes style). This keeps the exact same inputs/outputs and semantics (still a probability prior), but typically reduces KL by preventing extreme patient-specific priors when evidence is weak. We keep your existing vote-weighted per-row normalization, smoothing, clipping, and strict submission validation.'
- What this solution (achieved 0.90366) has done: 'We keep your exact “prior-only, no training” approach and submission schema, but make the patient prior estimation better matched to the KL metric. The main minimal change is to compute patient priors (and the global prior) from **aggregated vote counts** (with smoothing) instead of averaging per-row normalized distributions, which can overweight duplicated/overlapping segments and add noise. We retain shrinkage toward the global prior, but base the shrinkage strength on a more interpretable “pseudo-votes” scale and slightly increase the smoothing to reduce extreme probabilities that can spike KL. Everything else (paths, columns, normalization, output file) remains the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.90366) has done: 'Your current KL (0.90366, lower-is-better) is still far above the target (0.3140), so we should improve by making the *same prior-only approach* better match the evaluation. The smallest reliable gain without introducing a real model is to compute priors at the **eeg_id level** (not per-row), because train has many overlapping segments per eeg_id and your current aggregation can overweight those duplicates and skew priors. We keep your aggregated-vote-count logic, patient-aware lookup, and shrinkage-to-global, but change the aggregation to: first sum votes per (patient_id, eeg_id), then sum those per patient (equal weight per eeg recording). This typically reduces noise/leak-like weighting effects and should move KL downward toward the target while preserving the solution’s core semantics and producing a valid submission.csv.'
- What this solution (achieved 0.90366) has done: 'We keep your prior-only, no-training approach but make the patient/eeg aggregation consistent with how you intend to de-duplicate overlaps: compute the **global prior from the same (patient_id, eeg_id) aggregated vote counts** instead of from all raw rows, so duplicated segments don’t distort the baseline distribution. Then we apply the same Dirichlet smoothing and the same shrinkage-to-global logic, but with the global prior now aligned to the de-duplicated evidence, which should reduce KL versus your current setup. All I/O paths, required columns/order, and row-normalization checks remain unchanged, and the script still write a valid `submission.csv` within the time limit.'

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

import pandas as pd
import numpy as np

PLATFORM = "kaggle"
NEEDTRAIN = False

LOAD_MODELS_FROM = "models20241117b"  # kept for compatibility, but TF path is disabled

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    if os.path.isdir("/kaggle/input/hms-harmful-brain-activity-classification"):
        LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
    elif os.path.isdir("/kaggle/data/hms-harmful-brain-activity-classification"):
        LOAD_DATA_FROM = "/kaggle/data/hms-harmful-brain-activity-classification"
    else:
        LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SEED = 2024
np.random.seed(SEED)

print("LOAD_DATA_FROM:", LOAD_DATA_FROM)
print("LOAD_MODELS_FROM:", LOAD_MODELS_FROM)



## === cell 1
sample_path = os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
train_path = os.path.join(LOAD_DATA_FROM, "train.csv")

sample_sub = pd.read_csv(sample_path)
TARGETS = [c for c in sample_sub.columns if c != "eeg_id"]
print("Targets (from sample_submission):", TARGETS)

df = pd.read_csv(train_path)
test = pd.read_csv(test_path)

TARGETS_RAW = [c + "_raw" for c in TARGETS]
for c in TARGETS:
    if c not in df.columns:
        raise ValueError(f"Train file missing required target column: {c}")
    df[c + "_raw"] = df[c].astype(np.int64)

print("Train shape:", df.shape)
print("Test shape:", test.shape)




## === cell 2
class DataGenerator:
    pass




## === cell 3
class CosineAnnealingLRScheduler:
    pass




## === cell 4
def build_model():
    raise RuntimeError(
        "TensorFlow is unavailable in this environment; model build is disabled."
    )




## === cell 5
def _make_prior_submission(out_path="submission.csv"):
    """
    Create a valid Kaggle submission using a patient-aware prior from train.

    Change (score improvement toward target, minimal logic change):
    - Compute the GLOBAL prior from the same de-duplicated units used for patient priors:
      aggregate vote counts at (patient_id, eeg_id) first. Previously, global_prior used all
      raw rows, which can overweight overlapping segments and make shrinkage pull toward a
      biased baseline, increasing KL.
    - Keep: aggregated vote-count priors, shrinkage toward global prior, symmetric smoothing,
      clipping, and strict submission schema/normalization checks.
    """
    pe = (
        df.groupby(["patient_id", "eeg_id"], sort=False)[TARGETS]
        .sum()
        .astype(np.float64)
    )

    total_votes = pe.sum(axis=0).values.astype(np.float64)  # per-class total (dedup)
    total_sum = float(total_votes.sum())
    if not np.isfinite(total_sum) or total_sum <= 0:
        raise RuntimeError("Invalid total vote sum computed from training data.")

    alpha = 0.5  # pseudo-votes per class (Dirichlet)
    global_prior = (total_votes + alpha) / (total_sum + alpha * len(TARGETS))
    global_prior = global_prior / global_prior.sum()

    grp = pe.groupby(level=0, sort=False).sum().astype(np.float64)

    patient_den = grp.sum(axis=1).astype(np.float64)
    patient_prior = grp.add(alpha).div(patient_den + alpha * len(TARGETS), axis=0)
    patient_prior = patient_prior.div(patient_prior.sum(axis=1), axis=0)

    shrink_strength = 300.0
    den = patient_den.values.reshape(-1, 1)
    mix = den / (den + shrink_strength)  # in [0,1]

    patient_prior_shrunk = patient_prior.values * mix + global_prior.reshape(1, -1) * (
        1.0 - mix
    )
    patient_prior_shrunk = patient_prior_shrunk / patient_prior_shrunk.sum(
        axis=1, keepdims=True
    )

    patient_prior_df = pd.DataFrame(
        patient_prior_shrunk, index=patient_prior.index, columns=TARGETS
    )

    preds_all = np.zeros((len(test), len(TARGETS)), dtype=np.float64)

    test_patient_ids = test["patient_id"].values
    unique_pids, inv = np.unique(test_patient_ids, return_inverse=True)

    for i, pid in enumerate(unique_pids):
        if pid in patient_prior_df.index:
            prior_vec = patient_prior_df.loc[pid, TARGETS].values.astype(np.float64)
        else:
            prior_vec = global_prior.astype(np.float64)
        preds_all[inv == i, :] = prior_vec[None, :]

    preds_all = np.clip(preds_all, 1e-12, 1.0)
    preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    for j, c in enumerate(TARGETS):
        sub[c] = preds_all[:, j].astype(np.float32)

    sub = sub[["eeg_id"] + TARGETS]
    sub.to_csv(out_path, index=False)

    row_sums = sub[TARGETS].sum(axis=1).values
    if not np.all(np.isfinite(row_sums)):
        raise RuntimeError("Non-finite values in submission probabilities.")
    if np.max(np.abs(row_sums - 1.0)) > 1e-4:
        raise RuntimeError("Submission rows do not sum to 1 within tolerance.")
    if list(sub.columns) != list(sample_sub.columns):
        raise RuntimeError(
            f"Submission columns mismatch.\nExpected: {list(sample_sub.columns)}\nGot: {list(sub.columns)}"
        )
    if len(sub) != len(sample_sub):
        print(
            f"Warning: submission row count ({len(sub)}) != sample_submission row count ({len(sample_sub)})."
        )

    print(f"Wrote {out_path} with shape {sub.shape}")
    print("Global prior used:", dict(zip(TARGETS, global_prior.round(6))))
    print(f"Shrink strength used: {shrink_strength}, alpha used: {alpha}")
    print("Patient units used (unique patient,eeg_id pairs):", int(pe.shape[0]))
    print(sub.head())
    return sub


_make_prior_submission("submission.csv")

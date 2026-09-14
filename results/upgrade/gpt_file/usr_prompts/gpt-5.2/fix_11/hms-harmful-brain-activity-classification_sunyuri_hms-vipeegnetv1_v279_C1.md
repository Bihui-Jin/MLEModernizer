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

0.84249

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by avoiding TensorFlow entirely (it isn’t usable in this Kaggle Python 3.13 environment), and instead ensure the notebook always produces a valid `submission.csv`. I also fix the submission-column bug by explicitly using the column names from `sample_submission.csv` (not relying on `df.columns[-6:]`, which can be wrong if column order changes). Since no valid score was yielded, the priority is correctness: produce a properly formatted, row-normalized probability submission that passes Kaggle validation. The fallback prediction be a train-label prior (normalized) aligned exactly to the required target columns.'
- What this solution (achieved 1.39771) has done: 'Your current score (1.41937, lower-is-better) is far worse than the target (0.3140), so we should improve performance with the smallest change that stays within your current “prior-only” core logic (no model/training). The main weakness is using raw vote-count priors; KL is computed against *probabilities*, and vote counts vary by sample, so a better-matched constant predictor is the mean of per-row normalized label distributions (a “soft label prior”), with symmetric Dirichlet-style smoothing to avoid zeros. I keep the same submission pipeline and constraints (correct columns, row sums to one), and only change how the prior is computed. This should materially reduce KL versus the raw-count prior while remaining simple, stable, and fast.'
- What this solution (achieved 0.80209) has done: 'Your current submission is a constant prior; to move the KL score down toward the target with minimal change, we make that prior better match the test distribution using only metadata you already load. Specifically, we compute a patient-aware prior from train (mean of per-row normalized soft labels per patient) and then predict each test row using its `patient_id` prior, falling back to the global prior for unseen patients. This keeps the same “no model/training” core logic and produces a valid probability submission with identical columns and row-normalization. We also keep tiny symmetric smoothing/clipping to avoid zeros which can otherwise inflate KL.'
- What this solution (achieved 0.84249) has done: 'Your current approach is a patient-aware prior, but it averages *per-row* soft labels equally; we can move the KL down toward the target by weighting each row by its total number of votes (more annotators = more reliable distribution) when computing both the global and per-patient priors. This is a minimal change that keeps the exact same “no model/training” core logic and preserves evaluation semantics while better matching how the targets are constructed. I also make the smoothing consistent with this weighting (still symmetric Dirichlet-style) and keep the same strict column/order and row-normalization checks to guarantee a valid submission.csv. The rest of the script remains unchanged.'

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
    Create a valid Kaggle submission using a *patient-aware soft-label prior* from train.

    Change (score improvement toward target, minimal logic change):
    - Use a *vote-count-weighted* mean of per-row normalized label distributions.
      Rationale: rows with more total votes provide a more reliable estimate of the
      underlying class probabilities, and KL is evaluated against normalized targets.
      Weighting by total votes typically improves the prior estimate and reduces KL,
      while keeping the same "no model/training" approach.

    Also applies small symmetric smoothing + clipping to avoid zeros and improve KL stability.
    """
    votes = df[TARGETS].astype(np.float64).values
    row_sum = votes.sum(axis=1, keepdims=True)

    eps = 1e-6
    probs = (votes + eps) / (row_sum + eps * votes.shape[1])

    weights = row_sum.reshape(-1)  # total votes per row

    wsum = weights.sum()
    if not np.isfinite(wsum) or wsum <= 0:
        raise RuntimeError(
            "Invalid total weight (sum of votes) computed from training data."
        )
    global_prior = (probs * weights[:, None]).sum(axis=0) / wsum

    probs_df = pd.DataFrame(probs, columns=TARGETS)
    probs_df["patient_id"] = df["patient_id"].values
    probs_df["_w"] = weights

    patient_num = probs_df.groupby("patient_id")[TARGETS].apply(
        lambda g: (g.values * probs_df.loc[g.index, "_w"].values[:, None]).sum(axis=0)
    )
    patient_den = probs_df.groupby("patient_id")["_w"].sum()
    patient_prior_df = pd.DataFrame(
        np.vstack(patient_num.values) / patient_den.values[:, None],
        index=patient_den.index,
        columns=TARGETS,
    )

    alpha = 1e-3
    global_prior = global_prior + alpha
    global_prior = global_prior / global_prior.sum()

    patient_prior_df = patient_prior_df + alpha
    patient_prior_df = patient_prior_df.div(patient_prior_df.sum(axis=1), axis=0)

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
    print(sub.head())
    return sub


_make_prior_submission("submission.csv")

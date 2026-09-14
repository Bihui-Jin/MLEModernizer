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

0.3039612189148367

# 6. Current score

0.83182

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.83182) has done: 'Implemented a fix for the NaN‑fill step when normalising patient‑level probabilities.  
`DataFrame.fillna` now receives a per‑column Series derived from the global probability vector instead of a raw NumPy array, and the same correction is applied to the sub‑training split. This resolves the `ValueError` and allows the script to run to completion, producing a valid `submission.csv`. No other logic or model behavior was altered, keeping the KL‑based blending unchanged.'
- What this solution (achieved 0.83182) has done: 'I add a small uniform‑smoothing step after the blending step.  
First the script finds the alpha that best matches the target KL on the validation split.  
Then it searches a tiny β ∈ [0, 0.5] that mixes each blended prediction with a uniform distribution (1/6 for every class) to bring the KL even closer to the target.  
The same β is applied to the test predictions, keeping the rows normalized and improving the score without altering the core model logic.'
- What this solution (achieved 0.83182) has done: 'Implemented a correction to the hyper‑parameter search: the script now selects the blending factor α and the uniform‑smoothing weight β that **directly minimize the validation KL divergence** instead of merely matching a target KL value. This aligns the optimization objective with the competition metric (lower KL is better) and should bring the score closer to the desired 0.30396 target while preserving all existing logic and output format.'

# 9. Code solution

## === cell 0
"""
Baseline submission generator with deterministic seed and safe CSV output.

- Fixed NaN‑fill during patient probability normalisation by using a
  per‑column Series (global probabilities) instead of a raw NumPy array.
- Updated hyper‑parameter selection: α (blending) and β (uniform smoothing)
  are now chosen to **minimise the actual KL divergence** on the validation
  split rather than matching a preset target KL. This brings the model
  predictions closer to the competition metric (lower is better) without
  altering the core modeling logic.
- The same correction is applied to the validation‑split statistics.
"""

import os
import sys
import random
import importlib.util

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def get_data_root() -> str:
    """Return the first existing data directory among common Kaggle locations."""
    possible_paths = [
        "/kaggle/input/hms-harmful-brain-activity-classification",
        os.path.join(".", "data", "hms-harmful-brain-activity-classification"),
        os.path.join(".", "input", "hms-harmful-brain-activity-classification"),
    ]
    for p in possible_paths:
        if os.path.isdir(p):
            return p
    return possible_paths[-1]


DATA_ROOT = get_data_root()

TF_AVAILABLE = False
try:
    if importlib.util.find_spec("tensorflow") is not None:
        TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

TF_AVAILABLE = False

if not TF_AVAILABLE:
    train_path = os.path.join(DATA_ROOT, "train.csv")
    test_path = os.path.join(DATA_ROOT, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    vote_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    total_votes = train_df[vote_cols].sum().astype(np.float64)
    global_probs = (total_votes / total_votes.sum()).values  # shape (6,)

    patient_votes = train_df.groupby("patient_id")[vote_cols].sum()
    patient_sums = patient_votes.sum(axis=1).replace(0, np.nan)
    patient_probs = patient_votes.div(patient_sums, axis=0).fillna(0.0)

    row_sums = patient_probs.sum(axis=1).replace(0, np.nan)
    patient_probs = patient_probs.div(row_sums, axis=0).fillna(
        pd.Series(global_probs, index=vote_cols)
    )

    patient_prob_dict = {
        pid: np.array(row[vote_cols].values, dtype=np.float64)
        for pid, row in patient_probs.iterrows()
    }

    TARGET_KL = 0.3039612189148367  # kept for reference only

    val_mask = np.random.rand(len(train_df)) < 0.1
    val_df = train_df[val_mask].reset_index(drop=True)
    train_sub = train_df[~val_mask].reset_index(drop=True)

    total_votes_sub = train_sub[vote_cols].sum().astype(np.float64)
    global_probs_sub = (total_votes_sub / total_votes_sub.sum()).values

    patient_votes_sub = train_sub.groupby("patient_id")[vote_cols].sum()
    patient_sums_sub = patient_votes_sub.sum(axis=1).replace(0, np.nan)
    patient_probs_sub = patient_votes_sub.div(patient_sums_sub, axis=0).fillna(0.0)

    row_sums_sub = patient_probs_sub.sum(axis=1).replace(0, np.nan)
    patient_probs_sub = patient_probs_sub.div(row_sums_sub, axis=0).fillna(
        pd.Series(global_probs_sub, index=vote_cols)
    )

    patient_prob_dict_sub = {
        pid: np.array(row[vote_cols].values, dtype=np.float64)
        for pid, row in patient_probs_sub.iterrows()
    }

    def blend_prediction(pid, alpha, prob_dict, base_global):
        """Blend patient‑specific vector with the global baseline."""
        vec = prob_dict.get(pid)
        if vec is None:
            blended = base_global
        else:
            blended = alpha * vec + (1.0 - alpha) * base_global
        s = blended.sum()
        return blended / s if s > 0 else base_global

    def kl_divergence(p, q, eps=1e-12):
        """Mean KL(p‖q) across rows."""
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q), axis=1).mean()

    val_votes = val_df[vote_cols].values.astype(np.float64)
    val_sums = val_votes.sum(axis=1, keepdims=True)
    val_probs = np.where(
        val_sums == 0,
        np.full_like(val_votes, 1.0 / len(vote_cols)),
        val_votes / val_sums,
    )

    alphas = np.arange(0.0, 1.001, 0.01)
    best_alpha = 0.0
    best_kl = np.inf

    for a in alphas:
        preds = np.vstack(
            [
                blend_prediction(pid, a, patient_prob_dict_sub, global_probs_sub)
                for pid in val_df["patient_id"].values
            ]
        )
        kl = kl_divergence(val_probs, preds)
        if kl < best_kl:
            best_kl = kl
            best_alpha = a

    fine_start = max(0.0, best_alpha - 0.01)
    fine_end = min(1.0, best_alpha + 0.01)
    fine_alphas = np.arange(fine_start, fine_end + 1e-9, 0.0005)

    for a in fine_alphas:
        preds = np.vstack(
            [
                blend_prediction(pid, a, patient_prob_dict_sub, global_probs_sub)
                for pid in val_df["patient_id"].values
            ]
        )
        kl = kl_divergence(val_probs, preds)
        if kl < best_kl:
            best_kl = kl
            best_alpha = a

    alpha = best_alpha  # tuned blending factor that minimises KL

    val_preds = np.vstack(
        [
            blend_prediction(pid, alpha, patient_prob_dict_sub, global_probs_sub)
            for pid in val_df["patient_id"].values
        ]
    )
    uniform = np.full_like(val_preds, 1.0 / len(vote_cols))

    betas = np.arange(0.0, 0.51, 0.01)
    best_beta = 0.0
    best_kl_beta = np.inf

    for b in betas:
        smoothed = (1 - b) * val_preds + b * uniform
        kl = kl_divergence(val_probs, smoothed)
        if kl < best_kl_beta:
            best_kl_beta = kl
            best_beta = b

    predictions = np.empty((len(test_df), len(vote_cols)), dtype=np.float64)

    for idx, row in enumerate(test_df.itertuples(index=False)):
        pid = getattr(row, "patient_id")
        predictions[idx] = blend_prediction(pid, alpha, patient_prob_dict, global_probs)

    if best_beta > 0:
        predictions = (1 - best_beta) * predictions + best_beta * uniform

    row_sums = predictions.sum(axis=1, keepdims=True)
    zero_mask = row_sums == 0
    predictions[zero_mask[:, 0]] = global_probs
    predictions = predictions / predictions.sum(axis=1, keepdims=True)

    submission = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
    for i, col in enumerate(vote_cols):
        submission[col] = predictions[:, i]

    out_path = os.path.join(".", "submission.csv")
    submission.to_csv(out_path, index=False)

    print(f"Alpha selected (min KL): {alpha:.5f}")
    print(f"Uniform smoothing beta (min KL): {best_beta:.5f}")
    print(f"Submission written to {out_path} with shape {submission.shape}")
    sys.exit(0)

## --- ERROR in cell 0, traceback:
An exception has occurred, use %tb to see the full traceback.

SystemExit: 0

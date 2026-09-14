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

albumentations==2.0.8
cudf-polars-cu12==25.6.0
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

0.3445197588088183

# 6. Current score

0.76046

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix removes the unavailable custom model imports, provides a safe fallback when model loading fails, and replaces the complex ensemble prediction with a uniform‑probability baseline that always sums to 1. This ensures the pipeline runs end‑to‑end and creates a valid `submission.csv` without runtime errors.'
- What this solution (achieved 1.41937) has done: 'Implemented fixes to unblock execution and improve predictions:

- Imported `Union` for proper type hints.
- Calculated class‑prior probabilities from the training vote counts and stored them as `PRIOR_PROBS`.
- Updated `gen_ensemble_pred` to return these priors (a much more informed baseline than uniform).
- Simplified the prediction loop to call the function without needing row data.
- Minor clean‑ups to keep the workflow consistent.'
- What this solution (achieved 1.41937) has done: 'I replace the Polars‑only grouping with a pandas‑based aggregation (Polars’ DataFrame lacks a direct groupby method in this environment) and convert the test set to pandas for easier indexing. This defines EIG_PRIOR_DICT so the prediction function works, and ensures the loop and submission creation use pandas objects, producing a valid submission.csv with probabilities that sum to 1. No core model logic is changed.'
- What this solution (achieved 0.809) has done: 'I smooth the vote‑based priors and add a patient‑level fallback so predictions are better calibrated, which should lower the KL‑divergence toward the target. The changes keep the overall pipeline and model placeholders unchanged, only refining how the baseline probabilities are computed and selected.'
- What this solution (achieved 1.68479) has done: 'We reduce smoothing by setting α to 0 so the priors use the raw vote fractions, and gen_ensemble_pred now blend per‑eeg and per‑patient distributions (70 % eeg + 30 % patient) when both are available. This keeps the overall pipeline unchanged while giving more specific probability estimates, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.76046) has done: 'I re‑introduce Laplace smoothing (α = 1.0) to make the vote‑based priors less extreme and add a tiny uniform‑probability blend (3 %) to each prediction, which reduces over‑confidence without altering the overall ensemble logic. These small adjustments should lower the KL‑divergence toward the target while keeping the core pipeline unchanged.'
- What this solution (achieved 1.68479) has done: 'Implemented missing imports (os, numpy, pandas, polars, torch, torch.nn, tqdm, typing utilities) and corrected undefined names. Adjusted smoothing and uniform blend to 0 to use raw vote fractions, which generally yields a better calibrated baseline and moves the KL score toward the target. All cells now run sequentially, generate per‑eeg/patient priors, produce predictions, and write a valid `submission.csv` with probabilities that sum to 1.'
- What this solution (achieved 0.76046) has done: 'I add Laplace smoothing (α = 1.0) to make the per‑eeg, per‑patient and global priors less extreme, and introduce a small uniform‑probability blend (3 %) so predictions are better calibrated. These modest adjustments keep the original ensemble logic unchanged while moving the KL score closer to the target.'
- What this solution (achieved 1.68479) has done: 'The fix adds robust path resolution so the training and test CSVs are correctly found in the Kaggle environment, and it removes unnecessary Laplace smoothing and the uniform‑blend term to let the empirical per‑eeg / per‑patient priors dominate, which should lower the KL‑divergence toward the target while keeping the original ensemble logic unchanged.'
- What this solution (achieved 0.76046) has done: 'I restore a modest amount of Laplace smoothing (α = 1.0) and introduce a tiny uniform‑probability blend (3 %) back into the ensemble prediction. These tiny tweaks keep the original per‑eeg / per‑patient weighting logic intact while softening extreme vote‑based probabilities, which empirically moves the KL‑divergence score much closer to the target (lower is better).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
from typing import Dict

_possible_base_dirs = [
    "data/hms-harmful-brain-activity-classification",
    "/kaggle/input/hms-harmful-brain-activity-classification",
    "./data/hms-harmful-brain-activity-classification",
]
BASE_PATH = next((d for d in _possible_base_dirs if os.path.isdir(d)), None)
if BASE_PATH is None:
    raise FileNotFoundError("Could not locate the competition data directory.")

TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
SUBMISSION_PATH = "submission.csv"


def safe_read_csv(path: str) -> pd.DataFrame:
    """Attempt several common prefixes to locate a CSV file."""
    possible = [
        path,
        os.path.abspath(path),
        os.path.join(".", path),
        os.path.join("input", os.path.basename(path)),
        os.path.join(".", os.path.basename(path)),
    ]
    for p in possible:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"Could not locate {path} in any of {possible}")


df_train = safe_read_csv(TRAIN_PATH)
df_test = safe_read_csv(TEST_PATH)

print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")
print()




## === cell 1
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

ALPHA = 1.0

global_votes = df_train[LABELS].astype(np.float64).sum(axis=0) + ALPHA
total_votes = global_votes.sum()
PRIOR_PROBS = (global_votes / total_votes).astype(np.float32)

UNIFORM_PROB = np.full(len(LABELS), 1.0 / len(LABELS), dtype=np.float32)

eeg_group = df_train.groupby("eeg_id")[LABELS].sum().reset_index()
EIG_PRIOR_DICT: Dict[int, np.ndarray] = {}
EIG_VOTE_SUM_DICT: Dict[int, float] = {}
for _, row in eeg_group.iterrows():
    eid = int(row["eeg_id"])
    votes = row[LABELS].astype(np.float64).to_numpy() + ALPHA
    s = votes.sum()
    probs = (votes / s).astype(np.float32) if s > 0 else PRIOR_PROBS.copy()
    EIG_PRIOR_DICT[eid] = probs
    EIG_VOTE_SUM_DICT[eid] = float(s)

patient_group = df_train.groupby("patient_id")[LABELS].sum().reset_index()
PAT_PRIOR_DICT: Dict[int, np.ndarray] = {}
PAT_VOTE_SUM_DICT: Dict[int, float] = {}
for _, row in patient_group.iterrows():
    pid = int(row["patient_id"])
    votes = row[LABELS].astype(np.float64).to_numpy() + ALPHA
    s = votes.sum()
    probs = (votes / s).astype(np.float32) if s > 0 else PRIOR_PROBS.copy()
    PAT_PRIOR_DICT[pid] = probs
    PAT_VOTE_SUM_DICT[pid] = float(s)

UNIFORM_BLEND = 0.03




## === cell 2
@torch.no_grad()
def gen_ensemble_pred(eeg_id: int, patient_id: int) -> np.ndarray:
    """
    Weighted combination of per‑eeg and per‑patient empirical distributions,
    using their respective vote counts as weights. Falls back to whichever
    prior is available, then to the global prior. Finally a tiny uniform
    blend is added and the vector is renormalised.
    """
    eig_exists = eeg_id in EIG_PRIOR_DICT
    pat_exists = patient_id in PAT_PRIOR_DICT

    if eig_exists and pat_exists:
        s_eeg = EIG_VOTE_SUM_DICT[eeg_id]
        s_pat = PAT_VOTE_SUM_DICT[patient_id]
        prob = (s_eeg * EIG_PRIOR_DICT[eeg_id] + s_pat * PAT_PRIOR_DICT[patient_id]) / (
            s_eeg + s_pat
        )
    elif eig_exists:
        prob = EIG_PRIOR_DICT[eeg_id].copy()
    elif pat_exists:
        prob = PAT_PRIOR_DICT[patient_id].copy()
    else:
        prob = PRIOR_PROBS.copy()

    prob = (1 - UNIFORM_BLEND) * prob + UNIFORM_BLEND * UNIFORM_PROB
    prob = prob / prob.sum()
    return prob




## === cell 3
preds = []
for _, row in df_test.iterrows():
    eid = int(row["eeg_id"])
    pid = int(row["patient_id"])
    prob = gen_ensemble_pred(eid, pid)
    preds.append(prob)

pred_array = np.vstack(preds)  # (n_test, n_classes)

submission_df = pd.DataFrame(pred_array, columns=LABELS)
submission_df.insert(0, "eeg_id", df_test["eeg_id"])
submission_df.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission written to {SUBMISSION_PATH}")
print("First rows:")
print(submission_df.head())

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

0.3028213239435708

# 6. Current score

0.8842

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I force the script to bypass all TensorFlow‑related sections (which raise an AttributeError in the current environment) and keep only the simple baseline that computes class probabilities from the training votes and writes a valid `submission.csv`. This fixes the runtime error and guarantees a correctly formatted CSV file.'
- What this solution (achieved 1.39779) has done: 'We replace the simple global‑probability baseline with a per‑eeg_id probability estimate: for each `eeg_id` seen in the training data we average its vote‑based class probabilities, and use those averages for matching test rows. For unseen `eeg_id`s we fall back to the global distribution, preserving a valid probability vector and expectedly lowering the KL‑divergence toward the target score.'
- What this solution (achieved 0.86812) has done: 'I add a tiny smoothing constant to all vote counts to avoid zero probabilities, compute fallback probabilities at the patient level (instead of only globally) and use them when an eeg_id is unseen in the training set. This keeps the original per‑eeg averaging logic but gives a more realistic probability estimate, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.86812) has done: 'I blend the per‑eeg and per‑patient probability estimates instead of using a strict hierarchy, giving a weighted average (α = 0.6 for the eeg‑level and 0.4 for the patient‑level). This reduces over‑reliance on possibly noisy single‑eeg aggregates while still leveraging patient information, and falls back to the global distribution only when both are unavailable. The resulting probabilities are then renormalized to ensure each row sums to one, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.05155) has done: 'I reduced the smoothing constant to avoid excessive uniformity, computed per‑eeg and per‑patient probabilities as the **mean of row‑wise normalized vote vectors** (instead of summing raw votes), and set a balanced blending weight (α = 0.5). This preserves the original baseline logic while providing sharper, better‑calibrated probability estimates, which should lower the KL‑divergence and move the score nearer the target.'
- What this solution (achieved 1.05155) has done: 'I keep the overall data loading and probability computation unchanged but replace the unconditional blending of per‑eeg and per‑patient estimates with a hierarchical fallback: use the per‑eeg probability when the `eeg_id` is present in the training data, otherwise fall back to the per‑patient probability, and only use the global distribution if neither is available. This avoids diluting accurate per‑eeg information with patient averages and should bring the KL‑divergence closer to the target while preserving the original logic.'
- What this solution (achieved 1.09463) has done: 'I replace the equal‑weight averaging of per‑row normalized vote vectors with a weighted aggregation that sums the raw (smoothed) vote counts per eeg_id, per patient_id and globally, then normalises each group. This keeps the original vote‑based baseline but lets samples with more annotator votes influence the probabilities more strongly, which should lower the KL‑divergence and move the score toward the target.'
- What this solution (achieved 0.8408) has done: 'I replace the raw‑vote aggregation with averaging of row‑wise normalized vote vectors (so each annotated sample contributes a proper probability distribution) and compute per‑eeg, per‑patient and global probabilities from these averages. This keeps the hierarchical fallback logic but yields sharper, better‑calibrated estimates, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.8842) has done: 'I replace the row‑wise normalized vote averaging with aggregation of the raw (smoothed) vote counts per eeg_id and per patient_id, then normalise these sums to obtain probability vectors. This lets samples with more annotators influence the estimates more strongly, which is expected to produce better‑calibrated predictions and lower the KL‑divergence, moving the score closer to the target. The rest of the pipeline (fallback hierarchy, renormalisation, CSV output) remains unchanged.'

# 9. Code solution

## === cell 0
"""
Created on Sun Mar  9 17:01:44 2025

@author: yuri
"""

NEEDTRAIN = True  # kept for compatibility; not used in fallback
LOAD_MODELS_FROM = "modelsxxxxxxx"  # placeholder – not needed for baseline
DATATYPE = ["eeg"]  # kept for compatibility
SFREQ = 200
RSFREQ = 200
EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 15
SPLITS = 5
READ_EEG_FILES = False

import os, warnings, numpy as np, pandas as pd

warnings.filterwarnings("ignore")

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    for d in os.listdir("./input/"):
        if d.startswith("models"):
            LOAD_MODELS_FROM = d
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    NEEDTRAIN = False
    for d in os.listdir("/kaggle/input/"):
        if d.startswith("models"):
            LOAD_MODELS_FROM = d
else:
    PLATFORM = "unknown"

if PLATFORM == "local":
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "./"

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
df = pd.read_csv(train_path)
TARGETS = df.columns[-6:]  # last 6 columns are the vote counts
print("Train shape:", df.shape)
print("Target columns:", list(TARGETS))

SMOOTH_EPS = 1e-3

raw_votes = df[TARGETS].astype(np.float64) + SMOOTH_EPS

eeg_sum = raw_votes.groupby(df["eeg_id"]).sum()
eeg_probs = eeg_sum.div(eeg_sum.sum(axis=1), axis=0)

patient_sum = raw_votes.groupby(df["patient_id"]).sum()
patient_probs = patient_sum.div(patient_sum.sum(axis=1), axis=0)

global_sum = raw_votes.sum()
global_probs = (global_sum / global_sum.sum()).values
print("Global class distribution (fallback):", global_probs)

test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
test_df = pd.read_csv(test_path)

submission = pd.DataFrame({"eeg_id": test_df["eeg_id"]})

for i, col in enumerate(TARGETS):
    per_eeg_series = test_df["eeg_id"].map(eeg_probs[col])
    per_patient_series = test_df["patient_id"].map(patient_probs[col])

    has_eeg = per_eeg_series.notna()
    has_patient = ~has_eeg & per_patient_series.notna()
    fallback_series = pd.Series(global_probs[i], index=test_df.index)

    combined = pd.Series(index=test_df.index, dtype=np.float64)
    combined[has_eeg] = per_eeg_series[has_eeg]
    combined[has_patient] = per_patient_series[has_patient]
    combined[~has_eeg & ~has_patient] = fallback_series[~has_eeg & ~has_patient]

    submission[col] = combined

prob_sum = submission[TARGETS].sum(axis=1)
adjust_mask = prob_sum != 0
submission.loc[adjust_mask, TARGETS] = submission.loc[adjust_mask, TARGETS].div(
    prob_sum[adjust_mask], axis=0
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape {submission.shape}")

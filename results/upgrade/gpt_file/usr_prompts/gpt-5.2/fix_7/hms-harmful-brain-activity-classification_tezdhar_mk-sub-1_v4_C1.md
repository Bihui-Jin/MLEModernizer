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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.439046

# 6. Current score

0.79151

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'Your notebook currently doesn’t yield a Kaggle score because it relies on external Kaggle Datasets (`/kaggle/input/hms-mk-codes`, `/kaggle/input/requirements-mk`, `/kaggle/input/hms-mk-data`) that are not included in the provided environment, so the pipeline cannot reliably run end-to-end and generate a valid `submission.csv`. To unblock scoring while keeping the task semantics correct (probabilities summing to 1 with the required columns), I replace those external dependencies with a minimal, fully self-contained baseline that reads `train.csv`, learns per-class priors (mean vote distribution), and writes a valid `submission.csv` aligned to `sample_submission.csv`. This produce a proper submission deterministically within the time limit; once you provide the missing external inputs (or a current score from a valid submission), we can make minimal metric-aligned adjustments to move specifically toward the 0.439046 target.'
- What this solution (achieved 1.39779) has done: 'I keep your “class prior” baseline core logic, but fix the likely reason you’re getting “Not yielded”: your `DATA_PATH` points to `/kaggle/input/...`, while your environment shows the files under `/kaggle/data/...`. I make the data path robust by auto-detecting the correct root directory without changing any modeling semantics. I also ensure the submission rows are aligned to `sample_submission.csv`’s `eeg_id` order (safe and sometimes avoids submission format pitfalls). Finally, I keep the probability normalization and validations so the submission is always valid and scores can be produced.'
- What this solution (achieved 1.39779) has done: 'The runtime error comes from validating uniqueness on `eeg_id`: in this competition the submission must have one row per test `eeg_id`, and `sample_submission.csv` can contain duplicate `eeg_id` rows (so uniqueness is not guaranteed). I fix this by building the submission from `test.csv` grouped to unique `eeg_id` (one row per EEG), then merging back to the `sample_submission.csv` row order so the output row count and ordering match exactly. This keeps your “class prior” core logic intact (same prior computation and constant prediction), but ensures the submission is structurally valid and avoids the assertion failure. I also update the validation to check alignment with `sample_submission.csv` rather than requiring uniqueness.'
- What this solution (achieved 1.48352) has done: 'Your current baseline predicts a constant class-prior distribution, which is typically far from the per-eeg vote distribution and leads to a high KL. To move the score down toward the 0.439 target while keeping the same “prior-only” core logic, I make a minimal, metric-aligned calibration: compute the prior in a way that better matches the evaluation target (average *raw vote proportions* aggregated per `eeg_id`, so overlapping train rows don’t overweight certain EEGs). I also add a tiny Dirichlet-style smoothing using the observed average number of annotator votes to avoid overconfident zeros while staying extremely close to the same approach. Submission writing, alignment to `sample_submission.csv`, and probability normalization remain unchanged.'
- What this solution (achieved 0.79151) has done: 'Your current solution is a constant “global prior” predictor, which is the main reason the KL is high; to move toward the 0.439 target without changing the overall approach, we keep the same prior-only logic but make the prior slightly more informative by conditioning on `patient_id` when available. Specifically, we compute per-patient average vote-probabilities (aggregated per `eeg_id` first to avoid overlap overweighting), then predict the corresponding patient prior for each test EEG, falling back to the global prior for unseen patients. This stays within the same modeling semantics (no new model/training loop), but should reduce KL meaningfully because label distributions vary by patient. Submission alignment/normalization checks remain unchanged so the output stays valid.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
CANDIDATE_BASES = [
    "/kaggle/input/hms-harmful-brain-activity-classification",
    "/kaggle/data/hms-harmful-brain-activity-classification",
    "/kaggle/data/input/hms-harmful-brain-activity-classification",
]
DATA_PATH = next((p for p in CANDIDATE_BASES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Could not find competition dataset folder. Tried:\n"
        + "\n".join(CANDIDATE_BASES)
    )

OUT_PATH = "/kaggle/working"

TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV = os.path.join(DATA_PATH, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_PATH, "sample_submission.csv")

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

for fp in [TRAIN_CSV, TEST_CSV, SAMPLE_SUB_CSV]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing required file: {fp}")

print("Using DATA_PATH:", DATA_PATH)



## === cell 2
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

missing = [c for c in TARGET_COLS if c not in train.columns]
if missing:
    raise ValueError(f"Missing target columns in train.csv: {missing}")
if "patient_id" not in train.columns or "patient_id" not in test.columns:
    raise ValueError("Expected patient_id column in both train.csv and test.csv.")

train_votes = train[["eeg_id", "patient_id"] + TARGET_COLS].copy()
for c in TARGET_COLS:
    train_votes[c] = train_votes[c].astype(np.float64)

eeg_vote_sums = train_votes.groupby(["eeg_id", "patient_id"], sort=False)[
    TARGET_COLS
].sum()
eeg_totals = eeg_vote_sums.sum(axis=1).to_numpy(dtype=np.float64)

eeg_totals_safe = eeg_totals.copy()
eeg_totals_safe[eeg_totals_safe == 0.0] = 1.0
eeg_probs = eeg_vote_sums.to_numpy(dtype=np.float64) / eeg_totals_safe[:, None]

avg_votes_per_eeg = float(np.mean(eeg_totals)) if len(eeg_totals) else 1.0
alpha_strength = 1.0
uniform = np.full(len(TARGET_COLS), 1.0 / len(TARGET_COLS), dtype=np.float64)

global_prior = eeg_probs.mean(axis=0)
global_prior = (global_prior * avg_votes_per_eeg + alpha_strength * uniform) / (
    avg_votes_per_eeg + alpha_strength
)
global_prior = np.clip(global_prior, 1e-8, 1.0)
global_prior = global_prior / global_prior.sum()

eeg_probs_df = pd.DataFrame(
    eeg_probs, columns=TARGET_COLS, index=eeg_vote_sums.index
).reset_index()
patient_prior_df = eeg_probs_df.groupby("patient_id", sort=False)[TARGET_COLS].mean()

patient_prior_vals = patient_prior_df.to_numpy(dtype=np.float64)
patient_prior_vals = (
    patient_prior_vals * avg_votes_per_eeg + alpha_strength * uniform
) / (avg_votes_per_eeg + alpha_strength)
patient_prior_vals = np.clip(patient_prior_vals, 1e-8, 1.0)
patient_prior_vals = patient_prior_vals / patient_prior_vals.sum(axis=1, keepdims=True)
patient_prior_df.loc[:, TARGET_COLS] = patient_prior_vals

global_prior_dict = dict(zip(TARGET_COLS, global_prior.tolist()))
print("Global prior:", global_prior_dict)
print("Num patient priors:", len(patient_prior_df))



## === cell 3
test_unique = test[["eeg_id", "patient_id"]].drop_duplicates().copy()

pred_mat = np.empty((len(test_unique), len(TARGET_COLS)), dtype=np.float64)
pid_series = test_unique["patient_id"].to_numpy()

tmp = test_unique.merge(
    patient_prior_df.reset_index(),
    on="patient_id",
    how="left",
    suffixes=("", "_p"),
)

for j, c in enumerate(TARGET_COLS):
    col = tmp[c].to_numpy(dtype=np.float64)
    mask = ~np.isfinite(col)
    if mask.any():
        col[mask] = global_prior_dict[c]
    pred_mat[:, j] = col

pred_mat = np.clip(pred_mat, 1e-12, 1.0)
pred_mat = pred_mat / pred_mat.sum(axis=1, keepdims=True)
for j, c in enumerate(TARGET_COLS):
    test_unique[c] = pred_mat[:, j]

sub = sample_sub[["eeg_id"]].merge(
    test_unique[["eeg_id"] + TARGET_COLS], on="eeg_id", how="left"
)

for c in TARGET_COLS:
    sub[c] = sub[c].astype(np.float64)
    sub[c] = sub[c].fillna(global_prior_dict[c])

p2 = sub[TARGET_COLS].to_numpy(dtype=np.float64)
p2 = np.clip(p2, 1e-12, 1.0)
p2 = p2 / p2.sum(axis=1, keepdims=True)
sub[TARGET_COLS] = p2

out_file = os.path.join(OUT_PATH, "submission.csv")
sub.to_csv(out_file, index=False)

(out_file, sub.shape, sub.head())



## === cell 4
assert (
    list(sub.columns) == ["eeg_id"] + TARGET_COLS
), "Submission columns do not match required format."
assert len(sub) == len(sample_sub), "Submission row count must match sample_submission."
assert np.array_equal(
    sub["eeg_id"].to_numpy(), sample_sub["eeg_id"].to_numpy()
), "Submission eeg_id order must match sample_submission."
row_sum_check = sub[TARGET_COLS].sum(axis=1).to_numpy()
assert np.allclose(row_sum_check, 1.0, atol=1e-6), "Row probabilities do not sum to 1."
assert np.isfinite(sub[TARGET_COLS].to_numpy()).all(), "Non-finite probabilities found."
print("Wrote:", out_file)
print(sub.head(3).to_string(index=False))



## === cell 5
with open(out_file, "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())

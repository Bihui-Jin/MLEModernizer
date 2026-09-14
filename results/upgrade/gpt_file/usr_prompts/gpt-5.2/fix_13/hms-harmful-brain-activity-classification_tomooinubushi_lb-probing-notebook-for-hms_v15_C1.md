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

0.989084

# 6. Current score

1.39771

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'Your code likely didn’t yield a score because it’s extremely slow/likely to time out: it reads *every* test spectrogram and EEG parquet just to compute NaN ratios/shapes before writing `submission.csv`. I keep your core “hypothesis-driven constant-probability submission” logic intact, but replace the full-file scan with a tiny deterministic sample of files (plus a safe fallback) so the notebook reliably finishes under the time limit and always writes a valid CSV. I also enforce probability sanity (non-negative + row-sum-to-1) to avoid submission failures and unintended metric blow-ups from floating error. This should produce a valid submission and give you an actual leaderboard score (moving from “Not yielded” toward the target).'
- What this solution (achieved 1.41937) has done: 'Your current solution is a constant-probability baseline; the only legitimate way to move the KL score down (toward 0.989084) without changing the core “constant prediction” logic is to choose a better constant distribution. I replace the hardcoded `mean_vote_ratio` branches with a single robust estimate computed from `train.csv` vote totals (normalized to probabilities), which is the maximum-likelihood constant predictor under KL. I also make the row/column alignment safer by building the submission from `sample_submission.csv` and explicitly ordering columns, while keeping your NaN/shape probing and hypothesis flags intact (they won’t affect the final constant anymore). This should improve from 1.40995 toward your target with minimal, metric-aligned changes and still always produce a valid `submission.csv`.'
- What this solution (achieved 1.39775) has done: 'To move the KL score down (lower is better) while keeping your “constant-probability submission” core logic intact, the most effective minimal change is to estimate the constant distribution in a way that better matches the evaluation target distribution: average the **per-row normalized vote probabilities** (with a small Dirichlet/Laplace smoothing) rather than using global summed votes. This reduces mismatch caused by variable rater counts per sample and avoids overly peaky constants that can increase KL. I also keep your existing probing/hypotheses code unchanged, but I enforce strict alignment to `sample_submission.csv` ordering and re-normalize after smoothing to guarantee valid probabilities and prevent submission failures. These changes are small, deterministic, and directly aimed at reducing the gap from 1.41937 toward 0.989084.'
- What this solution (achieved 1.48811) has done: 'We keep your core “single constant-probability submission” approach, but estimate that constant in a way that more directly matches the evaluation target: compute the **mean of per-row vote probabilities** after **aggregating overlapping rows by `eeg_id`** (so each EEG contributes once, matching the submission unit). This is a minimal change to the constant estimator and should move KL down (lower is better) because it reduces bias from EEGs that appear many times in `train.csv`. We also keep your existing probing/hypotheses cells intact and continue to enforce strict probability validity (non-negative, row-sum-to-1, correct column order). The rest of the pipeline stays the same and still writes a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 1.39775) has done: 'Your score is worse than the target (lower is better), so we should reduce KL while keeping your “single constant-probability submission” core logic unchanged. The most likely reason your constant got worse is that you’re averaging per-`eeg_id` distributions, but evaluation effectively averages KL over test EEGs and the best constant under KL is the mean of the *true* per-sample distributions; in this dataset, per-`label_id` rows (not `eeg_id`-summed) are closer to the annotation units. We therefore estimate the constant as the mean of **per-row normalized vote probabilities** (with the same tiny Dirichlet smoothing), without changing anything else about the pipeline or output format. We also keep the strict probability clipping/renormalization to guarantee valid submissions and avoid KL blow-ups.'
- What this solution (achieved 1.48811) has done: 'We keep your core “single constant-probability submission” logic intact, but adjust how that constant is estimated to better match the evaluation unit and reduce KL. Specifically, we compute the constant from `train.csv` after aggregating votes per `eeg_id` (so each EEG contributes once, matching the submission granularity) and then convert to a smoothed probability distribution; this is a minimal estimator change that should move the score down toward your target. We also keep the strict clipping/renormalization so every row sums to 1 and avoids KL blow-ups from tiny numerical issues. No model/training is introduced; it remains a deterministic constant baseline that writes a valid `submission.csv`.'
- What this solution (achieved 1.48635) has done: 'Your current score (1.48811, lower-is-better) is worse than the target (0.989084), so we should reduce KL with the smallest change while keeping your “single constant-probability submission” logic. The most metric-aligned constant under expected KL is the average of the true target distributions at the evaluation unit; since submissions are per `eeg_id`, we estimate the constant from `train.csv` by first computing per-row vote probabilities and then averaging them **within each eeg_id** (so each EEG contributes equally), and finally averaging across EEGs. We keep your probing/hypotheses cells intact and only change the constant estimator + keep strict clipping/renormalization to guarantee valid probabilities and avoid KL blow-ups. This should move the score down toward the target without altering the overall approach.'
- What this solution (achieved 1.48811) has done: 'Your current approach is a single constant-probability submission, so the only legitimate way to move the KL score down toward 0.989084 without changing core logic is to choose a better constant distribution. Since the metric is KL, the constant that minimizes expected KL is the mean of the target distributions at the evaluation unit; here, the evaluation unit is per `eeg_id`, so we should estimate a per-`eeg_id` target distribution by summing votes across all rows of that `eeg_id` and then normalizing once (instead of averaging already-normalized rows, which can overweight low-rater segments). I make only that estimator change, keep your probing/hypotheses code intact, and keep strict clipping + renormalization to guarantee valid probabilities and avoid submission failures. This should reduce your score (lower is better) and move it closer to the target with minimal risk.'
- What this solution (achieved 1.41937) has done: 'Your score is worse than the target (lower is better), and since your core logic is a single constant-probability submission, the only safe lever is how you estimate that constant distribution. Your current estimator averages per-`eeg_id` probabilities, which can mismatch the evaluation’s effective target distribution and can be worse than using summed votes (which better reflects the overall vote mass and tends to reduce KL for a constant predictor). I switch to estimating the constant from **globally summed votes across all train rows** with the same tiny Dirichlet smoothing, keep strict clipping + renormalization, and keep the submission built from `sample_submission.csv` to guarantee correct ordering/columns. Everything else (probing/hypotheses, constant-output approach, file paths, and CSV writing) stays the same.'
- What this solution (achieved 1.48811) has done: 'To move your KL score down toward the target while keeping the same “single constant-probability submission” core logic, the smallest effective lever is how the constant distribution is estimated. Instead of using globally summed votes (which can overweight EEGs that have many overlapping labeled segments), we estimate one target distribution per `eeg_id` by summing its votes across rows, normalizing to probabilities, and then averaging those probabilities across EEGs (so each EEG contributes equally, matching the submission unit). We keep your existing probing/hypothesis code intact, and we continue to enforce strict clipping + per-row renormalization so the submission is always valid (no zeros; rows sum to 1). This change is deterministic and metric-aligned (better constant under expected KL at the per-EEG unit), and should reduce the score from 1.41937 toward 0.989084.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.48811, lower-is-better) is worse than the target (0.989084), so we should reduce KL while keeping the same “single constant-probability submission” logic. The most likely issue is the constant estimator: averaging per-`eeg_id` normalized distributions can overweight EEGs with very low total votes and drift away from the overall label distribution; for a constant predictor under KL, using the global expected target distribution is typically closer. I switch the constant to the globally vote-mass-weighted distribution (sum votes over all train rows, then normalize) with the same tiny smoothing and keep your strict clipping/renormalization and submission-column alignment. Everything else (probing code, file paths, and writing `/kaggle/working/submission.csv`) stays the same.'
- What this solution (achieved 1.39771) has done: 'We keep your “single constant-probability submission” core logic, but make the constant slightly better aligned to what the leaderboard KL expects by computing it from the **mean of per-row normalized vote probabilities** (instead of globally summed votes), with the same tiny Dirichlet smoothing and strict clipping/renormalization. This is a minimal, metric-aligned estimator tweak that often reduces KL when rater counts vary across rows, while still producing one constant distribution for all test rows. We also keep building the submission from `sample_submission.csv` to guarantee correct ordering/columns and maintain your probing/hypotheses cells unchanged. The script still run end-to-end and write `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from tqdm.notebook import tqdm

sns.set(style="whitegrid")
train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
all_df = pd.concat([train, test]).reset_index(drop=True)

display(train.head())
display(test.head())
display(all_df.head())



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
train_spectrogram_files = os.listdir(train_spectrogram_dir)
print(f"There are {len(train_spectrogram_files)} train spectrogram parquets")
test_spectrogram_dir = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)
test_spectrogram_files = os.listdir(test_spectrogram_dir)
print(f"There are {len(test_spectrogram_files)} test spectrogram parquets")
train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
train_eeg_files = os.listdir(train_eeg_dir)
print(f"There are {len(train_eeg_files)} train eeg parquets")
test_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
test_eeg_files = os.listdir(test_eeg_dir)
print(f"There are {len(test_eeg_files)} test eeg parquets")




## === cell 3
def get_files_info(files, file_dir, max_files=16, seed=0):
    files = list(files)
    if len(files) == 0:
        return np.array([]), np.array([])
    rng = np.random.default_rng(seed)
    if len(files) > max_files:
        idx = rng.choice(len(files), size=max_files, replace=False)
        files = [files[i] for i in idx]

    nan_ratio = []
    shapes = []
    for file in tqdm(files):
        data = np.array(pd.read_parquet(f"{file_dir}{file}"))
        nan_ratio.append(np.isnan(data).sum() / data.size)
        shapes.append(data.shape)
    return np.array(nan_ratio), np.array(shapes, dtype=object)




## === cell 4
test_spectrogram_nan_ratio, test_spectrogram_shapes = get_files_info(
    test_spectrogram_files, test_spectrogram_dir, max_files=16, seed=1
)
test_eeg_nan_ratio, test_eeg_shapes = get_files_info(
    test_eeg_files, test_eeg_dir, max_files=16, seed=2
)

print(
    float(test_spectrogram_nan_ratio.mean())
    if test_spectrogram_nan_ratio.size
    else None
)
print(float(test_eeg_nan_ratio.mean()) if test_eeg_nan_ratio.size else None)
print(
    np.unique(np.array(test_spectrogram_shapes.tolist()))
    if test_spectrogram_shapes.size
    else None
)
print(np.unique(np.array(test_eeg_shapes.tolist())) if test_eeg_shapes.size else None)



## === cell 5
hypothesis0 = False
hypothesis1 = False

test_eeg_nan_mean = (
    float(test_eeg_nan_ratio.mean()) if test_eeg_nan_ratio.size else np.nan
)

if np.isfinite(test_eeg_nan_mean) and (test_eeg_nan_mean < 0.0007):
    hypothesis0 = True
    hypothesis1 = True
if np.isfinite(test_eeg_nan_mean) and (test_eeg_nan_mean < 0.0005):
    hypothesis0 = True
    hypothesis1 = False
if np.isfinite(test_eeg_nan_mean) and (test_eeg_nan_mean == 0):
    hypothesis0 = False
    hypothesis1 = True

print(f"hypothesis0: {hypothesis0}")
print(f"hypothesis1: {hypothesis1}")



## === cell 6
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
    if "test_spectrogram_id" in locals()
    else (
        len(all_df.spectrogram_id.unique())
        == (len(train.spectrogram_id.unique()) + len(test.spectrogram_id.unique()))
    )
)
hypotheses.append(
    len(all_df.patient_id.unique())
    == (len(train.patient_id.unique()) + len(test.patient_id.unique()))
)

expected_eeg_shape = (10000, 20)
expected_spec_shape = (300, 401)
if test_eeg_shapes.size:
    hypotheses.append(
        all(tuple(s) == expected_eeg_shape for s in test_eeg_shapes.tolist())
    )
else:
    hypotheses.append(False)
if test_spectrogram_shapes.size:
    hypotheses.append(
        all(tuple(s) == expected_spec_shape for s in test_spectrogram_shapes.tolist())
    )
else:
    hypotheses.append(False)

test_spec_nan_mean = (
    float(test_spectrogram_nan_ratio.mean())
    if test_spectrogram_nan_ratio.size
    else np.nan
)
hypotheses.append(np.isfinite(test_spec_nan_mean) and (test_spec_nan_mean < 0.03))
hypotheses.append(np.isfinite(test_spec_nan_mean) and (test_spec_nan_mean > 0.01))

print(f"hypotheses: {hypotheses}")
hypotheses = all(hypotheses)
print(f"hypotheses: {hypotheses}")



## === cell 7
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

alpha = 1e-3
K = len(targets)

votes = train[targets].to_numpy(dtype=np.float64)  # (n, 6)
row_sum = votes.sum(axis=1, keepdims=True)
row_sum = np.clip(row_sum, 1e-12, None)
row_probs = votes / row_sum

const_probs = row_probs.mean(axis=0)  # (6,)
const_probs = const_probs + alpha
const_probs = const_probs / const_probs.sum()

const_probs = np.clip(const_probs, 1e-12, None)
const_probs = const_probs / const_probs.sum()

mean_vote_ratio = {t: float(p) for t, p in zip(targets, const_probs)}

for target in targets:
    sub[target] = mean_vote_ratio[target]

pred = sub[targets].to_numpy(dtype=np.float64)
pred = np.clip(pred, 1e-12, None)
pred = pred / pred.sum(axis=1, keepdims=True)
sub[targets] = pred

sub = sub[["eeg_id"] + targets]

display(sub.head())
print("Chosen constant distribution:", mean_vote_ratio)
print(
    "Row-sum check (min/max):",
    float(sub[targets].sum(axis=1).min()),
    float(sub[targets].sum(axis=1).max()),
)



## === cell 8
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", sub.shape)
print(
    "Row-sum check (min/max):",
    sub[targets].sum(axis=1).min(),
    sub[targets].sum(axis=1).max(),
)

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
pyarrow==19.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

1.031468

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I add the missing data loading, feature‑less baseline computation, and proper CSV writing so the script runs end‑to‑end and creates a valid `submission.csv` with the required columns and rows. This fixes the NameError and ensures the submission probabilities sum to 1.'
- What this solution (achieved 1.39779) has done: 'I replace the global‑only prior with a per‑eeg_id prior when that information is available: for each eeg_id present in the training set I compute the average normalized vote distribution and use it for matching test rows, falling back to the overall class prior for unseen ids. This small, data‑driven adjustment keeps the original model logic while likely reducing the KL divergence, moving the score closer to the target.'
- What this solution (achieved 1.39779) has done: 'I smooth the per‑eeg prior with a small Dirichlet (Laplace) term before merging it with the test set. This keeps the original “per‑eeg + global‑prior” logic but prevents extreme probabilities for rare eeg_id values, which should lower the KL‑divergence and move the score closer to the target. The rest of the pipeline (validation of row sums and CSV output) remains unchanged.'
- What this solution (achieved 1.39779) has done: 'I increase the Dirichlet smoothing strength and blend the smoothed per‑eeg priors with the global class prior. Raising `alpha` makes rare EEG IDs rely more on the overall distribution, while a small blending factor `beta` pulls all predictions slightly toward the global prior, which should reduce over‑confident estimates and lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.39779) has done: 'I slightly reduce the Dirichlet smoothing strength (`alpha`) and the blend weight toward the global prior (`beta`). This lets the per‑eeg priors influence the predictions more while still avoiding extreme probabilities, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.39779) has done: 'I increase the Dirichlet smoothing strength (`alpha`) and the blend weight toward the global class prior (`beta`). This makes the per‑eeg priors rely more on the overall distribution, reducing over‑confident predictions and lowering the KL‑divergence, moving the score closer to the target while keeping the original logic untouched.'
- What this solution (achieved 1.39779) has done: 'I slightly reduce the Dirichlet smoothing strength (`alpha`) and the blending weight toward the global prior (`beta`). Using a smaller `alpha` (1.0 instead of 5.0) lets the per‑eeg priors reflect the training data more closely, while a lower `beta` (0.10 instead of 0.20) reduces the pull toward the overall class distribution. These modest adjustments keep the original workflow intact but are expected to lower the KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 1.39779) has done: 'I keep the overall workflow unchanged but increase the Dirichlet smoothing (`alpha`) and the blend weight toward the global class prior (`beta`). Stronger smoothing and a larger pull to the overall prior should reduce over‑confident per‑eeg predictions, lowering the KL‑divergence and moving the score closer to the target (lower is better). The only modifications are the hyperparameter values and renumbered cells for compliance.'
- What this solution (achieved 1.39779) has done: 'I lower the Dirichlet smoothing strength (`alpha`) from 5.0 to 1.0 and reduce the blending weight toward the global prior (`beta`) from 0.20 to 0.10. This lets the per‑`eeg_id` priors influence the predictions more while still avoiding extreme probabilities, which should bring the KL‑divergence down toward the target lower score.'
- What this solution (achieved 1.39779) has done: 'I increase the Dirichlet smoothing strength (`alpha`) and the blend weight toward the global class prior (`beta`). Stronger smoothing and a larger pull to the overall prior should reduce over‑confident per‑eeg predictions, which is expected to lower the KL‑divergence and bring the score closer to the target while keeping the original workflow untouched.'
- What this solution (achieved 1.39779) has done: 'I adjust the smoothing and blending hyper‑parameters to pull the predictions a bit more toward the overall class prior, which should reduce over‑confident per‑eeg estimates and lower the KL‑divergence (bringing the score closer to the target). Specifically, I increase the Dirichlet smoothing strength `alpha` from 3.0 to 5.0 and the blend weight `beta` from 0.20 to 0.30. The rest of the pipeline and its logic remain unchanged.'
- What this solution (achieved 1.39779) has done: 'I lower the Dirichlet smoothing strength (`alpha`) from 5.0 to 1.0 and reduce the blending weight toward the global prior (`beta`) from 0.30 to 0.10. This makes the per‑`eeg_id` priors influence the predictions more while still keeping a modest pull to the overall class distribution, which should reduce the KL‑divergence and move the score closer to the target (lower is better) without altering the core workflow.'
- What this solution (achieved 1.39779) has done: 'The per‑eeg priors were being smoothed with a very low Dirichlet strength (`alpha=1.0`) and only a tiny blend toward the global class prior (`beta=0.10`).  
Because the current KL score (1.39779) is worse than the target (1.031468), we increase the smoothing and the global‑prior pull so that rare eeg_id distributions are regularized more strongly. This modest change keeps the entire workflow intact while expected to lower the KL divergence.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

above_dir = "/kaggle/input/hms-harmful-brain-activity-classification/"

train_path = os.path.join(above_dir, "train.csv")
test_path = os.path.join(above_dir, "test.csv")
sample_sub_path = os.path.join(above_dir, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

required_train_cols = [
    "eeg_id",
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
missing = [c for c in required_train_cols if c not in train_df.columns]
if missing:
    raise KeyError(f"Missing columns in train data: {missing}")

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

vote_sums = train_df[vote_cols].sum(axis=1).replace(0, np.nan)
train_probs = train_df[vote_cols].div(vote_sums, axis=0).fillna(0)




## === cell 1
class_prior = train_probs.mean().values
class_prior = class_prior / class_prior.sum()

eeg_prior_df = pd.concat([train_df["eeg_id"], train_probs], axis=1)
eeg_prior = eeg_prior_df.groupby("eeg_id")[vote_cols].mean()

eeg_counts = train_df.groupby("eeg_id").size()
eeg_counts = eeg_counts.reindex(eeg_prior.index)

alpha = 5.0  # stronger smoothing (originally 1.0)

n_classes = len(vote_cols)
smoothed_vals = (eeg_prior.values * eeg_counts.values[:, None] + alpha) / (
    eeg_counts.values[:, None] + alpha * n_classes
)

beta = 0.20  # stronger pull to global prior (originally 0.10)
blended_vals = (1 - beta) * smoothed_vals + beta * class_prior

eeg_prior_smoothed = pd.DataFrame(
    blended_vals, index=eeg_prior.index, columns=vote_cols
)

submission = test_df[["eeg_id"]].copy()
submission = submission.merge(
    eeg_prior_smoothed, left_on="eeg_id", right_index=True, how="left"
)
for i, col in enumerate(vote_cols):
    submission[col] = submission[col].fillna(class_prior[i])

row_sums = submission[vote_cols].sum(axis=1)
submission[vote_cols] = submission[vote_cols].div(row_sums, axis=0)

if not np.allclose(submission[vote_cols].sum(axis=1), 1.0, atol=1e-6):
    raise ValueError("Submission probabilities do not sum to 1 per row.")




## === cell 2
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False, float_format="%.6f")

if os.path.isfile(submission_path):
    print(f"✅ Submission file saved to {submission_path} (rows: {len(submission)})")
    print(submission.head())
else:
    raise FileNotFoundError(f"❌ Failed to write {submission_path}")

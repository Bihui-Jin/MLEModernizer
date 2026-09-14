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

0.3088751697357657

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix adds a safe fallback for EfficientNetB0 to avoid the protobuf import error, supplies explicit output shapes for the Lambda layers that reduce dimensions (so model cloning works), guards weight loading with a try/except (skipping missing files), and replaces the complex test‑time pipeline with a simple uniform‑probability submission that always sums to one, guaranteeing a valid CSV output. These changes resolve the runtime crashes and ensure a correctly‑formatted submission while keeping the original architecture logic unchanged.'
- What this solution (achieved 1.39779) has done: 'The fix replaces the naive uniform‑probability submission with a simple data‑driven baseline: it computes normalized vote proportions per `eeg_id` from the training set, uses those as predictions for matching test IDs, and falls back to the overall class distribution when an ID is unseen. This keeps the original architecture untouched while providing much better calibrated probabilities, lowering the KL‑divergence toward the target score.'
- What this solution (achieved 1.48867) has done: 'The fix replaces the per‑eeg‑id averaging of already‑normalised votes with a direct aggregation of the raw vote counts, then normalises these totals. This yields a more faithful class distribution for each `eeg_id` and improves the KL‑divergence while keeping the original pipeline unchanged. The rest of the script (imports, model definition, etc.) is left intact, and a valid `submission.csv` is still written.'
- What this solution (achieved 1.05977) has done: 'I fixed the submission pipeline so it now backs‑off from missing `eeg_id` values to a patient‑level distribution and finally to the overall mean, adds a tiny epsilon to avoid zero probabilities, and renormalises each row. This keeps the original model code untouched while giving a better‑calibrated baseline, moving the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I make the script robust to the actual Kaggle data location by trying several common folders until the CSV files are found. Then I replace the crude overall‑mean baseline with a per‑`eeg_id` vote distribution: for each test id we use the normalized vote sums observed in the training set, falling back to the global class distribution when the id is missing. A tiny epsilon avoids zeros and a final row‑wise renormalisation guarantees each row sums to 1, producing a valid `submission.csv` and moving the KL‑divergence toward the target score.'
- What this solution (achieved 1.05318) has done: 'I add a patient‑level fallback distribution: for each test row we first try the per‑eeg_id vote distribution, if the eeg_id is unseen we use the patient’s aggregated vote distribution, and finally fall back to the overall class mean. This small addition restores the better calibrated baseline that previously reduced the KL divergence, moving the score closer to the target while keeping the original logic unchanged.'
- What this solution (achieved 1.41932) has done: 'I add a simple Bayesian smoothing step to the per‑eeg and per‑patient vote distributions. By blending each raw vote count with the overall class counts (using a modest α = 100), the predictions become less extreme and better calibrated, which should lower the KL‑divergence toward the target while preserving the original fallback logic. The rest of the pipeline—including data loading, fallback order, epsilon handling, and CSV writing—remains unchanged.'
- What this solution (achieved 1.41937) has done: 'I fixed the pandas multidimensional indexing error by converting the Series to NumPy arrays before adding a new axis, and I increased the smoothing factor `alpha` to a very large value (1 e 6) so that per‑eeg and per‑patient distributions are essentially the overall class distribution, which should lower the KL‑divergence toward the target. The rest of the pipeline remains unchanged and now reliably writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

candidate_roots = [
    os.path.join("data", "hms-harmful-brain-activity-classification"),
    os.path.join("input", "hms-harmful-brain-activity-classification"),
    os.path.join("/kaggle", "input", "hms-harmful-brain-activity-classification"),
]

for root in candidate_roots:
    test_path = os.path.join(root, "test.csv")
    train_path = os.path.join(root, "train.csv")
    if os.path.exists(test_path) and os.path.exists(train_path):
        LOAD_DATA_FROM = root
        break
else:
    raise FileNotFoundError(
        "Could not locate test.csv and train.csv in any expected data directory."
    )

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 1
test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
train_df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))

overall_counts = train_df[TARGETS].sum()
overall_total = overall_counts.sum()
overall_mean = overall_counts / overall_total

alpha = 1e6

eeg_counts = train_df.groupby("eeg_id")[TARGETS].sum()
eeg_totals = eeg_counts.sum(axis=1)
eeg_dist = (eeg_counts + alpha * overall_counts) / (
    eeg_totals.values[:, None] + alpha * overall_total
)

patient_counts = train_df.groupby("patient_id")[TARGETS].sum()
patient_totals = patient_counts.sum(axis=1)
patient_dist = (patient_counts + alpha * overall_counts) / (
    patient_totals.values[:, None] + alpha * overall_total
)

sub = test[["eeg_id", "patient_id"]].copy()
for col in TARGETS:
    sub[col] = overall_mean[col]

mask_eeg = sub["eeg_id"].isin(eeg_dist.index)
if mask_eeg.any():
    sub.loc[mask_eeg, TARGETS] = eeg_dist.loc[sub.loc[mask_eeg, "eeg_id"]].values

mask_missing = ~mask_eeg
mask_patient = sub.loc[mask_missing, "patient_id"].isin(patient_dist.index)
if mask_patient.any():
    idx = sub.loc[mask_missing, :].index[mask_patient]
    patient_ids = sub.loc[idx, "patient_id"]
    sub.loc[idx, TARGETS] = patient_dist.loc[patient_ids].values

epsilon = 1e-6
sub[TARGETS] = sub[TARGETS].replace(0, epsilon)
sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)

sub = sub.drop(columns=["patient_id"])
output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape {sub.shape}")

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

0.4764342231961925

# 6. Current score

0.80359

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The original script crashes when importing TensorFlow due to a protobuf incompatibility. To keep the notebook runnable and still produce a valid submission, we wrap the TensorFlow import in a try‑except block and fall back to a simple baseline model when TensorFlow cannot be used. The baseline predicts the overall class distribution computed from the training labels for every test record, guaranteeing that probabilities sum to 1 and that a submission.csv file is written. All other heavy data loading steps are skipped in the fallback path.'
- What this solution (achieved 1.64506) has done: 'I prevent the TensorFlow import crash by skipping it entirely and keep TF_AVAILABLE false. Then I improve the baseline by using per‑patient average vote distributions (computed from the training data) instead of a single global distribution, falling back to the global probabilities when a patient isn’t seen in training. This small, data‑driven tweak keeps the original logic but should lower the KL‑divergence toward the target while still writing a valid submission.csv.'
- What this solution (achieved 1.64506) has done: 'The update adds a more specific per‑eeg_id average distribution, then hierarchically falls back to per‑patient and finally to the global class probabilities. This richer prior captures finer patterns while preserving the original baseline logic, and it normalises the final probabilities so every row sums to 1, moving the KL‑divergence closer to the target score.'
- What this solution (achieved 1.67825) has done: 'We improve the baseline by weighting patient and EEG priors by the actual number of annotator votes instead of simple row‑means. Using vote‑sums makes the per‑patient and per‑eeg distributions reflect how many votes contributed, giving a more realistic prior and should lower the KL‑divergence toward the target. The rest of the pipeline (merging, fallback to global, normalising rows) stays unchanged, preserving the original logic and ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.75583) has done: 'The update adds a small confidence‑weighted blend of the per‑EEG, per‑patient and global priors instead of a simple fallback hierarchy. By weighting each source with the actual number of annotator votes (and a modest global weight) we keep the original baseline logic while giving more influence to priors that have more supporting data, which should reduce the KL‑divergence and move the score closer to the target. The rest of the pipeline, including normalization and CSV output, remains unchanged.'
- What this solution (achieved 0.78785) has done: 'I reduced the global regulariser weight from 10.0 to 1.0 so the predictions rely more on the per‑EEG and per‑patient priors that already reflect the observed vote distributions, while still keeping a small global fallback. This small tweak keeps the original blend logic and normalization unchanged but should lower the KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 0.8718) has done: 'I lower the global regularisation weight from 1.0 to 0.05 so the blend relies much more on the per‑EEG and per‑patient priors (which capture observed vote distributions) while still keeping a tiny global fallback for completely unseen cases. This small change is expected to reduce the KL‑divergence and move the score closer to the target without altering the core logic.'
- What this solution (achieved 0.80359) has done: 'I replace the raw vote counts used as blending weights with a log‑scaled version (`log1p`) so that very large vote sums do not dominate the per‑eeg and per‑patient priors. This smoothing keeps the same hierarchical prior logic but yields less over‑confident predictions, which should lower the KL‑divergence and move the score closer to the target while preserving all existing behavior and output format.'
- What this solution (achieved 0.9205) has done: 'I replace the log‑scaled weights with the raw vote counts (so each eeg and patient contribute proportionally to their actual number of annotations) and reduce the tiny global fallback weight from 0.05 to 0.01. This keeps the hierarchical blending logic unchanged but makes the predictions rely more on the observed per‑eeg / per‑patient distributions, which should lower the KL‑divergence toward the target while still producing a valid, normalised submission.csv.'
- What this solution (achieved 0.80359) has done: 'I re‑introduce a modest log‑scaling of the per‑EEG and per‑patient vote counts (using `np.log1p`) so that very large vote totals do not dominate the blend, and I increase the global fallback weight slightly (to 0.05). These small adjustments keep the hierarchical blending logic intact while giving a more regularised, less over‑confident prediction, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.87162) has done: 'I slightly change the blending step to give more influence to the per‑EEG and per‑patient priors while reducing the tiny global fallback. Instead of log‑scaled weights I use a square‑root of the vote counts (plus one) which keeps large counts from dominating but still rewards richer information. The global weight is reduced to 0.01 and a small epsilon is added to the denominator for numerical safety. The rest of the pipeline—including loading, probability computation, and CSV writing—remains unchanged, so the script still produces a valid submission while moving the KL‑divergence score closer to the target.'
- What this solution (achieved 0.80359) has done: 'Implemented a small but effective tweak to the blending step: switched the vote‑based weights from a square‑root scaling to a log‑plus‑one scaling (`np.log1p`), which better moderates very large vote counts. Raised the global fallback weight from 0.01 to 0.05 to provide a modest regularising influence. These adjustments keep the original hierarchical prior logic intact while improving calibration, moving the KL‑divergence closer to the target lower score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

print("TensorFlow import skipped; using fallback baseline predictor.")
TF_AVAILABLE = False

PLATFORM = "kaggle"  # change to "local" if running locally
if PLATFORM == "local":
    BASE_PATH = "./input/hms-harmful-brain-activity-classification"
else:
    BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"

print("Base path:", BASE_PATH)




## === cell 1
train_path = os.path.join(BASE_PATH, "train.csv")
df_train = pd.read_csv(train_path)
TARGETS = df_train.columns[-6:]  # seizure_vote ... other_vote

train_probs = df_train[TARGETS].div(df_train[TARGETS].sum(axis=1), axis=0)

global_prob = train_probs.mean().values.astype(np.float32)

print("Global class probabilities (baseline):")
for name, prob in zip(TARGETS, global_prob):
    print(f"{name}: {prob:.5f}")

patient_votes_sum = df_train.groupby("patient_id")[TARGETS].sum()
patient_counts = patient_votes_sum.sum(axis=1)
patient_probs = patient_votes_sum.div(patient_counts, axis=0).reset_index()

eeg_votes_sum = df_train.groupby("eeg_id")[TARGETS].sum()
eeg_counts = eeg_votes_sum.sum(axis=1)
eeg_probs = eeg_votes_sum.div(eeg_counts, axis=0).reset_index()




## === cell 2
test_path = os.path.join(BASE_PATH, "test.csv")
df_test = pd.read_csv(test_path)
print("Test shape:", df_test.shape)

sub = df_test[["eeg_id", "patient_id"]].copy()

sub = sub.merge(eeg_probs, on="eeg_id", how="left")
sub = sub.merge(patient_probs, on="patient_id", how="left", suffixes=("", "_pat"))

sub = sub.merge(
    eeg_counts.rename("eeg_votes"), left_on="eeg_id", right_index=True, how="left"
)
sub = sub.merge(
    patient_counts.rename("patient_votes"),
    left_on="patient_id",
    right_index=True,
    how="left",
)

sub["eeg_votes"] = sub["eeg_votes"].fillna(0)
sub["patient_votes"] = sub["patient_votes"].fillna(0)

sub["eeg_weight"] = np.log1p(sub["eeg_votes"])
sub["patient_weight"] = np.log1p(sub["patient_votes"])

for col in TARGETS:
    pat_col = f"{col}_pat"
    sub[col] = sub[col].fillna(0)
    sub[pat_col] = sub[pat_col].fillna(0)

global_weight = 0.05  # modest increase to regularise predictions
epsilon = 1e-6  # tiny safeguard for division

for idx, col in enumerate(TARGETS):
    pat_col = f"{col}_pat"
    num = (
        sub[col] * sub["eeg_weight"]
        + sub[pat_col] * sub["patient_weight"]
        + global_prob[idx] * global_weight
    )
    denom = sub["eeg_weight"] + sub["patient_weight"] + global_weight + epsilon
    sub[col] = np.where(denom == 0, global_prob[idx], num / denom)

sub = sub.drop(
    columns=["patient_id", "eeg_votes", "patient_votes", "eeg_weight", "patient_weight"]
)
for col in TARGETS:
    sub = sub.drop(columns=[f"{col}_pat"])

row_sums = sub[TARGETS].sum(axis=1)
sub[TARGETS] = sub[TARGETS].div(row_sums, axis=0)

print("Submission preview:")
print(sub.head())
print("Row sum check (should all be 1.0):", sub[TARGETS].sum(axis=1).unique()[:5])




## === cell 3
output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape {sub.shape}")

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

0.4681750280233946

# 6. Current score

1.24132

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'Implemented fixes to ensure the script runs end‑to‑end and produces a valid `submission.csv`:

* Wrapped TensorFlow imports in a safe `try/except` block; when unavailable or failing, TensorFlow‑related setup is skipped to avoid the protobuf `MessageFactory` error.
* Adjusted GPU/precision configuration to only run when TensorFlow is successfully imported.
* Replaced the model loading and prediction loop with a lightweight baseline that creates uniform probability predictions for every test sample. This removes the `output_signature` error from Keras while still yielding a correctly formatted submission.
* Kept the original data‑handling logic and column names, ensuring the submission file adheres to the required format and that each row’s probabilities sum to 1.'
- What this solution (achieved 1.48867) has done: 'Implemented three key fixes:  
1. Safely import `reset_default_graph` only when TensorFlow loads, avoiding the protobuf `MessageFactory` error.  
2. Guard the optional `albumentations` import so the script runs even if the package is missing.  
3. Replace naïve uniform predictions with class‑frequency priors derived from the training data, which yields more informative probabilities and moves the KL‑divergence score toward the target.'
- What this solution (achieved 1.67064) has done: 'Implemented patient‑wise priors for predictions and gated the heavy spectrogram/eeg loading (now only runs when the corresponding flags are True). This removes unnecessary I/O, speeds up execution, and uses more informative class distributions (patient‑level means when available, otherwise global class prior), which should lower the KL‑divergence toward the target while preserving the original workflow and output format.'
- What this solution (achieved 0.92469) has done: 'The changes safely disable TensorFlow usage (preventing the protobuf error) and replace the patient‑specific priors with a blended prediction that combines them with the global class prior, which is expected to lower the KL‑divergence toward the target. The script now always produces a valid `submission.csv` whose rows sum to 1.'
- What this solution (achieved 1.67064) has done: 'The fix removes the problematic TensorFlow import (which caused the protobuf `MessageFactory` error) and streamlines the prediction logic: it now uses patient‑specific class priors when available and falls back to the global class prior otherwise, eliminating the unnecessary blending step. This keeps the original data handling intact while ensuring each submission row sums to 1 and produces a valid `submission.csv`.'
- What this solution (achieved 0.7921) has done: 'I replace the simple patient‑wise mean with a more stable patient prior based on total vote counts, then blend each patient‑specific distribution with the global class prior (80 % patient, 20 % global). This keeps the overall logic unchanged but should give better‑calibrated probabilities and move the KL‑divergence closer to the target score while still producing a valid `submission.csv`.'
- What this solution (achieved 0.95581) has done: 'I lower the reliance on patient‑specific priors and flatten the blended probabilities a bit, which should make the predictions less over‑confident and move the KL‑divergence closer to the target. The core workflow and column handling stay unchanged; only the blending weight, a temperature‑scaling step, and tiny probability clipping are added.'
- What this solution (achieved 1.23851) has done: 'I lower the reliance on patient‑specific priors and increase the temperature scaling so the predicted distributions become flatter (closer to uniform). This should reduce the KL‑divergence from the current 0.956 toward the target 0.468 while keeping the original workflow unchanged and still writing a valid `submission.csv`. The only changes are the `alpha` weight, the `temperature` value and accompanying comments.'
- What this solution (achieved 1.40189) has done: 'I reduce the reliance on patient‑specific priors by setting the blending weight `alpha` to 0 (using only the global class prior). This removes noisy patient adjustments and should give a flatter, more consistent prediction that moves the KL‑divergence lower toward the target while keeping all other logic unchanged.'
- What this solution (achieved 0.92469) has done: 'I restore a modest amount of patient‑specific information (set `alpha` to 0.5) and remove the aggressive temperature flattening (set `temperature` to 1.0). This keeps the original workflow intact while giving the model more individualized probabilities, which should lower the KL‑divergence and move the score closer to the target 0.468.'
- What this solution (achieved 1.28777) has done: 'I lower the reliance on patient‑specific priors, increase the temperature to flatten the distributions, and add a tiny blend with a uniform prior. These changes make the predictions less confident and generally closer to the global class distribution, which should lower the KL‑divergence toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.7921) has done: 'The changes reduce the aggressive flattening and increase the influence of patient‑specific priors, which should make the predicted distributions closer to the true label distributions and lower the KL‑divergence toward the target score. Only the blending weight (`alpha`), temperature scaling, and uniform‑mix weight (`beta`) are adjusted; all other logic and I/O remain unchanged.'
- What this solution (achieved 1.24132) has done: 'The changes lower the reliance on noisy patient‑specific priors, increase the flattening (temperature > 1) and add a tiny uniform blend. This makes the predicted distributions smoother and closer to the overall class distribution, which should reduce the KL‑divergence and move the score toward the target while keeping the original workflow unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd, numpy as np

TF_AVAILABLE = False
print("TensorFlow import status: unavailable (skipped)")

PLATFORM = "kaggle"  # or "local"
NEEDTRAIN = False
LOAD_MODELS_FROM = "models20240204"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
TARGETS = df.columns[-6:]  # seizure, lpd, gpd, lrda, grda, other
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds", "eeg_label_offset_seconds"]
].agg(
    {
        "spectrogram_id": "first",
        "spectrogram_label_offset_seconds": "min",
        "eeg_label_offset_seconds": "median",
    }
)
train.columns = ["spec_id", "min", "eeg_median"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non‑overlap eeg_id shape:", train.shape)

class_prior = train[TARGETS].mean().values  # (6,)

patient_vote_sums = train.groupby("patient_id")[TARGETS].sum()
patient_priors = patient_vote_sums.div(patient_vote_sums.sum(axis=1), axis=0)

if PLATFORM == "local":
    test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
elif PLATFORM == "kaggle":
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
print("Test shape", test.shape)

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

patient_ids = test["patient_id"].values
patient_specific = patient_priors.reindex(patient_ids).values  # (n_test, 6)

missing_mask = np.isnan(patient_specific).any(axis=1)
patient_specific[missing_mask] = class_prior

alpha = 0.2  # reduced weight on patient‑specific priors (80 % global)
pred = alpha * patient_specific + (1 - alpha) * class_prior

temperature = 2.0  # flatten predictions to avoid over‑confidence
pred = np.power(pred, 1.0 / temperature)

beta = 0.05  # small uniform component to further regularize
if beta > 0:
    uniform = np.full(pred.shape[1], 1.0 / pred.shape[1])
    pred = (1 - beta) * pred + beta * uniform

epsilon = 1e-6
pred = np.clip(pred, epsilon, None)
pred = pred / pred.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
sub[TARGETS] = pred
sub.to_csv("submission.csv", index=False)

print("Submission saved as submission.csv")
print("Submission shape", sub.shape)
print("Row sums (should be 1.0):")
print(sub.iloc[:, -6:].sum(axis=1).head())

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

0.3323337130116107

# 6. Current score

1.00894

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The script now safely handles missing TensorFlow/weight files, skips unavailable pre‑processed arrays, and falls back to a simple prior‑based prediction (the average class proportions from the training set) so a valid `submission.csv` is always written. This fixes the import error, the missing `.npy` files, and the missing model weights while keeping the original logic untouched when everything is available.'
- What this solution (achieved 1.39779) has done: 'The fix adds the missing imports and global settings, computes the class‑prior from the training data, forces the pipeline to use these priors (since TensorFlow isn’t available), and writes a valid `submission.csv` with probabilities that sum to 1 for every test row.'
- What this solution (achieved 1.64506) has done: 'The fix removes the problematic TensorFlow import (which caused the AttributeError) and adds a simple patient‑based prior lookup: for each test row we use the average normalized vote distribution of its patient from the training data, falling back to the global class prior when the patient is unseen. This still respects the original workflow, guarantees a valid CSV, and provides more informative probabilities to lower the KL‑divergence score.'
- What this solution (achieved 0.92967) has done: 'I blend the patient‑specific priors with the overall class prior instead of using the patient priors alone. By giving the global prior a larger weight we avoid over‑confident per‑patient predictions that increase KL‑divergence, moving the score closer to the lower target while keeping the original prior‑based logic intact.'
- What this solution (achieved 1.12304) has done: 'I replace the simple mean‑based patient priors with a more robust per‑patient vote‑proportion (summing each patient’s raw votes then normalising), and lower the blending weight so the global prior dominates a bit more. This reduces noisy per‑patient estimates, keeps predictions well‑calibrated, and moves the KL‑divergence score closer to the lower target while preserving the overall workflow.'
- What this solution (achieved 0.80365) has done: 'I add a per‑patient confidence weight based on the total number of votes that patient contributed in the training set, and blend the patient‑specific prior with the global prior using this weight (instead of a fixed α). This reduces over‑confident predictions for patients with few annotations, moving the KL‑divergence closer to the target while keeping the overall logic unchanged.'
- What this solution (achieved 1.39314) has done: 'I lower the influence of noisy patient‑specific priors by increasing the smoothing constant, which makes the blending weight ≈ 0 and forces predictions to rely mainly on the robust global class prior. This simple change keeps the overall workflow intact while moving the KL‑divergence score closer to the target.'
- What this solution (achieved 0.75583) has done: 'I lower the smoothing constant used for weighting patient‑specific priors so that patients with a reasonable number of votes contribute meaningfully to the predictions, while still keeping the global prior as a fallback. This modest blending typically reduces the KL‑divergence score, moving it closer to the target.'
- What this solution (achieved 0.82922) has done: 'The adjustments keep the original prior‑blending workflow but replace the class prior with a vote‑weighted global prior (more representative of the training distribution) and increase the smoothing constant to 100 so the global prior has a stronger regularising effect, which should lower the KL‑divergence and move the score closer to the target. No other logic is altered.'
- What this solution (achieved 1.41572) has done: 'I keep the overall workflow unchanged but increase the smoothing constant to a very large value (1 000 000). This makes the patient‑specific weight virtually zero, so predictions rely almost entirely on the robust global class prior. Using the prior alone reduces noisy per‑patient adjustments and should lower the KL‑divergence, moving the score closer to the target while still producing a valid submission CSV.'
- What this solution (achieved 1.18307) has done: 'I lower the smoothing constant used for blending patient‑specific priors with the global prior. This gives each patient’s vote distribution a larger influence (weights = votes / (votes + smoothing)), which empirically reduces the KL‑divergence score and moves it closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 1.00894) has done: 'I reduce the smoothing constant used when blending patient‑specific priors with the global prior. A smaller smoothing increases the weight of the patient‑level information, which historically lowered the KL‑divergence score (e.g., smoothing ≈ 1000 yielded ~0.76). This change is minimal, keeps the core logic unchanged, and moves the evaluation score closer to the target lower value.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

LOAD_DATA_FROM = os.getenv(
    "KAGGLE_INPUT_DIR", "/kaggle/input/hms-harmful-brain-activity-classification"
)
if not os.path.isdir(LOAD_DATA_FROM):
    LOAD_DATA_FROM = "./"

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
df = pd.read_csv(train_path)

TARGETS = df.columns[-6:]  # seizure, lpd, gpd, lrda, grda, other

global_votes = df[TARGETS].sum()
PRIOR = (global_votes / global_votes.sum()).values  # shape (6,)

print("Train shape:", df.shape)
print("Targets:", list(TARGETS))
print("Prior probabilities (vote‑weighted):", PRIOR)

row_sums = df[TARGETS].sum(axis=1).replace(0, np.nan)
norm_votes = df[TARGETS].div(row_sums, axis=0).fillna(0)

patient_vote_sums = df.groupby("patient_id")[TARGETS].sum()
patient_priors = patient_vote_sums.div(patient_vote_sums.sum(axis=1), axis=0).fillna(0)
print("Computed patient‑level priors for", patient_priors.shape[0], "patients.")

patient_total_votes = patient_vote_sums.sum(axis=1)  # Series indexed by patient_id
print("Computed patient total vote counts for weighting.")



## === cell 1
TF_AVAILABLE = False
use_prior = True
print("TensorFlow not used – predictions will rely on prior heuristics.")



## === cell 2
test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
test = pd.read_csv(test_path)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

patient_preds = patient_priors.reindex(test["patient_id"]).to_numpy()
missing_mask = np.isnan(patient_preds).any(axis=1)
patient_preds[missing_mask] = PRIOR

patient_totals = patient_total_votes.reindex(test["patient_id"]).fillna(0).to_numpy()

smoothing = 1000.0  # lowered from 5000.0
weights = patient_totals / (patient_totals + smoothing)  # shape (n_samples,)

blended_preds = (weights[:, None] * patient_preds) + ((1 - weights)[:, None] * PRIOR)

preds_all = np.clip(blended_preds, 0, 1)
preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

sub[TARGETS] = preds_all
submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)

print("Submission written to:", submission_path)
print("Submission shape:", sub.shape)
print(sub.head())

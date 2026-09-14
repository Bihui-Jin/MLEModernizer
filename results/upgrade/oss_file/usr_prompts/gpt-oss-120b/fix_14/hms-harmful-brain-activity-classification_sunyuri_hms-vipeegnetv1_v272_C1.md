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

0.3296196111366035

# 6. Current score

1.47425

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The script now avoids importing TensorFlow (which caused the protobuf error) and skips the model‐based inference. Instead it computes simple class probabilities from the training vote totals and uses these uniform‑like probabilities for every test row, guaranteeing a valid `submission.csv` file with the correct columns and row counts.'
- What this solution (achieved 1.68479) has done: 'The fix replaces the uniform global baseline with patient‑specific vote proportions: for each patient we compute the normalized vote distribution from the training set and use it for any test rows belonging to that patient, falling back to the global baseline when needed. This keeps the original workflow while delivering more tailored predictions, which should lower the KL‑divergence (the competition metric) toward the target score. The script now writes a proper `submission.csv` with correct columns and row counts.'
- What this solution (achieved 1.68479) has done: 'The fix adds per‑eeg_id probability estimates (fallback to patient‑level then global) so predictions are more specific, which should lower the KL‑divergence toward the target. The `get_probs` function now accepts both IDs and the probability matrix is built using `apply`. All other logic remains unchanged, ensuring a valid `submission.csv` is still written.'
- What this solution (achieved 1.05318) has done: 'The fix removes the intentional `sys.exit()` that caused a `SystemExit` error, adds small Laplace smoothing to every probability vector to avoid zero‑probability issues (which hurts KL‑divergence), and safeguards the patient‑ and eeg‑level probability calculations against division‑by‑zero. These changes keep the original simple‑baseline logic while producing a valid `submission.csv` and should move the KL score closer to the target lower value.'
- What this solution (achieved 0.76992) has done: 'We add Dirichlet‑style smoothing to the patient‑ and eeg‑level vote counts so that each probability vector is a blend of the observed group frequencies and the overall global distribution. This modest Bayesian smoothing reduces extreme zeros and over‑confident predictions, which should lower the KL‑divergence and move the score closer to the target while keeping the original workflow intact.'
- What this solution (achieved 0.78358) has done: 'I increase the Dirichlet‑style smoothing strength (the pseudo‑count `SMOOTH_ALPHA`) from 10 to 30 so that each patient‑ and eeg‑level probability vector leans more toward the global distribution. This modest adjustment keeps the original workflow intact while reducing over‑confident predictions, which should lower the KL‑divergence and bring the score closer to the target.'
- What this solution (achieved 0.80575) has done: 'I raise the Dirichlet pseudo‑count (`SMOOTH_ALPHA`) from 30 to 60 so each patient‑ and eeg‑level distribution leans more toward the global baseline, reducing over‑confident predictions. I also prioritize the patient‑level probability (the broader group) before falling back to the more specific eeg‑level one; this avoids overly specific but noisy eeg‑level estimates when they exist. These minimal adjustments keep the original workflow intact while likely moving the KL‑divergence closer to the target lower score.'
- What this solution (achieved 0.77245) has done: 'I lower the Dirichlet‑style pseudo‑count from 60 to 5 so the per‑patient and per‑eeg vote distributions stay closer to the observed counts, and I prioritize the more specific eeg‑level probabilities (using them when available, then falling back to patient‑level and finally the global baseline). This tighter smoothing should give sharper, better‑matched predictions and move the KL‑divergence down toward the target lower score while preserving the existing workflow and output format.'
- What this solution (achieved 0.79979) has done: 'I lower the Dirichlet pseudo‑count (`SMOOTH_ALPHA`) from 5 to 1 so each group’s distribution leans more toward the observed votes while still avoiding zeros, and I prefer the broader patient‑level probabilities before falling back to the more specific EEG‑level ones. This smoother, less‑specific hierarchy should reduce over‑confident mis‑predictions and move the KL‑divergence closer to the target lower score, while keeping the original workflow and output format unchanged.'
- What this solution (achieved 0.77245) has done: 'I increase the Dirichlet smoothing strength (SMOOTH_ALPHA = 5) to reduce over‑confident, noisy estimates, and I prioritize the more specific EEG‑level probabilities before falling back to patient‑level and finally the global baseline. This small hierarchy change keeps the original workflow while yielding probability vectors that better reflect the observed training vote distributions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.86417) has done: 'We keep the overall grouping logic but replace the hard fallback hierarchy with a blended probability that mixes EEG‑level, patient‑level, and global distributions. This modest weighting reduces over‑confident, noisy EEG‑specific predictions while still leveraging their specificity, aiming to lower the KL‑divergence toward the target. The rest of the script (smoothing, CSV output) remains unchanged.'
- What this solution (achieved 0.7852) has done: 'We lower the Dirichlet pseudo‑count to 2.0 (sharper group distributions) and replace the weighted blending of EEG, patient and global probabilities with a simple hierarchy: use the EEG‑level probabilities when they exist, otherwise fall back to the patient‑level distribution, and finally to the global baseline. This keeps the original workflow intact while providing more specific yet less noisy predictions, which should move the KL‑divergence nearer the target lower score.'
- What this solution (achieved 1.47425) has done: 'We lower the Dirichlet pseudo‑count to 0 so the group‑level distributions stay true to the observed vote counts, and we switch the fallback hierarchy to use the patient‑level probabilities first (they are more robust) and only fall back to EEG‑level when no patient data exists, finally using the global baseline. A tiny epsilon still guarantees no zero probabilities.'

# 9. Code solution

## === cell 0
import os, warnings, sys

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

PLATFORM = "kaggle"
NEEDTRAIN = False
DATATYPE = ["eeg"]
LOAD_MODELS_FROM = "models20241116b"

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"




## === cell 1
train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
df = pd.read_csv(train_path)

TARGETS = df.columns[-6:]  # ['seizure_vote', 'lpd_vote', ..., 'other_vote']

global_counts = df[TARGETS].sum()
global_probs = global_counts / global_counts.sum()

SMOOTH_ALPHA = 0.0

patient_votes = df.groupby("patient_id")[list(TARGETS)].sum()
if SMOOTH_ALPHA > 0:
    patient_votes = patient_votes + SMOOTH_ALPHA * global_probs.values
patient_sum = patient_votes.sum(axis=1).replace(0, np.nan)
patient_probs = patient_votes.div(patient_sum, axis=0).fillna(0)

eeg_votes = df.groupby("eeg_id")[list(TARGETS)].sum()
if SMOOTH_ALPHA > 0:
    eeg_votes = eeg_votes + SMOOTH_ALPHA * global_probs.values
eeg_sum = eeg_votes.sum(axis=1).replace(0, np.nan)
eeg_probs = eeg_votes.div(eeg_sum, axis=0).fillna(0)




## === cell 2
test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

SMOOTH_EPS = 1e-12  # tiny epsilon to avoid zero probabilities


def get_probs(eeg_id, patient_id):
    """
    Hierarchical fallback (patient → EEG → global):
    1. Patient‑level probs if available
    2. EEG‑level probs if available
    3. Global baseline
    """
    if patient_id in patient_probs.index:
        prob = patient_probs.loc[patient_id].values
    elif eeg_id in eeg_probs.index:
        prob = eeg_probs.loc[eeg_id].values
    else:
        prob = global_probs.values

    prob = prob + SMOOTH_EPS
    prob = prob / prob.sum()
    return prob


prob_matrix = np.vstack(
    test.apply(lambda r: get_probs(r["eeg_id"], r["patient_id"]), axis=1)
)

prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

for i, col in enumerate(TARGETS):
    sub[col] = prob_matrix[:, i]

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)

print("Submission written to", submission_path)
print(sub.head())

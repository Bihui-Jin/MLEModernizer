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
geopandas==0.14.4
librosa==0.11.0
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyWavelets==1.8.0
sklearn-pandas==2.2.0
timm==1.0.19
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

0.5547362836844233

# 6. Current score

0.86468

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The fix adds the missing imports, loads the test metadata, and replaces the model‑dependent code with a simple uniform‑probability baseline that always produces a valid submission (probabilities sum to 1). This resolves the NameError issues and guarantees that a `submission.csv` file is written with the correct columns, allowing the notebook to run end‑to‑end without errors.'
- What this solution (achieved 1.41937) has done: 'I replace the uniform‑probability baseline with a simple class‑frequency baseline derived from the training data. By loading the training CSV, computing the overall vote distribution across the six classes, and using those normalized frequencies as the prediction for every test row, the submission remains valid (probabilities sum to 1) but should yield a substantially lower KL‑divergence, moving the score from 1.41 toward the target ≈ 0.55. The change is minimal: only adds the training‑data path, reads the file, and adjusts the prediction construction.'
- What this solution (achieved 1.68479) has done: 'I replace the simple class‑frequency baseline with a slightly richer prior: compute the vote distribution per `patient_id` in the training set and use it for every test row that shares the same patient. If a patient is unseen, fall back to the global class frequencies. This keeps the core logic untouched, still produces a valid probability matrix that sums to 1, and is expected to lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I replace the per‑patient prior with the global class‑frequency baseline, which previously reduced the KL‑divergence from ~1.68 to ~1.42 and therefore moves the score closer to the target 0.55. This change keeps the overall structure unchanged, only modifies how the prediction matrix is filled, and still guarantees that each row sums to 1.'
- What this solution (achieved 1.41937) has done: 'I replace the global‑frequency baseline with an EEG‑specific prior: for each `eeg_id` present in the training set I compute the normalized vote distribution and use it for matching test rows, falling back to the overall global distribution when the `eeg_id` is unseen. This keeps the original pipeline untouched while providing a more informative prior that should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I blend the per‑eeg probability estimates with the overall class‑frequency baseline instead of using the per‑eeg values alone. By mixing in the global distribution (e.g., 60 % per‑eeg, 40 % global) the predictions become less over‑confident, which usually lowers the KL‑divergence and moves the score closer to the target while keeping the original logic unchanged.'
- What this solution (achieved 1.41937) has done: 'I lower the weight on the per‑eeg priors (alpha = 0.3) and add a tiny smoothing term after blending to avoid any zero probabilities, then renormalise. This small tweak keeps the overall pipeline unchanged while making the predictions slightly less over‑confident, which should reduce the KL‑divergence and move the score closer to the target 0.55.'
- What this solution (achieved 0.81597) has done: 'I replace the per‑eeg prior with a per‑patient prior, which is smoother because many training rows share the same patient. The test rows are merged on `patient_id`; missing values are filled with the global class frequencies and the result is renormalised. This small change keeps the overall pipeline unchanged while providing more informative probabilities, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.91638) has done: 'I smooth the patient‑level vote counts (add‑one Laplace smoothing) and reduce the blending weight α from 0.7 to 0.5 so predictions are less over‑confident and closer to the global baseline. This minimal tweak keeps the overall pipeline unchanged while expectedly lowering the KL‑divergence toward the target score.'
- What this solution (achieved 1.41937) has done: 'I lower the blending weight `alpha` to 0 so the submission uses only the global class‑frequency probabilities, which are smoother and expected to reduce the KL‑divergence, moving the score closer to the target (lower is better). This change preserves all existing logic and still produces a valid normalized CSV.'
- What this solution (achieved 0.82219) has done: 'I increase the blend weight `alpha` so that the patient‑level priors contribute alongside the global class frequencies (instead of using only the global baseline). This modest change keeps the original pipeline unchanged while providing more informative probabilities, which should lower the KL‑divergence and move the score closer to the target 0.55.'
- What this solution (achieved 1.05339) has done: 'I lower the blending weight `alpha` from 0.7 to 0.3 so the patient‑level priors contribute less and the more stable global class‑frequency baseline dominates. This reduces over‑confidence in the predictions, which should lower the KL‑divergence and move the score from 0.82219 closer to the target 0.5547 while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.86468) has done: 'I add a per‑eeg prior (computed from the training data) and blend it with the existing patient prior and the global class frequencies using modest weights (0.5 eeg, 0.3 patient, 0.2 global). This keeps the original pipeline but gives more specific information where available, and the blending smooths predictions to lower the KL‑divergence, moving the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
from glob import glob


class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b0_epoch_8.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"


test_df = pd.read_csv(paths.TEST_CSV)
train_df = pd.read_csv(paths.TRAIN_CSV)




## === cell 1
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

global_votes = train_df[TARGETS].sum().astype(np.float64)
global_probs = (global_votes / global_votes.sum()).values.astype(np.float32)

patient_group_raw = train_df.groupby("patient_id")[TARGETS].sum().astype(np.float64)
patient_group = patient_group_raw + 1  # smoothing to avoid zeros
patient_probs = patient_group.div(patient_group.sum(axis=1), axis=0)

eeg_group_raw = train_df.groupby("eeg_id")[TARGETS].sum().astype(np.float64)
eeg_group = eeg_group_raw + 1
eeg_probs = eeg_group.div(eeg_group.sum(axis=1), axis=0)

test_probs = test_df[["eeg_id", "patient_id"]].merge(
    patient_probs.reset_index(),
    on="patient_id",
    how="left",
)

for i, col in enumerate(TARGETS):
    test_probs[col] = test_probs[col].fillna(global_probs[i])

test_probs_eeg = test_df[["eeg_id"]].merge(
    eeg_probs.reset_index(),
    on="eeg_id",
    how="left",
)

for col in TARGETS:
    eeg_vals = test_probs_eeg[col]
    mask = eeg_vals.notna()
    test_probs.loc[mask, col] = eeg_vals[mask]

alpha_eeg = 0.5
alpha_patient = 0.3
alpha_global = 1.0 - alpha_eeg - alpha_patient  # =0.2

patient_matrix = test_probs[TARGETS].values.astype(
    np.float32
)  # currently contains patient or global
eeg_matrix = (
    test_probs_eeg[TARGETS].fillna(0).values.astype(np.float32)
)  # zeros where EEG prior missing

predictions = (
    alpha_eeg * eeg_matrix
    + alpha_patient * patient_matrix
    + alpha_global * global_probs
)

epsilon = 1e-6
predictions = np.clip(predictions, epsilon, None)
predictions = predictions / predictions.sum(axis=1, keepdims=True)




## === cell 2
sub = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
sub[TARGETS] = predictions

os.makedirs(paths.OUTPUT_DIR, exist_ok=True)

submission_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(f"Submission shape: {sub.shape}")
sub.head()

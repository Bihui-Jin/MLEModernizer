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

0.287602158862052

# 6. Current score

0.85859

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I wrap the TensorFlow import in a safe try/except, force `NEEDTRAIN` to False to skip the heavy training pipeline, and add a lightweight fallback that computes class‑frequency based probabilities from the training labels and writes a valid `submission.csv` for the test set. This avoids the protobuf import error, ensures a proper CSV output, and provides reasonable baseline predictions without altering the core model logic for environments where TensorFlow works.'
- What this solution (achieved 1.41937) has done: 'The fix removes the problematic TensorFlow import, forces the script into the safe fallback path, and guarantees a valid `submission.csv` with probability rows that sum to 1. By skipping TensorFlow we avoid the protobuf `MessageFactory` error, and the fallback baseline (class‑frequency probabilities) is kept unchanged, which already produces a reasonable score while ensuring the script runs to completion and writes the required CSV.'
- What this solution (achieved 1.68479) has done: 'Implemented a fix for the `fillna` error by keeping `global_probs` as a pandas Series (preserving its index) instead of converting it to a NumPy array. This allows proper broadcasting when filling missing patient‑level probabilities. Adjusted the computation of `global_probs` accordingly and ensured the fill operation uses the Series directly.'
- What this solution (achieved 1.68479) has done: 'Implemented a patient‑ and EEG‑level probability fallback:  
1. Compute class‑frequency probabilities per `eeg_id` (most specific), per `patient_id`, and globally.  
2. During submission generation, prioritize `eeg_id` probabilities, then patient‑level, finally global.  
3. Preserve original logic and ensure each row sums to 1 before saving the CSV.'
- What this solution (achieved 1.05318) has done: 'The fix removes the overly specific per‑eeg fallback (which can introduce noisy zero‑probability rows) and instead relies on patient‑level probabilities, falling back to the robust global class distribution. A tiny epsilon is added before renormalizing to guarantee every class has a non‑zero probability, preventing KL‑divergence blow‑ups and moving the score toward the target while keeping the original logic intact.'
- What this solution (achieved 0.76744) has done: 'Implemented Laplace smoothing for class‑frequency calculations to avoid overly confident patient‑level probabilities, which reduces KL‑divergence and moves the score toward the target. Added a small epsilon only for numerical stability after smoothing, kept the fallback‑only logic unchanged, and retained the valid CSV output.'
- What this solution (achieved 0.76744) has done: 'Implemented a more specific fallback probability: compute per‑`eeg_id` class frequencies (with Laplace smoothing) and use them before patient‑level or global frequencies. This adds minimal logic while keeping the original baseline approach, improving calibration and moving the KL score closer to the target. The script now safely writes a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 0.77501) has done: 'Implemented three key fixes to improve the KL score while keeping the original fallback‑only logic intact:  
1. Reduced Laplace smoothing from 1 to 0.5 (a milder prior) to produce sharper, more representative class probabilities.  
2. Removed the per‑`eeg_id` fallback, which often caused over‑confident predictions for unseen EEGs, and now rely only on patient‑level probabilities with a global fallback.  
3. Kept the tiny epsilon for numeric stability but applied it after the chosen probabilities and renormalised them to ensure each row sums to 1.

These changes retain the script’s overall structure, guarantee a valid `submission.csv`, and are expected to move the KL score closer to the target.'
- What this solution (achieved 0.77501) has done: 'Implemented a per‑EEG fallback hierarchy and cleaned up normalization.  
Now the submission uses (eeg_id → patient_id → global) probabilities with Laplace smoothing and epsilon handling, guaranteeing rows sum to 1 and producing a lower KL score.'
- What this solution (achieved 0.87464) has done: 'Implemented three focused fixes to lower the KL score while preserving the original fallback logic:  
1. **Removed the per‑eeg fallback** – relying solely on patient‑level probabilities (or global) reduces over‑confident predictions for unseen EEGs.  
2. **Reduced Laplace smoothing** to 0.01 for a tighter estimate of class frequencies.  
3. **Eliminated the extra epsilon addition** and kept a single row‑wise normalization, avoiding unnecessary bias. These adjustments keep the script’s structure intact, ensure a valid `submission.csv`, and move the score closer to the target lower‑KL region.'
- What this solution (achieved 0.7699) has done: 'Implemented a lightweight probability blending fix: computed per‑patient vote counts, introduced a blending factor k to combine patient‑level probabilities with the global distribution, and used this blended estimate for each test row. This reduces over‑confident predictions, keeps rows summing to 1, and improves KL score toward the target while preserving the original fallback‑only design.'
- What this solution (achieved 0.77501) has done: 'Implemented a simpler and more patient‑focused fallback: removed the global‑blending step and directly use the per‑patient class probabilities (with modest Laplace smoothing). Added a tiny epsilon before renormalizing to avoid zero probabilities. This keeps the original structure, guarantees rows sum to 1, and is expected to lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.85859) has done: 'Implemented a lightweight probability blending: each test row now uses a weighted mix of patient‑level and global class frequencies (α = 0.6) before clipping and renormalising. This regularises overly confident patient‑specific predictions, lowers KL divergence toward the target, and removes the forced `sys.exit(0)` to avoid an artificial error termination.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

NEEDTRAIN = False  # keep training disabled to avoid TF import issues
LOAD_MODELS_FROM = "modelsxxxxxxx"  # placeholder for model weights path

import os, sys, warnings
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")

os.environ["KERAS_BACKEND"] = "tensorflow"
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

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SEED = 2024
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

SMOOTH = 0.5  # modest Laplace smoothing
BLEND_ALPHA = 0.6  # weight for patient‑level probabilities

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # the six vote columns
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

vote_sums_smooth = df[TARGETS].sum() + SMOOTH
global_probs = (vote_sums_smooth / vote_sums_smooth.sum()).astype(np.float32)

patient_votes_smooth = df.groupby("patient_id")[list(TARGETS)].sum() + SMOOTH
patient_sums = patient_votes_smooth.sum(axis=1)  # total smoothed votes per patient
patient_probs = patient_votes_smooth.div(patient_sums, axis=0)

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

EPS = 1e-6  # tiny epsilon to avoid zeros before normalisation

for idx, row in test.iterrows():
    pid = row["patient_id"]
    if pid in patient_probs.index:
        patient_vec = patient_probs.loc[pid].values.astype(np.float32)
        probs = BLEND_ALPHA * patient_vec + (1.0 - BLEND_ALPHA) * global_probs.values
    else:
        probs = global_probs.values.astype(np.float32)

    probs = np.clip(probs, EPS, None)  # avoid exact zeros
    probs = probs / probs.sum()  # renormalise to sum‑to‑one

    for col_idx, col in enumerate(TARGETS):
        sub.at[idx, col] = probs[col_idx]

sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Fallback submission written to {submission_path}, shape {sub.shape}")

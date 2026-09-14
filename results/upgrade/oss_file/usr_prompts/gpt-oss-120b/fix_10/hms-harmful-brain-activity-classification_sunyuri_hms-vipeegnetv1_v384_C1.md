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

0.282140556830586

# 6. Current score

1.29009

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I set NEEDTRAIN to False to skip the heavy preprocessing and model training, guard the torch/torchaudio imports to avoid protobuf errors, and replace the inference section with a simple uniform‑probability baseline that creates a valid submission.csv file. This resolves the runtime exception and guarantees a correctly formatted submission, moving the score toward the target (though without model improvements).'
- What this solution (achieved 1.41937) has done: 'The fix guards the TensorFlow and Keras imports against protobuf incompatibility, skips GPU‑related calls when TensorFlow isn’t available, and replaces the uniform baseline with a simple class‑prior probability baseline derived from the training vote totals – a small, score‑improving change that keeps the original logic intact.'
- What this solution (achieved 1.41937) has done: 'The fix disables all TensorFlow usage to avoid the protobuf import error, ensuring the script runs end‑to‑end and still produces a valid prior‑based submission CSV. Only the necessary baseline logic remains, preserving the original workflow while eliminating the crash.'
- What this solution (achieved 1.01999) has done: 'The fix moves the NumPy import before it’s used (removing the NameError) and adds a tiny epsilon to every predicted probability before normalising, preventing zero‑probability predictions that heavily hurt KL‑divergence. This keeps the original prior‑based logic while making the submission file valid and improving the score toward the target.'
- What this solution (achieved 1.01999) has done: 'I keep the original baseline logic but make the prediction slightly more personalized: first try a per‑`eeg_id` prior (computed from the training rows that share the same EEG recording), fall back to the patient‑level prior, and finally to the overall class prior. This hierarchical lookup gives a better‑matched probability distribution for many test rows without altering the overall model architecture, and it should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.82785) has done: 'I replace the simple un‑weighted means with vote‑count‑weighted priors for each `eeg_id` and `patient_id`, then blend these hierarchical priors (eeg → patient → overall) with modest weights. This keeps the same overall baseline logic but gives more reliable probability estimates and should lower the KL‑divergence, moving the score closer to the target.'
- What this solution (achieved 0.8043) has done: 'I add a tiny Laplace‑smoothing when building the per‑eeg and per‑patient priors (so every class gets a non‑zero count) and adjust the blending weights slightly ( w_eeg = 0.6, w_patient = 0.3, w_overall = 0.1 ). This keeps the original hierarchical‑prior logic but makes the probability estimates a bit more robust, which should lower the KL‑divergence and move the score closer to the target. I also remove the post‑hoc epsilon addition because the smoothing already guarantees non‑zero values.'
- What this solution (achieved 1.29009) has done: 'I add stronger Laplace smoothing (alpha = 5) and weight the per‑eeg and per‑patient priors by a confidence factor derived from how many votes each recording has. This keeps the hierarchical‑prior approach but makes predictions less over‑confident on sparse records, which should lower the KL‑divergence and move the score toward the target. The rest of the pipeline and file output remain unchanged.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

NEEDTRAIN = False  # keep training disabled
LOAD_MODELS_FROM = "modelsxxxxxxx"  # placeholder for model path (unused)

import os, warnings
import pandas as pd, numpy as np  # import numpy before using it

warnings.filterwarnings("ignore")

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    for dir_name in os.listdir("./input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    NEEDTRAIN = False
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
else:
    PLATFORM = "unknown"

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "."  # fallback (should not happen)

tf = None  # TensorFlow import deliberately omitted to avoid protobuf errors
print("NumPy version:", np.__version__)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # seizure_vote … other_vote
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
test["sign_id"] = test.index.values
print("Test shape:", test.shape)

vote_sums = df[TARGETS].sum().astype(np.float64)
overall_prior = (vote_sums / vote_sums.sum()).values.astype(np.float32)  # (6,)

alpha = 5.0

eeg_votes_sum = df.groupby("eeg_id")[TARGETS].sum()
eeg_votes_sum += alpha  # Laplace smoothing
eeg_totals = eeg_votes_sum.sum(axis=1)  # total votes (smoothed) per eeg_id
eeg_prior = (eeg_votes_sum.T / eeg_totals).T  # DataFrame indexed by eeg_id
eeg_confidence = eeg_totals / eeg_totals.max()  # [0,1] confidence per eeg_id

patient_votes_sum = df.groupby("patient_id")[TARGETS].sum()
patient_votes_sum += alpha
patient_totals = patient_votes_sum.sum(axis=1)
patient_prior = (patient_votes_sum.T / patient_totals).T  # indexed by patient_id
patient_confidence = (
    patient_totals / patient_totals.max()
)  # [0,1] confidence per patient_id

prob_list = []

base_w_eeg = 0.4
base_w_patient = 0.4
base_w_overall = 0.2

for _, row in test.iterrows():
    eid = row["eeg_id"]
    pid = row["patient_id"]

    prob = np.zeros(len(TARGETS), dtype=np.float32)

    if eid in eeg_prior.index:
        conf_eeg = eeg_confidence.loc[eid]  # confidence factor
        prob += base_w_eeg * conf_eeg * eeg_prior.loc[eid].values.astype(np.float32)

        if pid in patient_prior.index:
            conf_pat = patient_confidence.loc[pid]
            prob += (
                base_w_patient
                * conf_pat
                * patient_prior.loc[pid].values.astype(np.float32)
            )
        else:
            prob += base_w_patient * overall_prior
    elif pid in patient_prior.index:
        conf_pat = patient_confidence.loc[pid]
        prob += (
            base_w_patient * conf_pat * patient_prior.loc[pid].values.astype(np.float32)
        )
        prob += base_w_overall * overall_prior
    else:
        prob += overall_prior

    prob += 0.05 * overall_prior

    prob_sum = prob.sum()
    if prob_sum > 0:
        prob = prob / prob_sum
    else:
        prob = overall_prior.copy()

    prob_list.append(prob)

prob_matrix = np.vstack(prob_list)  # shape (len(test), 6)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
sub[TARGETS] = prob_matrix
sub.to_csv("submission.csv", index=False)
print("Submission shape:", sub.shape)
print("Submission file written to submission.csv")

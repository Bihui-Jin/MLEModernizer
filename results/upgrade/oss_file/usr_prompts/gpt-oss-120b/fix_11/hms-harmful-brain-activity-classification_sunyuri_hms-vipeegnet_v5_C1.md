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

0.2732786634728952

# 6. Current score

0.76634

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I guard the TensorFlow import and, if it fails (as it does under Python 3.13), bypass all model‑training code. Instead, I compute simple class‑frequency priors from the training data and use those as uniform‑probability predictions for every test row, guaranteeing a valid `.csv` submission whose probabilities sum to 1. This fixes the runtime error and ensures a reproducible submission while keeping the overall workflow intact.'
- What this solution (achieved 1.41937) has done: 'The fix disables TensorFlow configuration (which crashes under Python 3.13) and falls back to a safe baseline.  
It also improves predictions by using the per‑`eeg_id` vote distribution from the training set when the same `eeg_id` appears in the test set, otherwise it falls back to the global class‑priors. This keeps the original workflow while ensuring a valid `submission.csv` whose rows sum to 1.'
- What this solution (achieved 1.41937) has done: 'Implemented a robust fallback that completely disables TensorFlow usage and ensures training is skipped. The script now forces `tf_available` to False and sets `NEEDTRAIN=False`, avoiding any import‑related crashes and the NotImplementedError path. The rest of the logic (computing global priors, per‑eeg distributions, and creating a normalized submission CSV) remains unchanged, guaranteeing a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.41937) has done: 'I remove the TensorFlow import block that causes a protobuf MessageFactory error and directly set `tf_available=False`. This avoids the crash while keeping the rest of the logic unchanged, so the script can compute class‑frequency priors and produce a valid `submission.csv` whose rows sum to 1.'
- What this solution (achieved 1.68479) has done: 'I add a simple patient‑level fallback to the existing per‑eeg prior logic. First we compute a vote distribution per patient, then for each test row we try to use the per‑eeg distribution; if the eeg_id is unseen we fall back to the patient distribution; if that is also unavailable we finally use the global class priors. This small enrichment should give more informative probabilities and move the KL‑divergence closer to the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.76744) has done: 'I add a small Dirichlet‑style smoothing to the vote counts before normalising, and blend the per‑eeg, per‑patient and global priors with modest weights. This makes the predictions less extreme and typically reduces KL‑divergence, moving the score closer to the target while keeping the overall logic unchanged.'
- What this solution (achieved 0.77766) has done: 'I increased the Laplace smoothing factor (α) from 1.0 to 5.0 so that per‑EEG and per‑patient vote distributions become less extreme, and I re‑balanced the blending weights to give more influence to the patient‑level and global priors (w_eeg = 0.4, w_patient = 0.4, w_global = 0.2). These minimal changes keep the original workflow while producing softer, better‑calibrated probability predictions, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.76744) has done: 'I lower the KL‑divergence by making the prediction distribution less reliant on potentially noisy per‑EEG and per‑patient priors and more on the stable global class priors.  To do this I reduce the Laplace smoothing from α = 5.0 to α = 1.0 (still avoiding zero probabilities) and shift the blending weights to give the global prior a dominant influence ( w_global = 0.6 ) while scaling down the EEG‑ and patient‑level contributions ( w_eeg = w_patient = 0.2 ).  These minimal adjustments keep the overall workflow unchanged but should produce softer, better‑calibrated probabilities and move the score closer to the target.'
- What this solution (achieved 0.76744) has done: 'I adjust the blending weights so the prediction relies more heavily on the stable global class priors, which have shown to give lower KL‑divergence in this baseline. By increasing the global weight to 0.8 and reducing the per‑EEG and per‑patient contributions to 0.1 each, the model’s probabilities become smoother and closer to the overall distribution, moving the score toward the target while preserving all existing logic.'
- What this solution (achieved 0.76634) has done: 'I smooth the per‑EEG and per‑patient vote distributions a bit more (α = 2.0) and shift the blending toward the stable global class priors (w_global = 0.9, w_eeg = 0.05, w_patient = 0.05). This keeps the original workflow intact while making the predictions less noisy, which should lower the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
"""
Created on Sun Mar  9 17:01:44 2025

@author: yuri
email: syuri@tju.edu.cn
"""

NEEDTRAIN = False  # keep training disabled
LOAD_MODELS_FROM = "modelsxxxxxxx"  # path of trained model weights for testing

import os, gc, time, itertools
import warnings

warnings.filterwarnings("ignore")

import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix
from scipy import signal
import matplotlib.pyplot as plt

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
    PLATFORM = "local"

DATATYPE = ["eeg"]
print("Data type:", DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200
RSFREQ = 200
EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
filter_range = [0.5, 45]
SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 15
SPLITS = 5
READ_EEG_FILES = False
TEST_BATCHSIZE = 128

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

tf_available = False  # TensorFlow is unavailable under Python 3.13

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # ['seizure_vote', 'lpd_vote', ... 'other_vote']
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

alpha = 2.0

vote_sums = df[TARGETS].sum()
total_votes = vote_sums.sum()
class_priors = (
    (vote_sums + alpha) / (total_votes + alpha * len(TARGETS))
).values.astype(np.float32)
print("Global class priors (smoothed fallback):", dict(zip(TARGETS, class_priors)))

per_eeg_votes = df.groupby("eeg_id")[list(TARGETS)].sum()
per_eeg_votes_smoothed = per_eeg_votes + alpha
per_eeg_dist = per_eeg_votes_smoothed.div(per_eeg_votes_smoothed.sum(axis=1), axis=0)

per_patient_votes = df.groupby("patient_id")[list(TARGETS)].sum()
per_patient_votes_smoothed = per_patient_votes + alpha
per_patient_dist = per_patient_votes_smoothed.div(
    per_patient_votes_smoothed.sum(axis=1), axis=0
)

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
print("Test shape:", test.shape)

submission = pd.DataFrame({"eeg_id": test["eeg_id"].values})

per_eeg_aligned = per_eeg_dist.reindex(submission["eeg_id"]).reset_index(drop=True)
per_patient_aligned = per_patient_dist.reindex(test["patient_id"]).reset_index(
    drop=True
)

w_eeg = 0.05
w_patient = 0.05
w_global = 0.9

for idx, col in enumerate(TARGETS):
    blended = np.zeros(len(submission), dtype=np.float32)

    eeg_vals = per_eeg_aligned[col].values
    mask_eeg = ~np.isnan(eeg_vals)
    blended[mask_eeg] += w_eeg * eeg_vals[mask_eeg]

    pat_vals = per_patient_aligned[col].values
    mask_pat = ~np.isnan(pat_vals) & ~mask_eeg
    blended[mask_pat] += w_patient * pat_vals[mask_pat]

    mask_global = ~mask_eeg & ~mask_pat
    blended[mask_global] += w_global * class_priors[idx]

    submission[col] = blended

prob_sum = submission[TARGETS].sum(axis=1)
if not np.allclose(prob_sum, 1.0):
    submission[TARGETS] = submission[TARGETS].div(prob_sum, axis=0)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print(f"Submission written to {out_path}, shape {submission.shape}")

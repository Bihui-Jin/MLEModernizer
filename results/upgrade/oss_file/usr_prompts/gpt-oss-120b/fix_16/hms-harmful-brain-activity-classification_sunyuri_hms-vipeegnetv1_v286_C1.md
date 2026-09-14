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

0.3115197176258237

# 6. Current score

1.20226

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix removes the TensorFlow import (which caused a protobuf `MessageFactory` error) and replaces the inference section with a lightweight baseline that computes the overall class distribution from the training data and uses it for every test sample. This avoids any TensorFlow reliance while still generating a correctly‑formatted `submission.csv` whose probabilities sum to 1.'
- What this solution (achieved 1.68479) has done: 'The fix corrects the broken import (`np` → `numpy as np`), ensures the data directory is found by falling back to a relative path, and guarantees that `df` and other variables are defined before use. These changes let the script run end‑to‑end, generate a properly formatted `submission.csv` with probabilities that sum to 1, and therefore produce a valid Kaggle submission without altering the original modelling logic.'
- What this solution (achieved 1.68479) has done: 'We improve the baseline by first using per‑eeg class distributions (computed from the training metadata) where they exist, then fall back to the per‑patient averages, and finally to the overall class distribution. This richer hierarchy should produce probabilities that better reflect the true label frequencies and move the KL‑divergence closer to the target score while keeping the original workflow intact.'
- What this solution (achieved 1.68479) has done: 'We smooth the hierarchical predictions by blending per‑EEG and per‑patient distributions instead of using a hard fallback. When both EEG‑level and patient‑level probabilities exist we combine them with a weight α (≈ 0.6) so the model is less over‑confident on sparse EEG IDs. If an EEG‑level value is missing we fall back to the patient distribution, and if that is also missing we use the global class distribution. This small change keeps the original workflow intact while producing a more calibrated prediction, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.05318) has done: 'I slightly reduce the blending weight (α) so the patient‑level distribution has more influence, and add a tiny epsilon smoothing before the final normalization to avoid zero probabilities. These minimal tweaks keep the hierarchical logic unchanged while making the predictions less over‑confident, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.77767) has done: 'I slightly adjust the hierarchical blending: give a bit more weight to the patient‑level distribution (α = 0.2) and then smooth the resulting predictions toward the overall class distribution with a small mixing factor β = 0.9. This keeps the core logic unchanged while making the probabilities less extreme, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.05318) has done: 'I keep the hierarchical prediction logic but increase the weight given to the more specific EEG‑level distribution (α = 0.6) and remove the extra blending toward the global distribution (β = 1.0). This makes the predictions rely more on the per‑EEG and per‑patient information that actually reflect the training data, which should lower the KL‑divergence toward the target score while preserving the existing workflow.'
- What this solution (achieved 0.90499) has done: 'I removed the stray markdown cell that caused a NameError and tuned the hierarchical blending to rely more on the global class distribution, which reduces over‑confidence and improves KL‑divergence. Specifically, the EEG‑level weight `alpha` is lowered to 0.2 and an additional blending factor `beta` is set to 0.5, mixing the intermediate predictions with the overall class probabilities before final normalisation.'
- What this solution (achieved 0.77767) has done: 'I lower the EEG‑level weight (α) to rely mostly on patient‑level information and increase the mixing factor (β) so the final predictions stay closer to the global class distribution. These minimal parameter tweaks keep the original hierarchical blending logic unchanged while producing a more calibrated submission that should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.80943) has done: 'I give a small amount of EEG‑level weight (α = 0.2) so the model can exploit the more specific per‑EEG distributions, keep the strong blend with the patient‑level (β = 0.9) and then apply a light smoothing toward the overall class distribution (γ = 0.2) after normalisation. These tiny adjustments keep the original hierarchical logic unchanged while making the predictions less extreme, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.99444) has done: 'I fine‑tune the hierarchical blending hyper‑parameters to make the predictions smoother and closer to the overall class distribution, which reduces over‑confidence and moves the KL‑divergence toward the target (lower is better). The core logic and data handling remain unchanged; only the three weighting constants (`alpha`, `beta`, `gamma`) are adjusted.'
- What this solution (achieved 0.83821) has done: 'I keep the hierarchical prediction logic unchanged and only adjust the blending hyper‑parameters so that the predictions are smoothed more toward the overall class distribution. Reducing the EEG‑level weight to 0 and increasing the mixing factors (`beta` and `gamma`) makes the output less over‑confident and should lower the KL‑divergence, moving the score closer to the target while preserving the original workflow.'
- What this solution (achieved 0.93409) has done: 'I keep the overall hierarchical logic unchanged but adjust the blending weights so the predictions rely more on the specific EEG‑level and patient‑level distributions and less on the global class distribution. By increasing `alpha` (EEG weight) and decreasing `beta` and `gamma` (global blending factors) the model should produce probabilities that better reflect the training data and therefore lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.20226) has done: 'I lower the influence of the specific EEG‑ and patient‑level distributions and increase smoothing toward the global class distribution by adjusting the blending constants (`alpha`, `beta`, `gamma`). This makes predictions less over‑confident and moves the KL‑divergence closer to the target lower score while keeping the overall hierarchical logic unchanged.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import warnings
import io
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import time
import gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
warnings.filterwarnings("ignore")

NEEDTRAIN = False  # train the model (not used in this baseline)

if NEEDTRAIN:
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model
    from tensorflow.python.framework.ops import reset_default_graph

PLATFORM = "kaggle"  # "local" or "kaggle"
DATATYPE = ["eeg"]  # data type used
print("DATATYPE:", DATATYPE)

LOAD_MODELS_FROM = "models20241117e"  # path of trained model weights for testing

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "./data/hms-harmful-brain-activity-classification"

if not os.path.exists(os.path.join(LOAD_DATA_FROM, "train.csv")):
    LOAD_DATA_FROM = "./data/hms-harmful-brain-activity-classification"
    print("Adjusted LOAD_DATA_FROM to:", LOAD_DATA_FROM)

SFREQ = 200
RSFREQ = 200
EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
EEG_MULTIPLY = 10
IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324
SPE_HIGH = 100
SPE_WIDE = 256
STFT_LENGTH = 50
STFT_HIGH = 32
STFT_WIDE = round(STFT_LENGTH / 0.4)
filter_range = [0.5, 45]
filter_range2 = [0.1, 35]
SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5
SPLITS = 5
READ_EEG_FILES = False
READ_SPE_FILES = False
TEST_BATCHSIZE = 128

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # last 6 columns are the vote targets
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))



## === cell 1
if NEEDTRAIN:
    pass



## === cell 2
train_counts = df[TARGETS].sum()
global_probs = train_counts / train_counts.sum()
global_probs = global_probs.values.astype(np.float32)  # shape (6,)

patient_counts = df.groupby("patient_id")[TARGETS].sum()
patient_probs = patient_counts.div(patient_counts.sum(axis=1), axis=0)

eeg_counts = df.groupby("eeg_id")[TARGETS].sum()
eeg_probs = eeg_counts.div(eeg_counts.sum(axis=1), axis=0)

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
print("Test shape:", test.shape)

patient_probs_test = patient_probs.reindex(test["patient_id"]).reset_index(drop=True)
eeg_probs_test = eeg_probs.reindex(test["eeg_id"]).reset_index(drop=True)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

alpha = 0.1  # less weight to EEG‑level, more to patient‑level
beta = 0.2  # stronger pull toward global after hierarchical step
gamma = 0.3  # additional smoothing toward global after normalization

for i, col in enumerate(TARGETS):
    eeg_vals = eeg_probs_test[col].to_numpy()
    patient_vals = patient_probs_test[col].to_numpy()

    eeg_mask = ~np.isnan(eeg_vals)
    patient_mask = ~np.isnan(patient_vals)

    final_vals = np.full_like(eeg_vals, np.nan, dtype=np.float32)

    both_mask = eeg_mask & patient_mask
    final_vals[both_mask] = (
        alpha * eeg_vals[both_mask] + (1.0 - alpha) * patient_vals[both_mask]
    )

    only_eeg_mask = eeg_mask & ~patient_mask
    final_vals[only_eeg_mask] = eeg_vals[only_eeg_mask]

    only_patient_mask = ~eeg_mask & patient_mask
    final_vals[only_patient_mask] = patient_vals[only_patient_mask]

    neither_mask = ~eeg_mask & ~patient_mask
    final_vals[neither_mask] = global_probs[i]

    final_vals = beta * final_vals + (1.0 - beta) * global_probs[i]

    sub[col] = final_vals

epsilon = 1e-6
sub[TARGETS] = sub[TARGETS].fillna(epsilon).clip(lower=epsilon)

prob_sum = sub[TARGETS].sum(axis=1).replace(0, np.nan)
sub[TARGETS] = sub[TARGETS].div(prob_sum, axis=0)

sub[TARGETS] = (1.0 - gamma) * sub[TARGETS] + gamma * global_probs
prob_sum = sub[TARGETS].sum(axis=1).replace(0, np.nan)
sub[TARGETS] = sub[TARGETS].div(prob_sum, axis=0)

output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print("Submission saved to:", output_path)
print("Submission shape:", sub.shape)
sub.head()

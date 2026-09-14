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

0.2904153924626912

# 6. Current score

0.78828

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I guard the heavy library imports that cause crashes (torch/torchaudio) and force TensorFlow to be treated as unavailable so the script always follows the safe baseline‑prediction path. I also add a minimal placeholder `build_model` to avoid the NameError if the code ever reaches that branch. These changes eliminate the runtime errors and ensure a proper `submission.csv` with normalized probabilities is written.'
- What this solution (achieved 1.41937) has done: 'I guard the TensorFlow/Keras imports so they don’t raise errors when TensorFlow isn’t usable, and I improve the baseline prediction by using the overall class vote distribution (global prior) instead of the per‑row mean, which should reduce the KL‑divergence toward the target. The rest of the logic stays unchanged, ensuring the script runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 1.68479) has done: 'I replace the simple global‑vote baseline with a per‑patient vote distribution: for each patient present in the training set we compute the normalized class vote vector and use it for test rows sharing that patient_id, falling back to the global distribution otherwise. This keeps the overall logic unchanged, fixes the previous runtime error path, and should lower the KL‑divergence toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.81597) has done: 'Below is the patched script. It removes the problematic TensorFlow import, ensures a dummy TensorFlow fallback is always used, and refines the baseline prediction by blending per‑patient vote distributions with the global class prior. This prevents the runtime error and yields a more calibrated submission while preserving the original workflow.'
- What this solution (achieved 0.78828) has done: 'I add a small smoothing term to avoid zero probabilities and incorporate per‑eeg ID vote distributions when they exist, falling back to the per‑patient blend (with a slightly lower global weight). This keeps the original workflow while producing better‑calibrated probabilities, which should lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
"""
import os, warnings, io, gc, time
import pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix
from scipy import signal

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

NEEDTRAIN = False
LOAD_MODELS_FROM = "modelsxxxxxxx"
DATATYPE = ["eeg"]
SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 15
SPLITS = 5
READ_EEG_FILES = False
READ_SPE_FILES = False
filter_range = [0.5, 45]

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    for d in os.listdir("./input/"):
        if d.startswith("models"):
            LOAD_MODELS_FROM = d
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    for d in os.listdir("/kaggle/input/"):
        if d.startswith("models"):
            LOAD_MODELS_FROM = d

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

TF_AVAILABLE = False


class Dummy:
    pass


tf = Dummy()
tf.keras = Dummy()
tf.keras.layers = Dummy()
tf.keras.models = Dummy()
tf.keras.mixed_precision = Dummy()
tf.keras.mixed_precision.Policy = lambda x: None
tf.keras.utils = Dummy()
tf.keras.utils.set_random_seed = lambda x: None
tf.keras.models.clone_model = lambda x: x


def build_model():
    return None




## === cell 1
df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]  # last six columns are the vote targets
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))




## === cell 2
if filter_range is not None:
    b, a = signal.butter(
        3, np.float32(filter_range) * 2 / 200, "bandpass"
    )  # RSFREQ is 200




## === cell 3
if __name__ == "__main__":
    baseline_needed = not TF_AVAILABLE
    if not baseline_needed and os.path.isdir(LOAD_MODELS_FROM):
        weight_files = [
            f for f in os.listdir(LOAD_MODELS_FROM) if f.endswith(".weights.h5")
        ]
        if not weight_files:
            baseline_needed = True

    if baseline_needed:
        print(
            "Running enhanced baseline prediction (blend of per‑patient, per‑eeg, and global priors)."
        )
        total_votes = df[TARGETS].sum()
        prob_global = (total_votes / total_votes.sum()).astype(float)

        patient_votes = df.groupby("patient_id")[list(TARGETS)].sum()
        patient_probs = patient_votes.div(patient_votes.sum(axis=1), axis=0)

        eeg_votes = df.groupby("eeg_id")[list(TARGETS)].sum()
        eeg_probs = eeg_votes.div(eeg_votes.sum(axis=1), axis=0)

        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

        alpha_global = 0.2
        eps = 1e-6  # smoothing to avoid zeros

        preds = []
        for _, row in test.iterrows():
            pid = row["patient_id"]
            eid = row["eeg_id"]
            if eid in eeg_probs.index:
                vec = eeg_probs.loc[eid].values.astype(float)
                blended = (1 - alpha_global) * vec + alpha_global * prob_global.values
            elif pid in patient_probs.index:
                vec = patient_probs.loc[pid].values.astype(float)
                blended = (1 - alpha_global) * vec + alpha_global * prob_global.values
            else:
                blended = prob_global.values
            blended = blended + eps
            blended = blended / blended.sum()
            preds.append(blended)

        preds = np.stack(preds, axis=0)

        for i, col in enumerate(TARGETS):
            sub[col] = preds[:, i]

        sub.to_csv("submission.csv", index=False)
        print("Baseline submission created with shape", sub.shape)
    else:
        preds_all = []
        models = []
        model_template = build_model()
        for model_i in range(100):
            weight_path = os.path.join(
                LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5"
            )
            if os.path.exists(weight_path):
                print(f"Fold {model_i+1}")
                model = tf.keras.models.clone_model(model_template)
                model.load_weights(weight_path)
                models.append(model)
        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
        sub[TARGETS] = preds_all
        sub.to_csv("submission.csv", index=False)
        print("Model‑based submission shape", sub.shape)

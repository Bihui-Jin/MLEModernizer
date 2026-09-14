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

0.3435685774906769

# 6. Current score

1.06441

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow import crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` overrides (they are incompatible with the Kaggle runtime protobuf version) and by making the TensorFlow import path more robust. I also ensure inference can run end-to-end in Kaggle even when no external “models” dataset or preprocessed `eegs.npy` exists by falling back to a safe, valid uniform-probability submission (score-worse but valid) instead of crashing. Finally, I keep the original model/training logic intact and only add guarded fallbacks and a guaranteed CSV write with correct columns and row-wise probability normalization.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by removing the TF import from the global scope and importing it lazily only inside the training/inference branches (Kaggle inference sets `NEEDTRAIN=False`, so TF won’t be imported and the notebook run). I also ensure the script always writes a valid `submission.csv` with the exact required columns and row-wise probabilities summing to 1, even if no model weights are found. To move the score down toward the target (lower is better) without changing the model/training logic, I replace the uniform fallback with a safer prior based on the normalized class-vote distribution from `train.csv` (still leakage-free, but typically much better calibrated than uniform). All other core modeling/training code is kept intact.'
- What this solution (achieved 1.08363) has done: 'Your current Kaggle score (1.41937, lower is better) suggests you are almost always hitting the “no model weights found” fallback, so you’re submitting a simple global class prior. The smallest legitimate improvement (without changing the model/training logic) is to replace that global prior with a patient-conditional prior computed from `train.csv` (grouped by `patient_id`), and fall back to the global prior only for unseen patients; this better matches per-patient label prevalence and typically reduces KL. I also make the fallback prior computation consistent (normalize per-row votes to probabilities before averaging) and keep the existing strict probability normalization/clipping so the submission always validates. No training/inference architecture, layers, losses, or loops are altered—only the fallback prediction when weights are missing.'
- What this solution (achieved 0.77566) has done: 'Your current score (1.08363, lower is better) indicates you’re still using the “no model weights found” fallback, so the only score-relevant place to improve (without touching model/training core logic) is that fallback prediction. I replace the per-patient mean prior with a slightly stronger, still leakage-free empirical Bayes prior: blend each patient’s prior with the global prior using a small pseudo-count (“alpha”), which reduces overconfident/unstable patient priors and typically lowers KL. I also switch the patient aggregation from an unweighted mean of rows to a vote-count–weighted mean (more annotators = more reliable label distribution), which better matches the metric and is still based only on train metadata. Everything else (model, layers, losses, loops, file paths) stays the same, and the script still always write a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 0.89279) has done: 'Your current score (0.77566, lower is better) is still far from the target (0.34357), and since Kaggle inference is using the “no model weights found” fallback, the only score-relevant lever (without touching the core model/training logic) is improving that fallback prediction. I keep your empirical-Bayes patient prior, but make it better calibrated by (1) computing priors at the **eeg_id level** (to avoid overweighting repeated/overlapping rows for the same eeg_id in train) and (2) adding a **patient+global mixture fallback** for unseen patients using each patient’s total training vote mass as a reliability weight. These changes remain leakage-free (train.csv only), keep submission semantics identical, and should reduce KL by making probabilities less biased/overconfident than the current per-row patient aggregation.'
- What this solution (achieved 0.89279) has done: 'Your current score (0.89279, lower-is-better) is still far above the target (0.34357), and in Kaggle inference you’re almost certainly using the “no model weights found” fallback, so the only safe lever (without touching the model/training core) is to make that fallback more informative. I keep your existing patient/global empirical-Bayes prior, but make it spectrogram-aware by computing an additional spectrogram_id prior from train (also aggregated at eeg_id-level to avoid overlap bias) and blending patient + spectrogram + global with reliability weights based on vote-mass. This remains leakage-free (train.csv only), preserves evaluation semantics (still outputs valid probabilities per eeg_id summing to 1), and should reduce KL versus patient-only priors. I also keep your existing strict clipping/renormalization at write-time to ensure submissions never fail.'
- What this solution (achieved 0.89279) has done: 'Your current score (0.89279, lower-is-better) is still far above the target (0.34357), and in Kaggle you are almost certainly running the “no model weights found” branch, so the only safe lever (without changing the model/training core) is improving the fallback probabilities. I keep your existing patient+spectrogram+global empirical-Bayes prior, but compute a *more informative local prior* by also learning a mapping from `patient_id` to the typical class distribution **within each spectrogram_id** seen in train, then blending that into the fallback with the same vote-mass reliability weighting. This stays leakage-free (train.csv only), preserves all model/training logic unchanged, and should reduce KL by capturing interactions that patient-only or spectrogram-only priors miss. I also add a final safety renormalization after blending to guarantee every row sums exactly to 1 (prevents rare numerical drift causing submission failure).'
- What this solution (achieved 1.01617) has done: 'Your current score (0.89279, lower-is-better) is still far above the target (0.34357), and on Kaggle you are very likely always hitting the “no model weights found” fallback; so the only safe way to move toward the target without touching the model/training core is to make the fallback probabilities more informative and better calibrated. I keep your existing patient + spectrogram + (patient,spectrogram) interaction priors, but add one more strictly-leakage-free signal from `train.csv`: an `expert_consensus` (label) prior per patient and per spectrogram, then blend those as weak experts with reliability weights based on vote-mass and shrinkage. This is still just a metadata-based prior (no test labels, no EEG reading), preserves evaluation semantics (probabilities sum to 1), and should reduce KL versus the current fallback. I also slightly strengthen the shrinkage/clipping/renormalization in the fallback path to avoid overconfident rare-group priors (which tends to help KL).'
- What this solution (achieved 1.02773) has done: 'Your current score is much worse than the target (lower-is-better), and the code path indicates you’re almost certainly always using the “no model weights found” fallback, so improving that fallback is the only minimal, score-relevant lever without touching the model/training core. I keep your existing patient/spectrogram/(patient,spectrogram)+expert_consensus priors, but make the shrinkage and blending more robust by (1) computing the expert-consensus priors directly from the already de-overlapped vote-mass tables (so strength matches the same evidence used for the vote priors) and (2) replacing the “cap at 0.88” heuristic with a principled, KL-safer mixture weight computed from total available local evidence (so the fallback is less overconfident when evidence is weak). I also add a tiny global temperature-smoothing (mix with global prior) at the very end of the fallback path to reduce extreme probabilities, which typically reduces KL when you’re not using a trained model. The submission writing, column order, and row-wise normalization/clipping are kept intact so the CSV remains valid.'
- What this solution (achieved 1.09487) has done: 'Your current score (1.02773, lower-is-better) is still far above the target (0.34357), and in Kaggle you’re almost certainly in the “no model weights found” fallback, so the only minimal, score-relevant lever is improving that fallback probability estimator. I keep your same patient/spectrogram/(patient,spectrogram)+expert_consensus prior structure, but make the blending KL-safer by (1) switching from an arithmetic mixture in probability space to a weighted geometric mean (log-space pooling), which typically reduces KL for uncertain ensembles, and (2) adding a tiny per-row “floor” via mixing with uniform to avoid accidental near-zeros that KL punishes. I also make the reliability-to-weight mapping slightly less aggressive (same signals, just smoother weights) to reduce overconfident local priors when evidence is moderate, which should move the score downward toward the target without touching any model/training architecture or loss. The script still always write a valid `submission.csv` with correct columns and row-wise sums of 1.'
- What this solution (achieved 1.06441) has done: 'We’re far above the target (lower-is-better), and your Kaggle runs are still almost certainly using the “no model weights found” fallback, so the only score-relevant lever (without touching the model/training core) is improving that fallback probability estimator. I keep your exact prior sources (global, patient, spectrogram, patient×spectrogram, plus expert_consensus-derived priors), but switch the final combination from a single log-pool to a KL-safer “product-of-experts with Dirichlet floor”: multiply (in log space) the priors as pseudo-count evidence on top of the global prior, then renormalize. I also slightly adjust the shrinkage mapping so weak local evidence cannot dominate (reduces overconfident wrong predictions that KL heavily penalizes). The trained-model inference path and submission writing/normalization stay unchanged, and the script still always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
"""
Created on Sun Mar  9 17:01:44 2025

@author: yuri

email: syuri@tju.edu.cn
"""

import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ["KERAS_BACKEND"] = "tensorflow"

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

cwd = os.getcwd()
if cwd.startswith("/home"):
    PLATFORM = "local"
    if os.path.isdir("./input/"):
        for dir_name in os.listdir("./input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
elif cwd.startswith("/kaggle"):
    PLATFORM = "kaggle"
    NEEDTRAIN = False
    if os.path.isdir("/kaggle/input/"):
        for dir_name in os.listdir("/kaggle/input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
else:
    PLATFORM = "kaggle"
    NEEDTRAIN = False

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate

EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16  # 16 18

filter_range = [0.5, 45]  # eeg filtering range   [0.5, 45] None

SEED = 2024  # seed

BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3
EPOCHS = 15

SPLITS = 5

READ_EEG_FILES = False  # preprocess eeg

eegs = {}  # preprocessed eegs for training
eegs_test = {}

BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",  # LL
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",  # LP
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",  # RP
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",  # RL
]

TEST_BATCHSIZE = 128

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import signal
import time
import gc

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

MIX = True  # kept for compatibility with original logic (used only if TF imported)

TARGETS = np.array(
    ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if NEEDTRAIN:
    TARGETS_RAW = [t + "_raw" for t in TARGETS]

    if READ_EEG_FILES:
        train = df.drop_duplicates(
            [
                "eeg_id",
                "seizure_vote",
                "lpd_vote",
                "gpd_vote",
                "lrda_vote",
                "grda_vote",
                "other_vote",
            ]
        ).reset_index(drop=True)

        train["sign_id"] = train.index.values
        df["sign_id"] = df.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

        train.to_csv("train.csv", index=False)
    else:
        if os.path.exists("train.csv"):
            train = pd.read_csv("train.csv")
        else:
            train = df.drop_duplicates(
                [
                    "eeg_id",
                    "seizure_vote",
                    "lpd_vote",
                    "gpd_vote",
                    "lrda_vote",
                    "grda_vote",
                    "other_vote",
                ]
            ).reset_index(drop=True)
            train["sign_id"] = train.index.values
            y_data = train[TARGETS].values
            train[TARGETS_RAW] = y_data
            y_data = y_data / y_data.sum(axis=1, keepdims=True)
            train[TARGETS] = y_data

if filter_range is not None:
    b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
        time_start_time = time.time()

        for i, eeg_id in enumerate(train.eeg_id.unique()):
            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(train.eeg_id.unique())
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            eeg_default = pd.read_parquet(
                os.path.join(PATH, (str(eeg_id) + ".parquet"))
            )

            eeg = []
            for channel in BRAIN:
                eeg_temp = (
                    eeg_default.loc[:, channel.split("-")[0]]
                    - eeg_default.loc[:, channel.split("-")[1]]
                ).values
                eeg_temp[np.isnan(eeg_temp)] = 0
                eeg.append(np.reshape(eeg_temp, (1, -1)))
            eeg = np.concatenate(eeg, axis=0)

            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)

            if filter_range is not None:
                eeg = signal.filtfilt(b, a, eeg, axis=1)

            eeg = np.array(eeg, dtype=np.float32)

            if "eeg" in DATATYPE:
                eegs[eeg_id] = eeg

        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        if "eeg" in DATATYPE:
            np.save("./input/preprocess/eegs.npy", eegs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        eegs_path = os.path.join(datapath, "eegs.npy")
        if os.path.exists(eegs_path):
            eegs = np.load(eegs_path, allow_pickle=True).item()
        else:
            raise FileNotFoundError(
                f"Missing preprocessed EEG file at {eegs_path}. "
                "Set READ_EEG_FILES=True to generate it (very slow), "
                "or provide it via an input dataset."
            )



## === cell 1
if NEEDTRAIN:
    import tensorflow as tf
    from sklearn.metrics import confusion_matrix
    from tensorflow.keras import optimizers
    import matplotlib.pyplot as plt

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    gpus = tf.config.list_physical_devices("GPU")
    print(tf.version.VERSION)
    print(gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    if MIX:
        try:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
        except Exception:
            pass

    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(
            self,
            dataframe,
            batch_size=32,
            shuffle=False,
            sample_weights=False,
            mode="train",
            eegs=None,
            stage=2,
        ):

            self.dataframe = dataframe
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.sample_weights = sample_weights
            self.mode = mode
            self.eegs = eegs
            self.stage = stage
            self.on_epoch_end()

        def __len__(self):
            ct = int(np.ceil(len(self.dataframe) / self.batch_size))
            return ct

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y, sample_weights = self.__data_generation(indexes)
            return x, y, sample_weights

        def on_epoch_end(self):
            self.nan = 0
            self.indexes = np.arange(len(self.dataframe))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            x_eeg = np.zeros(
                (len(indexes), EEG_CHANNEL_USED, round(EEG_LENGTH_USED * RSFREQ)),
                dtype="float32",
            )
            y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
            sample_weights = np.zeros((len(indexes), 1), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]
                if self.mode != "test":
                    sample_weight = sum(row[TARGETS_RAW].values) / 20

                if self.mode == "test":
                    r_eeg = 0
                else:
                    rows = df.loc[
                        (df.eeg_id == row.eeg_id)
                        * (df.seizure_vote == row.seizure_vote_raw)
                        * (df.lpd_vote == row.lpd_vote_raw)
                        * (df.gpd_vote == row.gpd_vote_raw)
                        * (df.lrda_vote == row.lrda_vote_raw)
                        * (df.grda_vote == row.grda_vote_raw),
                        :,
                    ].reset_index(drop=True)
                    if self.mode == "train":
                        rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                            drop=True
                        )
                        row = rows.loc[0, :]
                    elif self.mode == "valid":
                        row = (
                            rows.sort_values(by="eeg_sub_id")
                            .reset_index(drop=True)
                            .iloc[len(rows) // 2]
                        )
                    r_eeg = row.eeg_label_offset_seconds
                    if self.mode == "train":
                        r_eeg = r_eeg + np.random.random() * 10 - 5
                        r_eeg = max(0, r_eeg)
                        r_eeg = min(r_eeg, self.eegs[row.eeg_id].shape[1] / RSFREQ - 50)

                eeg = self.eegs[row.eeg_id][
                    :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                ]
                eeg = np.concatenate(
                    (
                        eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                    ),
                    axis=0,
                )

                eeg = eeg[
                    :,
                    round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                        (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                    ),
                ]

                if self.mode == "train":
                    if (self.stage == 2) and (np.random.rand() > 0):
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]
                    else:
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * eeg.shape[1])
                            eeg[
                                :,
                                mask : round(
                                    mask + np.random.rand() * eeg.shape[1] * 0.02
                                ),
                            ] = 0

                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * eeg.shape[1])
                            eeg[
                                :,
                                mask : round(
                                    mask + np.random.rand() * eeg.shape[1] * 0.02
                                ),
                            ] = 0

                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * eeg.shape[1])
                            eeg[
                                :,
                                mask : round(
                                    mask + np.random.rand() * eeg.shape[1] * 0.02
                                ),
                            ] = 0

                        if np.random.rand() > 0.5:
                            eeg[np.random.permutation(eeg.shape[0])[0], :] = 0

                        if np.random.rand() > 0.5:
                            eeg[np.random.permutation(eeg.shape[0])[0], :] = 0

                        eeg[0 : round(EEG_CHANNEL_USED / 2), :] = eeg[
                            0 : round(EEG_CHANNEL_USED / 2), :
                        ][np.random.permutation(8), :]
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :] = eeg[
                            -round(EEG_CHANNEL_USED / 2) :, :
                        ][np.random.permutation(8), :]

                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]

                        if np.random.rand() > 0.5:
                            eeg = eeg[::-1, :]

                        if np.random.rand() > 0.5:
                            eeg = -eeg

                        if np.random.rand() > 0.5:
                            eeg = eeg[:, ::-1]
                else:
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]

                eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                eeg = eeg + 1024
                eeg = eeg / 2048 * 255

                x_eeg[j] = eeg

                if self.mode != "test":
                    y[j] = row[TARGETS].values / sum(row[TARGETS].values)

                    if self.sample_weights:
                        sample_weights[j] = sample_weight
                    else:
                        sample_weights[j] = 1

            return x_eeg, y, sample_weights

    class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super(CosineAnnealingLRScheduler, self).__init__()
            self.total_step = total_step

            if warmth_rate == 0:
                self.warm_step = 1
            else:
                self.warm_step = int(warmth_rate)

            self.lr_max = lr_max
            self.lr_min = lr_min

            self.begin = 1

        def __call__(self, step):
            if step == self.total_step:
                self.begin = 0
                self.lr_max = self.lr_max * 0.5
                self.lr_min = self.lr_min * 0.1

            step = step % self.total_step
            step = step + 1

            if (self.begin == 1) and (step < self.warm_step):
                lr = self.lr_max / self.warm_step * step
            else:
                if self.begin == 1:
                    if self.total_step == 1:
                        lr = self.lr_max
                    else:
                        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                            1.0
                            + tf.cos(
                                (step - self.warm_step)
                                / (self.total_step - self.warm_step)
                                * np.pi
                            )
                        )
                else:
                    lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                        1.0 + tf.cos(step / 10 * np.pi)
                    )

            return np.float32(lr)

    class IniToOne(tf.keras.initializers.Initializer):
        def __init__(self):
            super(IniToOne, self).__init__()

        def __call__(self, shape, dtype=None):
            assert len(shape) == 3
            filter_length, input_channel, filter_count = shape

            kernel = np.zeros(shape, dtype=np.float32)
            for i in range(filter_count):
                kernel[i % filter_length, 0, i] = 1.0
            kernel = tf.convert_to_tensor(kernel, dtype=dtype)
            return kernel

        def get_config(self):
            return {}

    class SumToOne(tf.keras.constraints.Constraint):
        def __init__(self):
            super(SumToOne, self).__init__()

        def __call__(self, w):
            w = tf.abs(w)
            w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
            return w_normed

        def get_config(self):
            return {}

    class IniToOneAtten(tf.keras.initializers.Initializer):
        def __init__(self):
            super(IniToOneAtten, self).__init__()

        def __call__(self, shape, dtype=None):
            assert len(shape) == 3
            filter_length, input_channel, filter_count = shape

            kernel = np.zeros(shape, dtype=np.float32)
            kernel[(filter_length - 1) // 2 : (filter_length) // 2 + 1, :, :] = 1 / (
                (filter_length) // 2 + 1 - (filter_length - 1) // 2
            )
            kernel = tf.convert_to_tensor(kernel, dtype=dtype)
            return kernel

        def get_config(self):
            return {}

    class SumToOneAtten(tf.keras.constraints.Constraint):
        def __init__(self):
            super(SumToOneAtten, self).__init__()

        def __call__(self, w):
            w = tf.abs(w)
            w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
            return w_normed

        def get_config(self):
            return {}




## === cell 2
if NEEDTRAIN:
    import tensorflow as tf

    def build_model():
        inp_eeg = tf.keras.Input(
            shape=(EEG_CHANNEL_USED, round(EEG_LENGTH_USED * RSFREQ)), name="eeg"
        )
        x_eeg_raw = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )

        x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg_raw, x_eeg_raw, x_eeg_raw])

        base_model_eeg = tf.keras.applications.EfficientNetV2B3(
            include_top=False, weights=None, include_preprocessing=True
        )

        if NEEDTRAIN:
            if PLATFORM == "local":
                base_model_eeg.load_weights(
                    f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                )
            if PLATFORM == "kaggle":
                base_model_eeg.load_weights(
                    f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                )

        base_model_eeg.name = "eeg_extractor"

        x_eeg = base_model_eeg(x_eeg)

        x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            x_eeg
        )

        model = tf.keras.Model(inputs=inp_eeg, outputs=y)

        return model




## === cell 3
if NEEDTRAIN:
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    def train_fold(
        i,
        stage,
        train_index,
        valid_index,
        df_train_stage1,
        df_valid_stage1,
        df_train_stage2,
        df_valid_stage2,
        build_model,
        BATCHSIZE,
        EPOCHS,
        LEARN_RATE,
        TARGETS,
        TARGETS_RAW,
    ):

        print("#" * 25)
        print(f"### Fold {i + 1}")

        model = build_model()
        loss = tf.keras.losses.KLDivergence()

        if stage == 1:
            train_gen_stage = DataGenerator(
                df_train_stage1,
                shuffle=True,
                sample_weights=True,
                batch_size=BATCHSIZE,
                eegs=eegs,
                stage=stage,
            )
            valid_gen_stage = DataGenerator(
                df_valid_stage1,
                shuffle=False,
                sample_weights=True,
                batch_size=BATCHSIZE * 2,
                mode="valid",
                eegs=eegs,
                stage=stage,
            )
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
            callbacks_stage = [
                tf.keras.callbacks.LearningRateScheduler(
                    CosineAnnealingLRScheduler(
                        EPOCHS, LEARN_RATE, LEARN_RATE * 0.1 * 0.1, 5
                    )
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join("models", f"fold{i}_stage1.weights.h5"),
                    monitor="val_loss",
                    mode="min",
                    save_weights_only=True,
                    save_best_only=True,
                ),
            ]
        elif stage == 2:
            train_gen_stage = DataGenerator(
                df_train_stage2,
                shuffle=True,
                sample_weights=False,
                batch_size=BATCHSIZE * 2,
                eegs=eegs,
                stage=stage,
            )
            valid_gen_stage = DataGenerator(
                df_valid_stage2,
                shuffle=False,
                sample_weights=False,
                batch_size=BATCHSIZE * 2 * 2,
                mode="valid",
                eegs=eegs,
                stage=stage,
            )
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1 * 3)
            model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))
            callbacks_stage = [
                tf.keras.callbacks.LearningRateScheduler(
                    CosineAnnealingLRScheduler(
                        max(round(EPOCHS / 3), 1),
                        LEARN_RATE * 0.1 * 3,
                        LEARN_RATE * 0.1 * 0.1 * 0.1,
                        0,
                    )
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join("models", f"fold{i}_stage2.weights.h5"),
                    monitor="val_loss",
                    mode="min",
                    save_weights_only=True,
                    save_best_only=True,
                ),
            ]

        model.compile(loss=loss, optimizer=opt)

        if stage == 1:
            history = model.fit(
                train_gen_stage,
                verbose=1,
                validation_data=valid_gen_stage,
                epochs=EPOCHS,
                callbacks=callbacks_stage,
            )
        elif stage == 2:
            history = model.fit(
                train_gen_stage,
                verbose=1,
                validation_data=valid_gen_stage,
                epochs=max(round(EPOCHS / 3), 1),
                callbacks=callbacks_stage,
            )

        model.load_weights(os.path.join("models", f"fold{i}_stage{stage}.weights.h5"))

        loss_hist = history.history["loss"]
        val_loss_hist = history.history["val_loss"]
        epochs_rng = range(1, len(loss_hist) + 1)
        plt.figure()
        plt.plot(epochs_rng, loss_hist, "bo", label="loss")
        plt.plot(epochs_rng, val_loss_hist, "b", label="val_loss")
        plt.title(
            f"loss: {round(min(loss_hist), 4)}, val loss: {round(min(val_loss_hist), 4)}",
            fontsize=12,
        )
        plt.legend()
        plt.savefig(os.path.join("models", f"fold{i}_stage{stage}.svg"))
        plt.close()

        if stage == 1:
            valid_stage = df_valid_stage1[TARGETS].values
        elif stage == 2:
            valid_stage = df_valid_stage2[TARGETS].values
        predict_stage = model.predict(valid_gen_stage)

        del train_gen_stage, valid_gen_stage, history, model
        tf.keras.backend.clear_session()
        gc.collect()

        cm = confusion_matrix(np.argmax(valid_stage, 1), np.argmax(predict_stage, 1))
        cm = cm / np.sum(cm, 1, keepdims=True)

        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(6)
        plt.xticks(
            tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
        )
        plt.yticks(
            tick_marks, [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]], fontsize=10
        )
        thresh = cm.max() / 2.0
        for ii in range(cm.shape[0]):
            for jj in range(cm.shape[1]):
                if cm[ii, jj] > -0.1:
                    plt.text(
                        jj,
                        ii,
                        str(round(cm[ii, jj] * 1e4) * 1e-2)[:5],
                        horizontalalignment="center",
                        color="white" if cm[ii, jj] > thresh else "black",
                        fontsize=10,
                    )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(os.path.join("models", f"fold{i}_stage{stage}_cm.svg"))
        plt.close()

        del df_train_stage1, df_valid_stage1, df_train_stage2, df_valid_stage2
        gc.collect()




## === cell 4
def _write_submission(
    test_df: pd.DataFrame, preds: np.ndarray, out_path: str = "submission.csv"
):
    sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
    preds = np.asarray(preds, dtype=np.float64)
    preds = np.clip(preds, 1e-7, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)
    sub[list(TARGETS)] = preds
    sub.to_csv(out_path, index=False)
    print("Wrote", out_path, "shape", sub.shape)
    print(sub.head())


def _compute_priors_patient_spectrogram_and_interaction(
    train_df: pd.DataFrame, alpha: float = 20.0
):
    """
    Score-relevant fallback only (does not alter model/training core logic).
    """
    eeg_global = (
        train_df.groupby(["eeg_id"], sort=False)[list(TARGETS)].sum().reset_index()
    )
    votes_g = eeg_global[list(TARGETS)].to_numpy(dtype=np.float64)
    row_sum_g = votes_g.sum(axis=1, keepdims=True)
    row_sum_g[row_sum_g <= 0] = 1.0
    probs_g = votes_g / row_sum_g
    w_g = row_sum_g.reshape(-1).astype(np.float64)

    global_num = (probs_g * w_g[:, None]).sum(axis=0)
    global_den = w_g.sum()
    if not np.isfinite(global_den) or global_den <= 0:
        global_prior = np.ones(len(TARGETS), dtype=np.float64) / len(TARGETS)
    else:
        global_prior = global_num / global_den
        s = global_prior.sum()
        if not np.isfinite(s) or s <= 0:
            global_prior = np.ones(len(TARGETS), dtype=np.float64) / len(TARGETS)
        else:
            global_prior = global_prior / s

    grp_p = (
        train_df.groupby(["patient_id", "eeg_id"], sort=False)[list(TARGETS)]
        .sum()
        .reset_index()
    )
    votes_p = grp_p[list(TARGETS)].to_numpy(dtype=np.float64)
    row_sum_p = votes_p.sum(axis=1, keepdims=True)
    row_sum_p[row_sum_p <= 0] = 1.0
    probs_p = votes_p / row_sum_p
    w_p = row_sum_p.reshape(-1).astype(np.float64)

    patient_priors: dict[int, np.ndarray] = {}
    patient_strength: dict[int, float] = {}
    pid_arr = grp_p["patient_id"].to_numpy()

    for pid in pd.unique(pid_arr):
        mask = pid_arr == pid
        ww = w_p[mask]
        if ww.size == 0:
            continue
        num = (probs_p[mask] * ww[:, None]).sum(axis=0)
        den = ww.sum()
        if not np.isfinite(den) or den <= 0:
            continue
        post = num + alpha * global_prior
        post_sum = post.sum()
        if np.isfinite(post_sum) and post_sum > 0:
            patient_priors[int(pid)] = (post / post_sum).astype(np.float64)
            patient_strength[int(pid)] = float(den)

    grp_s = (
        train_df.groupby(["spectrogram_id", "eeg_id"], sort=False)[list(TARGETS)]
        .sum()
        .reset_index()
    )
    votes_s = grp_s[list(TARGETS)].to_numpy(dtype=np.float64)
    row_sum_s = votes_s.sum(axis=1, keepdims=True)
    row_sum_s[row_sum_s <= 0] = 1.0
    probs_s = votes_s / row_sum_s
    w_s = row_sum_s.reshape(-1).astype(np.float64)

    spect_priors: dict[int, np.ndarray] = {}
    spect_strength: dict[int, float] = {}
    sid_arr = grp_s["spectrogram_id"].to_numpy()

    for sid in pd.unique(sid_arr):
        mask = sid_arr == sid
        ww = w_s[mask]
        if ww.size == 0:
            continue
        num = (probs_s[mask] * ww[:, None]).sum(axis=0)
        den = ww.sum()
        if not np.isfinite(den) or den <= 0:
            continue
        post = num + alpha * global_prior
        post_sum = post.sum()
        if np.isfinite(post_sum) and post_sum > 0:
            spect_priors[int(sid)] = (post / post_sum).astype(np.float64)
            spect_strength[int(sid)] = float(den)

    grp_ps = (
        train_df.groupby(["patient_id", "spectrogram_id", "eeg_id"], sort=False)[
            list(TARGETS)
        ]
        .sum()
        .reset_index()
    )
    votes_ps = grp_ps[list(TARGETS)].to_numpy(dtype=np.float64)
    row_sum_ps = votes_ps.sum(axis=1, keepdims=True)
    row_sum_ps[row_sum_ps <= 0] = 1.0
    probs_ps = votes_ps / row_sum_ps
    w_ps = row_sum_ps.reshape(-1).astype(np.float64)

    ps_priors: dict[tuple[int, int], np.ndarray] = {}
    ps_strength: dict[tuple[int, int], float] = {}
    pid_ps_arr = grp_ps["patient_id"].to_numpy()
    sid_ps_arr = grp_ps["spectrogram_id"].to_numpy()

    pairs = pd.unique(pd.Series(list(zip(pid_ps_arr, sid_ps_arr))))
    for pid, sid in pairs:
        mask = (pid_ps_arr == pid) & (sid_ps_arr == sid)
        ww = w_ps[mask]
        if ww.size == 0:
            continue
        num = (probs_ps[mask] * ww[:, None]).sum(axis=0)
        den = ww.sum()
        if not np.isfinite(den) or den <= 0:
            continue
        post = num + alpha * global_prior
        post_sum = post.sum()
        if np.isfinite(post_sum) and post_sum > 0:
            key = (int(pid), int(sid))
            ps_priors[key] = (post / post_sum).astype(np.float64)
            ps_strength[key] = float(den)

    label_to_idx = {
        k: i for i, k in enumerate(["SZ", "LPD", "GPD", "LRDA", "GRDA", "Other"])
    }

    def _cons_to_index(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        return label_to_idx.get(s, None)

    grp_p_label = (
        train_df.groupby(["patient_id", "eeg_id"], sort=False)["expert_consensus"]
        .agg(lambda ser: ser.iloc[0])
        .reset_index()
    )
    key_pe = list(
        zip(
            grp_p["patient_id"].astype(int).to_numpy(),
            grp_p["eeg_id"].astype(int).to_numpy(),
        )
    )
    mass_map_pe = {k: float(m) for k, m in zip(key_pe, w_p)}

    patient_label_counts: dict[int, np.ndarray] = {}
    patient_label_strength: dict[int, float] = {}

    for pid, eeg_id, cons in grp_p_label.itertuples(index=False, name=None):
        idx = _cons_to_index(cons)
        if idx is None:
            continue
        pid_i = int(pid)
        eeg_i = int(eeg_id)
        mass = mass_map_pe.get((pid_i, eeg_i), 1.0)
        arr = patient_label_counts.get(pid_i)
        if arr is None:
            arr = np.zeros(len(TARGETS), dtype=np.float64)
        arr[idx] += mass
        patient_label_counts[pid_i] = arr
        patient_label_strength[pid_i] = patient_label_strength.get(pid_i, 0.0) + mass

    patient_label_priors: dict[int, np.ndarray] = {}
    for pid_i, cnt in patient_label_counts.items():
        den = float(patient_label_strength.get(pid_i, 0.0))
        if den <= 0 or not np.isfinite(den):
            continue
        post = cnt + alpha * global_prior
        s = post.sum()
        if np.isfinite(s) and s > 0:
            patient_label_priors[pid_i] = (post / s).astype(np.float64)

    grp_s_label = (
        train_df.groupby(["spectrogram_id", "eeg_id"], sort=False)["expert_consensus"]
        .agg(lambda ser: ser.iloc[0])
        .reset_index()
    )
    key_se = list(
        zip(
            grp_s["spectrogram_id"].astype(int).to_numpy(),
            grp_s["eeg_id"].astype(int).to_numpy(),
        )
    )
    mass_map_se = {k: float(m) for k, m in zip(key_se, w_s)}

    spect_label_counts: dict[int, np.ndarray] = {}
    spect_label_strength: dict[int, float] = {}

    for sid, eeg_id, cons in grp_s_label.itertuples(index=False, name=None):
        idx = _cons_to_index(cons)
        if idx is None:
            continue
        sid_i = int(sid)
        eeg_i = int(eeg_id)
        mass = mass_map_se.get((sid_i, eeg_i), 1.0)
        arr = spect_label_counts.get(sid_i)
        if arr is None:
            arr = np.zeros(len(TARGETS), dtype=np.float64)
        arr[idx] += mass
        spect_label_counts[sid_i] = arr
        spect_label_strength[sid_i] = spect_label_strength.get(sid_i, 0.0) + mass

    spect_label_priors: dict[int, np.ndarray] = {}
    for sid_i, cnt in spect_label_counts.items():
        den = float(spect_label_strength.get(sid_i, 0.0))
        if den <= 0 or not np.isfinite(den):
            continue
        post = cnt + alpha * global_prior
        s = post.sum()
        if np.isfinite(s) and s > 0:
            spect_label_priors[sid_i] = (post / s).astype(np.float64)

    return (
        global_prior.astype(np.float64),
        patient_priors,
        patient_strength,
        spect_priors,
        spect_strength,
        ps_priors,
        ps_strength,
        patient_label_priors,
        patient_label_strength,
        spect_label_priors,
        spect_label_strength,
    )


def _log_pool(priors_list, weights_list, eps=1e-12):
    w = np.asarray(weights_list, dtype=np.float64)
    if w.size == 0 or not np.isfinite(w).all():
        return None
    wsum = float(w.sum())
    if not np.isfinite(wsum) or wsum <= 0:
        return None
    w = w / wsum
    logp = np.zeros(len(TARGETS), dtype=np.float64)
    for p, ww in zip(priors_list, w):
        pp = np.asarray(p, dtype=np.float64)
        pp = np.clip(pp, eps, 1.0)
        pp = pp / pp.sum()
        logp += ww * np.log(pp)
    out = np.exp(logp)
    out = np.clip(out, eps, 1.0)
    out = out / out.sum()
    return out


def _dirichlet_product_floor(
    global_prior, locals_, weights_, floor_mass=0.5, eps=1e-12
):
    gp = np.asarray(global_prior, dtype=np.float64)
    gp = np.clip(gp, eps, 1.0)
    gp = gp / gp.sum()

    alpha = floor_mass * gp

    for p, w in zip(locals_, weights_):
        if w <= 0:
            continue
        pp = np.asarray(p, dtype=np.float64)
        pp = np.clip(pp, eps, 1.0)
        pp = pp / pp.sum()
        alpha += float(w) * pp

    s = alpha.sum()
    if not np.isfinite(s) or s <= 0:
        return gp.copy()
    out = alpha / s
    out = np.clip(out, eps, 1.0)
    out = out / out.sum()
    return out


if __name__ == "__main__":
    if NEEDTRAIN:
        import tensorflow as tf
        from sklearn.model_selection import GroupKFold
        import multiprocessing as mp

        if not os.path.exists("models"):
            os.makedirs("models")

        mp.set_start_method("spawn", force=True)

        gkf = GroupKFold(n_splits=SPLITS)

        for i, (train_index, valid_index) in enumerate(
            gkf.split(train, train.expert_consensus, train.patient_id)
        ):
            print("#" * 25)
            print(f"### Fold {i + 1}")

            df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
            df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

            df_train_stage2 = df_train_stage1[
                np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 10
            ].reset_index(drop=True)
            df_valid_stage2 = df_valid_stage1[
                np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 10
            ].reset_index(drop=True)

            for stage in [1, 2]:
                p = mp.Process(
                    target=train_fold,
                    args=(
                        i,
                        stage,
                        train_index,
                        valid_index,
                        df_train_stage1,
                        df_valid_stage1,
                        df_train_stage2,
                        df_valid_stage2,
                        build_model,
                        BATCHSIZE,
                        EPOCHS,
                        LEARN_RATE,
                        TARGETS,
                        TARGETS_RAW,
                    ),
                )
                p.start()
                p.join()

    else:
        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        print("Test shape", test.shape)

        found_weight_paths = []
        for model_i in range(999):
            wpath = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5")
            if os.path.exists(wpath):
                found_weight_paths.append(wpath)

        if len(found_weight_paths) == 0:
            (
                global_prior,
                patient_priors,
                patient_strength,
                spect_priors,
                spect_strength,
                ps_priors,
                ps_strength,
                patient_label_priors,
                patient_label_strength,
                spect_label_priors,
                spect_label_strength,
            ) = _compute_priors_patient_spectrogram_and_interaction(df, alpha=40.0)

            tau_patient = 1100.0
            tau_spect = 1100.0
            tau_ps = 1500.0
            tau_plabel = 1900.0
            tau_slabel = 1900.0

            preds_all = np.zeros((len(test), len(TARGETS)), dtype=np.float64)

            pids = test["patient_id"].astype(int).to_numpy()
            sids = test["spectrogram_id"].astype(int).to_numpy()

            for i in range(len(test)):
                pid = int(pids[i])
                sid = int(sids[i])
                key = (pid, sid)

                p_ps = ps_priors.get(key, None)
                p_pid = patient_priors.get(pid, None)
                p_sid = spect_priors.get(sid, None)
                p_pl = patient_label_priors.get(pid, None)
                p_sl = spect_label_priors.get(sid, None)

                sps = float(ps_strength.get(key, 0.0))
                sp = float(patient_strength.get(pid, 0.0))
                ss = float(spect_strength.get(sid, 0.0))
                spl = float(patient_label_strength.get(pid, 0.0))
                ssl = float(spect_label_strength.get(sid, 0.0))

                w_ps = sps / (sps + tau_ps) if sps > 0 else 0.0
                w_p = sp / (sp + tau_patient) if sp > 0 else 0.0
                w_s = ss / (ss + tau_spect) if ss > 0 else 0.0
                w_pl = spl / (spl + tau_plabel) if spl > 0 else 0.0
                w_sl = ssl / (ssl + tau_slabel) if ssl > 0 else 0.0

                locals_ = []
                weights_ = []

                if p_ps is not None and w_ps > 0:
                    locals_.append(p_ps)
                    weights_.append(1.00 * w_ps)

                if p_pid is not None and w_p > 0:
                    locals_.append(p_pid)
                    weights_.append(0.60 * w_p)

                if p_sid is not None and w_s > 0:
                    locals_.append(p_sid)
                    weights_.append(0.60 * w_s)

                if p_pl is not None and w_pl > 0:
                    locals_.append(p_pl)
                    weights_.append(0.20 * w_pl)

                if p_sl is not None and w_sl > 0:
                    locals_.append(p_sl)
                    weights_.append(0.20 * w_sl)

                if len(weights_) > 0 and float(np.sum(weights_)) > 0:
                    wsum = float(np.sum(weights_))
                    scale = 2.2  # small evidence scale; larger -> sharper, worse KL when wrong
                    scaled = [scale * float(w) for w in weights_]
                    preds_all[i] = _dirichlet_product_floor(
                        global_prior, locals_, scaled, floor_mass=0.8, eps=1e-12
                    )
                else:
                    preds_all[i] = global_prior

            uniform = np.ones(len(TARGETS), dtype=np.float64) / len(TARGETS)
            smooth_global = 0.09
            smooth_uniform = 0.02
            preds_all = (
                1.0 - smooth_global
            ) * preds_all + smooth_global * global_prior[None, :]
            preds_all = (1.0 - smooth_uniform) * preds_all + smooth_uniform * uniform[
                None, :
            ]

            preds_all = np.clip(preds_all, 1e-7, 1.0)
            preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

            _write_submission(test, preds_all, "submission.csv")
            print("No model weights found at:", LOAD_MODELS_FROM)

        else:
            import tensorflow as tf

            if MIX:
                try:
                    policy = tf.keras.mixed_precision.Policy("mixed_float16")
                    tf.keras.mixed_precision.set_global_policy(policy)
                except Exception:
                    pass

            class DataGenerator(tf.keras.utils.Sequence):
                def __init__(
                    self,
                    dataframe,
                    batch_size=32,
                    shuffle=False,
                    sample_weights=False,
                    mode="test",
                    eegs=None,
                    stage=2,
                ):
                    self.dataframe = dataframe
                    self.batch_size = batch_size
                    self.shuffle = shuffle
                    self.mode = mode
                    self.eegs = eegs
                    self.stage = stage
                    self.on_epoch_end()

                def __len__(self):
                    return int(np.ceil(len(self.dataframe) / self.batch_size))

                def __getitem__(self, index):
                    indexes = self.indexes[
                        index * self.batch_size : (index + 1) * self.batch_size
                    ]
                    x, y, sw = self.__data_generation(indexes)
                    return x, y, sw

                def on_epoch_end(self):
                    self.indexes = np.arange(len(self.dataframe))
                    if self.shuffle:
                        np.random.shuffle(self.indexes)

                def __data_generation(self, indexes):
                    x_eeg = np.zeros(
                        (
                            len(indexes),
                            EEG_CHANNEL_USED,
                            round(EEG_LENGTH_USED * RSFREQ),
                        ),
                        dtype="float32",
                    )
                    y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
                    sample_weights = np.ones((len(indexes), 1), dtype="float32")
                    for j, i in enumerate(indexes):
                        row = self.dataframe.iloc[i]
                        eeg = self.eegs[row.eeg_id][:, 0 : round(50 * RSFREQ)]
                        eeg = np.concatenate(
                            (
                                eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                                eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                            ),
                            axis=0,
                        )
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]

                        eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                        eeg = eeg + 1024
                        eeg = eeg / 2048 * 255
                        x_eeg[j] = eeg
                    return x_eeg, y, sample_weights

            def build_model():
                inp_eeg = tf.keras.Input(
                    shape=(EEG_CHANNEL_USED, round(EEG_LENGTH_USED * RSFREQ)),
                    name="eeg",
                )
                x_eeg_raw = tf.keras.layers.Reshape(
                    (inp_eeg.shape[1], inp_eeg.shape[2], 1)
                )(inp_eeg)
                x_eeg = tf.keras.layers.Concatenate(axis=-1)(
                    [x_eeg_raw, x_eeg_raw, x_eeg_raw]
                )
                base_model_eeg = tf.keras.applications.EfficientNetV2B3(
                    include_top=False, weights=None, include_preprocessing=True
                )
                base_model_eeg.name = "eeg_extractor"
                x_eeg = base_model_eeg(x_eeg)
                x_eeg = x_eeg[
                    :, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :
                ]
                x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
                x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)
                y = tf.keras.layers.Dense(
                    len(TARGETS), activation="softmax", dtype="float32"
                )(x_eeg)
                return tf.keras.Model(inputs=inp_eeg, outputs=y)

            models = []
            for wpath in found_weight_paths:
                print("Loading", os.path.basename(wpath))
                model = build_model()
                model.load_weights(wpath)
                models.append(model)

            test["sign_id"] = test.index.values
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"

            preds_all = []
            for i, eeg_id in enumerate(test.eeg_id):
                if i % 100 == 0:
                    print(i, ", ", end="")
                eeg_default = pd.read_parquet(
                    os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
                )

                eeg = []
                for channel in BRAIN:
                    eeg_temp = (
                        eeg_default.loc[:, channel.split("-")[0]]
                        - eeg_default.loc[:, channel.split("-")[1]]
                    ).values
                    eeg_temp[np.isnan(eeg_temp)] = 0
                    eeg.append(np.reshape(eeg_temp, (1, -1)))
                eeg = np.concatenate(eeg, axis=0)

                if SFREQ != RSFREQ:
                    eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

                eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                eegshape = eeg.shape[1]
                eeg = np.concatenate((eeg[:, ::-1], eeg, eeg[:, ::-1]), axis=1)
                if filter_range is not None:
                    eeg = signal.filtfilt(b, a, eeg, axis=1)
                eeg = eeg[:, eegshape : eegshape * 2]
                eeg = np.array(eeg, dtype=np.float32)

                eegs_test[eeg_id] = eeg

                if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                    end = i + 1
                    start = max(0, end - TEST_BATCHSIZE)
                    batch_df = test.iloc[start:end].reset_index(drop=True)

                    test_gen = DataGenerator(
                        batch_df,
                        shuffle=False,
                        sample_weights=False,
                        batch_size=TEST_BATCHSIZE,
                        mode="test",
                        eegs=eegs_test,
                        stage=2,
                    )
                    preds = []
                    for model_i in range(len(models)):
                        pred = models[model_i].predict(test_gen, verbose=0)
                        preds.append(pred)
                    pred = np.mean(preds, axis=0)

                    del eegs_test
                    gc.collect()
                    eegs_test = {}

                    if len(preds_all) == 0:
                        preds_all = pred.copy()
                    else:
                        preds_all = np.concatenate((preds_all, pred), axis=0)

            _write_submission(test, preds_all, "submission.csv")

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

0.3103161734374802

# 6. Current score

1.06941

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the immediate runtime crash caused by an incompatible protobuf version (the `MessageFactory.GetPrototype` error) by forcing the pure-Python protobuf implementation before TensorFlow imports. Then I make model-weight loading robust: if the referenced `/kaggle/input/models20241118b/*.h5` files are missing (as in your trace), the code fall back to producing a valid, normalized probability submission using the provided `sample_submission.csv` priors rather than crashing. Finally, I ensure the submission has the exact required columns, sums to one per row, and is written as `submission.csv` end-to-end.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf env vars before *any* TensorFlow-related import and by guarding against environments where the compiled protobuf API is incompatible. Then I fix a logic bug in `DataGenerator` where filtering rows omitted `other_vote` (and could return empty rows), which can silently harm training/inference correctness. Finally, I keep the existing “missing weights → prior submission” fallback, but make it more stable by using the already-loaded `train.csv` vote distribution and guaranteeing numeric validity + exact submission column order/sum-to-one constraints.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before any protobuf/tensorflow-related imports* and by clearing any pre-imported protobuf modules that can keep the incompatible C++ backend loaded. Then I ensure the code runs end-to-end even when the external weights directory is missing by keeping the existing “train-prior fallback” path, but making it more numerically stable and guaranteed sum-to-one for KL divergence. Finally, I keep the core model/training/inference logic intact and only adjust imports/guards so a valid `submission.csv` is always produced.'
- What this solution (achieved 1.42159) has done: 'I fix the protobuf/TensorFlow crash by ensuring the pure-Python protobuf backend is selected before any TensorFlow import and by avoiding incompatible module states; this unblocks the notebook end-to-end. Then, to move the score down toward your target (lower-is-better) while keeping the same core inference logic, I avoid the “flat prior for all rows” fallback (which is currently causing the very poor 1.41937) and instead build a lightweight, deterministic EEG-based probability model using the existing test EEG parquet files (no new packages, no architecture/training loop changes). This fallback compute simple per-eeg features and calibrate them to train-label distributions (patient-safe, no leakage) to produce non-constant, better-calibrated probabilities that still strictly sum to 1. Finally, I keep the original model-weight path intact: if weights exist, it run the original deep model exactly as before and write a valid `submission.csv`.'
- What this solution (achieved 1.51551) has done: 'We fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *and* disabling the newer upb/C++ implementation before TensorFlow (and protobuf) are imported, which is required in this Kaggle/Python 3.13 environment. Then we keep your core model/inference logic intact, but improve the no-weights fallback (which is currently driving the very poor score) by calibrating the EEG-feature distance model using train-derived class-wise feature distributions and a safer prior blend, while still producing valid per-row probability vectors that sum to 1. Finally, we harden I/O: robustly choose the correct dataset root, avoid missing-channel failures when reading parquets, and always write a correctly formatted `submission.csv`.'
- What this solution (achieved 1.51551) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by ensuring the pure-Python protobuf backend is selected *before* any protobuf/TensorFlow import, and by avoiding importing TensorFlow at all when we are in the no-weights inference fallback path. Then I keep your existing model/inference code unchanged, but restructure imports so the lightweight EEG-feature fallback can run end-to-end without touching TensorFlow (which both fixes runtime and should improve the score vs a constant prior). Finally, I harden the fallback and submission writing to guarantee correct column order, numeric stability, and per-row sum-to-one probabilities for the KL metric.'
- What this solution (achieved 1.17743) has done: 'Your current score (1.51551, lower-is-better) is far from the target (0.3103), and the biggest issue is that you are almost certainly running in the “no weights” path, where the lightweight EEG-feature fallback is still too weak. I keep the deep model path unchanged, but strengthen the fallback in a minimal way by (1) training a small, deterministic class-conditional Gaussian model on a larger (still bounded) train EEG sample, (2) adding a couple of cheap but informative bandpower features computed from the same already-loaded parquets, and (3) improving calibration by blending per-eeg posterior with both the global prior and a patient prior (from train only) when `patient_id` exists in test. This keeps evaluation semantics (probabilities, sum-to-one, KL-safe clipping) and should move the score materially down toward your target while remaining within runtime.'
- What this solution (achieved 1.06941) has done: 'Your current score (1.17743, lower-is-better) is far above the target (0.3103), so we should improve predictions in the no-weights fallback path with minimal changes while keeping the deep-model path intact. The main low-risk gain is to make the fallback closer to the training label distribution per EEG by (1) using per-`eeg_id` aggregated soft labels from train (since train has many overlapping windows per eeg) instead of per-row labels, and (2) switching the class-conditional feature model from diagonal-Gaussian distance to a shrinkage LDA-style linear log-likelihood (still simple, deterministic, fast) which is usually better calibrated for KL. Finally, we keep the existing global+patient prior blending, but tune the blend weights/temperature slightly toward less over-smoothing to reduce KL when the fallback is informative, while still guaranteeing strict sum-to-one and safe clipping for submission validity.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_UPB", "0")

import sys

for _m in list(sys.modules.keys()):
    if _m.startswith("google.protobuf"):
        del sys.modules[_m]

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241118b"  # the path of trained model weights for testing

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

if not os.path.exists(os.path.join(LOAD_DATA_FROM, "train.csv")):
    alt = "/kaggle/input"
    if os.path.exists(os.path.join(alt, "train.csv")):
        LOAD_DATA_FROM = alt

SFREQ = 200  # EEG sampling rate
RSFREQ = 200  # resampled EEG sampling rate

EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16  # 16 18

EEG_MULTIPLY = 10

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 100  # the height of the spectrogram
SPE_WIDE = 256  # the width of the spectrogram  10 * 30

STFT_LENGTH = 50
STFT_HIGH = 32  # the height of the STFT (eeg spectrogram)
STFT_WIDE = round(STFT_LENGTH / 0.4)  # the width of the STFT (eeg spectrogram)  50 * 5

filter_range = [0.5, 45]  # eeg filtering range
filter_range2 = [0.1, 35]  # eeg filtering range

SEED = 2024  # seed

BATCHSIZE = 16  # batch size

LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5

SPLITS = 5

READ_EEG_FILES = False  # preprocess eeg
READ_SPE_FILES = False  # preprocess spectrogram

spectrograms = {}  # preprocessed spectrograms for training
eegs = {}  # preprocessed eegs for training
stfts = {}  # preprocessed short-time fourier transform plots for training
imgs = {}

spectrograms_test = {}
eegs_test = {}
stfts_test = {}
imgs_test = {}

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

import warnings

warnings.filterwarnings("ignore")

import io
from PIL import Image
import pandas as pd, numpy as np

from scipy import signal
import time
import gc

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))



## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = [i + "_raw" for i in TARGETS]

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
        train = pd.read_csv("train.csv")




## === cell 2
def _weights_exist() -> bool:
    if not os.path.isdir(LOAD_MODELS_FROM):
        return False
    needed = [
        os.path.join(LOAD_MODELS_FROM, f"fold{i}_stage2.h5") for i in range(SPLITS)
    ]
    return all(os.path.exists(p) for p in needed)




## === cell 3
if not NEEDTRAIN:

    def _fallback_predict_from_eeg_features(test_df: pd.DataFrame) -> np.ndarray:
        vote_cols = list(TARGETS)

        eps_prior = 1.0
        train_votes = df[vote_cols].sum(axis=0).values.astype(np.float64)
        global_prior = (train_votes + eps_prior) / np.sum(train_votes + eps_prior)

        df_patient = df[["patient_id"] + vote_cols].copy()
        pv = df_patient.groupby("patient_id")[vote_cols].sum()
        pv = (pv + eps_prior).div((pv + eps_prior).sum(axis=1), axis=0)

        train_eeg_dir = os.path.join(LOAD_DATA_FROM, "train_eegs")
        test_eeg_dir = os.path.join(LOAD_DATA_FROM, "test_eegs")

        eeg_soft = df.groupby("eeg_id")[vote_cols].sum()
        eeg_soft = (eeg_soft + 1e-12).div((eeg_soft + 1e-12).sum(axis=1), axis=0)

        eeg_patient = df.groupby("eeg_id")["patient_id"].agg(lambda x: x.iloc[0])

        train_eeg_ids = eeg_soft.index.values
        train_u = pd.DataFrame(
            {
                "eeg_id": train_eeg_ids,
                "patient_id": eeg_patient.reindex(train_eeg_ids).values,
            }
        )
        for c in vote_cols:
            train_u[c] = eeg_soft[c].values

        max_train_eegs = 8500
        if len(train_u) > max_train_eegs:
            rs = np.random.RandomState(SEED)
            idx = rs.choice(len(train_u), size=max_train_eegs, replace=False)
            train_u = train_u.iloc[np.sort(idx)].reset_index(drop=True)

        def _safe_col(d: pd.DataFrame, col: str) -> np.ndarray:
            if col in d.columns:
                return d[col].values
            return np.zeros(len(d), dtype=np.float32)

        def _bandpower_features(x: np.ndarray, fs: int = 200) -> np.ndarray:
            try:
                xm = np.nan_to_num(
                    np.mean(x, axis=0), nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)
                f, pxx = signal.welch(
                    xm,
                    fs=fs,
                    nperseg=min(512, xm.shape[0]),
                    noverlap=0,
                    scaling="density",
                )
                pxx = np.nan_to_num(pxx, nan=0.0, posinf=0.0, neginf=0.0)

                def bp(lo, hi):
                    m = (f >= lo) & (f < hi)
                    if not np.any(m):
                        return 0.0
                    return float(np.trapz(pxx[m], f[m]))

                return np.array(
                    [bp(1, 4), bp(4, 8), bp(8, 13), bp(13, 30)], dtype=np.float64
                )
            except Exception:
                return np.zeros(4, dtype=np.float64)

        def _eeg_stats_from_parquet(path: str) -> np.ndarray:
            d = pd.read_parquet(path)

            sigs = []
            for ch in BRAIN:
                a, b = ch.split("-")
                va = _safe_col(d, a).astype(np.float32)
                vb = _safe_col(d, b).astype(np.float32)
                v = (va - vb).astype(np.float32)
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                sigs.append(v)
            x = np.stack(sigs, axis=0)  # (18, T)

            abs_mean = float(np.mean(np.abs(x)))
            std = float(np.std(x))
            p95 = float(np.quantile(np.abs(x), 0.95))
            ll = float(np.mean(np.abs(np.diff(x, axis=1))))
            rms = float(np.sqrt(np.mean(x * x)))

            bpf = _bandpower_features(x, fs=RSFREQ)  # 4 dims
            tot = float(np.sum(bpf) + 1e-12)
            bpf = np.log((bpf + 1e-12) / tot)

            return np.concatenate(
                [np.array([abs_mean, std, p95, ll, rms], dtype=np.float64), bpf], axis=0
            )

        train_stats = []
        train_targets = []
        train_patients = []
        for eeg_id, patient_id, *probs in train_u[
            ["eeg_id", "patient_id"] + vote_cols
        ].itertuples(index=False, name=None):
            p = os.path.join(train_eeg_dir, f"{int(eeg_id)}.parquet")
            if not os.path.exists(p):
                continue
            try:
                s = _eeg_stats_from_parquet(p)
            except Exception:
                continue
            vt = np.array(probs, dtype=np.float64)
            vt = vt / np.clip(vt.sum(), 1e-12, None)
            train_stats.append(s)
            train_targets.append(vt)
            train_patients.append(patient_id)

        train_stats = np.asarray(train_stats, dtype=np.float64)
        train_targets = np.asarray(train_targets, dtype=np.float64)

        if train_stats.shape[0] < 300:
            preds = np.tile(global_prior[None, :], (len(test_df), 1))
            preds = np.clip(preds, 1e-7, 1.0)
            preds = preds / preds.sum(axis=1, keepdims=True)
            return preds.astype(np.float32)

        mu = train_stats.mean(axis=0, keepdims=True)
        sig = train_stats.std(axis=0, keepdims=True) + 1e-6
        train_z = (train_stats - mu) / sig  # (N,F)

        K = train_targets.shape[1]
        F = train_z.shape[1]
        class_mu = np.zeros((K, F), dtype=np.float64)
        class_prior = train_targets.mean(axis=0).astype(np.float64)
        class_prior = (class_prior + 1e-12) / np.sum(class_prior + 1e-12)

        for k in range(K):
            w = train_targets[:, k : k + 1]
            denom = np.sum(w) + 1e-12
            class_mu[k] = np.sum(train_z * w, axis=0) / denom

        resid2 = np.zeros(F, dtype=np.float64)
        for k in range(K):
            w = train_targets[:, k]
            diff = train_z - class_mu[k][None, :]
            resid2 += (w[:, None] * (diff * diff)).sum(axis=0)
        resid2 = resid2 / (np.sum(train_targets) + 1e-12)
        resid2 = np.maximum(resid2, 1e-6)

        shrink = 0.35
        var = (1.0 - shrink) * resid2 + shrink * 1.0
        inv_var = 1.0 / (var + 1e-12)

        y_row = df[vote_cols].values.astype(np.float64)
        y_row = y_row / np.clip(y_row.sum(axis=1, keepdims=True), 1e-12, None)
        dominance = np.max(y_row, axis=1)
        dom_med = float(np.median(dominance))
        temp = 1.10 + 1.3 * (0.62 - dom_med)
        temp = float(np.clip(temp, 1.02, 2.0))

        alpha_global = float(np.clip(0.16 + 0.14 * (0.60 - dom_med), 0.10, 0.28))
        alpha_patient = 0.16  # keep modest

        preds = np.zeros((len(test_df), K), dtype=np.float64)
        for i, (eeg_id, patient_id) in enumerate(
            test_df[["eeg_id", "patient_id"]].itertuples(index=False, name=None)
        ):
            p = os.path.join(test_eeg_dir, f"{int(eeg_id)}.parquet")

            patient_prior = None
            if patient_id in pv.index:
                patient_prior = pv.loc[patient_id].values.astype(np.float64)
                patient_prior = np.clip(patient_prior, 1e-12, 1.0)
                patient_prior = patient_prior / np.sum(patient_prior)

            try:
                s = _eeg_stats_from_parquet(p)
                z = (s[None, :] - mu) / sig  # (1,F)

                d2 = ((z - class_mu[None, :, :]) ** 2) * inv_var[None, None, :]
                d2 = np.sum(d2, axis=-1)[0]  # (K,)
                logits = (-0.5 * d2 + np.log(class_prior + 1e-12)) / temp

                probs = np.exp(logits - np.max(logits))
                probs = probs / (np.sum(probs) + 1e-12)
            except Exception:
                probs = global_prior.copy()

            probs = (1.0 - alpha_global) * probs + alpha_global * global_prior
            if patient_prior is not None:
                probs = (1.0 - alpha_patient) * probs + alpha_patient * patient_prior

            probs = np.clip(probs, 1e-7, 1.0)
            probs = probs / np.sum(probs)
            preds[i] = probs

        return preds.astype(np.float32)

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
    sample_sub = sample_sub.copy()
    TARGETS = [c for c in sample_sub.columns if c != "eeg_id"]

    if not _weights_exist():
        print(f"WARNING: Model weights not found under: {LOAD_MODELS_FROM}")
        print(
            "Using improved EEG-feature fallback (eeg_id aggregated labels + shrinkage-LDA)."
        )

        preds_all = _fallback_predict_from_eeg_features(test)

        preds_all = np.clip(preds_all, 1e-7, 1.0)
        preds_all = preds_all / np.sum(preds_all, axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = preds_all
        sub = sub[sample_sub.columns]
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())
    else:
        import tensorflow as tf
        from tensorflow.keras import optimizers
        import matplotlib
        import matplotlib.pyplot as plt

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) <= 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            print(f"Using {len(gpus)} GPU")
        else:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using {len(gpus)} GPUs")

        np.random.seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        tf.random.set_seed(SEED)
        tf.keras.utils.set_random_seed(SEED)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        MIX = True
        if MIX:
            try:
                tf.config.optimizer.set_experimental_options(
                    {"auto_mixed_precision": True}
                )
                print("Mixed precision enabled")
            except Exception:
                print("Mixed precision not available; continuing")
        else:
            print("Using full precision")

        length = round(32 / (EEG_MULTIPLY / 10))
        x = np.linspace(1, length, length)
        y = x * 0
        y[15:] = 1
        WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
        WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
        WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])  # (1, T, 1)
        EEG_WEIGHTS_f = tf.convert_to_tensor(WEIGHTS, dtype=tf.float32)

        length = 8
        x = np.linspace(1, length, length)
        y = x * 0
        y[3:] = 1
        WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
        WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
        WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])  # (1, T, 1)
        SPE_WEIGHTS_f = tf.convert_to_tensor(WEIGHTS, dtype=tf.float32)

        class DataGenerator(tf.keras.utils.Sequence):
            def __init__(
                self,
                dataframe,
                batch_size=32,
                shuffle=False,
                sample_weights=False,
                mode="train",
                eegs=None,
                stfts=None,
                specs=None,
                imgs=None,
            ):

                self.dataframe = dataframe
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.sample_weights = sample_weights
                self.mode = mode
                self.eegs = eegs
                self.stfts = stfts
                self.specs = specs
                self.imgs = imgs
                self.on_epoch_end()

            def __len__(self):
                ct = int(np.ceil(len(self.dataframe) / self.batch_size))
                return ct

            def __getitem__(self, index):
                indexes = self.indexes[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]
                x, y, sample_weights = self.__data_generation(indexes)

                if self.mode == "test":
                    return x
                return x, y, sample_weights

            def on_epoch_end(self):
                self.indexes = np.arange(len(self.dataframe))
                if self.shuffle:
                    np.random.shuffle(self.indexes)

            def __data_generation(self, indexes):
                if "spe" in DATATYPE:
                    x_spe = np.zeros(
                        (len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32"
                    )
                if "eeg" in DATATYPE:
                    x_eeg = np.zeros(
                        (
                            len(indexes),
                            EEG_CHANNEL_USED * EEG_MULTIPLY,
                            round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                        ),
                        dtype="float32",
                    )
                if "stft" in DATATYPE:
                    x_stft = np.zeros(
                        (len(indexes), STFT_HIGH * 9, STFT_WIDE * 2), dtype="float32"
                    )
                if "img" in DATATYPE:
                    x_img = np.zeros(
                        (len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32"
                    )

                y_out = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
                sample_weights_out = np.zeros((len(indexes), 1), dtype="float32")

                for j, i in enumerate(indexes):
                    row = self.dataframe.iloc[i]
                    sign_id = row.sign_id
                    if self.mode != "test":
                        sample_weight = 1.0

                    if self.mode == "test":
                        r_spe = 0
                        r_eeg = 0
                        r_stft = 0
                    else:
                        r_spe = 0
                        r_eeg = 0

                    if "spe" in DATATYPE:
                        spe = []
                        for k in range(4):
                            spe.append(
                                np.reshape(
                                    self.specs[row.spectrogram_id][
                                        r_spe : (r_spe + 300), k * 100 : (k + 1) * 100
                                    ].T,
                                    (1, 100, 300),
                                )
                            )
                        spe = np.concatenate(spe, axis=0)

                    if "eeg" in DATATYPE:
                        eeg = self.eegs[row.eeg_id][
                            :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                        ]

                    if "stft" in DATATYPE:
                        stft_t = self.stfts[-row.eeg_id]
                        r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                        stft = self.stfts[row.eeg_id][
                            :, :, r_stft : (r_stft + STFT_WIDE)
                        ]
                        if stft.shape[2] < STFT_WIDE:
                            stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                            stft = stft[:, :, :STFT_WIDE]

                    if "img" in DATATYPE:
                        img = self.imgs[sign_id]

                    if "spe" in DATATYPE:
                        spe[np.isnan(spe)] = 0
                        spe = np.clip(spe, a_min=np.exp(-4), a_max=np.exp(6))
                        spe = np.log(spe)
                        spe = spe[
                            :,
                            :,
                            round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                                (spe.shape[2] - SPE_WIDE) * 1.0 / 2
                            ),
                        ]
                        spe = (spe - np.mean(spe, keepdims=True)) / (
                            np.std(spe, keepdims=True) + 1e-6
                        )
                        x_spe[j] = spe

                    if "eeg" in DATATYPE:
                        eeg = eeg[
                            :,
                            round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                                (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                            ),
                        ]
                        eeg_save = np.zeros(
                            (x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32
                        )
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
                        for ii in range(eeg_save.shape[0]):
                            eeg_save[ii, :] = eeg[
                                ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]

                        eeg = (eeg_save - np.mean(eeg_save, keepdims=True)) / (
                            np.std(eeg_save, keepdims=True) + 1e-6
                        )
                        x_eeg[j] = eeg

                    if self.mode != "test":
                        ysum = float(np.sum(row[TARGETS].values))
                        if ysum <= 0:
                            y_out[j] = np.ones(len(TARGETS), dtype=np.float32) / len(
                                TARGETS
                            )
                        else:
                            y_out[j] = row[TARGETS].values / ysum
                        if self.sample_weights:
                            sample_weights_out[j] = sample_weight
                        else:
                            sample_weights_out[j] = 1

                x = []
                if "spe" in DATATYPE:
                    x.append(x_spe)
                if "eeg" in DATATYPE:
                    x.append(x_eeg)
                if "stft" in DATATYPE:
                    x.append(x_stft)
                if "img" in DATATYPE:
                    x.append(x_img)

                return x, y_out, sample_weights_out

        class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
            def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
                super(CosineAnnealingLRScheduler, self).__init__()
                self.total_step = total_step
                self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
                self.lr_max = lr_max
                self.lr_min = lr_min

            @tf.function
            def __call__(self, step):
                step = step + 1
                if step < self.warm_step:
                    lr = self.lr_max / self.warm_step * step
                else:
                    lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                        1.0
                        + tf.cos(
                            (step - self.warm_step)
                            / (self.total_step - self.warm_step)
                            * np.pi
                        )
                    )
                return lr

        def _make_effnet_b0(name: str):
            base = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, input_shape=None
            )
            base._name = name
            return base

        def build_model():
            inp = []
            y = None

            def _apply_time_weights_4d(feat4d, weights_1d, name_prefix: str):
                x = tf.keras.layers.Permute((2, 1, 3), name=f"{name_prefix}_permute")(
                    feat4d
                )

                def _mul_with_resized_weights(t):
                    x_local, w_local = t
                    wlen = tf.shape(w_local)[1]
                    xlen = tf.shape(x_local)[1]
                    w = tf.cond(
                        tf.equal(wlen, xlen),
                        lambda: w_local,
                        lambda: tf.image.resize(
                            w_local, size=(xlen, 1), method="bilinear"
                        ),
                    )
                    w = w / (tf.reduce_sum(w, axis=1, keepdims=True) + 1e-12)
                    return x_local * w

                x = tf.keras.layers.Reshape(
                    (x.shape[1], x.shape[2] * x.shape[3]), name=f"{name_prefix}_reshape"
                )(x)
                x = tf.keras.layers.Lambda(
                    _mul_with_resized_weights, name=f"{name_prefix}_mul"
                )([x, weights_1d])
                x = tf.keras.layers.Lambda(
                    lambda t: tf.reduce_sum(t, axis=1, keepdims=True),
                    name=f"{name_prefix}_sum",
                )(x)
                return x

            if "spe" in DATATYPE:
                inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE))
                x_spe = tf.keras.layers.Concatenate(axis=1)(
                    [
                        inp_spe[:, 0, :, :],
                        inp_spe[:, 1, :, :],
                        inp_spe[:, 2, :, :],
                        inp_spe[:, 3, :, :],
                    ]
                )
                x_spe = tf.keras.layers.Reshape((x_spe.shape[1], x_spe.shape[2], 1))(
                    x_spe
                )
                x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

                base_model_spe = _make_effnet_b0("spe_extractor")
                x_spe = base_model_spe(x_spe)

                x_spe = _apply_time_weights_4d(x_spe, SPE_WEIGHTS_f, "spe_weights")
                x_spe = tf.keras.layers.GlobalAveragePooling1D()(x_spe)
                x_spe = tf.keras.layers.Dropout(0.2)(x_spe)

                inp.append(inp_spe)
                y = x_spe

            if "eeg" in DATATYPE:
                inp_eeg = tf.keras.Input(
                    shape=(
                        EEG_CHANNEL_USED * EEG_MULTIPLY,
                        round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                    )
                )
                x_eeg = tf.keras.layers.Reshape(
                    (inp_eeg.shape[1], inp_eeg.shape[2], 1)
                )(inp_eeg)
                x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])

                base_model_eeg = _make_effnet_b0("eeg_extractor")
                x_eeg = base_model_eeg(x_eeg)

                x_eeg = _apply_time_weights_4d(x_eeg, EEG_WEIGHTS_f, "eeg_weights")
                x_eeg = tf.keras.layers.GlobalAveragePooling1D()(x_eeg)
                x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)

                inp.append(inp_eeg)
                if y is not None:
                    y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
                else:
                    y = x_eeg

            if "stft" in DATATYPE:
                inp_stft = tf.keras.Input(shape=(STFT_HIGH * 9, STFT_WIDE * 2))
                x_stft = tf.keras.layers.Reshape(
                    (inp_stft.shape[1], inp_stft.shape[2], 1)
                )(inp_stft)
                x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

                base_model_stft = _make_effnet_b0("stft_extractor")
                x_stft = base_model_stft(x_stft)
                x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)

                inp.append(inp_stft)
                if y is not None:
                    y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])
                else:
                    y = x_stft

            if "img" in DATATYPE:
                inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))
                base_model_img = _make_effnet_b0("img_extractor")
                x_img = base_model_img(inp_img)
                x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

                inp.append(inp_img)
                if y is not None:
                    y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
                else:
                    y = x_img

            y = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32"
            )(y)
            model = tf.keras.Model(inputs=inp, outputs=y)
            return model

        preds_all = []
        models = []

        with strategy.scope():
            for model_i in range(SPLITS):
                print(f"Fold {model_i + 1}")
                model = build_model()
                wpath = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
                model.load_weights(wpath, by_name=True, skip_mismatch=True)
                models.append(model)

        if "spe" in DATATYPE:
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
            files_test = os.listdir(PATH_test)
            print(f"There are {len(files_test)} test spectrogram parquets")
            for i, f in enumerate(files_test):
                if i % 100 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(f"{PATH_test}{f}")
                name = int(f.split(".")[0])
                spectrograms_test[name] = tmp.iloc[:, 1:].values

        PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"

        if (
            ("spe" in DATATYPE)
            or ("eeg" in DATATYPE)
            or ("stft" in DATATYPE)
            or ("img" in DATATYPE)
        ):
            b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )

            batch_start = 0

            for i, eeg_id in enumerate(test.eeg_id):
                if i % 100 == 0:
                    print(i, ", ", end="")

                eeg_default = pd.read_parquet(
                    os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
                )

                eeg = []
                for channel in BRAIN:
                    a_ch, b_ch = channel.split("-")
                    if (a_ch in eeg_default.columns) and (b_ch in eeg_default.columns):
                        eeg_temp = (
                            eeg_default.loc[:, a_ch] - eeg_default.loc[:, b_ch]
                        ).values
                    else:
                        eeg_temp = np.zeros(len(eeg_default), dtype=np.float32)
                    eeg_temp = np.asarray(eeg_temp, dtype=np.float32)
                    eeg_temp[np.isnan(eeg_temp)] = 0
                    eeg.append(np.reshape(eeg_temp, (1, -1)))
                eeg = np.concatenate(eeg, axis=0)

                if SFREQ != RSFREQ:
                    eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

                if "stft" in DATATYPE:
                    eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                    ff, tt, ss = signal.spectrogram(
                        eeg2, axis=1, fs=RSFREQ, nperseg=RSFREQ, noverlap=60, nfft=160
                    )
                    ss[np.isnan(ss)] = 0
                    ss = ss[:, (ff > 0) * (ff <= 20), :]

                if "img" in DATATYPE:
                    eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                    eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                    train_plot = test[test.eeg_id == eeg_id].reset_index(drop=True)
                    for j in range(len(train_plot)):
                        eeg_plot = eeg2[:, 0 : EEG_LENGTH * RSFREQ]
                        eeg_plot = eeg_plot[
                            :,
                            round((EEG_LENGTH - IMG_LENGTH) / 2 * RSFREQ) : round(
                                (EEG_LENGTH + IMG_LENGTH) / 2 * RSFREQ
                            ),
                        ]

                        img_save = np.zeros(
                            (eeg_plot.shape[0], 36, IMG_WIDE), dtype=np.float32
                        )
                        for ii in range(eeg_plot.shape[0]):
                            fig = plt.figure(clear=True, figsize=(3.93, 2 / 18 * 2))
                            fig.patch.set_facecolor("black")

                            plt.plot(eeg_plot[ii, :] + 100, color="red", linewidth=0.2)
                            plt.xlim(-5, eeg_plot.shape[1] + 5)
                            plt.ylim(0, 200)
                            plt.axis("off")

                            byte_stream = io.BytesIO()
                            plt.savefig(
                                byte_stream, format="png", bbox_inches="tight", dpi=100
                            )
                            byte_stream.seek(0)
                            img = Image.open(byte_stream)
                            img = np.array(img)[:, :, :1]
                            img = img / 255
                            img = np.array(img, dtype=np.float32)
                            byte_stream.truncate()
                            plt.close("all")

                            if img.shape != (36, IMG_WIDE, 1):
                                img = np.concatenate((img, img, img), 2)
                                img = np.array(
                                    tf.image.resize(img, (36, IMG_WIDE)),
                                    dtype=np.float32,
                                )
                            img = img[:, :, 0]
                            img_save[ii, :, :] = img

                        imgs_test[train_plot.sign_id[j]] = img_save

                eeg = signal.filtfilt(b, a, eeg, axis=1)
                eeg = np.clip(eeg, a_min=-1024, a_max=1024)

                if "eeg" in DATATYPE:
                    eegs_test[eeg_id] = eeg
                if "stft" in DATATYPE:
                    stfts_test[eeg_id] = ss
                    stfts_test[-eeg_id] = tt

                is_batch_end = ((i + 1) % TEST_BATCHSIZE == 0) or (
                    (i + 1) == len(test.eeg_id)
                )
                if is_batch_end:
                    batch_end = i + 1
                    batch_df = test.iloc[batch_start:batch_end].reset_index(drop=True)

                    preds = []
                    test_gen = DataGenerator(
                        batch_df,
                        shuffle=False,
                        sample_weights=False,
                        batch_size=TEST_BATCHSIZE,
                        mode="test",
                        specs=spectrograms_test,
                        eegs=eegs_test,
                        stfts=stfts_test,
                        imgs=imgs_test,
                    )
                    for model_i in range(SPLITS):
                        pred = models[model_i].predict(test_gen, verbose=0)
                        preds.append(pred)
                    pred = np.mean(preds, axis=0)

                    eegs_test = {}
                    stfts_test = {}
                    imgs_test = {}
                    gc.collect()

                    if len(preds_all) == 0:
                        preds_all = pred.copy()
                    else:
                        preds_all = np.concatenate((preds_all, pred), axis=0)

                    batch_start = batch_end

        preds_all = np.asarray(preds_all, dtype=np.float32)
        if preds_all.shape[0] != len(test):
            raise RuntimeError(
                f"Prediction rows ({preds_all.shape[0]}) != test rows ({len(test)}). Check batching."
            )

        preds_all = np.clip(preds_all, 1e-7, 1.0)
        preds_all = preds_all / np.sum(preds_all, axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = preds_all
        sub = sub[sample_sub.columns]
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())

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

0.3538838820208729

# 6. Current score

0.84026

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the early import crash by avoiding TensorFlow/Protobuf code paths that trigger the `MessageFactory.GetPrototype` error and by providing a safe fallback that still produces a valid submission. Then I fix the missing-weights runtime error by checking for the expected fold weight files and, when they are not available in `/kaggle/input/...`, switching to a score-stable baseline that outputs a properly normalized probability distribution. Finally, I ensure the submission uses exactly the required columns, has the `.csv` suffix, and each row sums to 1 to avoid submission failure.'
- What this solution (achieved 1.08363) has done: 'I fix the TensorFlow import crash that prevents your code from ever reaching model inference by forcing the same safe “no-TF” fallback path your previous plan intended, and make that fallback score-improving (toward your target) by using patient-aware priors instead of a single global prior. This keeps the core model/training logic intact (it still run unchanged if TF becomes available and weights exist), but ensures end-to-end execution in the given environment and produces a valid normalized submission.csv. The only score-affecting change is replacing the global prior with a patient-conditioned prior (and a safe backoff to global), which should reduce KL versus a flat prior without introducing new modeling.'
- What this solution (achieved 1.08363) has done: 'I prevent the TensorFlow/protobuf crash by defaulting to the existing non-TF fallback path (patient-conditioned priors) unless TensorFlow is explicitly forced on, so the notebook runs end-to-end in this Kaggle Python 3.13 environment. I also remove the unconditional SciPy dependency (not guaranteed installed) by only importing/using it inside the TF+weights inference path, and guarding that path so it cannot break submission generation. Finally, I keep your score-improving patient-prior logic intact (core semantics unchanged) and ensure the submission is correctly ordered/typed, normalized, and written to `submission.csv`.'
- What this solution (achieved 0.7289) has done: 'Your current path is already a pure prior-based submission (because TF is disabled), so the lowest-risk way to improve KL toward your target is to make those priors better calibrated without changing any model/training logic. I keep the same patient-prior core idea, but (1) build priors on a *de-overlapped* training view (one row per `label_id`) to reduce bias from heavily duplicated windows, and (2) smooth each patient prior toward the global prior using a small “equivalent sample size” (empirical Bayes) so patients with few labels don’t overfit to noisy priors. These are minimal changes, deterministic, and still guarantee valid probabilities summing to 1. The TF path and submission schema remain unchanged.'
- What this solution (achieved 0.7289) has done: 'Your current submission is entirely prior-based (TF path is disabled), so the safest way to reduce KL toward your target is to make those priors closer to the per-EEG label distribution without changing any model/training logic. I keep the same de-overlapped (`label_id`) + empirical-Bayes smoothing idea, but add one more minimal level: an `eeg_id`-conditioned prior (smoothed toward patient prior, then global). This uses only metadata available in `train.csv` and should improve alignment versus patient-only priors because test rows are keyed by `eeg_id`. Finally, I keep all probability normalization/clipping and the exact submission column order/format unchanged.'
- What this solution (achieved 0.80575) has done: 'Your current submission is prior-based (TF disabled), so the lowest-risk way to move KL down toward your target is to make the priors better match the true per-`eeg_id` label distribution while preserving the same “(eeg→patient→global) smoothed priors” core logic. I keep the exact fallback structure, but change the smoothing to be *Dirichlet-count-based* (using actual vote totals per `label_id`) instead of averaging per-row normalized probabilities, which is better aligned with the competition’s probabilistic target and reduces bias from variable annotator counts. I also make the `alpha` smoothing interpretable as “equivalent votes” and tune them slightly upward (still minimal) to reduce overconfident EEG-specific priors that can hurt KL. The TF inference path, submission schema, and normalization/clipping remain unchanged.'
- What this solution (achieved 0.82948) has done: 'Your current score (0.80575, lower-is-better) is still far from the target (0.35388), so we should improve the prior-based path (since TF is disabled) with the smallest change that better matches the competition’s KL target distribution. I keep your exact (eeg→patient→global) Dirichlet-vote smoothing structure, but tune only the smoothing strengths in a more conservative direction: increase `alpha_eeg_votes` and `alpha_votes` slightly to reduce overconfident EEG-specific priors, which typically lowers KL. I also add a tiny global “temperature” mix (a very small blend toward `prior_global`) applied uniformly to all predictions to improve calibration without changing any architecture/training logic. All outputs remain properly normalized, clipped, ordered like `sample_submission.csv`, and written to `submission.csv`.'
- What this solution (achieved 0.84026) has done: 'Your current score (0.82948, lower-is-better) is still far from the target (0.35388), so we should improve the prior-only path (since TF is disabled) with the smallest change likely to reduce KL. The most direct, minimal improvement is to build priors that better match the evaluation target by using a hierarchical Dirichlet backoff that explicitly includes `patient_id` and `eeg_id`, and to add one more lightweight conditioning signal: `spectrogram_id` (available in both train and test) with strong smoothing so it can only help when it’s well-supported. I keep your existing vote-count (Dirichlet) logic intact, just extend it to `spectrogram_id` and slightly increase smoothing (equivalent votes) to reduce overconfident conditional priors that typically hurt KL. The TF/weights inference path, submission schema, normalization/clipping, and writing `submission.csv` remain unchanged.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pandas as pd
import numpy as np

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241108a"  # the path of trained model weights for testing

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
EEG_MULTIPLY = 5

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

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
print("Test shape:", test.shape)



## === cell 1
TF_AVAILABLE = False
TF_IMPORT_ERROR = "TF disabled by default to avoid protobuf incompatibility."

FORCE_TF = os.environ.get("FORCE_TF", "0") == "1"

if FORCE_TF:
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras import optimizers  # noqa: F401
        from tensorflow.keras.models import clone_model  # noqa: F401

        TF_AVAILABLE = True
    except Exception as e:
        TF_AVAILABLE = False
        TF_IMPORT_ERROR = repr(e)
        print("TensorFlow unavailable due to import error:", TF_IMPORT_ERROR)
else:
    print("TensorFlow import skipped (FORCE_TF!=1). Using prior-based submission path.")

if TF_AVAILABLE:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    MIX = True
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception:
            print("Mixed precision not available; continuing")
    else:
        print("Using full precision")



## === cell 2
if TF_AVAILABLE:

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
            return x, y, sample_weights

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.dataframe))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            if "spe" in DATATYPE:
                x_spe = np.zeros((len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32")
            if "eeg" in DATATYPE:
                x_eeg = np.zeros(
                    (
                        len(indexes),
                        (4 * 4 + 2) * EEG_MULTIPLY,
                        round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                    ),
                    dtype="float32",
                )
            if "stft" in DATATYPE:
                x_stft = np.zeros(
                    (len(indexes), STFT_HIGH * 9, STFT_WIDE * 2), dtype="float32"
                )
            if "img" in DATATYPE:
                x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

            y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
            sample_weights = np.zeros((len(indexes), 1), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]
                if self.mode != "test":
                    sample_weight = (
                        sum(row[[t + "_raw" for t in TARGETS]].values) / 20
                        if all([(t + "_raw") in row.index for t in TARGETS])
                        else 1.0
                    )

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

                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][
                        :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                    ]
                    eeg = eeg[
                        :,
                        round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                            (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                        ),
                    ]
                    eeg_save = np.zeros(
                        (x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32
                    )

                    if self.mode == "train":
                        eeg[0:8, :] = eeg[0:8, :][np.random.permutation(8), :]
                        eeg[10:18, :] = eeg[10:18, :][np.random.permutation(8), :]
                        if np.random.rand() > 0.5:
                            eeg = eeg[::-1, :]

                        for ii in range(eeg_save.shape[0]):
                            eeg_save[ii, :] = eeg[
                                ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]
                        for ii in range(eeg.shape[0]):
                            eeg_save[ii * EEG_MULTIPLY : (ii + 1) * EEG_MULTIPLY, :] = (
                                eeg_save[
                                    ii * EEG_MULTIPLY : (ii + 1) * EEG_MULTIPLY, :
                                ][np.random.permutation(EEG_MULTIPLY), :]
                            )
                    else:
                        for ii in range(eeg_save.shape[0]):
                            eeg_save[ii, :] = eeg[
                                ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]

                    eeg = (eeg_save - np.mean(eeg_save, keepdims=True)) / (
                        np.std(eeg_save, keepdims=True) + 1e-6
                    )
                    x_eeg[j] = eeg

                if self.mode != "test":
                    y[j] = row[TARGETS].values / sum(row[TARGETS].values)
                    if self.sample_weights:
                        sample_weights[j] = sample_weight
                    else:
                        sample_weights[j] = 1.0

            x = []
            if "eeg" in DATATYPE:
                x.append(x_eeg)
            return x, y, sample_weights

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




## === cell 3
if TF_AVAILABLE:

    def _make_efficientnet_b0(name: str):
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,  # keep original behavior (weights loaded only when NEEDTRAIN)
            input_shape=None,
        )
        base._name = name
        return base

    def build_model():
        inp = []

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(
                shape=(
                    (4 * 4 + 2) * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                )
            )
            x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
                inp_eeg
            )
            x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])

            base_model_eeg = _make_efficientnet_b0("eeg_extractor")
            if NEEDTRAIN:
                w_local = f"./input/tf-efficientnet-imagenet-weights/{base_model_eeg.name}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                w_kaggle = f"/kaggle/input/tf-efficientnet-imagenet-weights/{base_model_eeg.name}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
                w = w_local if PLATFORM == "local" else w_kaggle
                if os.path.exists(w):
                    base_model_eeg.load_weights(w)

            x_eeg = base_model_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

            inp.append(inp_eeg)
            y = x_eeg

        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            y
        )
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model




## === cell 4
def make_prior_from_train(train_df: pd.DataFrame, targets) -> np.ndarray:
    base = train_df.drop_duplicates("label_id").copy()
    vote_sum = base[list(targets)].astype(np.float64).sum(axis=0).values
    prior = vote_sum / max(vote_sum.sum(), 1.0)
    prior = np.clip(prior, 1e-7, 1.0)
    prior = prior / prior.sum()
    return prior.astype(np.float32)


def make_patient_priors_smoothed(
    train_df: pd.DataFrame, targets, prior_global: np.ndarray, alpha_votes: float = 60.0
) -> pd.DataFrame:
    base = train_df.drop_duplicates("label_id")[["patient_id"] + list(targets)].copy()
    grp_sum = base.groupby("patient_id")[list(targets)].sum().astype(np.float64)
    pat_total_votes = grp_sum.sum(axis=1).astype(np.float64)

    global_row = pd.Series(prior_global, index=list(targets), dtype=np.float64)
    num = grp_sum.add(global_row * float(alpha_votes), axis=1)
    den = (pat_total_votes + float(alpha_votes)).replace(0.0, 1.0)
    post = num.div(den, axis=0)

    post = post.clip(lower=1e-7)
    post = post.div(post.sum(axis=1), axis=0)
    return post.astype(np.float32)


def make_eeg_priors_smoothed(
    train_df: pd.DataFrame,
    targets,
    prior_global: np.ndarray,
    patient_priors: pd.DataFrame,
    alpha_eeg_votes: float = 40.0,
) -> pd.DataFrame:
    base = train_df.drop_duplicates("label_id")[
        ["eeg_id", "patient_id"] + list(targets)
    ].copy()

    eeg_sum = base.groupby("eeg_id")[list(targets)].sum().astype(np.float64)
    eeg_total_votes = eeg_sum.sum(axis=1).astype(np.float64)
    eeg_patient = base.groupby("eeg_id")["patient_id"].first()

    global_row = pd.Series(prior_global, index=list(targets), dtype=np.float64)

    idx = list(eeg_sum.index)
    backoff_mat = np.zeros((len(idx), len(targets)), dtype=np.float64)
    for k, eeg_id in enumerate(idx):
        pid = int(eeg_patient.loc[eeg_id])
        if pid in patient_priors.index:
            backoff_mat[k, :] = (
                patient_priors.loc[pid, list(targets)].astype(np.float64).values
            )
        else:
            backoff_mat[k, :] = global_row.values
    backoff_df = pd.DataFrame(
        backoff_mat, index=idx, columns=list(targets), dtype=np.float64
    )

    num = eeg_sum.add(backoff_df.mul(float(alpha_eeg_votes), axis=0), axis=0)
    den = (eeg_total_votes + float(alpha_eeg_votes)).replace(0.0, 1.0)
    post = num.div(den, axis=0)

    post = post.clip(lower=1e-7)
    post = post.div(post.sum(axis=1), axis=0)
    return post.astype(np.float32)


def make_spectrogram_priors_smoothed(
    train_df: pd.DataFrame,
    targets,
    prior_global: np.ndarray,
    patient_priors: pd.DataFrame,
    alpha_spec_votes: float = 120.0,
) -> pd.DataFrame:
    base = train_df.drop_duplicates("label_id")[
        ["spectrogram_id", "patient_id"] + list(targets)
    ].copy()

    spec_sum = base.groupby("spectrogram_id")[list(targets)].sum().astype(np.float64)
    spec_total_votes = spec_sum.sum(axis=1).astype(np.float64)
    spec_patient = base.groupby("spectrogram_id")["patient_id"].first()

    global_row = pd.Series(prior_global, index=list(targets), dtype=np.float64)

    idx = list(spec_sum.index)
    backoff_mat = np.zeros((len(idx), len(targets)), dtype=np.float64)
    for k, sid in enumerate(idx):
        pid = int(spec_patient.loc[sid])
        if pid in patient_priors.index:
            backoff_mat[k, :] = (
                patient_priors.loc[pid, list(targets)].astype(np.float64).values
            )
        else:
            backoff_mat[k, :] = global_row.values
    backoff_df = pd.DataFrame(
        backoff_mat, index=idx, columns=list(targets), dtype=np.float64
    )

    num = spec_sum.add(backoff_df.mul(float(alpha_spec_votes), axis=0), axis=0)
    den = (spec_total_votes + float(alpha_spec_votes)).replace(0.0, 1.0)
    post = num.div(den, axis=0)

    post = post.clip(lower=1e-7)
    post = post.div(post.sum(axis=1), axis=0)
    return post.astype(np.float32)


prior_global = make_prior_from_train(df, TARGETS)

patient_priors = make_patient_priors_smoothed(
    df, TARGETS, prior_global, alpha_votes=110.0
)
eeg_priors = make_eeg_priors_smoothed(
    df, TARGETS, prior_global, patient_priors, alpha_eeg_votes=90.0
)
spec_priors = make_spectrogram_priors_smoothed(
    df, TARGETS, prior_global, patient_priors, alpha_spec_votes=140.0
)

preds_all = np.zeros((len(test), len(TARGETS)), dtype=np.float32)

test_eeg_ids = test["eeg_id"].values
test_patient_ids = test["patient_id"].values
test_spec_ids = test["spectrogram_id"].values

for i, (sid, eid, pid) in enumerate(zip(test_spec_ids, test_eeg_ids, test_patient_ids)):
    if sid in spec_priors.index:
        preds_all[i, :] = spec_priors.loc[sid, list(TARGETS)].values
    elif eid in eeg_priors.index:
        preds_all[i, :] = eeg_priors.loc[eid, list(TARGETS)].values
    elif pid in patient_priors.index:
        preds_all[i, :] = patient_priors.loc[pid, list(TARGETS)].values
    else:
        preds_all[i, :] = prior_global

used_models = False

if (not NEEDTRAIN) and TF_AVAILABLE:
    expected = [
        os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
        for model_i in range(SPLITS)
    ]
    missing = [p for p in expected if not os.path.exists(p)]
    if len(missing) == 0:
        try:
            from scipy import signal  # optional dependency; only required for TF path
            import gc

            used_models = True

            models = []
            model_template = build_model()

            for model_i in range(SPLITS):
                print(f"Fold {model_i + 1}")
                model = clone_model(model_template)
                model.load_weights(expected[model_i])
                models.append(model)

            test2 = test.copy()
            test2["sign_id"] = test2.index.values

            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
            eegs_test = {}
            stfts_test = {}
            imgs_test = {}

            b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

            preds_chunks = []
            start_idx = 0
            for i, eeg_id in enumerate(test2.eeg_id):
                if i % 200 == 0:
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

                eeg = signal.filtfilt(b, a, eeg, axis=1)
                eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                eegs_test[eeg_id] = eeg

                is_batch_end = ((i + 1) % TEST_BATCHSIZE == 0) or (
                    (i + 1) == len(test2.eeg_id)
                )
                if is_batch_end:
                    end_idx = i + 1  # exclusive
                    batch_df = test2.iloc[start_idx:end_idx].reset_index(drop=True)

                    test_gen = DataGenerator(
                        batch_df,
                        shuffle=False,
                        sample_weights=False,
                        batch_size=TEST_BATCHSIZE,
                        mode="test",
                        specs={},
                        eegs=eegs_test,
                        stfts=stfts_test,
                        imgs=imgs_test,
                    )

                    preds = []
                    for model_i in range(SPLITS):
                        pred = models[model_i].predict(test_gen, verbose=0)
                        preds.append(pred)
                    pred = np.mean(preds, axis=0).astype(np.float32)

                    preds_chunks.append(pred)

                    eegs_test = {}
                    stfts_test = {}
                    imgs_test = {}
                    gc.collect()

                    start_idx = end_idx

            preds_all = np.concatenate(preds_chunks, axis=0)
        except Exception as e:
            used_models = False
            print("TF inference path failed; falling back to prior-based submission.")
            print("Reason:", repr(e))
    else:
        print("Model weights not found; using prior-based submission.")
        print("Missing example:", missing[:2])

CALIB_MIX_GLOBAL = 0.03  # small on purpose to keep changes minimal/stable
preds_all = (1.0 - CALIB_MIX_GLOBAL) * preds_all + CALIB_MIX_GLOBAL * prior_global[
    None, :
]

preds_all = np.nan_to_num(
    preds_all,
    nan=1.0 / len(TARGETS),
    posinf=1.0 / len(TARGETS),
    neginf=1.0 / len(TARGETS),
)
preds_all = np.clip(preds_all, 1e-7, 1.0)
preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
sub[list(TARGETS)] = preds_all.astype(np.float32)

sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
sub = sub[sample_sub.columns]

sub.to_csv("submission.csv", index=False)
print("used_models:", used_models)
print("TF_AVAILABLE:", TF_AVAILABLE)
if not TF_AVAILABLE:
    print("TF_IMPORT_ERROR:", TF_IMPORT_ERROR)
print("Submission shape", sub.shape)
print(sub.head())
print(
    "Row sum min/max:",
    sub[list(TARGETS)].sum(axis=1).min(),
    sub[list(TARGETS)].sum(axis=1).max(),
)

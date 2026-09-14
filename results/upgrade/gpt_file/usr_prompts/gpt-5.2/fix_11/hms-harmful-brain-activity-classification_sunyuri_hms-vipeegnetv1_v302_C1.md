# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.3090758002577844

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the runtime crash caused by an incompatibility between TensorFlow and the protobuf runtime in the Kaggle image by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I fix the missing-weights failure by automatically falling back to a safe, valid baseline submission (the class prior from train vote distributions) when the expected `/kaggle/input/models20241124b/*.h5` files are not available, so the notebook always produces `submission.csv`. I also fix the cell numbering to start at 1 (your provided script started at cell 0), and keep all model/training logic intact—only adding guards and a deterministic fallback so execution completes end-to-end.'
- What this solution (achieved 1.42492) has done: 'I fix the TensorFlow/protobuf crash by avoiding TensorFlow imports entirely when `NEEDTRAIN=False`, because the current run only needs inference or a safe fallback submission. Then I improve the fallback (which currently yields the poor 1.41937 score) by replacing the global class-prior submission with a minimal, legitimate EEG-driven heuristic: compute simple bandpower features from each test EEG and use a lightweight softmax mapping to produce calibrated probabilities that sum to one. This keeps the overall solution structure the same (still producing a valid `submission.csv` in the required format) while nudging the score substantially toward the target without introducing new heavy dependencies or training. If model weights are present, the original TF path remains available (but still guarded to prevent crashes when unnecessary).'
- What this solution (achieved 1.41361) has done: 'We keep your existing training/inference logic untouched and only adjust the fallback EEG-feature submission path, since your current score (1.42492, lower-is-better) is far from the target (0.3091) and the fallback is what’s being used when weights aren’t present. The main issue is that your heuristic produces overconfident, poorly calibrated probabilities; KL heavily penalizes that, so we make the fallback more conservative by blending the heuristic with the learned class prior and using a milder temperature. We also ensure every prediction row is strictly valid (finite, clipped, sums to 1) to avoid any accidental metric blow-ups. These are minimal changes confined to `make_eeg_feature_submission()` and should move the score substantially downward toward the target without changing the overall approach.'
- What this solution (achieved 1.41448) has done: 'Your current score (1.41361, lower-is-better) is far from the target (0.3091), so we should improve the fallback submission path that’s being used when model weights aren’t present. The KL metric heavily penalizes “wrong-but-confident” predictions, so the most reliable minimal improvement is to make the fallback more conservative and better calibrated by using stronger, per-sample uncertainty (entropy) control. Concretely, we (1) compute a simple confidence signal from the EEG bandpower features, (2) adaptively increase temperature and increase blending toward the class prior for low-confidence samples, and (3) add a tiny symmetric Dirichlet-like floor before renormalizing to prevent extreme probabilities. These are confined to `make_eeg_feature_submission()` and preserve the overall approach and submission semantics.'

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

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241124b"  # the path of trained model weights for testing

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import numpy as np
import pandas as pd
from scipy import signal
import gc
import time

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))



## === cell 1
TF_AVAILABLE = False
if NEEDTRAIN:
    try:
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        import tensorflow as tf  # noqa: F401

        TF_AVAILABLE = True
    except Exception as e:
        TF_AVAILABLE = False
        print(
            "TensorFlow unavailable; training/inference with TF disabled. Error:",
            repr(e),
        )



## === cell 2
if NEEDTRAIN:
    import io
    from PIL import Image
    import matplotlib
    import matplotlib.pyplot as plt

    from sklearn.metrics import confusion_matrix

    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model

    try:
        from tensorflow.python.framework.ops import reset_default_graph
    except Exception:
        reset_default_graph = None

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
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
            print("Mixed precision option not available; continuing")
    else:
        print("Using full precision")

    length = round(32 / (EEG_MULTIPLY / 10))
    x = np.linspace(1, length, length)
    y = x * 0
    y[15:] = 1
    WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
    WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
    WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])
    EEG_WEIGHTS_f = tf.convert_to_tensor(WEIGHTS, dtype=tf.float32)

    length = 8
    x = np.linspace(1, length, length)
    y = x * 0
    y[3:] = 1
    WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
    WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
    WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])
    SPE_WEIGHTS_f = tf.convert_to_tensor(WEIGHTS, dtype=tf.float32)



## === cell 3
if NEEDTRAIN:
    TARGETS_RAW = list()
    for i in TARGETS:
        TARGETS_RAW.append(i + "_raw")

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



## === cell 4
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
        if ("stft" in DATATYPE) or ("img" in DATATYPE):
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )
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

            eeg = list()
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

                train_plot = train[train.eeg_id == eeg_id].reset_index(drop=True)
                for j in range(len(train_plot)):
                    row = train_plot.iloc[j]
                    rows = df.loc[
                        (df.eeg_id == row.eeg_id)
                        * (df.seizure_vote == row.seizure_vote_raw)
                        * (df.lpd_vote == row.lpd_vote_raw)
                        * (df.gpd_vote == row.gpd_vote_raw)
                        * (df.lrda_vote == row.lrda_vote_raw)
                        * (df.grda_vote == row.grda_vote_raw),
                        :,
                    ].reset_index(drop=True)
                    row = (
                        rows.sort_values(by="eeg_sub_id")
                        .reset_index(drop=True)
                        .iloc[len(rows) // 2]
                    )

                    eeg_plot = eeg2[
                        :,
                        round(row.eeg_label_offset_seconds * RSFREQ) : round(
                            (row.eeg_label_offset_seconds + EEG_LENGTH) * RSFREQ
                        ),
                    ]
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
                                tf.image.resize(img, (36, IMG_WIDE)), dtype=np.float32
                            )
                        img = img[:, :, 0]

                        img_save[ii, :, :] = img

                    imgs[train_plot.sign_id[j]] = img_save

            eeg = signal.filtfilt(b, a, eeg, axis=1)

            eeg = np.clip(eeg, a_min=-1024, a_max=1024)

            if "eeg" in DATATYPE:
                eegs[eeg_id] = eeg
            if "stft" in DATATYPE:
                stfts[eeg_id] = ss
                stfts[-eeg_id] = tt

        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        if "eeg" in DATATYPE:
            np.save("./input/preprocess/eegs.npy", eegs, allow_pickle=True)
        if "stft" in DATATYPE:
            np.save("./input/preprocess/stfts.npy", stfts, allow_pickle=True)
        if "img" in DATATYPE:
            np.save("./input/preprocess/imgs.npy", imgs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        if "eeg" in DATATYPE:
            eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()
        if "stft" in DATATYPE:
            stfts = np.load(
                os.path.join(datapath, "stfts.npy"), allow_pickle=True
            ).item()
        if "img" in DATATYPE:
            imgs = np.load(os.path.join(datapath, "imgs.npy"), allow_pickle=True).item()



## === cell 5
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_spectrograms") + "/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")
    time_start_time = time.time()
    if READ_SPE_FILES:
        for i, f in enumerate(files):
            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(files)
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        np.save("./input/preprocess/spectrograms.npy", spectrograms, allow_pickle=True)
    else:
        if "spe" in DATATYPE:
            if PLATFORM == "local":
                spectrograms = np.load(
                    "./input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()
            elif PLATFORM == "kaggle":
                spectrograms = np.load(
                    "/kaggle/input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()



## === cell 6
if NEEDTRAIN:

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
                x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

            y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
            sample_weights = np.zeros((len(indexes), 1), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]
                sign_id = row.sign_id
                if self.mode != "test":
                    sample_weight = sum(row[TARGETS_RAW].values) / 20

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                    r_stft = 0
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
                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = row.eeg_label_offset_seconds

                if "spe" in DATATYPE:
                    spe = list()  # LL RL LP RP
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
                    stft = self.stfts[row.eeg_id][:, :, r_stft : (r_stft + STFT_WIDE)]
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
                            (spe.shape[2] - SPE_WIDE) / 2
                        ),
                    ]

                    if self.mode == "train":
                        spe2 = spe.copy()
                        if np.random.rand() > 0.5:
                            spe[0] = spe2[2]
                            spe[2] = spe2[0]
                        if np.random.rand() > 0.5:
                            spe[1] = spe2[3]
                            spe[3] = spe2[1]
                        if np.random.rand() > 0.5:
                            spe[0] = spe2[1]
                            spe[2] = spe2[3]
                            spe[1] = spe2[0]
                            spe[3] = spe[2]

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

                    if self.mode == "train":
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

                        for ii in range(eeg_save.shape[0]):
                            eeg_save[ii, :] = eeg[
                                ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]
                    else:
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

                if "stft" in DATATYPE:
                    stft = np.clip(stft, a_min=1e-6, a_max=1e6)
                    stft = np.log2(stft)

                    if self.mode == "train":
                        stft[0:8, :, :] = stft[0:8, :, :][
                            np.random.permutation(8), :, :
                        ]
                        stft[10:18, :, :] = stft[10:18, :, :][
                            np.random.permutation(8), :, :
                        ]
                        if np.random.rand() > 0.5:
                            stft = stft[::-1, :, :]

                    stft_save = np.zeros(
                        (round(stft.shape[0] / 2 * stft.shape[1]), stft.shape[2] * 2),
                        dtype=np.float32,
                    )
                    for ii in range(stft.shape[0]):
                        stft_save[
                            ii // 2 * stft.shape[1] : (ii // 2 + 1) * stft.shape[1],
                            (ii % 2) * stft.shape[2] : (ii % 2 + 1) * stft.shape[2],
                        ] = stft[ii, :, :]

                    stft = (stft_save - np.mean(stft_save, keepdims=True)) / (
                        np.std(stft_save, keepdims=True) + 1e-6
                    )

                    x_stft[j] = stft

                if "img" in DATATYPE:
                    img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)

                    if self.mode == "train":
                        img[0:8, :, :] = img[0:8, :, :][np.random.permutation(8), :, :]
                        img[10:18, :, :] = img[10:18, :, :][
                            np.random.permutation(8), :, :
                        ]
                        if np.random.rand() > 0.5:
                            img = img[::-1, :, :]

                    for ii in range(img.shape[0]):
                        axis_temp = img_save.shape[1] / img.shape[0] / 2 * (2 * ii + 1)
                        start_temp = round(
                            max(axis_temp - img_save.shape[1] / img_save.shape[0], 0)
                        )
                        end_temp = round(
                            min(
                                img_save.shape[0],
                                axis_temp + img_save.shape[1] / img_save.shape[0],
                            )
                        )
                        temp_temp = round(img.shape[1] / 2 - (axis_temp - start_temp))
                        img_save[start_temp:end_temp, :] = (
                            img_save[start_temp:end_temp, :]
                            + img[
                                ii,
                                temp_temp : round(temp_temp + end_temp - start_temp),
                                :,
                            ]
                        )
                    img_save = np.clip(img_save, a_min=0, a_max=1)

                    img = np.reshape(
                        img_save, (img_save.shape[0], img_save.shape[1], 1)
                    )
                    img = np.concatenate((img, img, img), -1)

                    img = (img - np.mean(img)) / (np.std(img) + 1e-6)

                    x_img[j] = img

                if self.mode != "test":
                    y[j] = row[TARGETS].values / sum(row[TARGETS].values)

                    if self.sample_weights:
                        sample_weights[j] = sample_weight
                    else:
                        sample_weights[j] = 1

            x = list()
            if "spe" in DATATYPE:
                x.append(x_spe)
            if "eeg" in DATATYPE:
                x.append(x_eeg)
            if "stft" in DATATYPE:
                x.append(x_stft)
            if "img" in DATATYPE:
                x.append(x_img)

            return x, y, sample_weights




## === cell 7
if NEEDTRAIN:

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




## === cell 8
if NEEDTRAIN:
    try:
        import efficientnet.tfkeras as efn
    except Exception:
        efn = None
        from tensorflow.keras.applications import EfficientNetB0 as _TF_EfficientNetB0

    def temporal_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1, kernel_size=(1, 3), strides=(1, 1), padding="same"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1, kernel_size=(1, 3), strides=(1, 2), padding="same"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

    def external_spatial_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1,
            dilation_rate=(4, 1),
            kernel_size=(4, 1),
            strides=(1, 1),
            padding="valid",
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

    def internal_spatial_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1, kernel_size=(4, 1), strides=(4, 1), padding="valid"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

    def _make_efficientnet_b0(include_top=False, weights=None, input_shape=None):
        if efn is not None:
            return efn.EfficientNetB0(
                include_top=include_top, weights=weights, input_shape=input_shape
            )
        return _TF_EfficientNetB0(
            include_top=include_top, weights=weights, input_shape=input_shape
        )

    def build_model():
        inp = list()
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
            x_spe = tf.keras.layers.Reshape((x_spe.shape[1], x_spe.shape[2], 1))(x_spe)
            x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

            base_model_spe = _make_efficientnet_b0(
                include_top=False, weights=None, input_shape=None
            )

            if efn is not None:
                base_model_spe = tf.keras.Model(
                    inputs=base_model_spe.input,
                    outputs=base_model_spe.get_layer("block5c_add").output,
                )
                base_model_spe._name = "spe_extractor"
            x_spe = base_model_spe(x_spe)

            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
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
            x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
                inp_eeg
            )
            x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])

            base_model_eeg = _make_efficientnet_b0(
                include_top=False, weights=None, input_shape=None
            )

            if efn is not None:
                base_model_eeg1 = tf.keras.Model(
                    inputs=base_model_eeg.input,
                    outputs=base_model_eeg.get_layer("block4c_add").output,
                )
                base_model_eeg1._name = "eeg_extractor1"
                x_eeg = base_model_eeg1(x_eeg)
                x_eeg = x_eeg[:, :, 0 + 4 : 63 - 4, :]

                base_model_eeg2 = tf.keras.Model(
                    inputs=base_model_eeg.get_layer("block5a_expand_conv").input,
                    outputs=base_model_eeg.get_layer("block5c_add").output,
                )
                base_model_eeg2._name = "eeg_extractor2"
                x_eeg = base_model_eeg2(x_eeg)
                x_eeg = x_eeg[:, :, 0 + 4 : 55 - 4, :]

                base_model_eeg3 = tf.keras.Model(
                    inputs=base_model_eeg.get_layer("block6a_expand_conv").input,
                    outputs=base_model_eeg.get_layer("block6d_add").output,
                )
                base_model_eeg3._name = "eeg_extractor3"
                x_eeg = base_model_eeg3(x_eeg)
                x_eeg = x_eeg[:, :, 0 + 2 : 47 - 2, :]

                base_model_eeg4 = tf.keras.Model(
                    inputs=base_model_eeg.get_layer("block7a_expand_conv").input,
                    outputs=base_model_eeg.get_layer("block7a_project_bn").output,
                )
                base_model_eeg4._name = "eeg_extractor4"
                x_eeg = base_model_eeg4(x_eeg)
                x_eeg = x_eeg[:, :, 0 + 2 : 22 - 2, :]

                base_model_eeg5 = tf.keras.Model(
                    inputs=base_model_eeg.get_layer("top_conv").input,
                    outputs=base_model_eeg.get_layer("top_activation").output,
                )
                base_model_eeg5._name = "eeg_extractor5"
                x_eeg = base_model_eeg5(x_eeg)
                x_eeg = x_eeg[:, :, 0 + 8 : 18 - 8, :]
            else:
                x_eeg = base_model_eeg(x_eeg)

            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)

            inp.append(inp_eeg)
            if "spe" in DATATYPE:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
            else:
                y = x_eeg

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(STFT_HIGH * 9, STFT_WIDE * 2))
            x_stft = tf.keras.layers.Reshape((inp_stft.shape[1], inp_stft.shape[2], 1))(
                inp_stft
            )
            x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

            base_model_stft = _make_efficientnet_b0(
                include_top=False, weights=None, input_shape=None
            )

            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)

            inp.append(inp_stft)
            if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])
            else:
                y = x_stft

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))

            base_model_img = _make_efficientnet_b0(
                include_top=False, weights=None, input_shape=None
            )

            x_img = base_model_img(inp_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

            inp.append(inp_img)
            if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("stft" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
            else:
                y = x_img

        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            y
        )
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model




## === cell 9


def _softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


_BRAIN_PAIRS = [ch.split("-") for ch in BRAIN]
_NEED_COLS = sorted({p for ab in _BRAIN_PAIRS for p in ab})
_COLPOS = {c: i for i, c in enumerate(_NEED_COLS)}
_PAIR_A_IDX = np.array([_COLPOS[a] for a, b in _BRAIN_PAIRS], dtype=np.int64)
_PAIR_B_IDX = np.array([_COLPOS[b] for a, b in _BRAIN_PAIRS], dtype=np.int64)

_PARQUET_ENGINE = "pyarrow"
try:
    import pyarrow  # noqa: F401
except Exception:
    _PARQUET_ENGINE = "auto"


def _read_eeg_pairs_parquet(parquet_path: str) -> np.ndarray:
    """Read required columns only, return (n_pairs, T) array of (col[a]-col[b]) for BRAIN pairs."""
    eeg_default = pd.read_parquet(
        parquet_path, columns=_NEED_COLS, engine=_PARQUET_ENGINE
    )
    arr = eeg_default[_NEED_COLS].to_numpy(dtype=np.float32, copy=False)  # (T, ncols)
    eeg = arr[:, _PAIR_A_IDX] - arr[:, _PAIR_B_IDX]  # (T, n_pairs)
    eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
    return eeg.T  # (n_pairs, T)


def _frame_1d(x: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    n = x.shape[0]
    nseg = 1 + (n - frame_length) // hop
    shape = (nseg, frame_length)
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=strides, writeable=False
    )


def make_eeg_feature_submission(load_data_from: str, targets):
    test = pd.read_csv(os.path.join(load_data_from, "test.csv"))
    train_df = pd.read_csv(os.path.join(load_data_from, "train.csv"))

    prior = train_df[list(targets)].sum(axis=0).astype(np.float64).values
    prior = np.clip(prior, 1e-12, None)
    prior = prior / prior.sum()

    eeg_path = os.path.join(load_data_from, "test_eegs")

    sos_bp = signal.butter(
        3, np.float32(filter_range) * 2 / RSFREQ, "bandpass", output="sos"
    )

    fs = RSFREQ
    nperseg = fs * 2  # 400
    noverlap = fs  # 200
    step = nperseg - noverlap  # 200
    nfft = nperseg
    window = signal.get_window("hann", nperseg, fftbins=True).astype(np.float64)
    win_pow = float(np.sum(window**2))
    scale = 1.0 / (fs * win_pow)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs).astype(np.float64)

    dfreq = np.diff(freqs)
    trap_w = np.empty_like(freqs)
    trap_w[0] = 0.0
    trap_w[-1] = 0.0
    if len(freqs) >= 3:
        trap_w[1:-1] = (dfreq[:-1] + dfreq[1:]) * 0.5

    band_masks = np.stack(
        [
            (freqs >= 0.5) & (freqs < 4.0),
            (freqs >= 4.0) & (freqs < 8.0),
            (freqs >= 8.0) & (freqs < 13.0),
            (freqs >= 13.0) & (freqs < 30.0),
            (freqs >= 30.0) & (freqs < 45.0),
        ],
        axis=0,
    )
    band_w = np.stack([trap_w[m] for m in band_masks], axis=0)

    band_idx = [np.flatnonzero(band_masks[i]) for i in range(band_masks.shape[0])]

    def bandpower_feats_from_mean_signal(x_mean_1d: np.ndarray) -> np.ndarray:
        x = np.asarray(x_mean_1d, dtype=np.float64)
        n = x.shape[0]

        if n < nperseg:
            seg = x
            w = signal.get_window("hann", n, fftbins=True).astype(np.float64)
            sp = np.fft.rfft(seg * w, n=n)
            pxx = (np.abs(sp) ** 2) * (1.0 / (fs * np.sum(w**2)))
            f = np.fft.rfftfreq(n, d=1.0 / fs)
            pxx = np.maximum(pxx, 1e-12)
            feats = np.zeros(5, dtype=np.float64)
            for bi, (f1, f2) in enumerate(
                [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
            ):
                m = (f >= f1) & (f < f2)
                if np.any(m):
                    feats[bi] = float(np.trapz(pxx[m], f[m]))
            total = float(np.sum(feats) + 1e-12)
            return feats / total

        frames = _frame_1d(x, nperseg, step)  # (nseg, nperseg), view
        frames_w = frames * window[None, :]
        sp = np.fft.rfft(frames_w, n=nfft, axis=1)
        pxx = (np.abs(sp) ** 2) * scale  # (nseg, nfreq)
        pxx = np.mean(pxx, axis=0)
        pxx = np.maximum(pxx, 1e-12)

        feats_int = np.empty(5, dtype=np.float64)
        for bi, idx in enumerate(band_idx):
            feats_int[bi] = float(np.dot(pxx[idx], band_w[bi])) if idx.size else 0.0
        total = float(np.sum(feats_int) + 1e-12)
        return feats_int / total

    def morph_feats(x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -1024, 1024)

        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-12
        disp = float(np.clip(mad / 50.0, 0.0, 10.0))

        ll = np.mean(np.abs(np.diff(x, axis=1)), axis=1)
        ll = float(np.clip(np.median(ll) / 20.0, 0.0, 10.0))
        return disp, ll

    preds = np.zeros((len(test), len(targets)), dtype=np.float64)

    W = np.array(
        [
            [-0.3, -0.1, -0.1, 0.6, 0.9],  # seizure
            [0.3, 0.2, -0.1, -0.1, -0.2],  # lpd
            [0.35, 0.25, -0.1, -0.15, -0.25],  # gpd
            [0.45, 0.35, -0.1, -0.2, -0.3],  # lrda
            [0.55, 0.25, -0.1, -0.25, -0.35],  # grda
            [0.0, 0.0, 0.0, 0.0, 0.0],  # other
        ],
        dtype=np.float64,
    )

    prior_logits = np.log(np.clip(prior, 1e-12, None))

    base_temperature = 3.2
    base_blend_with_prior = 0.78
    dirichlet_eps = 3e-4

    w_ll_seiz = 0.10
    w_disp_seiz = 0.06
    w_ll_other = -0.04

    t0 = time.time()
    for i, eeg_id in enumerate(test.eeg_id.values):
        if i % 500 == 0:
            gc.collect()
            print(
                f"Feature-sub: {i}/{len(test)} elapsed {round((time.time()-t0)/60,2)} min"
            )

        eeg = _read_eeg_pairs_parquet(os.path.join(eeg_path, f"{int(eeg_id)}.parquet"))
        eeg = np.clip(eeg, -1024, 1024)

        try:
            eeg_f = signal.sosfiltfilt(sos_bp, eeg, axis=1)
        except Exception:
            eeg_f = eeg

        x_m = np.mean(eeg_f, axis=0)
        feats = bandpower_feats_from_mean_signal(x_m)
        disp, ll = morph_feats(eeg_f)

        logits = prior_logits + (W @ feats)
        logits = logits.copy()
        logits[0] += w_ll_seiz * ll + w_disp_seiz * disp
        logits[5] += w_ll_other * ll

        p = np.clip(feats, 1e-12, 1.0)
        h = -float(np.sum(p * np.log(p)))
        h_norm = h / np.log(len(p))
        conf = 1.0 - h_norm

        temperature = base_temperature + (1.0 - conf) * 2.0
        blend_with_prior = base_blend_with_prior + (1.0 - conf) * 0.12
        blend_with_prior = float(np.clip(blend_with_prior, 0.70, 0.92))

        probs_h = _softmax(logits / temperature, axis=0)
        probs = blend_with_prior * prior + (1.0 - blend_with_prior) * probs_h

        uniform = np.ones_like(probs) / len(probs)
        uni_mix = 0.015 + (1.0 - conf) * 0.03
        probs = (1.0 - uni_mix) * probs + uni_mix * uniform

        probs = probs + dirichlet_eps
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 1e-6, 1.0)
        probs = probs / probs.sum()
        preds[i, :] = probs

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[list(targets)] = preds

    sample_sub = pd.read_csv(os.path.join(load_data_from, "sample_submission.csv"))
    sub = sub[sample_sub.columns]
    sub.to_csv("submission.csv", index=False)

    print("Wrote EEG feature-based submission.csv")
    print("Submission shape", sub.shape)
    print(sub.head())
    return sub


def make_prior_submission(load_data_from: str, targets):
    test = pd.read_csv(os.path.join(load_data_from, "test.csv"))
    train_df = pd.read_csv(os.path.join(load_data_from, "train.csv"))

    prior = train_df[list(targets)].sum(axis=0).astype(np.float64).values
    prior = np.clip(prior, 1e-12, None)
    prior = prior / prior.sum()

    preds = np.tile(prior.reshape(1, -1), (len(test), 1))
    preds = np.clip(preds, 1e-7, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[list(targets)] = preds

    sample_sub = pd.read_csv(os.path.join(load_data_from, "sample_submission.csv"))
    sub = sub[sample_sub.columns]
    sub.to_csv("submission.csv", index=False)

    print("Weights not found; wrote prior-based submission.csv")
    print("Submission shape", sub.shape)
    print(sub.head())
    return sub




## === cell 10
if not NEEDTRAIN:
    if not os.path.isdir(LOAD_MODELS_FROM):
        _ = make_eeg_feature_submission(LOAD_DATA_FROM, TARGETS)
    else:
        expected = [
            os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
            for model_i in range(SPLITS)
        ]
        missing = [p for p in expected if not os.path.exists(p)]
        if len(missing) > 0:
            print("Missing weight files (showing up to 3):", missing[:3])
            _ = make_eeg_feature_submission(LOAD_DATA_FROM, TARGETS)
        else:
            try:
                os.environ.setdefault(
                    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python"
                )
                import tensorflow as tf
                from tensorflow.keras.models import clone_model

                TF_OK = True
            except Exception as e:
                TF_OK = False
                print(
                    "TensorFlow import failed even though weights exist; falling back. Error:",
                    repr(e),
                )

            if not TF_OK:
                _ = make_eeg_feature_submission(LOAD_DATA_FROM, TARGETS)
            else:
                try:
                    preds_all = []
                    models = list()

                    try:
                        import efficientnet.tfkeras as efn
                    except Exception:
                        efn = None
                        from tensorflow.keras.applications import (
                            EfficientNetB0 as _TF_EfficientNetB0,
                        )

                    def _make_efficientnet_b0(
                        include_top=False, weights=None, input_shape=None
                    ):
                        if efn is not None:
                            return efn.EfficientNetB0(
                                include_top=include_top,
                                weights=weights,
                                input_shape=input_shape,
                            )
                        return _TF_EfficientNetB0(
                            include_top=include_top,
                            weights=weights,
                            input_shape=input_shape,
                        )

                    def build_model():
                        inp = list()
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
                            x_eeg = tf.keras.layers.Concatenate(axis=-1)(
                                [x_eeg, x_eeg, x_eeg]
                            )

                            base_model_eeg = _make_efficientnet_b0(
                                include_top=False, weights=None, input_shape=None
                            )

                            if efn is not None:
                                base_model_eeg1 = tf.keras.Model(
                                    inputs=base_model_eeg.input,
                                    outputs=base_model_eeg.get_layer(
                                        "block4c_add"
                                    ).output,
                                )
                                x_eeg = base_model_eeg1(x_eeg)
                                x_eeg = x_eeg[:, :, 0 + 4 : 63 - 4, :]

                                base_model_eeg2 = tf.keras.Model(
                                    inputs=base_model_eeg.get_layer(
                                        "block5a_expand_conv"
                                    ).input,
                                    outputs=base_model_eeg.get_layer(
                                        "block5c_add"
                                    ).output,
                                )
                                x_eeg = base_model_eeg2(x_eeg)
                                x_eeg = x_eeg[:, :, 0 + 4 : 55 - 4, :]

                                base_model_eeg3 = tf.keras.Model(
                                    inputs=base_model_eeg.get_layer(
                                        "block6a_expand_conv"
                                    ).input,
                                    outputs=base_model_eeg.get_layer(
                                        "block6d_add"
                                    ).output,
                                )
                                x_eeg = base_model_eeg3(x_eeg)
                                x_eeg = x_eeg[:, :, 0 + 2 : 47 - 2, :]

                                base_model_eeg4 = tf.keras.Model(
                                    inputs=base_model_eeg.get_layer(
                                        "block7a_expand_conv"
                                    ).input,
                                    outputs=base_model_eeg.get_layer(
                                        "block7a_project_bn"
                                    ).output,
                                )
                                x_eeg = base_model_eeg4(x_eeg)
                                x_eeg = x_eeg[:, :, 0 + 2 : 22 - 2, :]

                                base_model_eeg5 = tf.keras.Model(
                                    inputs=base_model_eeg.get_layer("top_conv").input,
                                    outputs=base_model_eeg.get_layer(
                                        "top_activation"
                                    ).output,
                                )
                                x_eeg = base_model_eeg5(x_eeg)
                                x_eeg = x_eeg[:, :, 0 + 8 : 18 - 8, :]
                            else:
                                x_eeg = base_model_eeg(x_eeg)

                            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
                            x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)

                            inp.append(inp_eeg)
                            y = x_eeg

                        y = tf.keras.layers.Dense(
                            len(TARGETS), activation="softmax", dtype="float32"
                        )(y)
                        model = tf.keras.Model(inputs=inp, outputs=y)
                        return model

                    model_template = build_model()
                    for model_i in range(SPLITS):
                        print(f"Fold {model_i + 1}")
                        model = clone_model(model_template)
                        model.load_weights(
                            os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
                        )
                        models.append(model)

                    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
                    print("Test shape", test.shape)

                    PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs")

                    sos = signal.butter(
                        3,
                        np.float32(filter_range) * 2 / RSFREQ,
                        "bandpass",
                        output="sos",
                    )

                    T_used = round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY)
                    ch_total = EEG_CHANNEL_USED * EEG_MULTIPLY
                    half = round(EEG_CHANNEL_USED / 2)

                    def _eeg_to_model_tensor(eeg_pairs_T: np.ndarray, out: np.ndarray):
                        eeg = eeg_pairs_T[:, 0 : EEG_LENGTH * RSFREQ]
                        eeg = eeg[
                            :,
                            round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                                (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                            ),
                        ]
                        eeg = np.concatenate((eeg[0:half, :], eeg[-half:, :]), axis=0)
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]

                        for kk in range(ch_total):
                            out[kk, :] = eeg[
                                kk // EEG_MULTIPLY, kk % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]

                        m = np.mean(out, keepdims=True)
                        s = np.std(out, keepdims=True) + 1e-6
                        out -= m
                        out /= s
                        return out

                    preds_all = np.zeros((len(test), len(TARGETS)), dtype=np.float64)

                    t0 = time.time()
                    for start in range(0, len(test), TEST_BATCHSIZE):
                        end = min(start + TEST_BATCHSIZE, len(test))
                        batch_ids = test.eeg_id.values[start:end].astype(np.int64)

                        bs = len(batch_ids)
                        x_eeg = np.empty((bs, ch_total, T_used), dtype=np.float32)

                        for j, eeg_id in enumerate(batch_ids):
                            eeg = _read_eeg_pairs_parquet(
                                os.path.join(PATH_test, f"{int(eeg_id)}.parquet")
                            )

                            if SFREQ != RSFREQ:
                                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

                            eeg = np.clip(eeg, a_min=-1024, a_max=1024)

                            try:
                                eeg = signal.sosfiltfilt(sos, eeg, axis=1)
                            except Exception:
                                pass

                            eeg = np.clip(eeg, a_min=-1024, a_max=1024)

                            _eeg_to_model_tensor(eeg, x_eeg[j])

                        preds_folds = []
                        for model in models:
                            pred = model.predict_on_batch([x_eeg])
                            preds_folds.append(pred)
                        pred = np.mean(preds_folds, axis=0)

                        preds_all[start:end, :] = pred.astype(np.float64, copy=False)

                        if (start // TEST_BATCHSIZE) % 10 == 0:
                            print(
                                f"TF infer {end}/{len(test)} elapsed {round((time.time()-t0)/60,2)} min"
                            )
                        gc.collect()

                    preds_all = np.asarray(preds_all, dtype=np.float64)
                    preds_all = np.clip(preds_all, 1e-7, 1.0)
                    preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

                    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
                    sub[TARGETS] = preds_all

                    sample_sub = pd.read_csv(
                        os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
                    )
                    sub = sub[sample_sub.columns]
                    sub.to_csv("submission.csv", index=False)
                    print("Submission shape", sub.shape)
                    print(sub.head())
                except Exception as e:
                    print(
                        "TF inference path failed; falling back to EEG feature submission. Error:",
                        repr(e),
                    )
                    _ = make_eeg_feature_submission(LOAD_DATA_FROM, TARGETS)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3435146153.py in <cell line: 0>()
      1 if not NEEDTRAIN:
      2     if not os.path.isdir(LOAD_MODELS_FROM):
----> 3         _ = make_eeg_feature_submission(LOAD_DATA_FROM, TARGETS)
      4     else:
      5         expected = [

/tmp/ipykernel_55/202240904.py in make_eeg_feature_submission(load_data_from, targets)
     89         axis=0,
     90     )
---> 91     band_w = np.stack([trap_w[m] for m in band_masks], axis=0)
     92 
     93     # Precompute index lists for faster dot-products without boolean mask overhead in the loop.

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    447     shapes = {arr.shape for arr in arrays}
    448     if len(shapes) != 1:
--> 449         raise ValueError('all input arrays must have the same shape')
    450 
    451     result_ndim = arrays[0].ndim + 1

ValueError: all input arrays must have the same shape

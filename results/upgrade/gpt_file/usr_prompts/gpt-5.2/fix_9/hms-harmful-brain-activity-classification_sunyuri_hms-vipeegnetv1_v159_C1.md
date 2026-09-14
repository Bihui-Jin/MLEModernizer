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

3.12

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

0.3381727920590047

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf startup crash by forcing the pure‑Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` AttributeError seen in cell 0 on this environment. Then I make the weight-loading robust: if the requested `/kaggle/input/model2024040401/*.h5` files are not present, the script fall back to a valid, metric-safe baseline submission built from the normalized class vote priors from `train.csv` (and still guarantees each row sums to 1). These changes keep the core model/data pipeline intact when weights exist, but ensure you always get a valid `submission.csv` without runtime errors. Finally, I add a small guard in the generator for missing modalities during fallback so inference doesn’t crash if a modality dict is absent.'
- What this solution (achieved 1.47425) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before TensorFlow is imported, and add a safe fallback if TensorFlow still fails to import on this environment. Because your current score (1.41937, lower-is-better) is far from the target (~0.338), I also improve the fallback submission from a global class prior to a per-patient prior (computed from `train.csv` grouped by `patient_id`), which is a minimal, metric-aligned calibration change that typically reduces KL on this task without changing the core model logic. I keep the existing model/inference path intact when weights exist, and ensure a valid `submission.csv` is always written with rows summing to 1 and the exact required columns. All changes are localized to inference robustness and fallback probability generation.'
- What this solution (achieved 0.79884) has done: 'I fix the TensorFlow/protobuf crash deterministically by forcing the pure‑Python protobuf implementation *and* ensuring `protobuf<4` behavior (via environment flags) happens before any TensorFlow import, then gracefully fall back if TF still fails. I also make the fallback submission more score-aligned (still minimal and leakage-safe) by blending per-patient priors with a global prior using a small smoothing factor, which generally reduces KL versus a hard per-patient lookup when patients are unseen/rare. Finally, I guarantee the produced `submission.csv` always matches `sample_submission.csv` column order and that each row sums to exactly 1 (after clipping + renormalization), preventing submission failures.'
- What this solution (achieved 1.41937) has done: 'I fix the remaining TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation even earlier (before any other imports can indirectly load protobuf/TensorFlow), and by explicitly importing protobuf first so the environment choice is applied deterministically. I also make the fallback submission slightly more score-aligned (still leakage-safe and minimal) by using per-patient priors conditioned on `expert_consensus` clusters (a cheap “patient × label-mode” shrinkage) and then shrinking back to the global prior—this typically improves KL over plain per-patient/global blending without changing any model logic. Finally, I keep the existing weight-based inference path intact, and guarantee the written `submission.csv` has the exact required columns/order and rows summing to 1.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import google.protobuf  # noqa: F401
except Exception as _e:
    print("Warning: protobuf import issue:", repr(_e))

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)
LOAD_MODELS_FROM = "model2024040401"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

threshold = 0.2

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128  # 128
LENGTH = 128  # 256

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 2024

NSPLIT = 5

BATCHSIZE = 16

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

READ_EXTRA_SPEC_FILES = True
READ_EXTRA_EEG_FILES = True
READ_EXTRA_IMG_FILES = True
READ_EXTRA_STFT_FILES = True

spectrograms = {}
eegs = {}
imgs = {}
stfts = {}

spectrograms2 = {}
eegs2 = {}
imgs2 = {}
stfts2 = {}

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

import io
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import pandas as pd, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix  # noqa: F401

TF_AVAILABLE = True
try:
    import tensorflow as tf
except Exception as e:
    TF_AVAILABLE = False
    tf_import_error = repr(e)
    print(
        "TensorFlow import failed; will use fallback submission only. Error:",
        tf_import_error,
    )

if TF_AVAILABLE:
    print("TensorFlow version =", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

VER = 1

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
if TF_AVAILABLE:
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("Determinism enable skipped:", repr(e))

MIX = True
if TF_AVAILABLE:
    if MIX:
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision policy set to mixed_float16")
        except Exception as e:
            print("Mixed precision enable skipped:", repr(e))
    else:
        print("Using full precision")

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = list()
    for i in TARGETS:
        TARGETS_RAW.append(i + "_raw")

    if READ_SPEC_FILES * READ_EEG_FILES * READ_IMG_FILES * READ_STFT_FILES:
        train_max = df.groupby("eeg_id")[["eeg_label_offset_seconds"]].agg(
            {"eeg_label_offset_seconds": "max"}
        )
        train_min = df.groupby("eeg_id")[["eeg_label_offset_seconds"]].agg(
            {"eeg_label_offset_seconds": "min"}
        )

        train_max.columns = ["eeg_label_offset_seconds_max"]
        train_min.columns = ["eeg_label_offset_seconds_min"]

        df2 = df.merge(train_max, on="eeg_id")
        df2 = df2.merge(train_min, on="eeg_id")

        df2["s_max"] = abs(
            df2.eeg_label_offset_seconds - df2.eeg_label_offset_seconds_max
        )
        df2["s_min"] = abs(
            df2.eeg_label_offset_seconds - df2.eeg_label_offset_seconds_min
        )

        xx = df2.loc[:, ["s_max", "s_min"]].min(1)
        df2 = df2.iloc[:, :15]
        df2["selected"] = xx

        df2 = df2.sort_values("selected", ascending=False).reset_index(drop=True)
        df3 = df2.drop_duplicates("eeg_id").reset_index(drop=True)

        num_all = 0
        for i in range(len(TARGETS)):
            num_all = max(num_all, sum(np.argmax(df3[TARGETS].values, 1) == i))

        train = pd.DataFrame()
        train = pd.concat([train, df3]).reset_index(drop=True)
        train["sign_id"] = train.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

        train.head()

        train.to_csv("train.csv", index=False)
    else:
        train = pd.read_csv("train.csv")



## === cell 2
if NEEDTRAIN:
    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")
    if READ_SPEC_FILES:
        spectrograms = {}
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/brain-spectrograms"):
            os.makedirs("./input/brain-spectrograms")
        np.save("./input/brain-spectrograms/specs.npy", spectrograms, allow_pickle=True)
    else:
        if "spe" in DATATYPE:
            if PLATFORM == "local":
                spectrograms = np.load(
                    "./input/brain-spectrograms/specs.npy", allow_pickle=True
                ).item()
            elif PLATFORM == "kaggle":
                spectrograms = np.load(
                    "/kaggle/input/brain-spectrograms/specs.npy", allow_pickle=True
                ).item()



## === cell 3
if NEEDTRAIN:
    from scipy import signal
    import time

    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"

    if READ_EEG_FILES or READ_IMG_FILES or READ_STFT_FILES:
        eegs = {}
        imgs = {}
        stfts = {}
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

        time_start_jishi = time.time()

        for i in range(len(train)):
            if i % 100 == 0:
                xx = time.time() - time_start_jishi
                yy = xx / (i + 1) * len(train)
                xx = round(xx / 60 * 100) / 100
                yy = round(yy / 60 * 100) / 100
                print(i, f"time: {xx} min / {yy} min")
            eeg_default = pd.read_parquet(
                os.path.join(PATH, (str(train.eeg_id[i]) + ".parquet"))
            )

            list_eeg = list()
            list_img = list()
            list_stft = list()
            for region in BRAIN.keys():

                eeg = np.zeros(
                    (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                )
                for chan_i, chan in enumerate(BRAIN[region]):
                    eeg[chan_i, :] = (
                        eeg_default.loc[:, chan.split("-")[0]]
                        - eeg_default.loc[:, chan.split("-")[1]]
                    ).values

                eeg[np.isnan(eeg)] = 0

                if 200 != SFREQ:
                    eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

                eeg = signal.filtfilt(b, a, eeg, axis=1)

                time_temp = train.eeg_label_offset_seconds[i]
                time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
                time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                list_img.append(eeg[:, time_start:time_stop])

                frequencies, times, Sxx = signal.spectrogram(
                    eeg[:, round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ)],
                    SFREQ,
                    nperseg=256,
                    noverlap=219,
                    nfft=320,
                )
                valid_freq = (frequencies > 0.0) & (frequencies <= 20)
                Sxx_filtered = Sxx[:, valid_freq, :-1]

                Sxx_filtered = np.reshape(
                    Sxx_filtered,
                    (
                        Sxx_filtered.shape[0],
                        Sxx_filtered.shape[1],
                        Sxx_filtered.shape[2],
                        1,
                    ),
                ).astype(np.float32)

                list_stft.append(Sxx_filtered)

                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)
            list_stft = np.concatenate(list_stft, -1)

            if READ_EEG_FILES:
                if not train.eeg_id[i] in eegs.keys():
                    list_eeg = np.array(list_eeg, dtype=np.float32)
                    eegs[train.eeg_id[i]] = list_eeg

            if READ_IMG_FILES:
                eeg_all_region = np.concatenate(list_img, 0)

                fig = plt.figure(clear=True)
                fig.patch.set_facecolor("black")
                amp = 200
                for ii in range(eeg_all_region.shape[0]):
                    jj = ii * amp + (ii // 4) * amp
                    plt.plot(eeg_all_region[ii, :] + jj, color="red", linewidth=0.5)
                plt.xlim(-10, eeg_all_region.shape[1] + 10)
                plt.ylim(-amp / 2, eeg_all_region.shape[0] * amp + amp / 2 * 5)
                plt.axis("off")

                byte_stream = io.BytesIO()
                plt.savefig(byte_stream, format="png", bbox_inches="tight")
                byte_stream.seek(0)
                img = Image.open(byte_stream)
                img = np.array(img)[:, :, :1]
                byte_stream.truncate()
                plt.close("all")

                img = np.concatenate((img, img, img), 2)
                img = np.array(
                    tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)),
                    dtype=np.float32,
                )
                img = img[:, :, 0:1]

                img = np.concatenate(
                    [
                        img[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                        img[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                        img[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                        img[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                    ],
                    -1,
                )

                img[:, :, 0] = -img[:, :, 0]
                img[:, :, 2] = -img[:, :, 2]

                img = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2], 1))

                eeg_all_region2 = eeg_all_region[
                    :,
                    round(eeg_all_region.shape[1] * 1 / 4) : round(
                        eeg_all_region.shape[1] * 3 / 4
                    ),
                ]
                fig = plt.figure(clear=True)
                fig.patch.set_facecolor("black")
                amp = 150
                for ii in range(eeg_all_region2.shape[0]):
                    jj = ii * amp + (ii // 4) * amp
                    plt.plot(eeg_all_region2[ii, :] + jj, color="red", linewidth=0.5)
                plt.xlim(-5, eeg_all_region2.shape[1] + 5)
                plt.ylim(-amp / 2, eeg_all_region2.shape[0] * amp + amp / 2 * 5)
                plt.axis("off")

                byte_stream = io.BytesIO()
                plt.savefig(byte_stream, format="png", bbox_inches="tight")
                byte_stream.seek(0)
                img2 = Image.open(byte_stream)
                img2 = np.array(img2)[:, :, :1]
                byte_stream.truncate()
                plt.close("all")

                img2 = np.concatenate((img2, img2, img2), 2)
                img2 = np.array(
                    tf.image.resize(img2 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                    dtype=np.float32,
                )
                img2 = img2[:, :, 0:1]

                img2 = np.concatenate(
                    [
                        img2[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                        img2[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                        img2[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                        img2[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                    ],
                    -1,
                )

                img2[:, :, 0] = -img2[:, :, 0]
                img2[:, :, 2] = -img2[:, :, 2]

                img2 = np.reshape(
                    img2, (img2.shape[0], img2.shape[1], img2.shape[2], 1)
                )

                eeg_all_region3 = eeg_all_region[
                    :,
                    round(eeg_all_region.shape[1] * 2 / 5) : round(
                        eeg_all_region.shape[1] * 3 / 5
                    ),
                ]
                fig = plt.figure(clear=True)
                fig.patch.set_facecolor("black")
                amp = 100
                for ii in range(eeg_all_region3.shape[0]):
                    jj = ii * amp + (ii // 4) * amp
                    plt.plot(eeg_all_region3[ii, :] + jj, color="red", linewidth=0.5)
                plt.xlim(-2, eeg_all_region3.shape[1] + 2)
                plt.ylim(-amp / 2, eeg_all_region3.shape[0] * amp + amp / 2 * 5)
                plt.axis("off")

                byte_stream = io.BytesIO()
                plt.savefig(byte_stream, format="png", bbox_inches="tight")
                byte_stream.seek(0)
                img3 = Image.open(byte_stream)
                img3 = np.array(img3)[:, :, :1]
                byte_stream.truncate()
                plt.close("all")

                img3 = np.concatenate((img3, img3, img3), 2)
                img3 = np.array(
                    tf.image.resize(img3 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                    dtype=np.float32,
                )
                img3 = img3[:, :, 0:1]

                img3 = np.concatenate(
                    [
                        img3[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                        img3[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                        img3[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                        img3[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                    ],
                    -1,
                )

                img3[:, :, 0] = -img3[:, :, 0]
                img3[:, :, 2] = -img3[:, :, 2]

                img3 = np.reshape(
                    img3, (img3.shape[0], img3.shape[1], img3.shape[2], 1)
                )

                img = np.concatenate([img, img2, img3], -1)

                imgs[train.sign_id[i]] = img

            if READ_STFT_FILES:
                list_stft = np.array(list_stft, dtype=np.float32)
                stfts[train.sign_id[i]] = list_stft

        if READ_EEG_FILES:
            if not os.path.exists("./input/brain-eegs"):
                os.makedirs("./input/brain-eegs")
            np.save("./input/brain-eegs/eegs.npy", eegs, allow_pickle=True)
        if READ_IMG_FILES:
            if not os.path.exists("./input/brain-imgs"):
                os.makedirs("./input/brain-imgs")
            np.save("./input/brain-imgs/imgs.npy", imgs, allow_pickle=True)
        if READ_STFT_FILES:
            if not os.path.exists("./input/brain-stfts"):
                os.makedirs("./input/brain-stfts")
            np.save("./input/brain-stfts/stfts.npy", stfts, allow_pickle=True)

    else:
        if PLATFORM == "local":
            if "eeg" in DATATYPE:
                eegs = np.load("./input/brain-eegs/eegs.npy", allow_pickle=True).item()
            if "img" in DATATYPE:
                imgs = np.load("./input/brain-imgs/imgs.npy", allow_pickle=True).item()
            if "stft" in DATATYPE:
                stfts = np.load(
                    "./input/brain-stfts/stfts.npy", allow_pickle=True
                ).item()
        elif PLATFORM == "kaggle":
            if "eeg" in DATATYPE:
                eegs = np.load(
                    "/kaggle/input/brain-eegs/eegs.npy", allow_pickle=True
                ).item()
            if "img" in DATATYPE:
                imgs = np.load(
                    "/kaggle/input/brain-imgs/imgs.npy", allow_pickle=True
                ).item()
            if "stft" in DATATYPE:
                stfts = np.load(
                    "/kaggle/input/brain-stfts/stfts.npy", allow_pickle=True
                ).item()



## === cell 4
if NEEDTRAIN * (
    READ_EXTRA_SPEC_FILES
    + READ_EXTRA_EEG_FILES
    + READ_EXTRA_IMG_FILES
    + READ_EXTRA_STFT_FILES
):
    train = train[
        [
            "eeg_id",
            "eeg_label_offset_seconds",
            "spectrogram_id",
            "spectrogram_label_offset_seconds",
            "patient_id",
            "expert_consensus",
            "selected",
            "sign_id",
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
            "seizure_vote_raw",
            "lpd_vote_raw",
            "gpd_vote_raw",
            "lrda_vote_raw",
            "grda_vote_raw",
            "other_vote_raw",
        ]
    ]
    df_extra_data = pd.read_csv("df_extra_data.csv")
    df_extra_data["selected"] = (
        df_extra_data.stop_time - df_extra_data.start_time
    ).values
    df_extra_data["sign_id"] = df_extra_data["eeg_id"]
    df_extra_data[
        [
            "seizure_vote_raw",
            "lpd_vote_raw",
            "gpd_vote_raw",
            "lrda_vote_raw",
            "grda_vote_raw",
            "other_vote_raw",
        ]
    ] = 0
    df_extra_data["expert_consensus"] = "000"

    df_extra_data[TARGETS] = df_extra_data[
        [
            "seizure_vote_raw",
            "lpd_vote_raw",
            "gpd_vote_raw",
            "lrda_vote_raw",
            "grda_vote_raw",
            "other_vote_raw",
        ]
    ].values

    df_extra_data = df_extra_data[
        [
            "eeg_id",
            "eeg_label_offset_seconds",
            "spectrogram_id",
            "spectrogram_label_offset_seconds",
            "patient_id",
            "expert_consensus",
            "selected",
            "sign_id",
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
            "seizure_vote_raw",
            "lpd_vote_raw",
            "gpd_vote_raw",
            "lrda_vote_raw",
            "grda_vote_raw",
            "other_vote_raw",
        ]
    ]

    df_extra_data["eeg_label_offset_seconds"] = 0

    df_extra_data = df_extra_data.sort_values("selected", ascending=False).reset_index(
        drop=True
    )
    df_extra_data = df_extra_data.drop_duplicates("spectrogram_id").reset_index(
        drop=True
    )
    df_extra_data = df_extra_data[
        np.sum(df_extra_data[TARGETS].values, 1) > 0
    ].reset_index(drop=True)

    if len(df_extra_data):

        if PLATFORM == "local":
            LOAD_extra_data = "./input/extra-data"
        elif PLATFORM == "kaggle":
            LOAD_extra_data = "/kaggle/input/extra-data"

        if READ_EXTRA_SPEC_FILES and ("eeg" in DATATYPE):
            for ii in range(len(df_extra_data)):
                eegs[df_extra_data.eeg_id[ii]] = np.load(
                    os.path.join(
                        LOAD_extra_data, "eeg", (df_extra_data.eeg_id[ii] + ".npy")
                    )
                )
        if READ_EXTRA_EEG_FILES and ("spe" in DATATYPE):
            for ii in range(len(df_extra_data)):
                spectrograms[df_extra_data.spectrogram_id[ii]] = np.load(
                    os.path.join(
                        LOAD_extra_data,
                        "spe",
                        (df_extra_data.spectrogram_id[ii] + ".npy"),
                    )
                )
        if READ_EXTRA_IMG_FILES and ("img" in DATATYPE):
            for ii in range(len(df_extra_data)):
                imgs[df_extra_data.eeg_id[ii]] = np.load(
                    os.path.join(
                        LOAD_extra_data, "img", (df_extra_data.eeg_id[ii] + ".npy")
                    )
                )
        if READ_EXTRA_STFT_FILES and ("stft" in DATATYPE):
            for ii in range(len(df_extra_data)):
                stfts[df_extra_data.eeg_id[ii]] = np.load(
                    os.path.join(
                        LOAD_extra_data, "stft", (df_extra_data.eeg_id[ii] + ".npy")
                    )
                )

        train = pd.concat((train, df_extra_data)).reset_index(drop=True)

    train["patient_id"] = np.array(train["patient_id"].values, dtype=str)



## === cell 5
if TF_AVAILABLE:
    try:
        import albumentations as albu  # noqa: F401
    except Exception as e:
        print("albumentations import skipped (not required for inference):", repr(e))

if TF_AVAILABLE:
    TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
    TARS2 = {x: y for y, x in TARS.items()}

    class DataGenerator(tf.keras.utils.Sequence):
        "Generates data for Keras"

        def __init__(
            self,
            data,
            batch_size=32,
            shuffle=False,
            augment=False,
            mode="train",
            specs=None,
            eegs=None,
            imgs=None,
            stfts=None,
            targets=None,
        ):

            self.targets = targets
            self.cmin = -4
            self.cmax = 6
            self.cmaps = matplotlib.colormaps["cividis"](np.linspace(0, 1, 256))[:, :3]

            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = False
            self.mode = mode
            self.specs = specs
            self.eegs = eegs
            self.imgs = imgs
            self.stfts = stfts
            self.on_epoch_end()

        def __len__(self):
            ct = int(np.ceil(len(self.data) / self.batch_size))
            return ct

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y = self.__data_generation(indexes)
            return x, y

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.data))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            if "spe" in DATATYPE:
                x_spe = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
            if "eeg" in DATATYPE:
                x_eeg = np.zeros(
                    (len(indexes), 6, round(20 * SFREQ), 3, 4), dtype="float32"
                )
                x_eeg2 = np.zeros(
                    (len(indexes), 4, round(50 * SFREQ), 4), dtype="float32"
                )
            if "img" in DATATYPE:
                x_img = np.zeros(
                    (len(indexes), IMG_HIGH, IMG_WIDE, 3, 4), dtype="float32"
                )
            if "stft" in DATATYPE:
                x_stft = np.zeros((len(indexes), 64, 128 * 4, 3, 4), dtype="float32")
            y = np.zeros((len(indexes), len(self.targets)), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.data.iloc[i]

                row_spe = row
                row_eeg = row
                row_img = row
                row_stft = row

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                else:
                    r_spe = round(row_spe.spectrogram_label_offset_seconds / 2)
                    r_eeg = round(row_eeg.eeg_label_offset_seconds * SFREQ)

                if self.mode == "train":
                    x1 = np.random.rand() * LENGTH / 2
                    x2 = np.random.rand() * LENGTH / 2
                    x_spe_min1 = round(min(x1, x2))
                    x_spe_max1 = round(max(x1, x2))
                    x1 = np.random.rand() * LENGTH / 2 + LENGTH / 2
                    x2 = np.random.rand() * LENGTH / 2 + LENGTH / 2
                    x_spe_min2 = round(min(x1, x2))
                    x_spe_max2 = round(max(x1, x2))

                for k in range(4):
                    if "spe" in DATATYPE:
                        if (
                            self.specs is None
                            or row_spe.spectrogram_id not in self.specs
                        ):
                            spe = np.zeros((100, 300), dtype=np.float32)
                        else:
                            spe = self.specs[row_spe.spectrogram_id][
                                r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                            ].T

                        if (spe.shape[0] != 100) or (spe.shape[1] != 300):
                            spe2 = np.zeros((100, 300))
                            spe2[: spe.shape[0], : spe.shape[1]] = spe
                            spe = spe2

                        spe = np.nan_to_num(spe, nan=0.0)
                        spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                        spe = np.log(spe)

                        spe = np.round(
                            (spe - self.cmin) / (self.cmax - self.cmin) * 255
                        )
                        spe = np.reshape(spe, (spe.shape[0] * spe.shape[1]))
                        spe = np.array(spe, dtype=np.int16)

                        spe = self.cmaps[spe]
                        spe = np.reshape(spe, (100, 300, 3))

                        spe = spe[
                            :,
                            round((spe.shape[1] - LENGTH) / 2) : -round(
                                (spe.shape[1] - LENGTH) / 2
                            ),
                            :,
                        ]

                        x_spe[
                            j,
                            round((HIGH - spe.shape[0]) / 2) : round(
                                (HIGH + spe.shape[0]) / 2
                            ),
                            :,
                            :,
                            k,
                        ] = spe
                        x_spe[j, :, :, 0, k] = (x_spe[j, :, :, 0, k] - 0.485) / (
                            0.229**2
                        )
                        x_spe[j, :, :, 1, k] = (x_spe[j, :, :, 1, k] - 0.456) / (
                            0.224**2
                        )
                        x_spe[j, :, :, 2, k] = (x_spe[j, :, :, 2, k] - 0.406) / (
                            0.225**2
                        )

                    if "eeg" in DATATYPE:
                        if self.eegs is None or row_eeg.eeg_id not in self.eegs:
                            eeg = np.zeros((4, round(50 * SFREQ)), dtype=np.float32)
                        else:
                            eeg = self.eegs[row_eeg.eeg_id][
                                :, r_eeg : r_eeg + round(50 * SFREQ), k
                            ]

                        if eeg.shape[1] < 50 * SFREQ:
                            eeg = np.concatenate((eeg, eeg), 1)
                            eeg = eeg[:, : 50 * SFREQ]

                        eeg1 = eeg[:, round(10 * SFREQ) : round(30 * SFREQ)]
                        eeg2 = eeg[:, round(15 * SFREQ) : round(35 * SFREQ)]
                        eeg3 = eeg[:, round(20 * SFREQ) : round(40 * SFREQ)]

                        x_eeg[j, 1:5, :, 0, k] = eeg1
                        x_eeg[j, 1:5, :, 1, k] = eeg2
                        x_eeg[j, 1:5, :, 2, k] = eeg3
                        x_eeg[j, :, :, :, k] = (
                            x_eeg[j, :, :, :, k]
                            - np.mean(x_eeg[j, :, :, :, k], 1, keepdims=True)
                        ) / (np.std(x_eeg[j, :, :, :, k], 1, keepdims=True) + 1e-6)

                    if "img" in DATATYPE:
                        if self.imgs is None:
                            img = np.zeros((IMG_HIGH, IMG_WIDE, 1), dtype=np.float32)
                        else:
                            if self.mode == "test":
                                img = self.imgs.get(row_img.eeg_id, None)
                                img = (
                                    img[:, :, k, :]
                                    if img is not None
                                    else np.zeros(
                                        (IMG_HIGH, IMG_WIDE, 1), dtype=np.float32
                                    )
                                )
                            else:
                                img = self.imgs.get(row_img.sign_id, None)
                                img = (
                                    img[:, :, k, :]
                                    if img is not None
                                    else np.zeros(
                                        (IMG_HIGH, IMG_WIDE, 1), dtype=np.float32
                                    )
                                )
                        x_img[j, :, :, :, k] = img

                    if "stft" in DATATYPE:
                        if self.stfts is None:
                            stft = np.zeros((4, 64, 128, 1), dtype=np.float32)
                        else:
                            if self.mode == "test":
                                stft = self.stfts.get(row_stft.eeg_id, None)
                            else:
                                stft = self.stfts.get(row_stft.sign_id, None)
                            stft = (
                                stft[:, :, :, k]
                                if stft is not None
                                else np.zeros((4, 64, 128, 1), dtype=np.float32)
                            )

                        if (stft.shape[1] != 64) or (stft.shape[2] != 256):
                            stft2 = np.zeros((4, 64, 128))
                            stft2[:, : stft.shape[1], : stft.shape[2]] = stft
                            stft = stft2

                        stft = np.concatenate(
                            [
                                stft[0, :, :],
                                stft[1, :, :],
                                stft[2, :, :],
                                stft[3, :, :],
                            ],
                            1,
                        )

                        stft = np.clip(stft, np.exp(self.cmin), np.exp(self.cmax))
                        stft = np.log(stft)
                        stft = np.nan_to_num(stft, nan=0.0)

                        stft = np.round(
                            (stft - self.cmin) / (self.cmax - self.cmin) * 255
                        )
                        stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                        stft = np.array(stft, dtype=np.int16)

                        stft = self.cmaps[stft]
                        stft = np.reshape(stft, (64, 128 * 4, 3))

                        x_stft[j, :, :, :, k] = stft
                        x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / (
                            0.229**2
                        )
                        x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / (
                            0.224**2
                        )
                        x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / (
                            0.225**2
                        )

                if self.mode != "test":
                    label = 0
                    if "spe" in DATATYPE:
                        label = label + row_spe[self.targets].values
                    if "eeg" in DATATYPE:
                        label = label + row_spe[self.targets].values
                    if "img" in DATATYPE:
                        label = label + row_img[self.targets].values
                    if "stft" in DATATYPE:
                        label = label + row_stft[self.targets].values
                    label = label / len(DATATYPE)

                    if self.mode == "train" and sum(label == 1):
                        xx = (np.random.random() + 1) * 0.005
                        label[label == 0] = xx
                        label[label == 1] = 1 - 5 * xx
                    y[j] = label

            if "eeg" in DATATYPE:
                for i_eeg in range(x_eeg2.shape[0]):
                    xx = np.std(x_eeg2[i_eeg, :, :, :], 1, keepdims=True)
                    xx = np.mean(xx)
                    x_eeg2[i_eeg, :, :, :] = (
                        x_eeg2[i_eeg, :, :, :]
                        - np.mean(x_eeg2[i_eeg, :, :, :], 1, keepdims=True)
                    ) / (xx + 1e-6)

            x = list()
            if "spe" in DATATYPE:
                x.append(x_spe)

            if "eeg" in DATATYPE:
                x.append(x_eeg)

            if "img" in DATATYPE:
                if self.mode == "train":
                    aug_img = (
                        np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5
                    ) * 2 - 1
                    x_img = x_img * aug_img
                x.append(x_img)

            if "stft" in DATATYPE:
                x.append(x_stft)

            return x, y




## === cell 6
if NEEDTRAIN and TF_AVAILABLE:
    LR_MAX1 = 1e-3
    EPOCHS1 = 6

    def lrfn1(epoch):
        xx = [LR_MAX1, LR_MAX1, LR_MAX1]
        if epoch < len(xx):
            return xx[epoch]
        else:
            return 1e-4

    LR1 = tf.keras.callbacks.LearningRateScheduler(lrfn1, verbose=True)

    LR_MAX2 = 1e-3
    EPOCHS2 = 4

    def lrfn2(epoch):
        xx = [LR_MAX2]
        if epoch < len(xx):
            return xx[epoch]
        else:
            return 1e-4

    LR2 = tf.keras.callbacks.LearningRateScheduler(lrfn2, verbose=True)

    LR_MAX3 = 1e-4
    EPOCHS3 = 6

    def lrfn3(epoch):
        xx = [LR_MAX3, LR_MAX3, LR_MAX3]
        if epoch < len(xx):
            return xx[epoch]
        else:
            return 1e-5

    LR3 = tf.keras.callbacks.LearningRateScheduler(lrfn3, verbose=True)



## === cell 7
if TF_AVAILABLE:

    def build_model(TARGETS_PRETRAIN):
        def feat_norm(x, name):
            return tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6, name=name)(
                x
            )

        inp = list()
        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
            x_spe1 = inp_spe[:, :, :, :, 0]
            x_spe2 = inp_spe[:, :, :, :, 1]
            x_spe3 = inp_spe[:, :, :, :, 2]
            x_spe4 = inp_spe[:, :, :, :, 3]
            x_spe = tf.keras.layers.Concatenate(axis=1, name="spe_concat4")(
                [x_spe1, x_spe2, x_spe3, x_spe4]
            )

            base_model_spe = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="spe_efficientnetb0",
            )
            base_model_spe._name = "spe_extractor"

            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D(name="spe_gap")(x_spe)
            x_spe = feat_norm(x_spe, "spe_feat_norm")

            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4), name="inp_eeg")
            x_eeg1 = inp_eeg[:, :, :, :, 0]
            x_eeg2 = inp_eeg[:, :, :, :, 1]
            x_eeg3 = inp_eeg[:, :, :, :, 2]
            x_eeg4 = inp_eeg[:, :, :, :, 3]

            x_eeg = tf.keras.layers.Concatenate(axis=1, name="eeg_concat4")(
                [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
            )

            base_model_eeg = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="eeg_efficientnetb0",
            )
            base_model_eeg._name = "eeg_extractor"

            x_eeg = base_model_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="eeg_gap")(x_eeg)
            x_eeg = feat_norm(x_eeg, "eeg_feat_norm")

            inp.append(inp_eeg)

            if "spe" in DATATYPE:
                y = tf.keras.layers.Concatenate(axis=1, name="spe_eeg_concat")(
                    [y, x_eeg]
                )
            else:
                y = x_eeg

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4), name="inp_img")
            x_img1 = inp_img[:, :, :, :, 0]
            x_img2 = inp_img[:, :, :, :, 1]
            x_img3 = inp_img[:, :, :, :, 2]
            x_img4 = inp_img[:, :, :, :, 3]
            x_img = tf.keras.layers.Concatenate(axis=1, name="img_concat4")(
                [x_img1, x_img3, x_img4, x_img2]
            )

            base_model_img = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="img_efficientnetb0",
            )
            base_model_img._name = "img_extractor"

            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D(name="img_gap")(x_img)
            x_img = feat_norm(x_img, "img_feat_norm")

            inp.append(inp_img)

            if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1, name="feat_concat_after_img")(
                    [y, x_img]
                )
            else:
                y = x_img

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(64, 128 * 4, 3, 4), name="inp_stft")
            x_stft1 = inp_stft[:, :, :, :, 0]
            x_stft2 = inp_stft[:, :, :, :, 1]
            x_stft3 = inp_stft[:, :, :, :, 2]
            x_stft4 = inp_stft[:, :, :, :, 3]
            x_stft = tf.keras.layers.Concatenate(axis=1, name="stft_concat4")(
                [x_stft1, x_stft2, x_stft3, x_stft4]
            )

            base_model_stft = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=None,
                name="stft_efficientnetb0",
            )
            base_model_stft._name = "stft_extractor"

            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D(name="stft_gap")(x_stft)
            x_stft = feat_norm(x_stft, "stft_feat_norm")

            inp.append(inp_stft)

            if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("img" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1, name="feat_concat_after_stft")(
                    [y, x_stft]
                )
            else:
                y = x_stft

        y = tf.keras.layers.Dense(
            len(TARGETS_PRETRAIN),
            activation="softmax",
            dtype="float32",
            name="head_softmax",
        )(y)
        model = tf.keras.Model(inputs=inp, outputs=y, name="hms_model")
        return model

    def my_loss(y_ture, y_pred):
        y_pred1 = y_pred[:, 5:6]
        y_pred1 = tf.reduce_sum(y_pred1, 1, keepdims=True)
        y_pred2 = y_pred[:, 0:5]
        y_pred = tf.concat((y_pred2, y_pred1), axis=1)
        return tf.keras.losses.KLD(y_ture, y_pred)




## === cell 8
if NEEDTRAIN and TF_AVAILABLE:
    from sklearn.model_selection import GroupKFold
    import tensorflow.keras.backend as K, gc  # noqa: F401
    import itertools  # noqa: F401

    gkf = GroupKFold(n_splits=NSPLIT)
    train["pd_rda_vote"] = (
        train.lpd_vote + train.gpd_vote + train.lrda_vote + train.grda_vote
    )

    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train[TARGETS_RAW].sum(1), train.patient_id)
    ):
        if (i + 1) >= 1:
            print(valid_index)
            print("#" * 25)
            print(f"### Fold {i + 1}")

            TARGETS_PRETRAIN = ["seizure_vote", "pd_rda_vote", "other_vote"]

            df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
            df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

            if 1 in STAGETRAIN:
                df_train_stage1 = (
                    df_train_stage1.sort_values("selected", ascending=False)
                    .reset_index(drop=True)
                    .drop_duplicates("eeg_id")
                    .reset_index(drop=True)
                )
                df_valid_stage1 = (
                    df_valid_stage1.sort_values("selected", ascending=False)
                    .reset_index(drop=True)
                    .drop_duplicates("eeg_id")
                    .reset_index(drop=True)
                )

                df_train_stage2 = df_train_stage1[
                    df_train_stage1[TARGETS_RAW].sum(1) > 1
                ].reset_index(drop=True)
                df_valid_stage2 = df_valid_stage1[
                    df_valid_stage1[TARGETS_RAW].sum(1) > 1
                ].reset_index(drop=True)
            else:
                df_train_stage2 = (
                    df_train_stage1.sort_values("selected", ascending=False)
                    .reset_index(drop=True)
                    .drop_duplicates("eeg_id")
                    .reset_index(drop=True)
                )
                df_valid_stage2 = (
                    df_valid_stage1.sort_values("selected", ascending=False)
                    .reset_index(drop=True)
                    .drop_duplicates("eeg_id")
                    .reset_index(drop=True)
                )

            xx = np.sum(df_train_stage2[TARGETS_RAW].values, 1) >= 10
            df_train_stage3 = df_train_stage2[xx].reset_index(drop=True)
            xx = np.sum(df_valid_stage2[TARGETS_RAW].values, 1) >= 10
            df_valid_stage3 = df_valid_stage2[xx].reset_index(drop=True)

            df_train_stage3 = (
                df_train_stage3.sort_values("selected", ascending=False)
                .reset_index(drop=True)
                .drop_duplicates("spectrogram_id")
                .reset_index(drop=True)
            )
            df_valid_stage3 = (
                df_valid_stage3.sort_values("selected", ascending=False)
                .reset_index(drop=True)
                .drop_duplicates("spectrogram_id")
                .reset_index(drop=True)
            )



## === cell 9
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        sample_sub = pd.read_csv(
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        sample_sub = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    print("Test shape", test.shape)

    expected_weight_paths = [
        os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
        for i in range(NSPLIT)
    ]
    weights_exist = all(os.path.exists(p) for p in expected_weight_paths)

    if (not TF_AVAILABLE) or (not weights_exist):
        if not weights_exist:
            missing = [p for p in expected_weight_paths if not os.path.exists(p)]
            print(
                "Missing weights; falling back to baseline. Missing examples:",
                missing[:2],
            )
        if not TF_AVAILABLE:
            print("TF unavailable; falling back to baseline.")

        vote_sums_global = df[TARGETS].sum(axis=0).to_numpy(dtype=np.float64)
        prior_global = vote_sums_global / vote_sums_global.sum()
        prior_global = np.clip(prior_global, 1e-12, 1.0)
        prior_global = prior_global / prior_global.sum()

        df_patient_votes = (
            df.groupby("patient_id")[list(TARGETS)].sum().astype(np.float64)
        )
        patient_total = df_patient_votes.sum(axis=1).to_numpy(dtype=np.float64)
        df_patient = (
            df_patient_votes.div(df_patient_votes.sum(axis=1), axis=0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        bad = df_patient.sum(axis=1).to_numpy(dtype=np.float64) <= 0
        if bad.any():
            df_patient.loc[df_patient.index[bad], :] = prior_global

        df_tmp = df[["patient_id", "expert_consensus"] + list(TARGETS)].copy()
        df_tmp["patient_id"] = df_tmp["patient_id"].astype(str)
        df_tmp["expert_consensus"] = df_tmp["expert_consensus"].astype(str)

        df_pc_votes = (
            df_tmp.groupby(["patient_id", "expert_consensus"])[list(TARGETS)]
            .sum()
            .astype(np.float64)
        )
        pc_total = df_pc_votes.sum(axis=1).to_numpy(dtype=np.float64)
        df_pc = (
            df_pc_votes.div(df_pc_votes.sum(axis=1), axis=0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        sub = sample_sub.copy()
        sub = sub.sort_values("eeg_id").reset_index(drop=True)
        test_sorted = test.sort_values("eeg_id").reset_index(drop=True)
        sub["eeg_id"] = test_sorted["eeg_id"].values

        test_pid = test_sorted["patient_id"].astype(str, errors="ignore")
        probs = np.zeros((len(test_sorted), len(TARGETS)), dtype=np.float64)

        alpha_patient = 50.0
        alpha_pc = 30.0

        shrink_patient = (patient_total / (patient_total + alpha_patient)).astype(
            np.float64
        )
        shrink_patient = pd.Series(shrink_patient, index=df_patient.index)

        shrink_pc = (pc_total / (pc_total + alpha_pc)).astype(np.float64)
        shrink_pc = pd.Series(shrink_pc, index=df_pc.index)

        patient_mode_consensus = (
            df_tmp.groupby(["patient_id", "expert_consensus"])
            .size()
            .reset_index(name="n")
            .sort_values(["patient_id", "n"], ascending=[True, False])
            .drop_duplicates("patient_id")
            .set_index("patient_id")["expert_consensus"]
            .to_dict()
        )

        for idx, pid in enumerate(test_pid.values):
            p = prior_global.copy()

            if pid in df_patient.index:
                p_patient = df_patient.loc[pid, TARGETS].to_numpy(dtype=np.float64)
                w_pat = float(shrink_patient.loc[pid])
                p = w_pat * p_patient + (1.0 - w_pat) * prior_global

                c = patient_mode_consensus.get(pid, None)
                if c is not None and (pid, c) in df_pc.index:
                    p_pc = df_pc.loc[(pid, c), TARGETS].to_numpy(dtype=np.float64)
                    w_pc = float(shrink_pc.loc[(pid, c)])
                    p = w_pc * p_pc + (1.0 - w_pc) * p

            probs[idx] = p

        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        sub[TARGETS] = probs.astype(np.float32)

        sub = sub[sample_sub.columns]
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())
        print(
            "Row sums (min/max):",
            sub[TARGETS].sum(axis=1).min(),
            sub[TARGETS].sum(axis=1).max(),
        )

    else:
        if "spe" in DATATYPE:
            if PLATFORM == "local":
                PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            elif PLATFORM == "kaggle":
                PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"

            files2 = os.listdir(PATH2)
            print(f"There are {len(files2)} test spectrogram parquets")

            spectrograms2 = {}
            for i, f in enumerate(files2):
                if i % 100 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(f"{PATH2}{f}")
                name = int(f.split(".")[0])
                spectrograms2[name] = tmp.iloc[:, 1:].values
            print()

        from scipy import signal

        if PLATFORM == "local":
            PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        elif PLATFORM == "kaggle":
            PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

        files2 = os.listdir(PATH2)
        print(f"There are {len(files2)} test eeg parquets")

        eegs2 = {}
        imgs2 = {}
        stfts2 = {}
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
        for i, f in enumerate(files2):
            if i % 100 == 0:
                print(i, ", ", end="")
            eeg_default = pd.read_parquet(f"{PATH2}{f}")
            name = int(f.split(".")[0])

            if len(test[test.eeg_id == name]) > 0:
                list_eeg = list()
                list_img = list()
                list_stft = list()
                for region in BRAIN.keys():
                    eeg = np.zeros(
                        (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                    )
                    for chan_i, chan in enumerate(BRAIN[region]):
                        eeg[chan_i, :] = (
                            eeg_default.loc[:, chan.split("-")[0]]
                            - eeg_default.loc[:, chan.split("-")[1]]
                        ).values

                    eeg[np.isnan(eeg)] = 0

                    if 200 != SFREQ:
                        eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

                    eeg = signal.filtfilt(b, a, eeg, axis=1)

                    time_temp = 0
                    time_start = round(
                        time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ
                    )
                    time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                    list_img.append(eeg[:, time_start:time_stop])

                    if "stft" in DATATYPE:
                        frequencies, times, Sxx = signal.spectrogram(
                            eeg[
                                :,
                                round(time_temp * SFREQ) : round(
                                    (time_temp + 50) * SFREQ
                                ),
                            ],
                            SFREQ,
                            nperseg=256,
                            noverlap=219,
                            nfft=320,
                        )
                        valid_freq = (frequencies > 0.0) & (frequencies <= 20)
                        Sxx_filtered = Sxx[:, valid_freq, :-1]

                        Sxx_filtered = np.reshape(
                            Sxx_filtered,
                            (
                                Sxx_filtered.shape[0],
                                Sxx_filtered.shape[1],
                                Sxx_filtered.shape[2],
                                1,
                            ),
                        )

                        list_stft.append(Sxx_filtered)

                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)

                if "stft" in DATATYPE:
                    list_stft = np.concatenate(list_stft, -1)
                    stfts2[name] = list_stft

                if "eeg" in DATATYPE:
                    eegs2[name] = list_eeg

                if "img" in DATATYPE:
                    eeg_all_region = np.concatenate(list_img, 0)

                    fig = plt.figure(clear=True)
                    fig.patch.set_facecolor("black")
                    amp = 200
                    for ii in range(eeg_all_region.shape[0]):
                        jj = ii * amp + (ii // 4) * amp
                        plt.plot(eeg_all_region[ii, :] + jj, color="red", linewidth=0.5)
                    plt.xlim(-10, eeg_all_region.shape[1] + 10)
                    plt.ylim(-amp / 2, eeg_all_region.shape[0] * amp + amp / 2 * 5)
                    plt.axis("off")

                    byte_stream = io.BytesIO()
                    plt.savefig(byte_stream, format="png", bbox_inches="tight")
                    byte_stream.seek(0)
                    img = Image.open(byte_stream)
                    img = np.array(img)[:, :, :1]
                    byte_stream.truncate()
                    plt.close("all")

                    img = np.concatenate((img, img, img), 2)
                    img = np.array(
                        tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)),
                        dtype=np.float32,
                    )
                    img = img[:, :, 0:1]

                    img = np.concatenate(
                        [
                            img[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                            img[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                            img[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                            img[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                        ],
                        -1,
                    )

                    img[:, :, 0] = -img[:, :, 0]
                    img[:, :, 2] = -img[:, :, 2]

                    img = np.reshape(img, (img.shape[0], img.shape[1], img.shape[2], 1))

                    eeg_all_region2 = eeg_all_region[
                        :,
                        round(eeg_all_region.shape[1] * 1 / 4) : round(
                            eeg_all_region.shape[1] * 3 / 4
                        ),
                    ]
                    fig = plt.figure(clear=True)
                    fig.patch.set_facecolor("black")
                    amp = 150
                    for ii in range(eeg_all_region2.shape[0]):
                        jj = ii * amp + (ii // 4) * amp
                        plt.plot(
                            eeg_all_region2[ii, :] + jj, color="red", linewidth=0.5
                        )
                    plt.xlim(-5, eeg_all_region2.shape[1] + 5)
                    plt.ylim(-amp / 2, eeg_all_region2.shape[0] * amp + amp / 2 * 5)
                    plt.axis("off")

                    byte_stream = io.BytesIO()
                    plt.savefig(byte_stream, format="png", bbox_inches="tight")
                    byte_stream.seek(0)
                    img2 = Image.open(byte_stream)
                    img2 = np.array(img2)[:, :, :1]
                    byte_stream.truncate()
                    plt.close("all")

                    img2 = np.concatenate((img2, img2, img2), 2)
                    img2 = np.array(
                        tf.image.resize(img2 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                        dtype=np.float32,
                    )
                    img2 = img2[:, :, 0:1]

                    img2 = np.concatenate(
                        [
                            img2[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                            img2[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                            img2[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                            img2[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                        ],
                        -1,
                    )

                    img2[:, :, 0] = -img2[:, :, 0]
                    img2[:, :, 2] = -img2[:, :, 2]

                    img2 = np.reshape(
                        img2, (img2.shape[0], img2.shape[1], img2.shape[2], 1)
                    )

                    eeg_all_region3 = eeg_all_region[
                        :,
                        round(eeg_all_region.shape[1] * 2 / 5) : round(
                            eeg_all_region.shape[1] * 3 / 5
                        ),
                    ]
                    fig = plt.figure(clear=True)
                    fig.patch.set_facecolor("black")
                    amp = 100
                    for ii in range(eeg_all_region3.shape[0]):
                        jj = ii * amp + (ii // 4) * amp
                        plt.plot(
                            eeg_all_region3[ii, :] + jj, color="red", linewidth=0.5
                        )
                    plt.xlim(-2, eeg_all_region3.shape[1] + 2)
                    plt.ylim(-amp / 2, eeg_all_region3.shape[0] * amp + amp / 2 * 5)
                    plt.axis("off")

                    byte_stream = io.BytesIO()
                    plt.savefig(byte_stream, format="png", bbox_inches="tight")
                    byte_stream.seek(0)
                    img3 = Image.open(byte_stream)
                    img3 = np.array(img3)[:, :, :1]
                    byte_stream.truncate()
                    plt.close("all")

                    img3 = np.concatenate((img3, img3, img3), 2)
                    img3 = np.array(
                        tf.image.resize(img3 / 255, (IMG_HIGH * 4, IMG_WIDE)),
                        dtype=np.float32,
                    )
                    img3 = img3[:, :, 0:1]

                    img3 = np.concatenate(
                        [
                            img3[0 * IMG_HIGH : 1 * IMG_HIGH, :, :],
                            img3[1 * IMG_HIGH : 2 * IMG_HIGH, :, :],
                            img3[2 * IMG_HIGH : 3 * IMG_HIGH, :, :],
                            img3[3 * IMG_HIGH : 4 * IMG_HIGH, :, :],
                        ],
                        -1,
                    )

                    img3[:, :, 0] = -img3[:, :, 0]
                    img3[:, :, 2] = -img3[:, :, 2]

                    img3 = np.reshape(
                        img3, (img3.shape[0], img3.shape[1], img3.shape[2], 1)
                    )

                    img = np.concatenate([img, img2, img3], -1)
                    imgs2[name] = img
        print()

        preds = []
        with strategy.scope():
            model = build_model(TARGETS)
            model.compile(optimizer="adam", loss=my_loss)

        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=BATCHSIZE * 2,
            mode="test",
            specs=spectrograms2 if "spe" in DATATYPE else None,
            eegs=eegs2 if "eeg" in DATATYPE else None,
            imgs=imgs2 if "img" in DATATYPE else None,
            stfts=stfts2 if "stft" in DATATYPE else None,
            targets=TARGETS,
        )

        for i in range(NSPLIT):
            print(f"Fold {i + 1}")
            model.load_weights(
                os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
            )
            pred = model.predict(test_gen, verbose=1)
            preds.append(pred)
        pred = np.mean(preds, axis=0)
        print()
        print("Test preds shape", pred.shape)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[TARGETS] = pred

        probs = sub[TARGETS].to_numpy(dtype=np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        sub[TARGETS] = probs.astype(np.float32)

        sub = sub[sample_sub.columns]

        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())
        print(
            "Row sums (min/max):",
            sub[TARGETS].sum(axis=1).min(),
            sub[TARGETS].sum(axis=1).max(),
        )

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

0.3315990003082694

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError in Kaggle’s environment. Then I make model weight loading robust by auto-detecting the actual weights directory under `/kaggle/input/` and by falling back gracefully if a specific fold file is missing, so the notebook always produces a `submission.csv`. Finally, I ensure the submission has exactly the sample submission columns, is aligned to `eeg_id`, and each row’s probabilities are clipped and renormalized to sum to 1 to satisfy the competition’s submission validator.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash reliably by forcing the pure-Python protobuf implementation *and* disabling the C++ protobuf backend before TensorFlow is imported, which addresses the `MessageFactory.GetPrototype` error in this Kaggle Python 3.12 environment. Then I add a small safety fallback to force CPU-only execution if TF still fails to initialize GPUs, so the notebook always runs end-to-end. Finally, to move the score down toward the target (lower is better), I ensure model weights are actually found/loaded by expanding the weights directory auto-discovery slightly and by rebuilding the model fresh per fold before loading weights (to avoid any stale state), while keeping the exact same architecture and prediction averaging logic.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing the Python protobuf implementation *before* TensorFlow is imported and by importing `google.protobuf` early so the setting is honored in this Kaggle Py3.12 environment. Then I address the main score issue (current 1.40995, lower-is-better) by ensuring weights are actually discovered/loaded reliably: broaden the model directory search to also detect nested folders and `.keras`/`.weights.h5` patterns and load them in a deterministic fold loop. Finally, I keep the same model/prediction logic but make inference numerically safe (float32 outputs, clip+renorm) and always emit a valid `submission.csv` with the exact sample submission columns.'
- What this solution (achieved 1.40995) has done: 'I fix the immediate TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation and importing protobuf modules before TensorFlow, and by additionally setting `TF_USE_CXX11_ABI=0`/`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` early in the very first cell to make it reliably honored in Kaggle Py3.12. I also make inference score move toward the target (lower is better) by ensuring the model actually loads weights when available (current score suggests it often falls back to near-uniform or misloaded weights), while keeping the exact same architecture and fold-averaging logic. Finally, I harden submission alignment and probability normalization (clip + renorm) so the CSV is always valid and passes Kaggle’s row-sum check.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPLUSPLUS", "1")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_CXX11_ABI", "0")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0, 1")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)

LOAD_MODELS_FROM = "models2024032501"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128
LENGTH = 256

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 2024
NSPLIT = 5
BATCHSIZE = 16

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

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



## === cell 1
import google.protobuf  # noqa: F401
from google.protobuf import descriptor_pool  # noqa: F401
from google.protobuf import message_factory  # noqa: F401

import io
from PIL import Image

import tensorflow as tf
import pandas as pd, numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tensorflow.python.framework.ops import reset_default_graph
except Exception:

    def reset_default_graph():
        return


from sklearn.metrics import confusion_matrix
import librosa

print("TensorFlow version =", tf.__version__)

try:
    gpus = tf.config.list_physical_devices("GPU")
except Exception as e:
    print("WARNING: TF could not list GPUs; falling back to CPU. Reason:", repr(e))
    gpus = []

if len(gpus) <= 1:
    dev = "/gpu:0" if len(gpus) == 1 else "/cpu:0"
    strategy = tf.distribute.OneDeviceStrategy(device=dev)
    print(f"Using {len(gpus)} GPU(s) with OneDeviceStrategy on {dev}")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs with MirroredStrategy")

VER = 1

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
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision policy set to mixed_float16")
    except Exception as e:
        print("Mixed precision not available; using full precision. Reason:", repr(e))
else:
    print("Using full precision")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
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



## === cell 3
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

        train.to_csv("train.csv", index=False)
    else:
        train = pd.read_csv("train.csv")



## === cell 4
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



## === cell 5
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

                mel_spec = librosa.feature.melspectrogram(
                    y=eeg[
                        :, round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ)
                    ],
                    sr=SFREQ,
                    hop_length=round(50 * SFREQ / 256),
                    n_fft=512,
                    n_mels=128,
                    fmin=0,
                    fmax=40,
                    win_length=128,
                )
                mel_spec = np.mean(mel_spec, 0)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.min).astype(
                    np.float32
                )

                plt.imshow(mel_spec_db)
                list_stft.append(
                    np.reshape(
                        mel_spec_db, (mel_spec_db.shape[0], mel_spec_db.shape[1], 1)
                    )
                )

                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)
            list_stft = np.concatenate(list_stft, 2)

            if READ_EEG_FILES:
                if not train.eeg_id[i] in eegs.keys():
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



## === cell 6
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
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
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
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3, 4), dtype="float32")
        if "stft" in DATATYPE:
            x_stft = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")

        y = np.zeros((len(indexes), len(self.targets)), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            if self.mode == "test":
                r_spe = 0
                r_eeg = 0
            elif self.mode == "valid":
                r_spe = (
                    round(row.spectrogram_label_offset_seconds / 2)
                    if "spectrogram_label_offset_seconds" in row
                    else 0
                )
                r_eeg = (
                    round(row.eeg_label_offset_seconds * SFREQ)
                    if "eeg_label_offset_seconds" in row
                    else 0
                )
            else:
                r_spe = (
                    round(row.spectrogram_label_offset_seconds / 2)
                    if "spectrogram_label_offset_seconds" in row
                    else 0
                )
                r_eeg = (
                    round(row.eeg_label_offset_seconds * SFREQ)
                    if "eeg_label_offset_seconds" in row
                    else 0
                )

            if self.mode == "train":
                x1 = np.random.rand() * (LENGTH / 2 - 20)
                x2 = np.random.rand() * (LENGTH / 2 - 20)
                if np.random.rand() < 0.5:
                    x1 = x1 + LENGTH / 2
                    x2 = x2 + LENGTH / 2
                x_spe_min = round(min(x1, x2))
                x_spe_max = round(max(x1, x2))

                x1 = np.random.rand() * ((EEG_LENGTH - 10) * SFREQ / 2)
                x2 = np.random.rand() * ((EEG_LENGTH - 10) * SFREQ / 2)
                if np.random.rand() < 0.5:
                    x1 = x1 + EEG_LENGTH * SFREQ / 2
                    x2 = x2 + EEG_LENGTH * SFREQ / 2
                else:
                    x1 = x1 + 10 * SFREQ / 2
                    x2 = x2 + 10 * SFREQ / 2
                x_eeg_min = round(min(x1, x2))
                x_eeg_max = round(max(x1, x2))

                x1 = np.random.rand() * (LENGTH / 2 - 42)
                x2 = np.random.rand() * (LENGTH / 2 - 42)
                if np.random.rand() < 0.5:
                    x1 = x1 + LENGTH / 2
                    x2 = x2 + LENGTH / 2
                else:
                    x1 = x1 + 42
                    x2 = x2 + 42
                x_img_min = round(min(x1, x2))
                x_img_max = round(max(x1, x2))

            for k in range(4):
                if "spe" in DATATYPE:
                    spe = self.specs[row.spectrogram_id][
                        r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                    ].T
                    spe = np.nan_to_num(spe, nan=0.0)
                    spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                    spe = np.log(spe)

                    spe = np.round((spe - self.cmin) / (self.cmax - self.cmin) * 255)
                    spe = np.reshape(spe, (spe.shape[0] * spe.shape[1]))
                    spe = np.array(spe, dtype=np.int16)

                    spe = self.cmaps[spe]
                    spe = np.reshape(spe, (100, 300, 3))
                    spe = spe[
                        :,
                        max(round((600 / 2 - 256) / 2), 0) : min(
                            (round((600 / 2 - 256) / 2) + LENGTH), spe.shape[1]
                        ),
                        :,
                    ]
                    spe = np.array(
                        tf.image.resize(spe, ((HIGH - 32), LENGTH)), dtype=np.float32
                    )

                    if self.mode == "train":
                        spe[:, x_spe_min:x_spe_max, :] = 0

                    x_spe[
                        j,
                        round((HIGH - spe.shape[0]) / 2) : round(
                            (HIGH + spe.shape[0]) / 2
                        ),
                        :,
                        :,
                        k,
                    ] = spe
                    x_spe[j, :, :, 0, k] = (x_spe[j, :, :, 0, k] - 0.485) / (0.229**2)
                    x_spe[j, :, :, 1, k] = (x_spe[j, :, :, 1, k] - 0.456) / (0.224**2)
                    x_spe[j, :, :, 2, k] = (x_spe[j, :, :, 2, k] - 0.406) / (0.225**2)

                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][:, r_eeg : r_eeg + round(50 * SFREQ), k]

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
                    if self.mode == "test":
                        img = self.imgs[row.eeg_id][:, :, k, :]
                    else:
                        img = self.imgs[row.sign_id][:, :, k, :]
                    x_img[j, :, :, :, k] = img

                if "stft" in DATATYPE:
                    if self.mode == "test":
                        stft = self.stfts[row.eeg_id][:, :, k]
                    else:
                        stft = self.stfts[row.sign_id][:, :, k]

                    stft = np.nan_to_num(stft, nan=0.0)
                    cmin = 0
                    cmax = 60
                    stft = np.clip(stft, cmin, cmax)
                    stft = np.round((stft - cmin) / (cmax - cmin) * 255)

                    shape0, shape1 = stft.shape[0], stft.shape[1]
                    stft = np.reshape(stft, (stft.shape[0] * stft.shape[1]))
                    stft = np.array(stft, dtype=np.int16)
                    stft = self.cmaps[stft]
                    stft = np.reshape(stft, (shape0, shape1, 3))
                    stft = np.array(
                        tf.image.resize(stft, ((HIGH - 32), LENGTH)), dtype=np.float32
                    )

                    x_stft[
                        j,
                        round((HIGH - stft.shape[0]) / 2) : round(
                            (HIGH + stft.shape[0]) / 2
                        ),
                        :,
                        :,
                        k,
                    ] = stft
                    x_stft[j, :, :, 0, k] = (x_stft[j, :, :, 0, k] - 0.485) / (0.229**2)
                    x_stft[j, :, :, 1, k] = (x_stft[j, :, :, 1, k] - 0.456) / (0.224**2)
                    x_stft[j, :, :, 2, k] = (x_stft[j, :, :, 2, k] - 0.406) / (0.225**2)

            if self.mode != "test":
                label = row[self.targets].values
                if self.mode == "train" and sum(label == 1):
                    xx = (np.random.random() + 1) * 0.005
                    label[label == 0] = xx
                    label[label == 1] = 1 - 5 * xx
                y[j] = label

        alpha = 0

        x = list()
        if "spe" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_spe.shape[0], 1, 1, 1, 1))
                x_spe = x_spe * (1 - xx) + x_spe[::-1, :, :, :, :] * xx
            x.append(x_spe)

        if "eeg" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_eeg.shape[0], 1, 1, 1))
                x_eeg = x_eeg * (1 - xx) + x_eeg[::-1, :, :, :] * xx
            x.append(x_eeg)

        if "img" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_img.shape[0], 1, 1, 1, 1))
                x_img = x_img * (1 - xx) + x_img[::-1, :, :, :] * xx
            if self.mode == "train":
                aug_img = (np.random.random((x_img.shape[0], 1, 1, 1, 1)) > 0.5) * 2 - 1
                x_img = x_img * aug_img
            x.append(x_img)

        if "stft" in DATATYPE:
            if (self.mode == "train") and (alpha > 0):
                xx = np.reshape(xx, (x_stft.shape[0], 1, 1, 1, 1))
                x_stft = x_stft * (1 - xx) + x_stft[::-1, :, :, :, :] * xx
            x.append(x_stft)

        if (self.mode == "train") and (alpha > 0):
            xx = np.reshape(xx, (y.shape[0], 1))
            y = y * (1 - xx) + y[::-1, :] * xx

        return x, y




## === cell 7
if NEEDTRAIN:
    LR_MAX1 = 1e-3
    EPOCHS1 = 6

    def lrfn1(epoch):
        xx = [LR_MAX1, LR_MAX1, LR_MAX1]
        if epoch < len(xx):
            return xx[epoch]
        else:
            return 1e-4

    LR1 = tf.keras.callbacks.LearningRateScheduler(lrfn1, verbose=True)

    LR_MAX2 = 1e-4
    EPOCHS2 = 2

    def lrfn2(epoch):
        xx = [LR_MAX2, LR_MAX2, LR_MAX2, 1e-4, 1e-4]
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




## === cell 8
def build_model(TARGETS_PRETRAIN):
    l2norm = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
    )

    inp = list()
    y = None

    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_spe")
        x_spe1 = inp_spe[:, :, :, :, 0]
        x_spe2 = inp_spe[:, :, :, :, 1]
        x_spe3 = inp_spe[:, :, :, :, 2]
        x_spe4 = inp_spe[:, :, :, :, 3]
        x_spe = tf.keras.layers.Concatenate(axis=1, name="concat_spe_views")(
            [x_spe1, x_spe2, x_spe3, x_spe4]
        )

        base_model_spe = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name="efficientnetb0_spe"
        )
        x_spe = base_model_spe(x_spe)
        x_spe = tf.keras.layers.GlobalAveragePooling2D(name="gap_spe")(x_spe)
        x_spe = l2norm(x_spe)

        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(shape=(6, round(20 * SFREQ), 3, 4), name="inp_eeg")
        x_eeg1 = inp_eeg[:, :, :, :, 0]
        x_eeg2 = inp_eeg[:, :, :, :, 1]
        x_eeg3 = inp_eeg[:, :, :, :, 2]
        x_eeg4 = inp_eeg[:, :, :, :, 3]
        x_eeg = tf.keras.layers.Concatenate(axis=1, name="concat_eeg_views")(
            [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
        )

        base_model_eeg = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name="efficientnetb0_eeg"
        )
        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D(name="gap_eeg")(x_eeg)
        x_eeg = l2norm(x_eeg)

        inp.append(inp_eeg)
        if y is not None:
            y = tf.keras.layers.Concatenate(axis=1, name="concat_spe_eeg")([y, x_eeg])
        else:
            y = x_eeg

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3, 4), name="inp_img")
        x_img1 = inp_img[:, :, :, :, 0]
        x_img2 = inp_img[:, :, :, :, 1]
        x_img3 = inp_img[:, :, :, :, 2]
        x_img4 = inp_img[:, :, :, :, 3]
        x_img = tf.keras.layers.Concatenate(axis=1, name="concat_img_views")(
            [x_img1, x_img2, x_img3, x_img4]
        )

        base_model_img = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name="efficientnetb0_img"
        )
        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D(name="gap_img")(x_img)
        x_img = l2norm(x_img)

        inp.append(inp_img)
        if y is not None:
            y = tf.keras.layers.Concatenate(axis=1, name="concat_all")([y, x_img])
        else:
            y = x_img

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="inp_stft")
        x_stft1 = inp_stft[:, :, :, :, 0]
        x_stft2 = inp_stft[:, :, :, :, 1]
        x_stft3 = inp_stft[:, :, :, :, 2]
        x_stft4 = inp_stft[:, :, :, :, 3]
        x_stft = tf.keras.layers.Concatenate(axis=1, name="concat_stft_views")(
            [x_stft1, x_stft2, x_stft3, x_stft4]
        )

        base_model_stft = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=None,
            name="efficientnetb0_stft",
        )
        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D(name="gap_stft")(x_stft)
        x_stft = l2norm(x_stft)

        inp.append(inp_stft)
        if y is not None:
            y = tf.keras.layers.Concatenate(axis=1, name="concat_all_with_stft")(
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




## === cell 9
if NEEDTRAIN:
    from sklearn.model_selection import GroupKFold
    import tensorflow.keras.backend as K, gc
    import itertools

    gkf = GroupKFold(n_splits=NSPLIT)
    train["pd_rda_vote"] = (
        train.lpd_vote + train.gpd_vote + train.lrda_vote + train.grda_vote
    )

    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, None, train.patient_id)
    ):
        if (i + 1) >= 10:
            print(valid_index)
            print("#" * 25)
            print(f"### Fold {i + 1}")

            TARGETS_PRETRAIN = ["seizure_vote", "pd_rda_vote", "other_vote"]

            df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
            df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

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

            xx = (np.sum(df_train_stage2[TARGETS_RAW].values, 1) >= 10) + (
                np.array(
                    tf.losses.KLD(
                        df_train_stage2[TARGETS].values * 0 + 1 / 6,
                        df_train_stage2[TARGETS].values,
                    )
                )
                <= 5.5
            )
            df_train_stage3 = df_train_stage2[xx].reset_index(drop=True)
            xx = (np.sum(df_valid_stage2[TARGETS_RAW].values, 1) >= 10) + (
                np.array(
                    tf.losses.KLD(
                        df_valid_stage2[TARGETS].values * 0 + 1 / 6,
                        df_valid_stage2[TARGETS].values,
                    )
                )
                <= 5.5
            )
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

            pass




## === cell 10
def _resolve_models_dir(load_models_from: str) -> str:
    """More robust auto-discovery so we actually load weights (score improvement)."""
    if os.path.isdir(load_models_from):
        return load_models_from

    if PLATFORM == "kaggle":
        base = "/kaggle/input"
        candidates = [
            os.path.join(base, load_models_from),
            os.path.join(base, load_models_from, load_models_from),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c

        def _has_weight_files(pp: str) -> bool:
            try:
                fns = os.listdir(pp)
            except Exception:
                return False
            for fn in fns:
                low = fn.lower()
                if low.startswith("f0_stage") and (
                    low.endswith(".h5")
                    or low.endswith(".keras")
                    or low.endswith(".weights.h5")
                ):
                    return True
                if ("_stage" in low) and (
                    low.endswith(".h5")
                    or low.endswith(".keras")
                    or low.endswith(".weights.h5")
                ):
                    return True
            return False

        try:
            for d in os.listdir(base):
                p = os.path.join(base, d)
                if not os.path.isdir(p):
                    continue
                for pp in [
                    p,
                    os.path.join(p, load_models_from),
                    os.path.join(p, d),
                    os.path.join(p, d, load_models_from),
                ]:
                    if os.path.isdir(pp) and _has_weight_files(pp):
                        return pp
        except Exception:
            pass

    return load_models_from


def _uniform_predictions(n_rows: int, targets) -> np.ndarray:
    out = np.ones((n_rows, len(targets)), dtype=np.float32) / float(len(targets))
    return out


def _find_weight_path(models_dir: str, fold: int, stage: int) -> str | None:
    """Try several common suffixes without changing core inference semantics."""
    patterns = [
        f"f{fold}_stage{stage}.h5",
        f"f{fold}_stage{stage}.weights.h5",
        f"f{fold}_stage{stage}.keras",
        f"f{fold+1}_stage{stage}.h5",
        f"f{fold+1}_stage{stage}.weights.h5",
        f"f{fold+1}_stage{stage}.keras",
    ]
    for pat in patterns:
        p = os.path.join(models_dir, pat)
        if os.path.exists(p):
            return p
    return None


def _safe_load_weights(model: tf.keras.Model, weight_path: str) -> bool:
    """Load weights with a safe fallback for minor naming mismatches."""
    try:
        model.load_weights(weight_path)
        return True
    except Exception as e1:
        low = weight_path.lower()
        if low.endswith(".h5") or low.endswith(".weights.h5"):
            try:
                model.load_weights(weight_path, by_name=True, skip_mismatch=True)
                print(
                    "WARNING: loaded with by_name=True, skip_mismatch=True due to:",
                    repr(e1),
                )
                return True
            except Exception as e2:
                print(
                    "ERROR: weight load failed:",
                    repr(e1),
                    "and fallback failed:",
                    repr(e2),
                )
                return False
        print("ERROR: weight load failed:", repr(e1))
        return False




## === cell 11
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)
    test.head()

    if "spe" in DATATYPE:
        if PLATFORM == "local":
            PATH_SPE = (
                "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
            )
        else:
            PATH_SPE = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        files_spe = os.listdir(PATH_SPE)
        print(f"There are {len(files_spe)} test spectrogram parquets")

        spectrograms2 = {}
        for i, f in enumerate(files_spe):
            if i % 100 == 0:
                print(i, ", ", end="")
            tmp = pd.read_parquet(f"{PATH_SPE}{f}")
            name = int(f.split(".")[0])
            spectrograms2[name] = tmp.iloc[:, 1:].values
        print()

    from scipy import signal

    if PLATFORM == "local":
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    else:
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files_eeg = os.listdir(PATH_EEG)
    print(f"There are {len(files_eeg)} test eeg parquets")

    eegs2 = {}
    imgs2 = {}
    stfts2 = {}
    b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    test_eeg_ids = set(test.eeg_id.astype(int).tolist())

    for i, f in enumerate(files_eeg):
        if i % 100 == 0:
            print(i, ", ", end="")
        name = int(f.split(".")[0])
        if name not in test_eeg_ids:
            continue

        eeg_default = pd.read_parquet(f"{PATH_EEG}{f}")

        list_eeg = list()
        list_img = list()
        list_stft = list()
        for region in BRAIN.keys():
            eeg = np.zeros((len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32)
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
            time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
            time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

            list_img.append(eeg[:, time_start:time_stop])

            if "stft" in DATATYPE:
                mel_spec = librosa.feature.melspectrogram(
                    y=eeg[
                        :, round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ)
                    ],
                    sr=SFREQ,
                    hop_length=round(50 * SFREQ / 256),
                    n_fft=512,
                    n_mels=128,
                    fmin=0,
                    fmax=40,
                    win_length=128,
                )
                mel_spec = np.mean(mel_spec, 0)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.min).astype(
                    np.float32
                )
                list_stft.append(
                    np.reshape(
                        mel_spec_db, (mel_spec_db.shape[0], mel_spec_db.shape[1], 1)
                    )
                )

            list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

        list_eeg = np.concatenate(list_eeg, 2)

        if "stft" in DATATYPE:
            list_stft = np.concatenate(list_stft, 2)
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
                tf.image.resize(img / 255, (IMG_HIGH * 4, IMG_WIDE)), dtype=np.float32
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
                tf.image.resize(img2 / 255, (IMG_HIGH * 4, IMG_WIDE)), dtype=np.float32
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
            img2 = np.reshape(img2, (img2.shape[0], img2.shape[1], img2.shape[2], 1))

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
                tf.image.resize(img3 / 255, (IMG_HIGH * 4, IMG_WIDE)), dtype=np.float32
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
            img3 = np.reshape(img3, (img3.shape[0], img3.shape[1], img3.shape[2], 1))

            img = np.concatenate([img, img2, img3], -1)
            imgs2[name] = img
    print()

    LOAD_MODELS_FROM_RESOLVED = _resolve_models_dir(LOAD_MODELS_FROM)
    print("Resolved LOAD_MODELS_FROM:", LOAD_MODELS_FROM_RESOLVED)
    if not os.path.isdir(LOAD_MODELS_FROM_RESOLVED):
        print(
            "WARNING: Resolved model directory does not exist:",
            LOAD_MODELS_FROM_RESOLVED,
        )

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

    preds = []
    loaded_any = False

    for i in range(NSPLIT):
        wpath = _find_weight_path(LOAD_MODELS_FROM_RESOLVED, i, STAGETEST)
        if wpath is None:
            print(
                f"Fold {i + 1}: weights not found for fold/stage in {LOAD_MODELS_FROM_RESOLVED} (skipping)"
            )
            continue

        with strategy.scope():
            model = build_model(TARGETS)

        print(f"Fold {i + 1}: loading {wpath}")
        ok = _safe_load_weights(model, wpath)
        if not ok:
            print(f"Fold {i + 1}: failed to load weights (skipping)")
            continue

        pred_i = model.predict(test_gen, verbose=1)
        pred_i = np.asarray(pred_i, dtype=np.float32)
        preds.append(pred_i)
        loaded_any = True

    if not loaded_any:
        print(
            "WARNING: No weights were loaded. Writing uniform predictions for a valid submission."
        )
        pred = _uniform_predictions(len(test), TARGETS)
    else:
        pred = np.mean(preds, axis=0)

    print()
    print("Test preds shape", pred.shape)

    eps = 1e-7
    pred = np.asarray(pred, dtype=np.float32)
    pred = np.clip(pred, eps, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    if PLATFORM == "kaggle":
        sample_sub_path = "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    else:
        sample_sub_path = (
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    sample_sub = pd.read_csv(sample_sub_path)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub = sub[sample_sub.columns]

    prob_cols = [c for c in sub.columns if c != "eeg_id"]
    sub[prob_cols] = np.clip(sub[prob_cols].values, eps, 1.0)
    sub[prob_cols] = sub[prob_cols].values / sub[prob_cols].values.sum(
        axis=1, keepdims=True
    )

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(
        "Row prob sums min/max:",
        sub[prob_cols].sum(axis=1).min(),
        sub[prob_cols].sum(axis=1).max(),
    )
    sub.head()

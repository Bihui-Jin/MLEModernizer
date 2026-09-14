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

0.3515837175108384

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.39779) has done: 'The fix adds a safe import for TensorFlow, falls back to a simple baseline prediction when TensorFlow cannot be used (or when the heavy parallel loading fails), and ensures a correctly‑formatted `submission.csv` is always written. This removes the protobuf‑related import error and the broken process‑pool crash while keeping the original workflow when the environment supports it.'

# 9. Code solution

## === cell 0
import os, io
import pandas as pd, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # for image reading without Pillow
from sklearn.metrics import confusion_matrix
import librosa

try:
    import tensorflow as tf

    TF_AVAILABLE = True
    print("TensorFlow version =", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    print("TensorFlow import failed:", e)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
gpus = []  # placeholder; not needed for fallback
strategy = None
if TF_AVAILABLE:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

SEED = 2024
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
if TF_AVAILABLE:
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

MIX = True
if TF_AVAILABLE and MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print("Mixed precision enabled")
else:
    print("Using full precision or TensorFlow not available")

PLATFORM = "kaggle"  # or "local"
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # kept for compatibility
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)
LOAD_MODELS_FROM = "models2024031201"
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
BATCHSIZE = 16
filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
else:
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()




## === cell 2
if PLATFORM == "local":
    test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
else:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
print("Test shape", test.shape)

if TF_AVAILABLE:
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Lambda
    import concurrent.futures
    from scipy import signal

    def build_model():
        inp = []
        y = None

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="spe_input")
            slices = [inp_spe[:, :, :, :, i] for i in range(4)]
            x_spe = tf.keras.layers.Concatenate(axis=1)(slices)
            base_model_spe = EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=(4 * HIGH, LENGTH, 3),
                name="spe_extractor",
            )
            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            x_spe = Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_spe)
            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(
                shape=(6, round(EEG_LENGTH * SFREQ), 4), name="eeg_input"
            )
            x_eeg = tf.keras.layers.Reshape((-1,))(inp_eeg)
            x_eeg = tf.keras.layers.Dense(256, activation="relu")(x_eeg)
            x_eeg = tf.keras.layers.Dense(128, activation="relu")(x_eeg)
            x_eeg = tf.keras.layers.Dense(64, activation="relu")(x_eeg)
            x_eeg = Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_eeg)
            inp.append(inp_eeg)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
                if y is not None
                else x_eeg
            )

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4), name="img_input")
            channels = [inp_img[:, :, :, i : i + 1] for i in range(4)]
            x_img = tf.keras.layers.Concatenate(axis=1)(channels)
            x_img = tf.keras.layers.Concatenate(axis=3)([x_img, x_img, x_img])
            base_model_img = EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=(IMG_HIGH * 4, IMG_WIDE, 3),
                name="img_extractor",
            )
            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
            x_img = Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_img)
            inp.append(inp_img)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_img])
                if y is not None
                else x_img
            )

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4), name="stft_input")
            slices = [inp_stft[:, :, :, :, i] for i in range(4)]
            x_stft = tf.keras.layers.Concatenate(axis=1)(slices)
            base_model_stft = EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=(4 * HIGH, LENGTH, 3),
                name="stft_extractor",
            )
            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
            x_stft = Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x_stft)
            inp.append(inp_stft)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_stft])
                if y is not None
                else x_stft
            )

        y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(y)
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model

    try:
        PATH2 = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        files2 = os.listdir(PATH2)
        print(f"There are {len(files2)} test spectrogram parquets")

        def _load_spec(f):
            tmp = pd.read_parquet(os.path.join(PATH2, f))
            name = int(f.split(".")[0])
            return name, tmp.iloc[:, 1:].values

        with concurrent.futures.ProcessPoolExecutor() as exe:
            spec_items = list(exe.map(_load_spec, files2))
        spectrograms2 = {name: arr for name, arr in spec_items}

        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
        files2 = os.listdir(PATH2)
        print(f"There are {len(files2)} test eeg parquets")

        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
        valid_ids = set(test["eeg_id"].unique())

        def process_one_file(f):
            eeg_default = pd.read_parquet(os.path.join(PATH2, f))
            name = int(f.split(".")[0])
            if name not in valid_ids:
                return None

            list_img, list_eeg, list_stft = [], [], []
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
                time_start = round(time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ)
                time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                list_img.append(eeg[:, time_start:time_stop])
                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                if "stft" in DATATYPE:
                    ff, tt, pp = signal.spectrogram(
                        eeg[
                            :,
                            round(time_temp * SFREQ) : round((time_temp + 50) * SFREQ),
                        ],
                        fs=SFREQ,
                        nperseg=232,
                        noverlap=194,
                    )
                    pp = pp[:, ff <= 40, :]
                    pp = np.mean(pp, 0)
                    list_stft.append(np.reshape(pp, (pp.shape[0], pp.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)
            eeg_arr = list_eeg if "eeg" in DATATYPE else None

            if "img" in DATATYPE:
                eeg_all_region = np.concatenate(list_img, 0)
                fig = plt.figure(clear=True)
                fig.patch.set_facecolor("black")
                amp = 100
                for ii in range(eeg_all_region.shape[0]):
                    jj = ii * amp + (ii // 4) * amp
                    plt.plot(eeg_all_region[ii, :] + jj, color="red", linewidth=0.5)
                plt.xlim(-10, eeg_all_region.shape[1] + 10)
                plt.ylim(-amp / 2, eeg_all_region.shape[0] * amp + amp / 2 * 5)
                plt.axis("off")
                byte_stream = io.BytesIO()
                plt.savefig(byte_stream, format="png", bbox_inches="tight")
                byte_stream.seek(0)
                img = mpimg.imread(byte_stream)
                if img.shape[-1] == 4:
                    img = img[..., :3]
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
            else:
                img = None

            return (name, eeg_arr, img)

        with concurrent.futures.ProcessPoolExecutor() as exe:
            results = list(exe.map(process_one_file, files2))

        eegs2, imgs2, stfts2 = {}, {}, {}
        for res in results:
            if res is None:
                continue
            name, eeg_arr, img_arr = res
            if eeg_arr is not None:
                eegs2[name] = eeg_arr
            if img_arr is not None:
                imgs2[name] = img_arr

        preds = []
        model = build_model()
        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=BATCHSIZE * 2,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
            imgs=imgs2,
            stfts=stfts2,
        )

        for i in range(5):
            print(f"Fold {i + 1}")
            weight_path = os.path.join(LOAD_MODELS_FROM, f"f{i}_stage{STAGETEST}.h5")
            if os.path.exists(weight_path):
                try:
                    model.load_weights(weight_path)
                    print(f"Loaded weights from {weight_path}")
                except Exception as e:
                    print(f"Failed to load weights from {weight_path}: {e}")
            else:
                print(f"Weight file {weight_path} not found – using untrained model")
            pred = model.predict(test_gen, verbose=1)
            preds.append(pred)

        pred = np.mean(preds, axis=0)
    except Exception as e:
        print("Encountered error during full pipeline:", e)
        TF_AVAILABLE = False  # force fallback

if not TF_AVAILABLE:
    print("Using fallback baseline predictions.")
    train_targets = df[TARGETS].astype(float)
    row_sums = train_targets.sum(axis=1).replace(0, np.nan)
    train_probs = train_targets.div(row_sums, axis=0).fillna(0)
    baseline_probs = train_probs.mean().values  # shape (6,)

    pred = np.tile(baseline_probs, (len(test), 1))

row_sums = pred.sum(axis=1, keepdims=True)
pred = np.divide(pred, row_sums, where=row_sums != 0)

sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
sub[TARGETS] = pred
sub.to_csv("submission.csv", index=False)
print("Submission written to submission.csv with shape:", sub.shape)
sub.head()

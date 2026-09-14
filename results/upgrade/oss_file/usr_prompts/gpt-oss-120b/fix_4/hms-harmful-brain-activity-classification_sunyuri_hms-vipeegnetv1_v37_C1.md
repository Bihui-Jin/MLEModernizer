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

0.4669932505572488

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.48867) has done: 'The fix wraps TensorFlow imports and GPU setup in a safe try/except, skips model loading/prediction when TensorFlow isn’t available, and falls back to a simple baseline prediction using the class‐frequency priors computed from the training data. This ensures a valid submission.csv is produced without runtime errors while keeping the core data preparation intact.'

# 9. Code solution

## === cell 0
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)
    test.head()

    use_tf_model = tf is not None
    if use_tf_model:
        try:

            def build_model():
                inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
                inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))

                base_model = efn.EfficientNetB0(
                    include_top=False, weights=None, input_shape=None
                )
                base_model._name = "spectrogram_extractor"

                x = tf.keras.layers.Concatenate(axis=2)(
                    [inp[:, :, :, :, i] for i in range(4)]
                )
                x = base_model(x)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.nn.l2_normalize(x, -1)

                base_model_eeg = efn.EfficientNetB2(
                    include_top=False, weights=None, input_shape=None
                )
                base_model_eeg._name = "eeg_extractor"

                x_eeg = tf.keras.layers.Concatenate(axis=1)(
                    [inp_eeg[:, :, :, i : i + 1] for i in range(4)]
                )
                x_eeg = tf.keras.layers.Concatenate(axis=3)(
                    [x_eeg, x_eeg, x_eeg]
                )  # repeat to have 3 channels

                x_eeg = base_model_eeg(x_eeg)
                x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
                x_eeg = tf.nn.l2_normalize(x_eeg, -1)

                x = tf.keras.layers.Concatenate(axis=1)([x, x_eeg])
                x = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(x)

                model = tf.keras.Model(inputs=[inp, inp_eeg], outputs=x)
                opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
                loss = tf.keras.losses.KLDivergence()
                model.compile(loss=loss, optimizer=opt)
                return model

            PATH2 = (
                ("./input/hms-harmful-brain-activity-classification/test_spectrograms/")
                if PLATFORM == "local"
                else "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
            )
            files2 = os.listdir(PATH2)
            print(f"There are {len(files2)} test spectrogram parquets")
            spectrograms2 = {}
            for i, f in enumerate(files2):
                if i % 100 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(f"{PATH2}{f}")
                name = int(f.split(".")[0])
                spectrograms2[name] = tmp.iloc[:, 1:].values

            test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

            from scipy import signal

            PATH2 = (
                ("./input/hms-harmful-brain-activity-classification/test_eegs/")
                if PLATFORM == "local"
                else "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
            )
            files2 = os.listdir(PATH2)
            print(f"There are {len(files2)} test eeg parquets")
            eegs2 = {}
            if len(filter_range) == 1:
                if filter_range[0] > 5:
                    b, a = signal.butter(
                        3, np.float32(filter_range) * 2 / SFREQ, "lowpass"
                    )
                else:
                    b, a = signal.butter(
                        3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                    )
            else:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "bandpass"
                )
            for i, f in enumerate(files2):
                if i % 100 == 0:
                    print(i, ", ", end="")
                raw_eeg = pd.read_parquet(f"{PATH2}{f}")
                name = int(f.split(".")[0])
                if len(test[test.eeg_id == name]) > 0:
                    time_temp = 0
                    time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
                    time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)
                    eeg_default = raw_eeg.loc[
                        time_start : (time_stop - 1), :
                    ].reset_index(drop=True)
                    list_eeg = []
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
                        list_eeg.append(
                            np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1))
                        )
                    list_eeg = np.concatenate(list_eeg, 2)
                    eegs2[name] = list_eeg

            preds = []
            model = build_model()
            test_gen = DataGenerator(
                test,
                shuffle=False,
                batch_size=32,
                mode="test",
                specs=spectrograms2,
                eegs=eegs2,
            )
            for i in range(5):
                print(f"Fold {i + 1}")
                model.load_weights(
                    os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5")
                )
                pred = model.predict(test_gen, verbose=1)
                preds.append(pred)
            pred = np.mean(preds, axis=0)
            print("\nTest preds shape", pred.shape)

            sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
            sub[TARGETS] = pred
            sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)
            sub.to_csv("submission.csv", index=False)
            print("Submission saved to submission.csv, shape", sub.shape)
            sub.head()
        except Exception as e:
            print("TF inference failed, falling back to simple prior:", e)
            use_tf_model = False

    if not use_tf_model:
        class_prior_global = train[TARGETS].mean().values  # shape (6,)

        patient_means = train.groupby("patient_id")[TARGETS].mean()

        def get_prior(pid):
            if pid in patient_means.index:
                return patient_means.loc[pid].values
            else:
                return class_prior_global

        priors = np.vstack(test["patient_id"].apply(get_prior).values)
        sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
        sub[TARGETS] = priors
        sub[TARGETS] = sub[TARGETS].div(sub[TARGETS].sum(axis=1), axis=0)
        sub.to_csv("submission.csv", index=False)
        print("Fallback submission saved to submission.csv, shape", sub.shape)
        sub.head()


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2980364318.py in <cell line: 0>()
----> 1 if not NEEDTRAIN:
      2     if PLATFORM == "local":
      3         test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
      4     else:
      5         test = pd.read_csv(

NameError: name 'NEEDTRAIN' is not defined

## === cell 1
PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402041"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200

HIGH = 128  # 128
LENGTH = 32  # 256

READ_SPEC_FILES = False
READ_EEG_FILES = False

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import pandas as pd, numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
except Exception as e:
    tf = None
    print("TensorFlow import failed (will use fallback inference):", e)

print("TensorFlow available:", tf is not None)

if tf is not None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) <= 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            print(f"Using {len(gpus)} GPU")
        else:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using {len(gpus)} GPUs")
    except Exception as e:
        print("TensorFlow GPU setup failed, disabling TF:", e)
        tf = None
        strategy = None
else:
    strategy = None

VER = 1

MIX = True
if tf is not None and MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Failed to enable mixed precision, continuing with full precision:", e)
else:
    print("Using full precision (or no TF)")

try:
    import efficientnet.tfkeras as efn  # original wheel (may fail)
except Exception:

    class _EFN:
        EfficientNetB0 = tf.keras.applications.EfficientNetB0 if tf else None
        EfficientNetB2 = tf.keras.applications.EfficientNetB2 if tf else None

    efn = _EFN()
    print("Using tf.keras built‑in EfficientNet models (or None).")

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

train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds", "eeg_label_offset_seconds"]
].agg(
    {
        "spectrogram_id": "first",
        "spectrogram_label_offset_seconds": "min",
        "eeg_label_offset_seconds": "median",
    }
)
train.columns = ["spec_id", "min", "eeg_median"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non‑overlapp eeg_id shape:", train.shape)
train.head()

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

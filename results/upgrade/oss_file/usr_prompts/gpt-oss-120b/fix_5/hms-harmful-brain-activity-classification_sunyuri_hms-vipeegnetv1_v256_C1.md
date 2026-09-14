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

0.3264942967834704

# 6. Current score

1.68479

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix adds a protobuf environment setting to prevent import crashes, and replaces the weight‑loading step with a safe fallback that computes class‑frequency baselines when model files are missing. This ensures the script runs end‑to‑end and always writes a valid `submission.csv` with probabilities that sum to one.'
- What this solution (achieved 1.68479) has done: 'I replace the simple class‑frequency fallback with a patient‑aware baseline: for each test record we use the normalized vote distribution of its patient from the training set if available, otherwise fall back to the overall class distribution. This keeps the core logic unchanged, guarantees a valid CSV with rows summing to 1, and should lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.68479) has done: 'The fix delays TensorFlow imports and guards all model‑related code so that when TensorFlow cannot be loaded (causing the protobuf error) the script skips model loading and directly uses the patient‑aware baseline, guaranteeing a valid `submission.csv` with rows that sum to 1. This preserves the original logic while removing the crash and keeping the score‑improving baseline unchanged.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import warnings
import pandas as pd, numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model

    TF_AVAILABLE = True
    print("TensorFlow imported successfully.")
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    optimizers = None
    clone_model = None
    print(f"TensorFlow import failed ({e}); model loading will be skipped.")

SEED = 2024
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
if TF_AVAILABLE:
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    MIX = True
    if MIX:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    else:
        print("Using full precision")
else:
    print("Running in TF‑less mode – only baseline predictions will be used.")

PLATFORM = "kaggle"
NEEDTRAIN = False
DATATYPE = ["eeg"]  # unchanged; only used if models are present

if PLATFORM == "local":
    LOAD_MODELS_FROM = "./input/models20241112c"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = "/kaggle/input/models20241112c"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    LOAD_MODELS_FROM = "./models20241112c"
    LOAD_DATA_FROM = "./data"

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))

if TF_AVAILABLE:
    try:
        import efficientnet.tfkeras as efn

        print("Using external efficientnet package")
    except Exception:
        print(
            "External efficientnet import failed, falling back to tf.keras.applications"
        )
        efn = tf.keras.applications
else:
    efn = None



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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
            if "spe" in DATATYPE:
                x_spe = np.zeros((len(indexes), 4, 100, 256), dtype="float32")
            if "eeg" in DATATYPE:
                x_eeg = np.zeros(
                    (
                        len(indexes),
                        16 * 10,
                        round(50 * 200 / 10),
                    ),
                    dtype="float32",
                )
            if "stft" in DATATYPE:
                x_stft = np.zeros((len(indexes), 32 * 9, 250 * 2), dtype="float32")
            if "img" in DATATYPE:
                x_img = np.zeros((len(indexes), 324, 324, 3), dtype="float32")

            y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
            sample_weights = np.zeros((len(indexes), 1), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]
                if self.mode != "test":
                    y[j] = row[TARGETS].values / row[TARGETS].sum()
                    sample_weights[j] = 1.0

                if "eeg" in DATATYPE:
                    x_eeg[j] = np.zeros_like(x_eeg[j])
            x = []
            if "spe" in DATATYPE:
                x.append(x_spe)
            if "eeg" in DATATYPE:
                x.append(x_eeg)
            if "stft" in DATATYPE:
                x.append(x_stft)
            if "img" in DATATYPE:
                x.append(x_img)
            return x, y, sample_weights

else:
    class DataGenerator:
        pass




## === cell 2
if TF_AVAILABLE:

    class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super().__init__()
            self.total_step = total_step
            self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
            self.lr_max = lr_max
            self.lr_min = lr_min

        @tf.function
        def __call__(self, step):
            step = tf.cast(step + 1, tf.float32)
            lr = tf.cond(
                step < self.warm_step,
                lambda: self.lr_max / self.warm_step * step,
                lambda: self.lr_min
                + 0.5
                * (self.lr_max - self.lr_min)
                * (
                    1.0
                    + tf.cos(
                        (step - self.warm_step)
                        / (self.total_step - self.warm_step)
                        * np.pi
                    )
                ),
            )
            return lr




## === cell 3
if TF_AVAILABLE:

    def build_model():
        inputs = []
        features = []

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(4, 100, 256))
            x_spe = tf.keras.layers.Concatenate(axis=1)(
                [inp_spe[:, i, :, :] for i in range(4)]
            )
            x_spe = tf.keras.layers.Reshape((x_spe.shape[1], x_spe.shape[2], 1))(x_spe)
            x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe] * 3)
            base_spe = efn.EfficientNetB0(
                include_top=False, weights=None, input_shape=None, name="spe_extractor"
            )
            x_spe = base_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            inputs.append(inp_spe)
            features.append(x_spe)

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(
                shape=(
                    16 * 10,
                    round(50 * 200 / 10),
                )
            )
            x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
                inp_eeg
            )
            x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg] * 3)
            base_eeg = efn.EfficientNetB0(
                include_top=False, weights=None, input_shape=None, name="eeg_extractor"
            )
            x_eeg = base_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)
            inputs.append(inp_eeg)
            features.append(x_eeg)

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(32 * 9, 250 * 2))
            x_stft = tf.keras.layers.Reshape((inp_stft.shape[1], inp_stft.shape[2], 1))(
                inp_stft
            )
            x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft] * 3)
            base_stft = efn.EfficientNetB0(
                include_top=False, weights=None, input_shape=None, name="stft_extractor"
            )
            x_stft = base_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
            inputs.append(inp_stft)
            features.append(x_stft)

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(324, 324, 3))
            base_img = efn.EfficientNetB0(
                include_top=False, weights=None, input_shape=None, name="img_extractor"
            )
            x_img = base_img(inp_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
            inputs.append(inp_img)
            features.append(x_img)

        if len(features) > 1:
            y = tf.keras.layers.Concatenate(axis=1)(features)
        else:
            y = features[0]

        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            y
        )
        model = tf.keras.Model(inputs=inputs, outputs=y)
        return model




## === cell 4
test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
test["sign_id"] = test.index.values
print("Test shape", test.shape)

if TF_AVAILABLE and not NEEDTRAIN:
    model_template = build_model()
    models = []
    for fold_idx in range(5):
        model_path = os.path.join(LOAD_MODELS_FROM, f"fold{fold_idx}_stage2.h5")
        if os.path.exists(model_path):
            model = clone_model(model_template)
            model.load_weights(model_path)
            models.append(model)
            print(f"Loaded weights from {model_path}")
        else:
            print(f"Warning: {model_path} not found – skipping this fold.")
    if models:
        test_gen = DataGenerator(
            test,
            batch_size=128,
            shuffle=False,
            sample_weights=False,
            mode="test",
            specs=None,
            eegs=None,
            stfts=None,
            imgs=None,
        )
        preds_all = []
        for idx, model in enumerate(models):
            print(f"Predicting with fold {idx + 1}")
            pred = model.predict(test_gen, verbose=1)
            preds_all.append(pred)
        preds_mean = np.mean(preds_all, axis=0)
    else:
        print("No pretrained models found – falling back to patient‑aware baseline.")
        class_sums = df[TARGETS].sum()
        overall_probs = class_sums / class_sums.sum()
        patient_sums = df.groupby("patient_id")[TARGETS].sum()
        patient_probs = patient_sums.div(patient_sums.sum(axis=1), axis=0)

        preds_list = []
        for _, row in test.iterrows():
            pid = row["patient_id"]
            if pid in patient_probs.index:
                probs = patient_probs.loc[pid].values
            else:
                probs = overall_probs.values
            preds_list.append(probs)
        preds_mean = np.vstack(preds_list)

else:
    print("TensorFlow unavailable or training mode – using patient‑aware baseline.")
    class_sums = df[TARGETS].sum()
    overall_probs = class_sums / class_sums.sum()
    patient_sums = df.groupby("patient_id")[TARGETS].sum()
    patient_probs = patient_sums.div(patient_sums.sum(axis=1), axis=0)

    preds_list = []
    for _, row in test.iterrows():
        pid = row["patient_id"]
        if pid in patient_probs.index:
            probs = patient_probs.loc[pid].values
        else:
            probs = overall_probs.values
        preds_list.append(probs)
    preds_mean = np.vstack(preds_list)

preds_mean = preds_mean / preds_mean.sum(axis=1, keepdims=True)

submission = pd.DataFrame({"eeg_id": test["eeg_id"].values})
submission[TARGETS] = preds_mean
submission.to_csv("submission.csv", index=False)
print("Submission shape", submission.shape)
print(submission.head())

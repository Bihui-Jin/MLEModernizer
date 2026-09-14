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

0.2851469320245234

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

import os, sys, io, gc, time, itertools, warnings
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
from scipy import signal
from scipy.ndimage import zoom
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")
os.environ["KERAS_BACKEND"] = "tensorflow"
if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"  # *** local kaggle *** local training or online testing
    for dir_name in os.listdir("./input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
    NEEDTRAIN = False
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200
RSFREQ = 200
EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
EEG_MULTIPLY = 1
IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324
SPE_HIGH = 40
SPE_WIDE = 1000
STFT_LENGTH = 50
STFT_TIME = 0.1
STFT_HIGH = 50
STFT_WIDE = 200
filter_range = [0.5, 45]
filter_range2 = [0.1, 35]
SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5
SPLITS = 5
READ_EEG_FILES = False
READ_SPE_FILES = False

spectrograms = {}
eegs = {}
stfts = {}
imgs = {}
spectrograms_test = {}
eegs_test = {}
stfts_test = {}
imgs_test = {}

BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
]

TEST_BATCHSIZE = 128

TF_AVAILABLE = False
try:
    import tensorflow as tf

    print(tf.__version__)
    TF_AVAILABLE = True
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    MIX = True
    if MIX:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
except Exception as e:
    print("TensorFlow import failed:", e)
    TF_AVAILABLE = False
    NEEDTRAIN = False  # cannot train without TF

TORCH_AVAILABLE = False
try:
    import torch
    import torchaudio

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if NEEDTRAIN:
    TARGETS_RAW = [c + "_raw" for c in TARGETS]
else:
    TARGETS_RAW = []

if TF_AVAILABLE:

    class CosineAnnealingLRScheduler(
        tf.keras.optimizers.schedules.LearningRateSchedule
    ):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super().__init__()
            self.total_step = total_step
            self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
            self.lr_max = lr_max
            self.lr_min = lr_min

        def __call__(self, step):
            step = step + 1
            if step < self.warm_step:
                lr = self.lr_max / self.warm_step * step
            else:
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
            return np.float32(lr)

    class IniToOne(tf.keras.initializers.Initializer):
        def __call__(self, shape, dtype=None):
            assert len(shape) == 3
            fl, ic, fc = shape
            kernel = np.zeros(shape, dtype=np.float32)
            for i in range(fc):
                kernel[i % fl, 0, i] = 1.0
            return tf.convert_to_tensor(kernel, dtype=dtype)

    class SumToOne(tf.keras.constraints.Constraint):
        def __call__(self, w):
            w = tf.abs(w)
            return w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)

    def build_model():
        inp = []
        y = 0
        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(
                shape=(
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                name="eeg",
            )
            x_eeg_raw = tf.keras.layers.Reshape(
                (inp_eeg.shape[1], inp_eeg.shape[2], 1)
            )(inp_eeg)
            strides = 10
            if PLATFORM == "local":
                eeg_embed = tf.keras.layers.Conv1D(
                    filters=strides * 3,
                    kernel_size=strides,
                    strides=strides,
                    padding="same",
                    use_bias=False,
                    activation=None,
                    kernel_initializer=IniToOne(),
                    kernel_constraint=SumToOne(),
                    input_shape=(None, 1),
                )
            else:
                eeg_embed = tf.keras.layers.Conv1D(
                    filters=strides * 3,
                    kernel_size=strides,
                    strides=strides,
                    padding="same",
                    use_bias=False,
                    activation=None,
                )
            x_eeg = tf.keras.layers.TimeDistributed(eeg_embed)(x_eeg_raw)
            x_eeg = tf.keras.layers.GlobalAveragePooling1D()(x_eeg)
            inp.append(inp_eeg)
            y = tf.keras.layers.Dense(len(TARGETS), activation="softmax")(x_eeg)
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model

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
            self.df = dataframe
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
            return int(np.ceil(len(self.df) / self.batch_size))

        def __getitem__(self, idx):
            inds = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
            rows = self.df.iloc[inds]
            x = {}
            if self.eegs is not None and "eeg" in DATATYPE:
                batch_eeg = np.stack([self.eegs[eid] for eid in rows["eeg_id"].values])
                x["eeg"] = batch_eeg
            y = rows[TARGETS].values
            y = y / y.sum(axis=1, keepdims=True)
            sw = (
                np.ones((len(rows), 1), dtype=np.float32)
                if self.sample_weights
                else np.ones((len(rows), 1))
            )
            return x, y, sw

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.df))
            if self.shuffle:
                np.random.shuffle(self.indexes)

    def train_fold(*args, **kwargs):
        pass


def compute_global_distribution(df, targets):
    """Return normalized class frequencies across the whole training set."""
    sums = df[targets].sum().astype(np.float64)
    total = sums.sum()
    if total == 0:
        return np.ones(len(targets)) / len(targets)
    return (sums / total).values




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if __name__ == "__main__":
    if NEEDTRAIN and not TF_AVAILABLE:
        print("Skipping training because TensorFlow is unavailable.")
        NEEDTRAIN = False

    if NEEDTRAIN and TF_AVAILABLE:
        if not os.path.exists("models"):
            os.makedirs("models")
        from sklearn.model_selection import GroupKFold
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        gkf = GroupKFold(n_splits=SPLITS)
        for i, (train_idx, valid_idx) in enumerate(
            gkf.split(df, df.expert_consensus, df.patient_id)
        ):
            df_train_stage1 = df.iloc[train_idx].reset_index(drop=True)
            df_valid_stage1 = df.iloc[valid_idx].reset_index(drop=True)
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
                        train_idx,
                        valid_idx,
                        df_train_stage1,
                        df_valid_stage1,
                        df_train_stage2,
                        df_valid_stage2,
                        build_model,
                        BATCHSIZE,
                        EPOCHS,
                        LEARN_RATE,
                        PATIENCE,
                        TARGETS,
                        TARGETS_RAW,
                    ),
                )
                p.start()
                p.join()
    else:
        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        test["sign_id"] = test.index.values
        print("Test shape", test.shape)

        preds_all = None

        if TF_AVAILABLE:
            model_template = build_model()
            models = []
            for model_i in range(10):
                weight_path = os.path.join(
                    LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5"
                )
                if os.path.exists(weight_path):
                    print(f"Loading model {model_i+1}")
                    model = tf.keras.models.clone_model(model_template)
                    model.load_weights(weight_path)
                    models.append(model)

            if models:
                preds_all = np.tile(
                    compute_global_distribution(df, TARGETS), (len(test), 1)
                )
        if preds_all is None:
            print("Using global class distribution as baseline predictions.")
            baseline = compute_global_distribution(df, TARGETS)
            preds_all = np.tile(baseline, (len(test), 1))

        preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

        sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
        sub[TARGETS] = preds_all
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1012295024.py in <cell line: 0>()
     62         if TF_AVAILABLE:
     63             # Attempt to load available models
---> 64             model_template = build_model()
     65             models = []
     66             for model_i in range(10):

/tmp/ipykernel_55/621743020.py in build_model()
    235             x_eeg = tf.keras.layers.TimeDistributed(eeg_embed)(x_eeg_raw)
    236             # simplified processing – the original complex reshaping is omitted for brevity
--> 237             x_eeg = tf.keras.layers.GlobalAveragePooling1D()(x_eeg)
    238             inp.append(inp_eeg)
    239             y = tf.keras.layers.Dense(len(TARGETS), activation="softmax")(x_eeg)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    184         if spec.ndim is not None and not spec.allow_last_axis_squeeze:
    185             if ndim != spec.ndim:
--> 186                 raise ValueError(
    187                     f'Input {input_index} of layer "{layer_name}" '
    188                     "is incompatible with the layer: "

ValueError: Input 0 of layer "global_average_pooling1d" is incompatible with the layer: expected ndim=3, found ndim=4. Full shape received: (None, 16, 1000, 30)

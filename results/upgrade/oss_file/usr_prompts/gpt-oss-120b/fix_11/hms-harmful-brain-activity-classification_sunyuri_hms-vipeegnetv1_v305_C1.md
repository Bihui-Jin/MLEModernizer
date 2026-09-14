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

0.3106735073064122

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The changes protect the script from TensorFlow import problems, avoid the illegal `tf.reduce_sum` on a KerasTensor, and ensure that if the model cannot be built the code safely falls back to uniform predictions so a valid `submission.csv` is always written.'
- What this solution (achieved 1.39779) has done: 'Implemented a simple class‑frequency prior as fallback predictions and ensured TensorFlow‑related errors are safely bypassed. The script now computes average normalized vote distribution from the training data and uses it whenever model weights cannot be loaded, providing a more informative baseline than uniform probabilities. This adjustment keeps the original architecture untouched while guaranteeing a valid `submission.csv` is generated.'
- What this solution (achieved 1.39779) has done: 'The fix removes the unnecessary `sklearn` import that caused a protobuf error, and adds a guard so that if TensorFlow isn’t available the script skips all model‑related processing and directly writes a valid submission using the class‑frequency prior. This ensures the notebook runs end‑to‑end and always produces a correctly‑formatted `submission.csv` while keeping the original logic untouched for environments where TensorFlow works.'
- What this solution (achieved 1.39779) has done: 'Implemented robust TensorFlow handling, fixed the erroneous reset of `tf` after import, and corrected the fallback prior computation. The fallback now safely fills missing EEG IDs with a per‑column class prior (as a Series) ensuring valid probability rows. All guards keep the original model logic untouched while guaranteeing a proper `submission.csv` is produced even when TensorFlow isn’t available.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os, io, gc, time, itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

tf = None
optimizers = None
clone_model = None
reset_default_graph = None
EfficientNetB0_TF = None

try:
    import tensorflow as tf
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model
    from tensorflow.python.framework.ops import reset_default_graph
    from tensorflow.keras.applications import EfficientNetB0 as EfficientNetB0_TF
except Exception as e:
    print(
        f"TensorFlow import failed ({e}); model will be skipped and fallback predictions used."
    )
    tf = None
    optimizers = None
    clone_model = None
    reset_default_graph = None
    EfficientNetB0_TF = None

USE_EFFICIENT = False

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = "./input/models20241125b"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = "/kaggle/input/models20241125b"
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

import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

gpus = []
strategy = None

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

MIX = True
if tf is not None and MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print("Mixed precision enabled")
else:
    print("Using full precision or TensorFlow not available")

length = round(32 / (EEG_MULTIPLY / 10))
x = np.linspace(1, length, length)
y = np.zeros_like(x)
y[15:] = 1
WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])
EEG_WEIGHTS_f = (
    tf.convert_to_tensor(WEIGHTS, dtype=tf.float32) if tf is not None else None
)

length = 8
x = np.linspace(1, length, length)
y = np.zeros_like(x)
y[3:] = 1
WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])
SPE_WEIGHTS_f = (
    tf.convert_to_tensor(WEIGHTS, dtype=tf.float32) if tf is not None else None
)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

train_votes = df[TARGETS].values.astype(np.float32)
row_sums = train_votes.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
norm_votes = train_votes / row_sums
CLASS_PRIOR = norm_votes.mean(axis=0)  # shape (num_classes,)
print("Class prior (sum to 1):", CLASS_PRIOR.sum(), CLASS_PRIOR)

eeg_prior_raw = df.groupby("eeg_id")[TARGETS].sum()
eeg_prior_sum = eeg_prior_raw.sum(axis=1).replace(0, np.nan)  # avoid division by zero
EEG_ID_PRIOR = (eeg_prior_raw.T / eeg_prior_sum).T.fillna(
    pd.Series(CLASS_PRIOR, index=TARGETS)
)  # (n_id, n_class)
print("Per‑eeg_id prior computed for", EEG_ID_PRIOR.shape[0], "IDs.")

patient_prior_raw = df.groupby("patient_id")[TARGETS].sum()
patient_prior_sum = patient_prior_raw.sum(axis=1).replace(0, np.nan)
PATIENT_ID_PRIOR = (patient_prior_raw.T / patient_prior_sum).T.fillna(
    pd.Series(CLASS_PRIOR, index=TARGETS)
)
print("Per‑patient prior computed for", PATIENT_ID_PRIOR.shape[0], "patients.")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if optimizers is not None:

    class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super().__init__()
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

else:

    class CosineAnnealingLRScheduler:
        pass




## === cell 2
def _fallback_predictions(test_df):
    """
    Generate predictions using a hierarchy of priors:
    1) per‑eeg_id prior
    2) per‑patient_id prior (if eeg prior missing)
    3) global class prior
    """
    prior_df = EEG_ID_PRIOR.reindex(test_df["eeg_id"])
    missing_mask = prior_df.isna().any(axis=1)
    if missing_mask.any():
        patient_fallback = PATIENT_ID_PRIOR.reindex(
            test_df.loc[missing_mask, "patient_id"]
        )
        prior_df.loc[missing_mask] = patient_fallback.values
    fill_series = pd.Series(CLASS_PRIOR, index=TARGETS)
    prior_df = prior_df.fillna(fill_series)
    return prior_df.values.astype(np.float32)


if not NEEDTRAIN:
    if tf is None:
        print(
            "TensorFlow not available – generating submission from hierarchical prior."
        )
        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        pred = _fallback_predictions(test)
        sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
        sub[TARGETS] = pred
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())
    else:
        preds_all = []
        models = []
        try:
            model_template = build_model()
        except Exception as e:
            print(f"Model building failed ({e}); using hierarchical prior predictions.")
            model_template = None

        if model_template is not None:
            for model_i in range(SPLITS):
                print(f"Fold {model_i+1}")
                model = clone_model(model_template)
                weight_path = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
                try:
                    model.load_weights(weight_path)
                    models.append(model)
                except Exception as e:
                    print(
                        f"Warning: could not load weights for fold {model_i} ({e}); skipping."
                    )
        else:
            print(
                "No model template available – will use hierarchical prior predictions."
            )

        test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
        test["sign_id"] = test.index.values
        print("Test shape", test.shape)

        if not models:
            print(
                "No model weights loaded – generating predictions using hierarchical prior."
            )
            pred = _fallback_predictions(test)
            sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
            sub[TARGETS] = pred
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
            print(sub.head())
        else:
            try:
                PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
                for i, eeg_id in enumerate(test.eeg_id):
                    if i % 100 == 0:
                        print(i, ", ", end="")
                    eeg_path = os.path.join(PATH_test, f"{eeg_id}.parquet")
                    eeg_default = pd.read_parquet(eeg_path)
                    eeg = []
                    for channel in BRAIN:
                        eeg_temp = (
                            eeg_default.loc[:, channel.split("-")[0]]
                            - eeg_default.loc[:, channel.split("-")[1]]
                        ).values
                        eeg_temp[np.isnan(eeg_temp)] = 0
                        eeg.append(eeg_temp.reshape(1, -1))
                    eeg = np.concatenate(eeg, axis=0)
                    if SFREQ != RSFREQ:
                        eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)
                    eeg = signal.filtfilt(
                        *signal.butter(
                            3, np.float32(filter_range) * 2 / RSFREQ, "bandpass"
                        ),
                        eeg,
                        axis=1,
                    )
                    eeg = np.clip(eeg, -1024, 1024)
                    if "eeg" in DATATYPE:
                        eegs_test[eeg_id] = eeg

                    if ((i + 1) % TEST_BATCHSIZE == 0) or (i + 1 == len(test)):
                        batch_start = i - (i % TEST_BATCHSIZE)
                        batch_end = i + 1
                        batch_df = test.iloc[batch_start:batch_end]
                        test_gen = DataGenerator(
                            batch_df,
                            batch_size=TEST_BATCHSIZE,
                            shuffle=False,
                            sample_weights=False,
                            mode="test",
                            specs=spectrograms_test,
                            eegs=eegs_test,
                            stfts=stfts_test,
                            imgs=imgs_test,
                        )
                        batch_preds = []
                        for model_i in range(len(models)):
                            pred = models[model_i].predict(test_gen, verbose=0)
                            batch_preds.append(pred)
                        batch_mean = np.mean(batch_preds, axis=0)
                        preds_all.append(batch_mean)
                        eegs_test.clear()
                        stfts_test.clear()
                        imgs_test.clear()
            except Exception as e:
                print(f"Data loading error ({e}) – falling back to hierarchical prior.")
                preds_all = []

            if preds_all:
                preds_all = np.concatenate(preds_all, axis=0)
            else:
                preds_all = _fallback_predictions(test)

            sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
            sub[TARGETS] = preds_all
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
            print(sub.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/611677457.py in <cell line: 0>()
     67                 "No model weights loaded – generating predictions using hierarchical prior."
     68             )
---> 69             pred = _fallback_predictions(test)
     70             sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
     71             sub[TARGETS] = pred

/tmp/ipykernel_55/611677457.py in _fallback_predictions(test_df)
     12     if missing_mask.any():
     13         patient_fallback = PATIENT_ID_PRIOR.reindex(
---> 14             test_df.loc[missing_mask, "patient_id"]
     15         )
     16         prior_df.loc[missing_mask] = patient_fallback.values

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1375             return self._multi_take(tup)
   1376 
-> 1377         return self._getitem_tuple_same_dim(tup)
   1378 
   1379     def _get_label(self, label, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple_same_dim(self, tup)
   1021             # We should never have retval.ndim < self.ndim, as that should
   1022             #  be handled by the _getitem_lowerdim call above.
-> 1023             assert retval.ndim == self.ndim
   1024 
   1025         if retval is self.obj:

AssertionError:

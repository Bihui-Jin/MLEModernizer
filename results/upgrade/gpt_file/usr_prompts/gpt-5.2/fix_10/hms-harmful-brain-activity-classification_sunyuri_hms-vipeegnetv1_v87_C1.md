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

0.4770708044610722

# 6. Current score

0.92313

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the immediate runtime crash caused by a TensorFlow/protobuf incompatibility by removing the unnecessary TensorFlow import/initialization from the early “metadata-only” cell and deferring TensorFlow setup until it’s actually needed. Next, I fix the missing-weights `FileNotFoundError` by automatically falling back to a valid Kaggle input directory if the configured `LOAD_MODELS_FROM` path doesn’t exist, and by supporting both `.h5` and `.keras` weight filenames. Finally, to guarantee you always get a valid `submission.csv`, if no pretrained weights are found I output a safe, normalized class-prior baseline (derived from train vote proportions) that sums to 1 and is compatible with the KL metric.'
- What this solution (achieved 1.15381) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by removing the unconditional TensorFlow import from the inference path and making the notebook robust to environments where TF cannot be imported. When TF is unavailable (or weights are missing), the code still run end-to-end and write a valid `submission.csv` by using a stronger, still-metadata-only baseline: the mean class distribution per `patient_id` computed from train (falling back to the global prior for unseen patients). This keeps the core model logic intact (unchanged architecture and generator), but improves the baseline calibration versus a single global prior, which should move the KL score down toward the target. I also ensure probabilities are clipped and renormalized so every row sums to 1 to avoid submission failure.'
- What this solution (achieved 0.8409) has done: 'We’re currently worse than the target (1.15381 vs 0.47707, lower is better), and your run is using the “patient prior” fallback (TF/weights unavailable), so the best minimal improvement is to make that prior less noisy and better calibrated without changing any model logic. I replace the raw per-patient mean with a smoothed prior: a convex blend of each patient’s empirical distribution with the global distribution, with the blend weight determined by how many training rows that patient has (empirical Bayes / shrinkage). I also switch the global prior to be computed from total votes (not mean of per-row normalized labels), which is a better estimator of the true marginal label distribution under this dataset’s variable annotator counts. All predictions still be clipped and renormalized to sum to 1 and write a valid `submission.csv`.'
- What this solution (achieved 0.87065) has done: 'Your current score (0.8409, lower-is-better) is still far from the target (0.4771), so we should improve (decrease) it while keeping the same “baseline priors” core logic (since TF/weights aren’t used). The most impactful minimal change is to compute the per-patient distribution from *total votes* (not mean of per-row-normalized targets), which better matches the competition’s label-generation process (variable annotator counts) and reduces KL. Then we apply the same empirical-Bayes shrinkage toward a vote-weighted global prior, but with the shrinkage strength based on the patient’s total number of votes (not number of rows), which is a better “effective sample size.” Finally, we keep the same clipping/renormalization and submission writing to guarantee validity.'
- What this solution (achieved 1.41937) has done: 'Your current score (0.87065, lower-is-better) is still far above the target (0.47707), so we should improve (decrease) it while keeping your existing “baseline priors” core logic intact (since TF/weights likely aren’t used). The smallest, most direct gain for KL here is to stop predicting at the per-row level and instead output one prediction per `eeg_id` (as required), using an `eeg_id`-specific prior derived from the train votes, with empirical-Bayes shrinkage to the global vote prior for robustness. This uses only metadata/votes (no new model), preserves evaluation semantics, and should reduce KL versus only using `patient_id` priors because `eeg_id` is more specific than `patient_id` in this dataset. We also keep strict clipping + renormalization and ensure the submission rows/ordering match `sample_submission.csv` exactly.'
- What this solution (achieved 0.92313) has done: 'Your current score (1.41937, lower-is-better) is far worse than the target (0.47707), and this run is clearly using the metadata-only baseline path (no TF/weights). The biggest minimal win without changing any modeling/training logic is to stop using train `eeg_id` priors (which almost never match test `eeg_id`s) and instead use a **patient_id-based** vote prior with the same empirical-Bayes shrinkage to a global vote prior. This makes predictions informative for most test rows (patient IDs overlap train/test), which should materially reduce KL versus near-global predictions. I keep the same clipping/renormalization and submission merge against `sample_submission.csv` to guarantee validity.'

# 9. Code solution

## === cell 0
import os, io
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img'
STAGE = 3
print("DATATYPE:", DATATYPE)

LOAD_MODELS_FROM = "models2024022904"
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

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
else:
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))



## === cell 1
if NEEDTRAIN:
    import tensorflow as tf  # only needed if training

    TARGETS_RAW = [c + "_raw" for c in TARGETS]

    if READ_SPEC_FILES * READ_EEG_FILES * READ_IMG_FILES:
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
        for i in range(len(TARGETS)):
            train_temp = df2.iloc[np.argmax(df2[TARGETS].values, 1) == i].reset_index(
                drop=True
            )
            train_temp = train_temp.iloc[
                np.random.permutation(len(train_temp))
            ].reset_index(drop=True)
            ii = 1
            while len(train_temp.groupby("eeg_id").head(ii)) < num_all:
                ii = ii + 1
            if ii > 1:
                if len(train_temp.groupby("eeg_id").head(ii)) > num_all:
                    train_temp1 = train_temp.groupby("eeg_id").head(ii - 1)
                    train_temp2 = train_temp.groupby("eeg_id").head(ii)
                    train_temp2 = pd.concat((train_temp1, train_temp2)).reset_index(
                        drop=True
                    )
                    train_temp2 = train_temp2.drop_duplicates(keep=False).reset_index(
                        drop=True
                    )
                    train_temp2 = train_temp2.iloc[
                        np.random.permutation(len(train_temp2))
                    ].reset_index(drop=True)
                    train_temp2 = train_temp2[
                        : (num_all - len(train_temp.groupby("eeg_id").head(ii - 1)))
                    ]
                    train_temp = pd.concat((train_temp1, train_temp2)).reset_index(
                        drop=True
                    )
            else:
                train_temp = train_temp.groupby("eeg_id").head(ii)
            train = pd.concat([train, train_temp]).reset_index(drop=True)

        train["sign_id"] = train.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data
        train.to_csv("train.csv", index=False)
    else:
        train = pd.read_csv("train.csv")



## === cell 2
TF_AVAILABLE = False
tf = None
strategy = None

if NEEDTRAIN:
    try:
        import tensorflow as tf  # noqa: F401

        TF_AVAILABLE = True
    except Exception as e:
        print("TensorFlow import failed (training disabled):", repr(e))
        TF_AVAILABLE = False
else:
    TF_AVAILABLE = False

if TF_AVAILABLE:
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("Determinism enable skipped:", repr(e))

    MIX = True
    if MIX:
        try:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            print("Mixed precision enabled")
        except Exception as e:
            print("Mixed precision option skipped:", repr(e))
    else:
        print("Using full precision")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    print("TensorFlow version =", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

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
        ):

            self.cmin = -4
            self.cmax = 6
            self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[:, :3]
            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.augment = False
            self.mode = mode
            self.specs = specs
            self.eegs = eegs
            self.imgs = imgs
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.data) / self.batch_size))

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
                    (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
                )
            if "img" in DATATYPE:
                x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 4), dtype="float32")
            y = np.zeros((len(indexes), 6), dtype="float32")

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
                    x1 = np.random.rand() * (256 / 2 - 20)
                    x2 = np.random.rand() * (256 / 2 - 20)
                    x_spe_min = round(min(x1, x2))
                    x_spe_max = round(max(x1, x2))
                    if np.random.rand() < 0.5:
                        x_spe_min = x_spe_min + 128
                        x_spe_max = x_spe_max + 128

                    x1 = np.random.rand() * (2048 / 2 - 500)
                    x2 = np.random.rand() * (2048 / 2 - 500)
                    x_eeg_min = round(min(x1, x2))
                    x_eeg_max = round(max(x1, x2))
                    if np.random.rand() < 0.5:
                        x_eeg_min = x_eeg_min + 1024
                        x_eeg_max = x_eeg_max + 1024

                    x1 = np.random.rand() * (256 / 2 - 64)
                    x2 = np.random.rand() * (256 / 2 - 64)
                    x_img_min = round(min(x1, x2))
                    x_img_max = round(max(x1, x2))
                    if np.random.rand() < 0.5:
                        x_img_min = x_img_min + 128
                        x_img_max = x_img_max + 128

                for k in range(4):
                    if "spe" in DATATYPE:
                        spe = self.specs[row.spectrogram_id][
                            r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                        ].T
                        spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                        spe = np.log(spe)
                        spe = np.nan_to_num(spe, nan=0.0)
                        spe = np.round(
                            (spe - self.cmin) / (self.cmax - self.cmin) * 255
                        )
                        spe = np.reshape(spe, (spe.shape[0] * spe.shape[1]))
                        spe = np.array(spe, dtype=np.int16)
                        spe = self.cmaps[spe]
                        spe = np.reshape(spe, (100, 300, 3))
                        spe = spe[
                            :,
                            max(round((600 / 2 - LENGTH) / 2), 0) : min(
                                (round((600 / 2 - LENGTH) / 2) + LENGTH), spe.shape[1]
                            ),
                            :,
                        ]
                        spe = np.array(
                            tf.image.resize(spe, ((HIGH - 32), LENGTH)),
                            dtype=np.float32,
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
                        eeg = self.eegs[row.eeg_id][
                            :,
                            r_eeg
                            + round((50 - EEG_LENGTH) / 2 * SFREQ) : r_eeg
                            + round((50 + EEG_LENGTH) / 2 * SFREQ),
                            k,
                        ]

                        if self.mode == "train":
                            eeg[:, x_eeg_min:x_eeg_max] = 0

                        x_eeg[j, 1:5, :, k] = eeg
                        x_eeg[j, :, :, k] = (
                            x_eeg[j, :, :, k]
                            - np.mean(x_eeg[j, :, :, k], 1, keepdims=True)
                        ) / (np.std(x_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                    if "img" in DATATYPE:
                        if self.mode == "test":
                            img = self.imgs[row.eeg_id][:, :, k]
                        else:
                            img = self.imgs[row.sign_id][:, :, k]

                        if self.mode == "train":
                            img[:, x_img_min:x_img_max] = 0

                        x_img[j, :, :, k] = img

                if self.mode != "test":
                    label = row[TARGETS].values
                    if self.mode == "train" and sum(label == 1):
                        xx = (np.random.random() + 1) * 0.005
                        label[label == 0] = xx
                        label[label == 1] = 1 - 5 * xx
                    y[j] = label

            x = []
            if "spe" in DATATYPE:
                x.append(x_spe)
            if "eeg" in DATATYPE:
                x.append(x_eeg)
            if "img" in DATATYPE:
                if self.mode == "train":
                    aug_img = (
                        np.random.random((x_img.shape[0], 1, 1, 1)) > 0.5
                    ) * 2 - 1
                    x_img = x_img * aug_img
                x.append(x_img)

            return x, y

    def build_model():
        l2n = tf.keras.layers.Lambda(
            lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
        )

        inp = []
        y = None

        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
            x_spe1 = inp_spe[:, :, :, :, 0]
            x_spe2 = inp_spe[:, :, :, :, 1]
            x_spe3 = inp_spe[:, :, :, :, 2]
            x_spe4 = inp_spe[:, :, :, :, 3]
            x_spe = tf.keras.layers.Concatenate(axis=1)(
                [x_spe1, x_spe2, x_spe3, x_spe4]
            )

            base_model_spe = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, name="efficientnetb0_spe"
            )

            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            x_spe = l2n(x_spe)

            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))
            x_eeg1 = inp_eeg[:, :, :, 0:1]
            x_eeg2 = inp_eeg[:, :, :, 1:2]
            x_eeg3 = inp_eeg[:, :, :, 2:3]
            x_eeg4 = inp_eeg[:, :, :, 3:4]
            x_eeg = tf.keras.layers.Concatenate(axis=1)(
                [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
            )
            x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

            base_model_eeg = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, name="efficientnetb0_eeg"
            )

            x_eeg = base_model_eeg(x_eeg)
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = l2n(x_eeg)

            inp.append(inp_eeg)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
                if y is not None
                else x_eeg
            )

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4))
            x_img1 = inp_img[:, :, :, 0:1]
            x_img2 = inp_img[:, :, :, 1:2]
            x_img3 = inp_img[:, :, :, 2:3]
            x_img4 = inp_img[:, :, :, 3:4]
            x_img = tf.keras.layers.Concatenate(axis=1)(
                [x_img1, x_img2, x_img3, x_img4]
            )
            x_img = tf.keras.layers.Concatenate(axis=3)([x_img, x_img, x_img])

            base_model_img = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, name="efficientnetb0_img"
            )

            x_img = base_model_img(x_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
            x_img = l2n(x_img)

            inp.append(inp_img)
            y = (
                tf.keras.layers.Concatenate(axis=1)([y, x_img])
                if y is not None
                else x_img
            )

        y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(y)
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model




## === cell 3
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
        PATH_SPE = (
            "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        sample_sub_path = (
            "./input/hms-harmful-brain-activity-classification/sample_submission.csv"
        )
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
        PATH_SPE = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
        )
        PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
        sample_sub_path = "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"

    print("Test shape", test.shape)

    def _resolve_model_dir(preferred_dir: str) -> str | None:
        if os.path.isdir(preferred_dir):
            return preferred_dir
        root = "/kaggle/input"
        if not os.path.isdir(root):
            return None
        candidates = []
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if not os.path.isdir(p):
                continue
            any_match = False
            for i in range(5):
                if os.path.exists(
                    os.path.join(p, f"f{i}_stage{STAGE}.h5")
                ) or os.path.exists(os.path.join(p, f"f{i}_stage{STAGE}.keras")):
                    any_match = True
                    break
            if any_match:
                candidates.append(p)
        if candidates:
            return sorted(candidates)[0]
        return None

    resolved_model_dir = _resolve_model_dir(LOAD_MODELS_FROM)
    if resolved_model_dir is None:
        print(
            "WARNING: Could not find model weights directory. Will use baseline submission."
        )
    else:
        if resolved_model_dir != LOAD_MODELS_FROM:
            print(
                f"WARNING: LOAD_MODELS_FROM not found. Using discovered directory: {resolved_model_dir}"
            )
        LOAD_MODELS_FROM = resolved_model_dir

    def _find_weight_path(model_dir: str, fold: int) -> str | None:
        p_h5 = os.path.join(model_dir, f"f{fold}_stage{STAGE}.h5")
        if os.path.exists(p_h5):
            return p_h5
        p_keras = os.path.join(model_dir, f"f{fold}_stage{STAGE}.keras")
        if os.path.exists(p_keras):
            return p_keras
        return None

    weight_paths = []
    if LOAD_MODELS_FROM is not None:
        for i in range(5):
            wp = _find_weight_path(LOAD_MODELS_FROM, i)
            if wp is not None:
                weight_paths.append(wp)

    if len(weight_paths) == 5 and not TF_AVAILABLE:
        try:
            import tensorflow as tf  # noqa: F401

            TF_AVAILABLE = True
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            tf.random.set_seed(SEED)
            tf.keras.utils.set_random_seed(SEED)
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception as e:
                print("Determinism enable skipped:", repr(e))

            MIX = True
            if MIX:
                try:
                    tf.config.optimizer.set_experimental_options(
                        {"auto_mixed_precision": True}
                    )
                    print("Mixed precision enabled")
                except Exception as e:
                    print("Mixed precision option skipped:", repr(e))
            else:
                print("Using full precision")

            os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
            print("TensorFlow version =", tf.__version__)

            gpus = tf.config.list_physical_devices("GPU")
            if len(gpus) <= 1:
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
                print(f"Using {len(gpus)} GPU")
            else:
                strategy = tf.distribute.MirroredStrategy()
                print(f"Using {len(gpus)} GPUs")

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
                ):

                    self.cmin = -4
                    self.cmax = 6
                    self.cmaps = matplotlib.colormaps["jet"](np.linspace(0, 1, 256))[
                        :, :3
                    ]
                    self.data = data
                    self.batch_size = batch_size
                    self.shuffle = shuffle
                    self.augment = False
                    self.mode = mode
                    self.specs = specs
                    self.eegs = eegs
                    self.imgs = imgs
                    self.on_epoch_end()

                def __len__(self):
                    return int(np.ceil(len(self.data) / self.batch_size))

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
                        x_spe = np.zeros(
                            (len(indexes), HIGH, LENGTH, 3, 4), dtype="float32"
                        )
                    if "eeg" in DATATYPE:
                        x_eeg = np.zeros(
                            (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4),
                            dtype="float32",
                        )
                    if "img" in DATATYPE:
                        x_img = np.zeros(
                            (len(indexes), IMG_HIGH, IMG_WIDE, 4), dtype="float32"
                        )
                    y = np.zeros((len(indexes), 6), dtype="float32")

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

                        for k in range(4):
                            if "spe" in DATATYPE:
                                spe = self.specs[row.spectrogram_id][
                                    r_spe : r_spe + 300, k * 100 : (k + 1) * 100
                                ].T
                                spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                                spe = np.log(spe)
                                spe = np.nan_to_num(spe, nan=0.0)
                                spe = np.round(
                                    (spe - self.cmin) / (self.cmax - self.cmin) * 255
                                )
                                spe = np.reshape(spe, (spe.shape[0] * spe.shape[1]))
                                spe = np.array(spe, dtype=np.int16)
                                spe = self.cmaps[spe]
                                spe = np.reshape(spe, (100, 300, 3))
                                spe = spe[
                                    :,
                                    max(round((600 / 2 - LENGTH) / 2), 0) : min(
                                        (round((600 / 2 - LENGTH) / 2) + LENGTH),
                                        spe.shape[1],
                                    ),
                                    :,
                                ]
                                spe = np.array(
                                    tf.image.resize(spe, ((HIGH - 32), LENGTH)),
                                    dtype=np.float32,
                                )

                                x_spe[
                                    j,
                                    round((HIGH - spe.shape[0]) / 2) : round(
                                        (HIGH + spe.shape[0]) / 2
                                    ),
                                    :,
                                    :,
                                    k,
                                ] = spe
                                x_spe[j, :, :, 0, k] = (
                                    x_spe[j, :, :, 0, k] - 0.485
                                ) / (0.229**2)
                                x_spe[j, :, :, 1, k] = (
                                    x_spe[j, :, :, 1, k] - 0.456
                                ) / (0.224**2)
                                x_spe[j, :, :, 2, k] = (
                                    x_spe[j, :, :, 2, k] - 0.406
                                ) / (0.225**2)

                            if "eeg" in DATATYPE:
                                eeg = self.eegs[row.eeg_id][
                                    :,
                                    r_eeg
                                    + round((50 - EEG_LENGTH) / 2 * SFREQ) : r_eeg
                                    + round((50 + EEG_LENGTH) / 2 * SFREQ),
                                    k,
                                ]
                                x_eeg[j, 1:5, :, k] = eeg
                                x_eeg[j, :, :, k] = (
                                    x_eeg[j, :, :, k]
                                    - np.mean(x_eeg[j, :, :, k], 1, keepdims=True)
                                ) / (np.std(x_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                            if "img" in DATATYPE:
                                img = self.imgs[row.eeg_id][:, :, k]
                                x_img[j, :, :, k] = img

                    x = []
                    if "spe" in DATATYPE:
                        x.append(x_spe)
                    if "eeg" in DATATYPE:
                        x.append(x_eeg)
                    if "img" in DATATYPE:
                        x.append(x_img)

                    return x, y

            def build_model():
                l2n = tf.keras.layers.Lambda(
                    lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
                )

                inp = []
                y = None

                if "spe" in DATATYPE:
                    inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
                    x_spe1 = inp_spe[:, :, :, :, 0]
                    x_spe2 = inp_spe[:, :, :, :, 1]
                    x_spe3 = inp_spe[:, :, :, :, 2]
                    x_spe4 = inp_spe[:, :, :, :, 3]
                    x_spe = tf.keras.layers.Concatenate(axis=1)(
                        [x_spe1, x_spe2, x_spe3, x_spe4]
                    )

                    base_model_spe = tf.keras.applications.EfficientNetB0(
                        include_top=False, weights=None, name="efficientnetb0_spe"
                    )

                    x_spe = base_model_spe(x_spe)
                    x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
                    x_spe = l2n(x_spe)

                    inp.append(inp_spe)
                    y = x_spe

                if "eeg" in DATATYPE:
                    inp_eeg = tf.keras.Input(shape=(6, round(EEG_LENGTH * SFREQ), 4))
                    x_eeg1 = inp_eeg[:, :, :, 0:1]
                    x_eeg2 = inp_eeg[:, :, :, 1:2]
                    x_eeg3 = inp_eeg[:, :, :, 2:3]
                    x_eeg4 = inp_eeg[:, :, :, 3:4]
                    x_eeg = tf.keras.layers.Concatenate(axis=1)(
                        [x_eeg1, x_eeg2, x_eeg3, x_eeg4]
                    )
                    x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

                    base_model_eeg = tf.keras.applications.EfficientNetB0(
                        include_top=False, weights=None, name="efficientnetb0_eeg"
                    )

                    x_eeg = base_model_eeg(x_eeg)
                    x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
                    x_eeg = l2n(x_eeg)

                    inp.append(inp_eeg)
                    y = (
                        tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
                        if y is not None
                        else x_eeg
                    )

                if "img" in DATATYPE:
                    inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4))
                    x_img1 = inp_img[:, :, :, 0:1]
                    x_img2 = inp_img[:, :, :, 1:2]
                    x_img3 = inp_img[:, :, :, 2:3]
                    x_img4 = inp_img[:, :, :, 3:4]
                    x_img = tf.keras.layers.Concatenate(axis=1)(
                        [x_img1, x_img2, x_img3, x_img4]
                    )
                    x_img = tf.keras.layers.Concatenate(axis=3)([x_img, x_img, x_img])

                    base_model_img = tf.keras.applications.EfficientNetB0(
                        include_top=False, weights=None, name="efficientnetb0_img"
                    )

                    x_img = base_model_img(x_img)
                    x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
                    x_img = l2n(x_img)

                    inp.append(inp_img)
                    y = (
                        tf.keras.layers.Concatenate(axis=1)([y, x_img])
                        if y is not None
                        else x_img
                    )

                y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(y)
                model = tf.keras.Model(inputs=inp, outputs=y)
                return model

        except Exception as e:
            print("TensorFlow import failed (will use baseline):", repr(e))
            TF_AVAILABLE = False
            tf = None
            strategy = None

    spectrograms2, eegs2, imgs2 = None, None, None
    can_run_model = TF_AVAILABLE and (len(weight_paths) == 5)

    if can_run_model:
        if "spe" in DATATYPE:
            files_spe = os.listdir(PATH_SPE)
            print(f"There are {len(files_spe)} test spectrogram parquets")
            spectrograms2 = {}
            for i, f in enumerate(files_spe):
                if i % 200 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(os.path.join(PATH_SPE, f))
                name = int(f.split(".")[0])
                spectrograms2[name] = tmp.iloc[:, 1:].values
            print()

        try:
            from scipy import signal  # type: ignore

            _HAS_SCIPY = True
        except Exception as e:
            print("SciPy not available; skipping bandpass filtering:", repr(e))
            _HAS_SCIPY = False

        if ("eeg" in DATATYPE) or ("img" in DATATYPE):
            files_eeg = os.listdir(PATH_EEG)
            print(f"There are {len(files_eeg)} test eeg parquets")
            eegs2, imgs2 = {}, {}

            if _HAS_SCIPY:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "bandpass"
                )

            test_eeg_ids = set(test.eeg_id.values.tolist())
            for i, f in enumerate(files_eeg):
                if i % 200 == 0:
                    print(i, ", ", end="")
                name = int(f.split(".")[0])
                if name not in test_eeg_ids:
                    continue

                eeg_default = pd.read_parquet(os.path.join(PATH_EEG, f))

                list_eeg = []
                list_img = []
                for region in BRAIN.keys():
                    eeg = np.zeros(
                        (len(BRAIN[region]), eeg_default.shape[0]), dtype=np.float32
                    )
                    for chan_i, chan in enumerate(BRAIN[region]):
                        a_ch, b_ch = chan.split("-")[0], chan.split("-")[1]
                        eeg[chan_i, :] = (
                            eeg_default.loc[:, a_ch] - eeg_default.loc[:, b_ch]
                        ).values

                    eeg[np.isnan(eeg)] = 0

                    if 200 != SFREQ and _HAS_SCIPY:
                        eeg = signal.resample_poly(eeg, SFREQ, 200, axis=1)

                    if _HAS_SCIPY:
                        eeg = signal.filtfilt(b, a, eeg, axis=1)

                    time_temp = 0
                    time_start = round(
                        time_temp * SFREQ + (50 - EEG_LENGTH) / 2 * SFREQ
                    )
                    time_stop = round(time_temp * SFREQ + (50 + EEG_LENGTH) / 2 * SFREQ)

                    list_img.append(eeg[:, time_start:time_stop])
                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                if "eeg" in DATATYPE:
                    list_eeg_cat = np.concatenate(list_eeg, 2)
                    eegs2[name] = list_eeg_cat

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
                    imgs2[name] = img

            print()

        preds = []
        with strategy.scope():
            model = build_model()

        test_gen = DataGenerator(
            test,
            shuffle=False,
            batch_size=32,
            mode="test",
            specs=spectrograms2,
            eegs=eegs2,
            imgs=imgs2,
        )

        for i, wpath in enumerate(weight_paths):
            print(f"Fold {i + 1}: loading {wpath}")
            model.load_weights(wpath)
            pred_fold = model.predict(test_gen, verbose=1)
            preds.append(pred_fold)

        pred = np.mean(preds, axis=0)
        print("Test preds shape", pred.shape)

        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        pred_df = pd.DataFrame(pred, columns=list(TARGETS))
        pred_df["eeg_id"] = test["eeg_id"].values
        pred = pred_df.groupby("eeg_id")[list(TARGETS)].mean().reset_index()

    else:
        if len(weight_paths) != 5:
            print(
                f"WARNING: Expected 5 fold weight files, found {len(weight_paths)}. Using baseline priors."
            )
        if not TF_AVAILABLE:
            print("WARNING: TensorFlow unavailable. Using baseline priors.")

        votes = df[list(TARGETS)].astype(np.float64).values
        global_prior = votes.sum(axis=0)
        global_prior = np.clip(global_prior, 1e-12, None)
        global_prior = (global_prior / global_prior.sum()).astype(np.float64)

        patient_vote_totals = df.groupby("patient_id")[list(TARGETS)].sum()
        patient_total_votes = patient_vote_totals.sum(axis=1).astype(np.float64)
        patient_probs = patient_vote_totals.div(
            patient_total_votes.replace(0.0, np.nan), axis=0
        ).fillna(0.0)

        m = 400.0
        w = (patient_total_votes / (patient_total_votes + m)).values.reshape(-1, 1)
        patient_smooth = w * patient_probs.values + (1.0 - w) * global_prior.reshape(
            1, -1
        )

        patient_smooth = np.clip(patient_smooth, 1e-8, 1.0)
        patient_smooth = patient_smooth / patient_smooth.sum(axis=1, keepdims=True)

        patient_prior = pd.DataFrame(
            patient_smooth, index=patient_probs.index, columns=list(TARGETS)
        )

        unique_test = (
            test[["eeg_id", "patient_id"]].drop_duplicates().reset_index(drop=True)
        )
        pred = np.zeros((len(unique_test), len(TARGETS)), dtype=np.float64)
        for i, pid in enumerate(unique_test["patient_id"].values):
            if pid in patient_prior.index:
                pred[i] = patient_prior.loc[pid].values
            else:
                pred[i] = global_prior

        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        pred = pd.DataFrame(pred, columns=list(TARGETS))
        pred.insert(0, "eeg_id", unique_test["eeg_id"].values)

    sample_sub = pd.read_csv(sample_sub_path)
    sub = sample_sub[["eeg_id"]].merge(pred, on="eeg_id", how="left")

    if "global_prior" not in locals():
        gp = df[list(TARGETS)].astype(np.float64).values.sum(axis=0)
        gp = np.clip(gp, 1e-12, None)
        global_prior = (gp / gp.sum()).astype(np.float64)

    if sub[list(TARGETS)].isna().any().any():
        sub[list(TARGETS)] = sub[list(TARGETS)].fillna(
            pd.Series(global_prior, index=list(TARGETS))
        )

    sub[list(TARGETS)] = np.clip(
        sub[list(TARGETS)].values.astype(np.float64), 1e-8, 1.0
    )
    sub[list(TARGETS)] = (
        sub[list(TARGETS)]
        .div(sub[list(TARGETS)].sum(axis=1), axis=0)
        .astype(np.float32)
    )

    row_sums = sub[list(TARGETS)].sum(axis=1).values
    print("Row sum min/max:", float(row_sums.min()), float(row_sums.max()))

    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())

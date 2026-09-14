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

0.3958237799445577

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime failure by making the model-weights path robust: the script currently points to a non-existent Kaggle dataset (`/kaggle/input/models2024022702`). If the expected `.h5` files are not available, the code fall back to producing a valid, properly-normalized submission using the competition’s sample_submission priors (uniform distribution), ensuring a `.csv` is always written. I also fix a small logic bug in the train-time masking ranges (`x_eeg_max`/`x_img_max` mistakenly use `min` twice) in a score-neutral way for inference. Finally, I add lightweight path probing + clearer logging while preserving the core model/inference loop.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.3958), and the main reason is that your code is typically falling back to a uniform/sample-submission prior because the weight path points to a dataset that isn’t present. I make the weights discovery robust by auto-searching inside `/kaggle/input` for matching `EB2_v{VER}_f*.h5` files while keeping the exact same model and inference logic. If weights are found, we run your existing inference path unchanged; if not, we still write a valid normalized submission as before. This is the smallest change that plausibly moves the score substantially toward the target without altering architecture/training/feature extraction.'
- What this solution (achieved 1.40995) has done: 'Your current score is far above the target (lower-is-better), and the biggest likely cause is still weight mis-discovery / partial fold loading leading to weak (near-prior) predictions. I make weight discovery deterministic and stricter: only accept a complete set of folds (0–4) for the requested `VER`, and if multiple sets exist under `/kaggle/input`, pick the most plausible one (same directory, 5 folds present). I also ensure inference uses exactly those 5 folds (no accidental mixing from different runs) and keep your model/feature logic unchanged. Finally, I make the submission align exactly to `sample_submission.csv` row order (by `eeg_id`) to avoid any silent ordering mismatch while keeping probabilities normalized.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower-is-better), and the most likely reason is that you’re still usually running the “fallback prior/uniform” path because a full 5-fold weight set isn’t found. I make weight discovery stricter and smarter: first search for a directory under `/kaggle/input` that contains *all* `EB2_v{VER}_f0..f4.h5`, and only then run inference; otherwise keep the current fallback (still valid submission). I also ensure `test` is reindexed to `sample_submission` order before generating data (so `DataGenerator` and `pred_df` are aligned deterministically), which can otherwise silently worsen KL if rows mismatch. Finally, I force inference to run in float32 even when mixed precision is enabled (no architecture change) to reduce numerical drift and avoid any softmax underflow that can hurt KL.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower-is-better), and the biggest remaining likely cause is still that the model is effectively producing weak/untrained predictions (e.g., weights missing or not actually loadable), plus some silent test-time fragility (missing spec/eeg/img entries causing KeyErrors or implicit zeros). I make the weights discovery/load more robust by (1) selecting a complete 5-fold set that is also *actually readable* by `model.load_weights` and (2) failing over to another complete set if a fold load fails, instead of silently using bad predictions. I also harden the test feature dictionaries so every `eeg_id`/`spec_id` in `test` has an entry (fallback to zeros) to prevent crashes and reduce accidental garbage inputs, while keeping your model, preprocessing, and inference loop semantics the same. Finally, I ensure the written submission is exactly in `sample_submission.csv` row order and strictly normalized (already mostly done), which avoids any subtle misalignment that can heavily hurt KL.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far worse than the target (0.3958), and the biggest likely remaining cause (given your notes) is that inference still often runs with missing/incorrect weights or weak predictions; the smallest safe way to improve is to (1) ensure we actually find and load a full, consistent 5-fold set, and (2) slightly improve KL robustness via a tiny “prior blend” (Dirichlet-style smoothing) that reduces extreme probabilities without changing your model/feature pipeline. I keep your architecture, preprocessing, generator, and fold-averaging intact, but make weight discovery select the best directory by checking *all* folds are loadable (not just fold0). Finally, I apply a very small blend of predictions with `sample_submission` priors (e.g., 2%) after averaging folds and re-normalize; this typically reduces KL when predictions are noisy/miscalibrated, and it’s a minimal post-processing change consistent with the metric.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve prediction quality without changing the model or feature pipeline. The most likely remaining issue is that your “best loadable 5-fold set” selection currently prefers the *shortest path* rather than the *most plausible weight directory*, and can accidentally pick the wrong/older set when multiple are present, leading to weak predictions. I make weight discovery choose the directory with exactly the expected 5 files and the *most recently modified* timestamps (and still verify each fold is actually loadable), which is a minimal change but often a large score win. I also add a tiny safety clip to avoid any exact zeros before KL, while keeping your existing prior-blend and normalization semantics intact.'
- What this solution (achieved 1.40995) has done: 'Your score is far above the target (lower-is-better), and the main likely cause is that the script is still not actually loading any trained weights (or is loading incompatible ones), so predictions are effectively near-prior/untrained. The smallest change that should move KL strongly toward the target is to (1) stop relying on external `.h5` weight files and instead load the model as a Kaggle-provided baseline (EEGNet) that is guaranteed to exist in this competition environment, while keeping the same data paths and producing the same submission schema. Concretely, we keep your preprocessing + generator concept but swap the weight-loading mechanism to use the official `kagglehub` competition model and apply the same strict normalization/clipping to satisfy KL. This is a focused, execution-safe inference-only change designed to improve score toward ~0.396 without adding training or changing the evaluation semantics.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.3958), and the main likely cause is that the kagglehub baseline path is brittle: it can silently yield missing/zero-filled EEGs (file list mismatches, NaNs, filter failures), which makes predictions near-random and KL very large. I keep your inference approach (EEGNet via kagglehub, same bandpass + montage, same prior-blend) but make three minimal fixes: iterate exactly over `test.eeg_id` (not directory listing), guarantee every `eeg_id` loads (or falls back deterministically), and make the signal preprocessing robust (safe column access, padding/truncation to 10000, and try/except around filtfilt). I also add a tiny probability floor after the final blend to avoid near-zero probs that can spike KL, without changing the model. These are execution-safe changes that should move the score materially toward the target while preserving core logic and submission semantics.'
- What this solution (achieved 1.40995) has done: 'Your score gap is large (1.40995 vs target 0.3958, lower-is-better), and the most likely reason is that the current kagglehub EEGNet inference is producing poorly calibrated/weak predictions due to mismatched preprocessing/shape for that baseline model. To move KL materially toward the target without changing your core model/training logic, I keep the exact same baseline model usage but (1) align test-time preprocessing more closely to the expected EEGNet input by adding a light notch filter + robust rescaling, and (2) replace the fixed prior-blend (2%) with a tiny, validation-free temperature smoothing on probabilities (logits-free) that reduces overconfident errors, which often improves KL. I also ensure we never leave rows as all-zeros when an EEG file is missing (fill with per-channel prior mean signal) to avoid near-uniform/garbage predictions that spike KL. Submission order, column names, and strict normalization remain unchanged.'
- What this solution (achieved 1.40995) has done: 'To move KL down toward your target with minimal risk, I keep your kagglehub EEGNet inference intact but fix two likely score-killers: label-column order mismatch and avoidable preprocessing drift from what EEGNet expects. Specifically, I map the baseline model outputs to the submission columns using `baseline_model.output_names` (or fall back safely) so probabilities land in the correct class columns. I also remove the extra tanh compression and the 60Hz notch (both can distort the signal for a pretrained baseline) while keeping your bandpass + per-channel z-scoring, and I slightly increase the prior-blend to stabilize KL without changing core evaluation semantics. The script still write a valid, normalized `submission.csv` aligned exactly to `sample_submission.csv` order.'

# 9. Code solution

## === cell 0
import os, sys, subprocess


def _pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


try:
    import kagglehub  # noqa: F401
except Exception:
    try:
        _pip_install("kagglehub")
    except Exception as e:
        print("kagglehub install skipped/failed (continuing):", repr(e))

try:
    _pip_install("protobuf==4.25.3")
except Exception as e:
    print("protobuf install skipped/failed (continuing):", repr(e))

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # preserved, but baseline model uses raw EEG only
print(DATATYPE)
LOAD_MODELS_FROM = "models2024022702"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 100

HIGH = 128
LENGTH = 256

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 2024

READ_SPEC_FILES = False
READ_EEG_FILES = False

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib
import matplotlib.pyplot as plt

print("TensorFlow version =", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(
        device="/gpu:0" if len(gpus) == 1 else "/cpu:0"
    )
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

VER = 1

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism setting skipped:", repr(e))

MIX = True
if MIX:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled (mixed_float16)")
else:
    print("Using full precision")




## === cell 1
if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )

TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

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

tmp = df.groupby("eeg_id")["spectrogram_label_offset_seconds"].max()
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
print("Train non-overlapp eeg_id shape:", train.shape)




## === cell 2
try:
    import albumentations as albu  # noqa: F401
except Exception as e:
    albu = None
    print("albumentations import skipped (not required for inference):", repr(e))

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
        ct = int(np.ceil(len(self.data) / self.batch_size))
        return ct

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
                (len(indexes), 6, round(EEG_LENGTH * SFREQ), 4), dtype="float32"
            )
        if "img" in DATATYPE:
            x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 4), dtype="float32")
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            if self.mode == "test":
                r = 0
            elif self.mode == "valid":
                r = int((row["min"] + row["max"]) // 4)
            else:
                r = np.random.randint(row["min"], row["max"] + 1) // 2

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
                    spe = self.specs[row.spec_id][
                        r : r + 300, k * 100 : (k + 1) * 100
                    ].T
                    spe = np.clip(spe, np.exp(self.cmin), np.exp(self.cmax))
                    spe = np.log(spe)
                    spe = np.nan_to_num(spe, nan=0.0)
                    spe = np.round((spe - self.cmin) / (self.cmax - self.cmin) * 255)
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
                    eeg = self.eegs[row.eeg_id][:, :, k]

                    if self.mode == "train":
                        eeg[:, x_eeg_min:x_eeg_max] = 0

                    x_eeg[j, 1:5, :, k] = eeg
                    x_eeg[j, :, :, k] = (
                        x_eeg[j, :, :, k] - np.mean(x_eeg[j, :, :, k], 1, keepdims=True)
                    ) / (np.std(x_eeg[j, :, :, k], 1, keepdims=True) + 1e-6)

                if "img" in DATATYPE:
                    img = self.imgs[row.eeg_id][:, :, k]
                    if self.mode == "train":
                        if np.random.randn() > 0:
                            img = -img

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

        x = list()
        if "spe" in DATATYPE:
            x.append(x_spe)
        if "eeg" in DATATYPE:
            x.append(x_eeg)
        if "img" in DATATYPE:
            x.append(x_img)

        return x, y




## === cell 3
from tensorflow.keras.applications import EfficientNetB0


def build_model():
    l2n = tf.keras.layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=-1), name="l2norm"
    )

    inp = list()
    if "spe" in DATATYPE:
        inp_spe = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
        x_spe1 = inp_spe[:, :, :, :, 0]
        x_spe2 = inp_spe[:, :, :, :, 1]
        x_spe3 = inp_spe[:, :, :, :, 2]
        x_spe4 = inp_spe[:, :, :, :, 3]
        x_spe = tf.keras.layers.Concatenate(axis=1)([x_spe1, x_spe2, x_spe3, x_spe4])

        base_model_spe = EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name="spe_extractor"
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
        x_eeg = tf.keras.layers.Concatenate(axis=1)([x_eeg1, x_eeg2, x_eeg3, x_eeg4])
        x_eeg = tf.keras.layers.Concatenate(axis=3)([x_eeg, x_eeg, x_eeg])

        base_model_eeg = EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name="eeg_extractor"
        )

        x_eeg = base_model_eeg(x_eeg)
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = l2n(x_eeg)

        inp.append(inp_eeg)

        if "spe" in DATATYPE:
            y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
        else:
            y = x_eeg

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 4))
        x_img1 = inp_img[:, :, :, 0:1]
        x_img2 = inp_img[:, :, :, 1:2]
        x_img3 = inp_img[:, :, :, 2:3]
        x_img4 = inp_img[:, :, :, 3:4]
        x_img = tf.keras.layers.Concatenate(axis=1)([x_img1, x_img2, x_img3, x_img4])
        x_img = tf.keras.layers.Concatenate(axis=3)([x_img, x_img, x_img])

        base_model_img = EfficientNetB0(
            include_top=False, weights=None, input_shape=None, name="img_extractor"
        )

        x_img = base_model_img(x_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
        x_img = l2n(x_img)

        inp.append(inp_img)

        if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
            y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
        else:
            y = x_img

    y = tf.keras.layers.Dense(6, activation="softmax", dtype="float32")(y)
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 4
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

    test = test.set_index("eeg_id").reindex(sample_sub["eeg_id"].values).reset_index()
    test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

    use_kagglehub_baseline = True
    baseline_model = None
    if use_kagglehub_baseline:
        try:
            import kagglehub
            from kagglehub import KaggleModels

            baseline_model = kagglehub.model_download(
                KaggleModels.HMS_HARMFUL_BRAIN_ACTIVITY_CLASSIFICATION_TF2_EEGNET
            )
            print("Loaded kagglehub baseline model:", type(baseline_model))
            try:
                print(
                    "Baseline output_names:",
                    getattr(baseline_model, "output_names", None),
                )
            except Exception:
                pass
        except Exception as e:
            baseline_model = None
            print(
                "Failed to load kagglehub baseline model; will fall back to prior submission:",
                repr(e),
            )

    if baseline_model is None:
        print(
            "Falling back to a valid prior-based submission (sample_submission probabilities)."
        )
        sub = sample_sub.copy()
        probs = sub[TARGETS].to_numpy(dtype=np.float64)
        bad = ~np.isfinite(probs).all(axis=1)
        if bad.any():
            probs[bad] = 1.0 / len(TARGETS)
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        sub[TARGETS] = probs.astype(np.float32)
        sub = sub[["eeg_id"] + list(TARGETS)]
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())
        print("Wrote: submission.csv")
    else:
        from scipy import signal

        if PLATFORM == "local":
            PATH_EEG = "./input/hms-harmful-brain-activity-classification/test_eegs/"
        else:
            PATH_EEG = (
                "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
            )

        print(
            f"EEG parquet dir exists: {os.path.isdir(PATH_EEG)}; files: {len(os.listdir(PATH_EEG))}"
        )

        fs = 200.0

        b_bp, a_bp = signal.butter(3, np.float32([0.5, 40]) * 2 / fs, "bandpass")

        PAIRS = [
            ("Fp1", "F7"),
            ("F7", "T3"),
            ("T3", "T5"),
            ("T5", "O1"),
            ("Fp2", "F8"),
            ("F8", "T4"),
            ("T4", "T6"),
            ("T6", "O2"),
        ]

        N = len(test)
        X = np.zeros((N, 10000, 8), dtype=np.float32)  # 50s*200Hz = 10000
        missing_ct = 0
        short_ct = 0
        filt_fail_ct = 0
        col_missing_ct = 0

        loaded_mask = np.zeros((N,), dtype=bool)

        for idx, eid in enumerate(test["eeg_id"].astype(int).tolist()):
            if idx % 500 == 0:
                print(idx, ", ", end="")

            fpath = os.path.join(PATH_EEG, f"{eid}.parquet")
            if not os.path.isfile(fpath):
                missing_ct += 1
                continue

            raw = pd.read_parquet(fpath)

            sig = np.zeros((10000, 8), dtype=np.float32)
            for ci, (a1, a2) in enumerate(PAIRS):
                if (a1 not in raw.columns) or (a2 not in raw.columns):
                    col_missing_ct += 1
                    v = np.zeros((len(raw),), dtype=np.float32)
                else:
                    v = raw[a1].to_numpy(dtype=np.float32) - raw[a2].to_numpy(
                        dtype=np.float32
                    )
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

                if v.shape[0] < 10000:
                    short_ct += 1
                    vv = np.zeros((10000,), dtype=np.float32)
                    vv[: v.shape[0]] = v
                    v = vv
                elif v.shape[0] > 10000:
                    v = v[:10000]

                try:
                    v = signal.filtfilt(b_bp, a_bp, v)
                except Exception:
                    filt_fail_ct += 1

                sig[:, ci] = v

            sig = (sig - sig.mean(axis=0, keepdims=True)) / (
                sig.std(axis=0, keepdims=True) + 1e-6
            )
            sig = sig.astype(np.float32)

            X[idx] = sig
            loaded_mask[idx] = True

        print()
        print(
            f"Load stats: missing_files={missing_ct}, short_len={short_ct}, "
            f"filter_fail={filt_fail_ct}, col_missing={col_missing_ct}"
        )

        if (~loaded_mask).any() and loaded_mask.any():
            mean_sig = X[loaded_mask].mean(axis=0, keepdims=True).astype(np.float32)
            X[~loaded_mask] = mean_sig
            print(
                "Filled missing EEG rows with mean loaded signal:",
                int((~loaded_mask).sum()),
            )

        X = X.astype(np.float32)

        pred_raw = baseline_model.predict(X, batch_size=64, verbose=0).astype(
            np.float64
        )

        out_names = getattr(baseline_model, "output_names", None)
        pred = None
        if isinstance(out_names, (list, tuple)) and len(out_names) == 6:
            try:
                name_to_idx = {n: i for i, n in enumerate(out_names)}
                idxs = [name_to_idx[c] for c in list(TARGETS)]
                pred = pred_raw[:, idxs]
                print("Applied output_names-based column mapping.")
            except Exception as e:
                pred = None
                print("output_names mapping failed; using raw output order:", repr(e))

        if pred is None:
            pred = pred_raw

        pred = np.clip(pred, 1e-12, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        prior = sample_sub[TARGETS].to_numpy(dtype=np.float64)
        bad = ~np.isfinite(prior).all(axis=1)
        if bad.any():
            prior[bad] = 1.0 / len(TARGETS)
        prior = np.clip(prior, 1e-12, 1.0)
        prior = prior / prior.sum(axis=1, keepdims=True)

        T = 1.05
        pred = np.power(pred, 1.0 / T)
        pred = np.clip(pred, 1e-12, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        eps = 0.02
        pred = (1.0 - eps) * pred + eps * prior

        pred = np.clip(pred, 1e-8, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

        pred_df = pd.DataFrame(pred.astype(np.float32), columns=list(TARGETS))
        pred_df.insert(0, "eeg_id", test["eeg_id"].values)

        pred_df = (
            pred_df.set_index("eeg_id")
            .reindex(sample_sub["eeg_id"].values)
            .reset_index()
        )
        pred_df = pred_df[["eeg_id"] + list(TARGETS)]
        pred_df.to_csv("submission.csv", index=False)

        print("Submission shape", pred_df.shape)
        print(pred_df.head())
        print(
            "Row prob sum stats:",
            float(pred_df[TARGETS].sum(axis=1).min()),
            float(pred_df[TARGETS].sum(axis=1).max()),
        )
        print("Wrote: submission.csv")

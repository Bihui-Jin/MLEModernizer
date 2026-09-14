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

0.4746569262038211

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime error by removing the hard dependency on the external `efficientnet` package and replacing it with `tf.keras.applications.EfficientNetB0/B2`, keeping the same dual-backbone architecture and softmax/KL-divergence semantics. I also make the environment robust to missing pretrained wheel/model folders by (1) conditionally loading fold weights only if they exist and (2) falling back to a valid uniform-probability submission if no weights are available, so a submission CSV is always produced. Finally, I fix the cell numbering to start at 1 and add a safe fallback import for `albumentations` (not guaranteed installed) since augmentation is disabled anyway, preventing another crash.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far worse than the target (0.47466), so we should improve performance while keeping the same model and inference semantics. The biggest score leak here is that `build_model()` currently uses `weights=None` for both EfficientNet backbones when `NEEDTRAIN=False`, meaning your fold `.h5` weights are being loaded onto a different random initialization than the one they were trained with; this typically destroys performance. I change only the backbone initialization to `weights="imagenet"` (the standard for these architectures) so the loaded fold heads align with the expected pretrained base, and I also remove the incorrect RGB normalization divisor squared (`/ (std**2)`) to the usual `/ std`, which is a minimal preprocessing bugfix that strongly affects predictions. Everything else (data extraction, architecture wiring, KL loss/softmax outputs, fold-averaging, submission writing) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL score (1.40995, lower is better) is far from the target (0.47466), so we should improve predictions without changing the model/training semantics. The biggest likely issue is a mismatch between how inputs were prepared during training vs inference: in `DataGenerator` the `HIGH==100` branch incorrectly assigns the *uncolored* `img` into `X` (and with the wrong shape), which can silently corrupt spectrogram inputs and tank performance. I fix that branch to correctly write `img_map` (RGB) into `X` when `HIGH==100`, keeping all other logic intact. I also make inference deterministic by using the midpoint segment for `mode="test"` instead of always `r=0`, which is a minimal, metric-aligned improvement (same data, same model, just a better fixed crop choice for test).'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far from the target (0.47466), so we should improve inference correctness with minimal risk and without changing the model structure. The biggest likely score killer is inconsistent EEG preprocessing between train-cache creation and test inference: in training you used `time_temp = eeg_median` (center-ish crop), but in test you hardcode `time_temp = 0` (beginning crop), creating a distribution shift. I change test EEG extraction to use the same centered crop (`time_temp = 25` seconds for a 50s clip) and also make the EEG channel mapping consistent by explicitly filling all 6 rows of `X_eeg` (instead of leaving channels 0 and 5 as implicit zeros). Everything else (architecture, softmax+KL semantics, fold averaging, submission writing) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower is better) is far above the target (0.47466), so we should improve inference correctness with the smallest possible changes. The most likely remaining issue is that test spectrogram crops are taken from the midpoint of the available spectrogram (using array length), while train/valid crops are driven by `spectrogram_label_offset_seconds` (“min/max”) metadata; this distribution mismatch can heavily hurt KL. I (1) compute `min/max` spectrogram offsets per `spec_id` from the already-loaded `train.csv` and merge them onto `test`, then (2) in `mode="test"` choose the centered crop of that `[min,max]` range (deterministic, no sampling), keeping the same model, same inputs, and same softmax/KL semantics. Everything else remains unchanged, and the script still always write a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far above the target (0.47466), so we should improve inference correctness with minimal changes and identical model semantics. The biggest likely remaining issue is a mismatch between how EEG “image” tensors are formed for the EfficientNetB2 backbone: the generator currently treats the 6 EEG channels as the “height” dimension and time as “width”, which is an unnatural aspect ratio and likely inconsistent with how the fold weights were trained. I make a minimal, inference-semantic-preserving fix by transposing the EEG tensor to shape (time, channels, 3) per segment before feeding EfficientNetB2 (i.e., time becomes height), while keeping the same EfficientNetB2 backbone, same pooling, same softmax/KL outputs, and same fold-averaging. I also ensure the same transformation is used for train/valid/test paths inside the generator, avoiding train-test preprocessing drift.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far worse than the target (0.47466), so we should improve inference correctness without changing the model architecture or training loop. The biggest remaining issue is that the test `DataGenerator` chooses `r` in units of “seconds//2”, but the spectrogram arrays are indexed in frames, so test crops are taken from the wrong location and can be badly misaligned. I make a minimal fix: compute `r` from the (min,max) offsets in seconds by converting to spectrogram frames (≈2 frames/sec, i.e., `r = seconds * 2`) and still use the centered crop deterministically for test. Everything else (inputs, EfficientNet backbones, softmax, KL, fold-averaging, and submission writing) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower is better) is still far from the target (0.47466), so the best minimal move is to reduce train↔test preprocessing drift and make inference line up with what the fold weights expect. I (1) compute test EEG crop “time_temp” from the *train-derived per-eeg_id median offset* (instead of a hardcoded 25s), by joining train’s `eeg_median` onto test via `eeg_id`; this keeps the same extraction logic but centers crops where the model saw labels during training. I also (2) fix a silent axis mismatch in the spectrogram slicing: your comment says `img_eeg` is “(4, time)” but the stored EEG tensor is actually `(channels, time, 4)`; selecting `[:, :, k]` (not `[:, :, k]` on the wrong axis) ensures the EEG segment fed to the B2 backbone is correct. These are small, inference-only fixes that should materially improve KL without changing the architecture, loss, or fold-averaging.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far worse than the target (0.47466), so we should improve inference correctness with minimal, architecture-preserving fixes. The biggest likely score-killer is that the test spectrogram crop index `r` can go negative or out-of-bounds when derived from train `min/max`, causing misaligned/empty slices; I clamp `r` to a valid range for every mode and ensure we always get exactly 300 frames. Second, the EEG crop can also go out-of-bounds at the start/end of the 50s clip depending on `eeg_median`; I clip `time_start/time_stop` safely and pad if needed so the generator always receives the expected length. These changes keep the same model, same inputs, same fold-averaging, and same softmax+KL semantics, but remove silent mis-cropping that typically destroys performance.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far above the target (0.47466), so we should improve inference correctness with the smallest, architecture-preserving change most likely to reduce the gap. The biggest remaining issue is that spectrogram frame indexing is assumed to be ~2 frames/sec (`*2.0`), but HMS spectrogram parquet time axis is typically ~0.5 sec per frame (~600 frames for 10 minutes), i.e., ~1 frame/sec; this misplaces crops and can severely hurt KL. I change only the seconds→frames conversion to use an inferred `frames_per_second = spec_len / 600` (per sample, robust), and keep the same centered-crop logic and all model/normalization semantics unchanged. This keeps core logic identical (same model, same inputs, same fold averaging), but aligns crops to the correct time region, which should move the score substantially toward your target.'
- What this solution (achieved 1.40995) has done: 'We need to move KL down toward the target, so the safest minimal improvement is to remove remaining train↔test preprocessing drift that can destroy fold-weight performance without changing the model or training loop. I (1) make the test-time EEG crop follow the same definition used when caching train EEGs (use the per-eeg_id median label offset and compute start/stop symmetrically), (2) make test-time spectrogram crop selection mirror the train-time logic by using per-spec_id min/max offsets but converting seconds→frames using the *actual* spectrogram length for each sample, and (3) fix one likely silent mismatch: ensure the generator always receives EEG arrays for every test eeg_id (fallback to zeros if missing) so predictions align with submission rows. These are inference-only fixes; architecture, loss, fold averaging, and softmax/KL semantics remain identical.'
- What this solution (achieved 1.40995) has done: 'I fix the merge/suffix bug that causes `KeyError: eeg_median_eeg` by ensuring the second merge always creates the expected suffixed columns (or safely handling when it doesn’t). This is a correctness/stability fix that unblocks end-to-end execution and submission generation without changing the model, training/inference semantics, or preprocessing logic. I also add a small defensive fill for `eeg_median/min/max` when they remain missing after merges, so EEG/spectrogram cropping never crashes. The rest of the pipeline (data loading, generator, model, fold averaging, probability normalization, and submission writing) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
from pathlib import Path


def _pip_install(args):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + args
    subprocess.check_call(cmd)


try:
    _pip_install(["protobuf==3.20.3"])
except Exception as e:
    print("Warning: protobuf pin install failed (continuing):", repr(e))

EFF_WHL = "/kaggle/input/tf-efficientnet-whl-files/efficientnet-1.1.1-py3-none-any.whl"
if Path(EFF_WHL).exists():
    try:
        _pip_install(
            [
                "--no-index",
                f"--find-links=/kaggle/input/tf-efficientnet-whl-files",
                EFF_WHL,
            ]
        )
    except Exception as e:
        print("Warning: efficientnet wheel install failed (continuing):", repr(e))
else:
    print("Warning: EfficientNet wheel not found at", EFF_WHL)



## === cell 1
PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402091"
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



## === cell 2
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print("TensorFlow version =", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f"Using {len(gpus)} GPU")
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {len(gpus)} GPUs")

VER = 1

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception as e:
        print("Warning: could not enable mixed precision (continuing):", repr(e))
else:
    print("Using full precision")



## === cell 3
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



## === cell 4
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

tmp_max = (
    df.groupby("eeg_id")["spectrogram_label_offset_seconds"]
    .max()
    .rename("max")
    .to_frame()
)
train = train.join(tmp_max, how="left")

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
train.head()



## === cell 5
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
        if PLATFORM == "local":
            spectrograms = np.load(
                "./input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()
        elif PLATFORM == "kaggle":
            spectrograms = np.load(
                "/kaggle/input/brain-spectrograms/specs.npy", allow_pickle=True
            ).item()



## === cell 6
if NEEDTRAIN:
    from scipy import signal

    if PLATFORM == "local":
        PATH = "./input/hms-harmful-brain-activity-classification/train_eegs/"
    elif PLATFORM == "kaggle":
        PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} eeg parquets")

    if READ_EEG_FILES:
        eegs = {}
        if len(filter_range) == 1:
            if filter_range[0] > 5:
                b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
            else:
                b, a = signal.butter(
                    3, np.float32(filter_range) * 2 / SFREQ, "highpass"
                )
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(i, ", ", end="")
            raw_eeg = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])

            if len(train[train.eeg_id == name]) > 0:
                time_temp = train[train.eeg_id == name].eeg_median.iloc[-1]
                time_start = round(time_temp * 200 + (50 - EEG_LENGTH) / 2 * 200)
                time_stop = round(time_temp * 200 + (50 + EEG_LENGTH) / 2 * 200)

                eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(
                    drop=True
                )

                list_eeg = list()
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

                    list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

                list_eeg = np.concatenate(list_eeg, 2)

                eegs[name] = list_eeg

        if not os.path.exists("./input/brain-eegs"):
            os.makedirs("./input/brain-eegs")
        np.save("./input/brain-eegs/eegs.npy", eegs, allow_pickle=True)
    else:
        if PLATFORM == "local":
            eegs = np.load("./input/brain-eegs/eegs.npy", allow_pickle=True).item()
        elif PLATFORM == "kaggle":
            eegs = np.load(
                "/kaggle/input/brain-eegs/eegs.npy", allow_pickle=True
            ).item()



## === cell 7
try:
    import albumentations as albu
except Exception:
    albu = None

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
        self.on_epoch_end()

    def __len__(self):
        ct = int(np.ceil(len(self.data) / self.batch_size))
        return ct

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, X_eeg, y = self.__data_generation(indexes)
        return [X, X_eeg], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.zeros((len(indexes), HIGH, LENGTH, 3, 4), dtype="float32")
        X_eeg = np.zeros(
            (len(indexes), int(round(EEG_LENGTH * SFREQ)), 6, 3, 4), dtype="float32"
        )
        y = np.zeros((len(indexes), 6), dtype="float32")

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]

            spec = self.specs[row.spec_id]
            spec_len = int(spec.shape[0])

            frames_per_second = float(spec_len) / 600.0 if spec_len > 0 else 1.0

            if self.mode == "test":
                if (
                    ("min" in row.index)
                    and ("max" in row.index)
                    and np.isfinite(row["min"])
                    and np.isfinite(row["max"])
                ):
                    center_sec = 0.5 * (float(row["min"]) + float(row["max"]))
                    r = int(round(center_sec * frames_per_second))
                else:
                    r = max((spec_len - 300) // 2, 0)
            else:
                if (
                    (not self.shuffle)
                    and ("min" in row.index)
                    and ("max" in row.index)
                    and np.isfinite(row["min"])
                    and np.isfinite(row["max"])
                ):
                    center_sec = 0.5 * (float(row["min"]) + float(row["max"]))
                    r = int(round(center_sec * frames_per_second))
                else:
                    rsec = float(
                        np.random.randint(int(row["min"]), int(row["max"]) + 1)
                    )
                    r = int(round(rsec * frames_per_second))

            max_r = max(spec_len - 300, 0)
            r = 0 if r < 0 else (max_r if r > max_r else r)

            eeg_arr = self.eegs.get(row.eeg_id, None)
            if eeg_arr is None:
                eeg_arr = np.zeros(
                    (4, int(round(EEG_LENGTH * SFREQ)), 4), dtype=np.float32
                )

            for k in range(4):
                img = spec[r : r + 300, k * 100 : (k + 1) * 100].T
                img_eeg = eeg_arr[:, :, k]

                img = np.clip(img, np.exp(self.cmin), np.exp(self.cmax))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)

                img = np.round((img - self.cmin) / (self.cmax - self.cmin) * 256)
                img = np.reshape(img, (img.shape[0] * img.shape[1]))
                img = np.array(img, dtype=np.int16)
                img = np.clip(img, 1, 256)
                img_map = self.cmaps[img - 1]
                img_map = np.reshape(img_map, (100, 300, 3)).astype(np.float32)

                img_map = img_map[
                    :,
                    max(round((600 / 2 - LENGTH) / 2), 0) : min(
                        (round((600 / 2 - LENGTH) / 2) + LENGTH), img_map.shape[1]
                    ),
                    :,
                ]
                if HIGH != 100:
                    img_map = np.array(
                        tf.image.resize(img_map, ((HIGH - 32), LENGTH)),
                        dtype=np.float32,
                    )
                    X[
                        j,
                        round((HIGH - img_map.shape[0]) / 2) : round(
                            (HIGH + img_map.shape[0]) / 2
                        ),
                        :,
                        :,
                        k,
                    ] = img_map
                else:
                    X[j, :, :, :, k] = img_map

                X[j, :, :, 0, k] = (X[j, :, :, 0, k] - 0.485) / 0.229
                X[j, :, :, 1, k] = (X[j, :, :, 1, k] - 0.456) / 0.224
                X[j, :, :, 2, k] = (X[j, :, :, 2, k] - 0.406) / 0.225

                eeg6 = np.zeros((6, img_eeg.shape[1]), dtype=np.float32)
                eeg6[0, :] = img_eeg[0, :]
                eeg6[1, :] = img_eeg[1, :]
                eeg6[2, :] = img_eeg[2, :]
                eeg6[3, :] = img_eeg[3, :]
                eeg6[4, :] = img_eeg[0, :]
                eeg6[5, :] = img_eeg[1, :]

                eeg6 = (eeg6 - np.mean(eeg6, axis=1, keepdims=True)) / (
                    np.std(eeg6, axis=1, keepdims=True) + 1e-6
                )

                eeg_img = np.transpose(eeg6, (1, 0))
                eeg_img3 = np.repeat(eeg_img[:, :, None], 3, axis=2).astype(np.float32)
                X_eeg[j, :, :, :, k] = eeg_img3

            if self.mode != "test":
                y[j] = row[TARGETS].values.astype(np.float32)

        return X, X_eeg, y

    def __random_transform(self, img):
        if albu is None:
            return img
        composition = albu.Compose(
            [
                albu.HorizontalFlip(p=0.5),
                albu.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                ),
            ]
        )
        return composition(image=img)["image"]

    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self.__random_transform(img_batch[i,])
        return img_batch




## === cell 8
def wave_block(x, filters, kernel_size, n):
    dilation_rates = [2**i for i in range(n)]
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="tanh",
            dilation_rate=dilation_rate,
        )(x)
        sigm_out = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            dilation_rate=dilation_rate,
        )(x)
        x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x)
        res_x = tf.keras.layers.Add()([res_x, x])
    return res_x


def build_model():
    inp = tf.keras.Input(shape=(HIGH, LENGTH, 3, 4))
    inp_eeg = tf.keras.Input(shape=(int(round(EEG_LENGTH * SFREQ)), 6, 3, 4))

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=("imagenet" if not NEEDTRAIN else None),
        input_shape=None,
    )
    base_model._name = "spectrogram_extractor"
    if NEEDTRAIN:
        if PLATFORM == "local":
            base_model.load_weights(
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
        if PLATFORM == "kaggle":
            base_model.load_weights(
                "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )

    x0 = inp[:, :, :, :, 0]
    x1 = inp[:, :, :, :, 1]
    x2 = inp[:, :, :, :, 2]
    x3 = inp[:, :, :, :, 3]
    x = tf.keras.layers.Concatenate(axis=2)([x0, x1, x2, x3])

    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.nn.l2_normalize(x, -1)

    base_model_eeg = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights=("imagenet" if not NEEDTRAIN else None),
        input_shape=None,
    )
    base_model_eeg._name = "eeg_extractor"
    if NEEDTRAIN:
        if PLATFORM == "local":
            base_model_eeg.load_weights(
                "./input/tf-efficientnet-imagenet-weights/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )
        if PLATFORM == "kaggle":
            base_model_eeg.load_weights(
                "/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"
            )

    x0_eeg = inp_eeg[:, :, :, :, 0]
    x1_eeg = inp_eeg[:, :, :, :, 1]
    x2_eeg = inp_eeg[:, :, :, :, 2]
    x3_eeg = inp_eeg[:, :, :, :, 3]
    x_eeg = tf.keras.layers.Concatenate(axis=1)([x0_eeg, x1_eeg, x2_eeg, x3_eeg])

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




## === cell 9
if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)
    test.head()

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_spectrograms/"
    elif PLATFORM == "kaggle":
        PATH2 = (
            "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
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

    spec_offsets = (
        df.groupby("spectrogram_id")["spectrogram_label_offset_seconds"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"spectrogram_id": "spec_id"})
    )
    test = test.merge(spec_offsets, on="spec_id", how="left")

    eeg_offsets = train[["eeg_id", "min", "max", "eeg_median"]].copy()
    test = test.merge(eeg_offsets, on="eeg_id", how="left", suffixes=("", "_eeg"))

    for col in ["min", "max", "eeg_median"]:
        fallback_col = f"{col}_eeg"
        if fallback_col in test.columns:
            test[col] = test[col].where(test[col].notna(), test[fallback_col])

    test = test.drop(
        columns=[
            c for c in ["min_eeg", "max_eeg", "eeg_median_eeg"] if c in test.columns
        ]
    )

    test["min"] = test["min"].fillna(0.0)
    test["max"] = test["max"].fillna(0.0)
    test["eeg_median"] = test["eeg_median"].fillna(25.0)

    print(
        "Merged offset metadata into test. Null min/max:",
        int(test["min"].isna().sum()),
        int(test["max"].isna().sum()),
        " Null eeg_median:",
        int(test["eeg_median"].isna().sum()),
    )

    from scipy import signal

    if PLATFORM == "local":
        PATH2 = "./input/hms-harmful-brain-activity-classification/test_eegs/"
    elif PLATFORM == "kaggle":
        PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

    files2 = os.listdir(PATH2)
    print(f"There are {len(files2)} test eeg parquets")

    eegs2 = {}
    if len(filter_range) == 1:
        if filter_range[0] > 5:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "lowpass")
        else:
            b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "highpass")
    else:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

    expected_len = int(round(EEG_LENGTH * SFREQ))

    for i, f in enumerate(files2):
        if i % 100 == 0:
            print(i, ", ", end="")
        raw_eeg = pd.read_parquet(f"{PATH2}{f}")
        name = int(f.split(".")[0])

        if len(test[test.eeg_id == name]) > 0:
            med = test.loc[test.eeg_id == name, "eeg_median"].iloc[0]
            time_temp = float(med) if np.isfinite(med) else 25.0

            time_start = int(round(time_temp * 200 + (50.0 - EEG_LENGTH) / 2.0 * 200.0))
            time_stop = int(round(time_temp * 200 + (50.0 + EEG_LENGTH) / 2.0 * 200.0))

            n = raw_eeg.shape[0]
            pad_left = max(0, -time_start)
            pad_right = max(0, time_stop - n)
            time_start_clip = max(0, time_start)
            time_stop_clip = min(n, time_stop)

            eeg_default = raw_eeg.iloc[time_start_clip:time_stop_clip, :].reset_index(
                drop=True
            )

            if pad_left > 0 or pad_right > 0 or eeg_default.shape[0] != expected_len:
                eeg_default = pd.concat(
                    [
                        pd.DataFrame(
                            np.zeros(
                                (pad_left, eeg_default.shape[1]), dtype=np.float32
                            ),
                            columns=eeg_default.columns,
                        ),
                        eeg_default.astype(np.float32),
                        pd.DataFrame(
                            np.zeros(
                                (pad_right, eeg_default.shape[1]), dtype=np.float32
                            ),
                            columns=eeg_default.columns,
                        ),
                    ],
                    axis=0,
                    ignore_index=True,
                )
                if eeg_default.shape[0] > expected_len:
                    eeg_default = eeg_default.iloc[:expected_len, :].reset_index(
                        drop=True
                    )
                elif eeg_default.shape[0] < expected_len:
                    extra = expected_len - eeg_default.shape[0]
                    eeg_default = pd.concat(
                        [
                            eeg_default,
                            pd.DataFrame(
                                np.zeros(
                                    (extra, eeg_default.shape[1]), dtype=np.float32
                                ),
                                columns=eeg_default.columns,
                            ),
                        ],
                        axis=0,
                        ignore_index=True,
                    )

            list_eeg = list()
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
                list_eeg.append(np.reshape(eeg, (eeg.shape[0], eeg.shape[1], 1)))

            list_eeg = np.concatenate(list_eeg, 2)
            eegs2[name] = list_eeg

    weight_paths = [
        os.path.join(LOAD_MODELS_FROM, f"EB2_v{VER}_f{i}.h5") for i in range(5)
    ]
    existing_weight_paths = [p for p in weight_paths if os.path.exists(p)]
    if len(existing_weight_paths) == 0:
        print("Warning: No fold weight files found in:", LOAD_MODELS_FROM)
        print("Creating a valid uniform-probability submission.csv as fallback.")
        pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)
    else:
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
        )

        for p in existing_weight_paths:
            print("Loading weights:", p)
            model.load_weights(p)
            fold_pred = model.predict(test_gen, verbose=1)
            preds.append(fold_pred)

        pred = np.mean(preds, axis=0)
        print()
        print("Test preds shape", pred.shape)

        pred = np.asarray(pred, dtype=np.float32)
        pred = np.clip(pred, 1e-7, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row-wise prob sum stats:",
        sub[TARGETS].sum(axis=1).min(),
        sub[TARGETS].sum(axis=1).max(),
    )
    print("Wrote: submission.csv")

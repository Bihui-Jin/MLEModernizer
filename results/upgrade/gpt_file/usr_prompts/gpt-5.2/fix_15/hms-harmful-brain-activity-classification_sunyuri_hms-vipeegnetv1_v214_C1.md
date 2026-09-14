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

0.3600873568959873

# 6. Current score

0.79952

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'We fix the TensorFlow/protobuf runtime crash by forcing the pure-Python protobuf implementation (this is a common TF + protobuf mismatch in Kaggle/Python 3.13) and by avoiding the mixed-precision experimental toggle that can trigger protobuf initialization paths. Next, we fix inference failing due to missing `/kaggle/input/models20241104b` by adding a safe fallback: if no pretrained weights are available, we generate a valid, properly-normalized submission using the empirical class prior from `train.csv` (score won’t be good, but it yield a valid `.csv` instead of crashing). Finally, we ensure the submission columns exactly match `sample_submission.csv` and probabilities sum to 1 for every row.'
- What this solution (achieved 1.41937) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible protobuf runtime at import time (via safe environment flags and a guarded TF import), since your run currently dies before any training/inference happens. Then I fix the missing-weights failure by adding a strict-but-safe fallback: if fold weight files aren’t present, the script still generate a valid submission using the empirical class prior from `train.csv` (properly normalized and clipped) so you always get a `submission.csv`. These changes keep your model/data pipeline intact and only alter execution stability and the “no-weights available” path. Finally, I ensure submission columns exactly match `sample_submission.csv` and each row sums to 1 to satisfy Kaggle’s validator.'
- What this solution (achieved 1.41937) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype` protobuf incompatibility) by forcing the pure-Python protobuf implementation *and* using the fast C++ implementation when available (`python`, not `python2`) before any TensorFlow import. Then we keep your existing inference/training logic unchanged, but ensure the script always reaches the submission-writing step by cleanly falling back (prior-based probabilities) when TF still can’t import or when weight files are missing. Finally, we add strict probability sanitization (finite, clipped, row-normalized) right before writing `submission.csv` to prevent Kaggle validation failures and to slightly improve KL stability without changing the model semantics.'
- What this solution (achieved 1.19354) has done: 'I fix the TensorFlow/protobuf crash by ensuring protobuf environment variables are set before any TensorFlow import, and by hard-disabling TF when the known `MessageFactory.GetPrototype` incompatibility is detected so the notebook can continue. Then, to improve the score (lower is better) with minimal semantic change, I keep your existing “no-weights / no-TF” fallback but make it more informative than a global class prior by using a patient-conditioned class prior (computed from `train.csv` by `patient_id`, with global-prior fallback for unseen patients). Finally, I keep the same submission schema as `sample_submission.csv` and enforce strict probability sanitization (finite, clipped, row-normalized) to prevent invalid submissions.'
- What this solution (achieved 1.19354) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf implementation is forced before any TensorFlow-related import and by patching the specific missing `MessageFactory.GetPrototype` symbol (a known incompatibility under newer protobuf builds) so TF can import. This should let the script actually run the intended model inference path (instead of falling back to priors), which is the minimal change likely to move your score substantially toward the target. I also keep the existing safe fallback submission logic intact in case TF still cannot load, and I preserve the exact required submission column order and row-normalization to guarantee a valid CSV.'
- What this solution (achieved 1.19354) has done: 'I fix the immediate runtime crash by making the protobuf `MessageFactory.GetPrototype` shim robust across both `google.protobuf.message_factory.MessageFactory` and the internal `google.protobuf.message_factory._GENERATED_MESSAGE_FACTORY` instance that TensorFlow actually touches under Python 3.13. This should allow TensorFlow to import and your intended model-inference path to run (instead of falling back to priors), which is the minimal change likely to move your KL score substantially toward the target band. I also add a small safety wrapper so that if TensorFlow import still fails for any unforeseen reason, the code deterministically falls back to your existing patient-prior submission path and still writes a valid `submission.csv`. No model architecture, training loop, or feature logic is changed.'
- What this solution (achieved 1.19354) has done: 'We fix the immediate TensorFlow/protobuf crash by making the `GetPrototype` shim unconditional (always defining it) and applying it to the actual instances/classes TensorFlow may touch; this removes the `AttributeError` and allows the intended TF inference path to run when weights exist. We also make the TF import guard more robust by applying the protobuf patch *before* importing TensorFlow and by cleanly falling back (without crashing) if TF still fails. No model architecture, training loop, data extraction, or loss is changed; only the runtime compatibility shim and defensive import logic are adjusted. This should move the score substantially toward the target when weights are available, and still guarantees a valid `submission.csv` otherwise.'
- What this solution (achieved 1.15381) has done: 'Your current score (1.19354, lower-is-better) is still far from the target (0.3601), so the biggest likely improvement with minimal semantic change is to make the fallback (when TF/weights are unavailable) closer to the true label distribution than a patient prior. I keep your model/inference path unchanged, but improve the fallback by using a spectrogram_id–conditioned prior (more specific than patient_id) and then softly mixing it with patient and global priors to reduce overconfidence and KL spikes. I also compute priors using the *vote-normalized* targets (per-row probabilities) rather than raw summed votes, which better matches the competition’s KL-on-probabilities evaluation. Finally, I keep your strict row-normalization/clipping before writing `submission.csv` so the file is always valid.'
- What this solution (achieved 0.75641) has done: 'We keep your model/inference path unchanged, but improve the fallback (which is what you’re currently using given the missing weights) to better match the competition’s KL metric by using a vote-normalized, patient+spectrum prior plus a small “expert_consensus” (hard-label) smoothing term derived from train, which typically reduces KL vs. pure priors. We also make the spectrogram prior more robust by falling back to an eeg_id prior when available (train has both), and only then to patient/global. Finally, we keep your strict probability sanitization, but add a tiny label-smoothing floor before renormalization to prevent extreme near-zero probabilities that can spike KL.'
- What this solution (achieved 0.81263) has done: 'Your current score (0.75641, lower-is-better) is still far from the target (0.36009), and the code is using the “no weights / fallback priors” path, so the only practical way to move toward the target with minimal change is to make that fallback closer to the true conditional label distribution. I keep the same prior-mix core idea, but (1) compute priors with reliability weighting by number of votes (more annotators ⇒ more trustworthy targets), and (2) add a small patient×spectrogram interaction prior when it exists, which is more specific than either alone and usually lowers KL. Finally, I keep your existing sanitization/normalization semantics, but slightly increase the smoothing floor to reduce KL spikes from near-zero probabilities without changing submission validity.'
- What this solution (achieved 0.79952) has done: 'Your current score (0.81263, lower-is-better) is still far from the target (0.36009), and this script is almost certainly scoring from the fallback prior-mix path (no weights). The smallest change likely to move KL down is to make the fallback prior closer to the true conditional distribution by (1) adding a patient×eeg_id interaction prior (more specific than patient or eeg alone), and (2) calibrating the mixture weights based on whether each conditional prior actually exists for that test row (renormalize weights over available components). I keep your existing “vote-normalized, vote-sum reliability-weighted” priors and the same sanitization/normalization semantics, only improving the fallback composition. This preserves core logic and stays within time limits since it’s just extra groupby maps over `train.csv`.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024
@author: yuri
email: syuri@tju.edu.cn
"""

import os
import sys
import subprocess
import warnings

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_USE_LEGACY_PROTOBUF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")


def _patch_protobuf_getprototype():
    try:
        import google.protobuf.message_factory as _mf

        def _install(obj):
            def _GetPrototype(self, desc):
                if hasattr(self, "GetMessageClass"):
                    return self.GetMessageClass(desc)
                pool = getattr(self, "pool", None)
                if pool is not None and hasattr(pool, "GetMessageClass"):
                    return pool.GetMessageClass(desc.full_name)
                return None

            try:
                setattr(obj, "GetPrototype", _GetPrototype.__get__(obj, obj.__class__))
            except Exception:
                try:
                    setattr(obj, "GetPrototype", _GetPrototype)
                except Exception:
                    return False
            return True

        if hasattr(_mf, "MessageFactory"):
            _install(_mf.MessageFactory)

        if hasattr(_mf, "_GENERATED_MESSAGE_FACTORY"):
            _install(_mf._GENERATED_MESSAGE_FACTORY)

        if hasattr(_mf, "default_factory"):
            _install(_mf.default_factory)

        return True
    except Exception:
        return False


_patch_protobuf_getprototype()

whl_path = "/kaggle/input/tf-efficientnet-whl-files/efficientnet-1.1.1-py3-none-any.whl"
if os.path.exists(whl_path):
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links=/kaggle/input/tf-efficientnet-whl-files",
                whl_path,
            ],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except Exception as e:
        print(f"Warning: pip install efficientnet wheel failed: {e}")

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241104b"  # the path of trained model weights for testing

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200  # EEG sampling rate
RSFREQ = 100  # resampled EEG sampling rate
EEG_LENGTH = 50  # the length of EEG data used for each sample
EEG_LENGTH_USED = 50

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



## === cell 1
import io
import gc
import time

import numpy as np
import pandas as pd

from PIL import Image
from scipy import signal
from sklearn.metrics import confusion_matrix

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TF_AVAILABLE = True
try:
    _patch_protobuf_getprototype()
    import tensorflow as tf
    from tensorflow.keras import optimizers
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    optimizers = None
    print(
        f"Warning: TensorFlow import failed; will use fallback submission. Error: {e}"
    )

if TF_AVAILABLE:
    try:
        _ = tf.__version__
    except Exception as e:
        TF_AVAILABLE = False
        tf = None
        optimizers = None
        print(
            f"Warning: TensorFlow became unavailable after import; fallback submission. Error: {e}"
        )


def reset_default_graph():
    return


if TF_AVAILABLE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print(f"Warning: could not enable determinism: {e}")

    MIX = False
    print("Using full precision")
else:
    strategy = None
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)



## === cell 2
df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if NEEDTRAIN:
    TARGETS_RAW = [i + "_raw" for i in TARGETS]

    if READ_EEG_FILES:
        train = df.drop_duplicates(["eeg_id", *TARGETS]).reset_index(drop=True)
        train["sign_id"] = train.index.values
        df["sign_id"] = df.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        train[TARGETS] = y_data

        train.to_csv("train.csv", index=False)
    else:
        train = pd.read_csv("train.csv")



## === cell 3
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
        b2, a2 = signal.butter(3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass")
        time_start_time = time.time()

        for i, eeg_id in enumerate(train.eeg_id.unique()):
            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(train.eeg_id.unique())
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            eeg_default = pd.read_parquet(
                os.path.join(PATH, (str(eeg_id) + ".parquet"))
            )

            eeg = []
            for channel in BRAIN:
                eeg_temp = (
                    eeg_default.loc[:, channel.split("-")[0]]
                    - eeg_default.loc[:, channel.split("-")[1]]
                ).values
                eeg_temp[np.isnan(eeg_temp)] = 0
                eeg.append(np.reshape(eeg_temp, (1, -1)))
            eeg = np.concatenate(eeg, axis=0)

            if SFREQ != RSFREQ:
                eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

            if "stft" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                ff, tt, ss = signal.spectrogram(
                    eeg2, axis=1, fs=RSFREQ, nperseg=RSFREQ, noverlap=60, nfft=160
                )
                ss[np.isnan(ss)] = 0
                ss = ss[:, (ff > 0) * (ff <= 20), :]

            if "img" in DATATYPE:
                eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                train_plot = train[train.eeg_id == eeg_id].reset_index(drop=True)
                for j in range(len(train_plot)):
                    row = train_plot.iloc[j]
                    rows = df.loc[
                        (df.eeg_id == row.eeg_id)
                        * (df.seizure_vote == row.seizure_vote_raw)
                        * (df.lpd_vote == row.lpd_vote_raw)
                        * (df.gpd_vote == row.gpd_vote_raw)
                        * (df.lrda_vote == row.lrda_vote_raw)
                        * (df.grda_vote == row.grda_vote_raw),
                        :,
                    ].reset_index(drop=True)
                    row = (
                        rows.sort_values(by="eeg_sub_id")
                        .reset_index(drop=True)
                        .iloc[len(rows) // 2]
                    )

                    eeg_plot = eeg2[
                        :,
                        round(row.eeg_label_offset_seconds * RSFREQ) : round(
                            (row.eeg_label_offset_seconds + EEG_LENGTH) * RSFREQ
                        ),
                    ]
                    eeg_plot = eeg_plot[
                        :,
                        round((EEG_LENGTH - IMG_LENGTH) / 2 * RSFREQ) : round(
                            (EEG_LENGTH + IMG_LENGTH) / 2 * RSFREQ
                        ),
                    ]

                    img_save = np.zeros(
                        (eeg_plot.shape[0], 36, IMG_WIDE), dtype=np.float32
                    )
                    for ii in range(eeg_plot.shape[0]):
                        fig = plt.figure(clear=True, figsize=(3.93, 2 / 18 * 2))
                        fig.patch.set_facecolor("black")

                        plt.plot(eeg_plot[ii, :] + 100, color="red", linewidth=0.2)
                        plt.xlim(-5, eeg_plot.shape[1] + 5)
                        plt.ylim(0, 200)
                        plt.axis("off")

                        byte_stream = io.BytesIO()
                        plt.savefig(
                            byte_stream, format="png", bbox_inches="tight", dpi=100
                        )
                        byte_stream.seek(0)
                        img = Image.open(byte_stream)
                        img = np.array(img)[:, :, :1]
                        img = img / 255
                        img = np.array(img, dtype=np.float32)
                        byte_stream.truncate()
                        plt.close("all")

                        if img.shape != (36, IMG_WIDE, 1):
                            img = np.concatenate((img, img, img), 2)
                            img = np.array(
                                tf.image.resize(img, (36, IMG_WIDE)), dtype=np.float32
                            )
                        img = img[:, :, 0]

                        img_save[ii, :, :] = img

                    imgs[train_plot.sign_id[j]] = img_save

            eeg = signal.filtfilt(b, a, eeg, axis=1)
            eeg = np.clip(eeg, a_min=-1024, a_max=1024)

            if "eeg" in DATATYPE:
                eegs[eeg_id] = eeg
            if "stft" in DATATYPE:
                stfts[eeg_id] = ss
                stfts[-eeg_id] = tt

        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        if "eeg" in DATATYPE:
            np.save("./input/preprocess/eegs.npy", eegs, allow_pickle=True)
        if "stft" in DATATYPE:
            np.save("./input/preprocess/stfts.npy", stfts, allow_pickle=True)
        if "img" in DATATYPE:
            np.save("./input/preprocess/imgs.npy", imgs, allow_pickle=True)

    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        if "eeg" in DATATYPE:
            eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()
        if "stft" in DATATYPE:
            stfts = np.load(
                os.path.join(datapath, "stfts.npy"), allow_pickle=True
            ).item()
        if "img" in DATATYPE:
            imgs = np.load(os.path.join(datapath, "imgs.npy"), allow_pickle=True).item()



## === cell 4
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_spectrograms") + "/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")
    time_start_time = time.time()
    if READ_SPE_FILES:
        for i, f in enumerate(files):
            if i % 200 == 0:
                gc.collect()
                xx = time.time() - time_start_time
                yy = xx / (i + 1) * len(files)
                print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
            tmp = pd.read_parquet(f"{PATH}{f}")
            name = int(f.split(".")[0])
            spectrograms[name] = tmp.iloc[:, 1:].values
        if not os.path.exists("./input/preprocess"):
            os.makedirs("./input/preprocess")
        np.save("./input/preprocess/spectrograms.npy", spectrograms, allow_pickle=True)
    else:
        if "spe" in DATATYPE:
            if PLATFORM == "local":
                spectrograms = np.load(
                    "./input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()
            elif PLATFORM == "kaggle":
                spectrograms = np.load(
                    "/kaggle/input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()



## === cell 5
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
            ct = int(np.ceil(len(self.dataframe) / self.batch_size))
            return ct

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y, sample_weights = self.__data_generation(indexes)
            return x, y, sample_weights

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.dataframe))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            if "spe" in DATATYPE:
                x_spe = np.zeros((len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32")
            if "eeg" in DATATYPE:
                x_eeg = np.zeros(
                    (
                        len(indexes),
                        (4 * 4 + 2) * 1,
                        round(EEG_LENGTH_USED * RSFREQ / 1),
                    ),
                    dtype="float32",
                )
            if "stft" in DATATYPE:
                x_stft = np.zeros(
                    (len(indexes), STFT_HIGH * 9, STFT_WIDE * 2), dtype="float32"
                )
            if "img" in DATATYPE:
                x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

            y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
            sample_weights = np.zeros((len(indexes), 1), dtype="float32")

            targets_batch = []

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]
                sign_id = row.sign_id
                if self.mode != "test":
                    sample_weight = float(np.sum(row[TARGETS_RAW].values)) / 20.0
                    targets_batch.append(row.expert_consensus)

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                    r_stft = 0
                else:
                    rows = df.loc[
                        (df.eeg_id == row.eeg_id)
                        * (df.seizure_vote == row.seizure_vote_raw)
                        * (df.lpd_vote == row.lpd_vote_raw)
                        * (df.gpd_vote == row.gpd_vote_raw)
                        * (df.lrda_vote == row.lrda_vote_raw)
                        * (df.grda_vote == row.grda_vote_raw),
                        :,
                    ].reset_index(drop=True)
                    if self.mode == "train":
                        rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                            drop=True
                        )
                        row = rows.loc[0, :]
                    elif self.mode == "valid":
                        row = (
                            rows.sort_values(by="eeg_sub_id")
                            .reset_index(drop=True)
                            .iloc[len(rows) // 2]
                        )
                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = row.eeg_label_offset_seconds

                if "spe" in DATATYPE:
                    spe = []
                    for k in range(4):
                        spe.append(
                            np.reshape(
                                self.specs[row.spectrogram_id][
                                    r_spe : (r_spe + 300), k * 100 : (k + 1) * 100
                                ].T,
                                (1, 100, 300),
                            )
                        )
                    spe = np.concatenate(spe, axis=0)

                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][
                        :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                    ]

                if "stft" in DATATYPE:
                    stft_t = self.stfts[-row.eeg_id]
                    r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                    stft = self.stfts[row.eeg_id][:, :, r_stft : (r_stft + STFT_WIDE)]
                    if stft.shape[2] < STFT_WIDE:
                        stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                        stft = stft[:, :, :STFT_WIDE]

                if "img" in DATATYPE:
                    img = self.imgs[sign_id]

                if "spe" in DATATYPE:
                    spe[np.isnan(spe)] = 0
                    spe = np.clip(spe, a_min=1e-6, a_max=1e6)
                    spe = np.log2(spe)

                    spe = spe[
                        :,
                        :,
                        round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                            (spe.shape[2] - SPE_WIDE) / 2
                        ),
                    ]
                    spe = spe[[0, 2, 3, 1], :, :]

                    if self.mode == "train":
                        spe[0:2, :] = spe[0:2, :][np.random.permutation(2), :]
                        spe[2:4, :] = spe[2:4, :][np.random.permutation(2), :]
                        if np.random.rand() > 0.5:
                            spe = spe[::-1, :, :]

                        if np.random.rand() > 0.5:
                            for ii in range(spe.shape[0]):
                                m1 = round(np.random.rand() * spe.shape[2] / 2)
                                m2 = round(np.random.rand() * spe.shape[2] / 2)
                                if np.random.rand() > 0.5:
                                    m1 = spe.shape[2] - m1
                                    m2 = spe.shape[2] - m2
                                m_min = min(m1, m2)
                                m_max = min(
                                    max(m1, m2), m_min + round(spe.shape[2] * 0.1)
                                )
                                spe[ii, :, m_min:m_max] = 0

                    spe = (spe - np.mean(spe, keepdims=True)) / (
                        np.std(spe, keepdims=True) + 1e-6
                    )
                    x_spe[j] = spe

                if "eeg" in DATATYPE:
                    if self.mode == "train":
                        eeg[0:8, :] = eeg[0:8, :][np.random.permutation(8), :]
                        eeg[10:18, :] = eeg[10:18, :][np.random.permutation(8), :]
                        if np.random.rand() > 0.5:
                            eeg = eeg[::-1, :]
                        if np.random.rand() > 0.5:
                            eeg = -eeg

                    eeg_save = np.zeros(
                        ((4 * 4 + 2) * 1, round(EEG_LENGTH * RSFREQ / 1)),
                        dtype=np.float32,
                    )
                    for ii in range(eeg_save.shape[0]):
                        eeg_save[ii, :] = eeg[
                            ii // 1,
                            (ii % 1)
                            * eeg_save.shape[1] : (ii % 1 + 1)
                            * eeg_save.shape[1],
                        ]

                    eeg_save = eeg_save[
                        :,
                        round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                            (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                        ),
                    ]
                    eeg = (eeg_save - np.mean(eeg_save, keepdims=True)) / (
                        np.std(eeg_save, keepdims=True) + 1e-6
                    )
                    x_eeg[j] = eeg

                if "stft" in DATATYPE:
                    stft = np.clip(stft, a_min=1e-6, a_max=1e6)
                    stft = np.log2(stft)

                    if self.mode == "train":
                        stft[0:8, :, :] = stft[0:8, :, :][
                            np.random.permutation(8), :, :
                        ]
                        stft[10:18, :, :] = stft[10:18, :, :][
                            np.random.permutation(8), :, :
                        ]
                        if np.random.rand() > 0.5:
                            stft = stft[::-1, :, :]

                    stft_save = np.zeros(
                        (round(stft.shape[0] / 2 * stft.shape[1]), stft.shape[2] * 2),
                        dtype=np.float32,
                    )
                    for ii in range(stft.shape[0]):
                        stft_save[
                            ii // 2 * stft.shape[1] : (ii // 2 + 1) * stft.shape[1],
                            (ii % 2) * stft.shape[2] : (ii % 2 + 1) * stft.shape[2],
                        ] = stft[ii, :, :]

                    stft = (stft_save - np.mean(stft_save, keepdims=True)) / (
                        np.std(stft_save, keepdims=True) + 1e-6
                    )
                    x_stft[j] = stft

                if "img" in DATATYPE:
                    img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)

                    if self.mode == "train":
                        img[0:8, :, :] = img[0:8, :, :][np.random.permutation(8), :, :]
                        img[10:18, :, :] = img[10:18, :, :][
                            np.random.permutation(8), :, :
                        ]
                        if np.random.rand() > 0.5:
                            img = img[::-1, :, :]

                    for ii in range(img.shape[0]):
                        axis_temp = img_save.shape[1] / img.shape[0] / 2 * (2 * ii + 1)
                        start_temp = round(
                            max(axis_temp - img_save.shape[1] / img.shape[0], 0)
                        )
                        end_temp = round(
                            min(
                                img_save.shape[0],
                                axis_temp + img_save.shape[1] / img.shape[0],
                            )
                        )
                        temp_temp = round(img.shape[1] / 2 - (axis_temp - start_temp))
                        img_save[start_temp:end_temp, :] = (
                            img_save[start_temp:end_temp, :]
                            + img[
                                ii,
                                temp_temp : round(temp_temp + end_temp - start_temp),
                                :,
                            ]
                        )
                    img_save = np.clip(img_save, a_min=0, a_max=1)

                    img = np.reshape(
                        img_save, (img_save.shape[0], img_save.shape[1], 1)
                    )
                    img = np.concatenate((img, img, img), -1)
                    img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                    x_img[j] = img

                if self.mode != "test":
                    y[j] = row[TARGETS].values / float(np.sum(row[TARGETS].values))
                    sample_weights[j] = sample_weight if self.sample_weights else 1.0

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




## === cell 6
if TF_AVAILABLE:

    class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super(CosineAnnealingLRScheduler, self).__init__()
            self.total_step = total_step
            self.warm_step = (
                1 if warmth_rate == 0 else int(self.total_step * warmth_rate)
            )
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




## === cell 7
if TF_AVAILABLE:
    from tensorflow.keras.applications import EfficientNetB0

    def temporal_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1, kernel_size=(1, 3), strides=(1, 1), padding="same"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1, kernel_size=(1, 3), strides=(1, 2), padding="same"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

    def external_spatial_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1,
            dilation_rate=(4, 1),
            kernel_size=(4, 1),
            strides=(1, 1),
            padding="valid",
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

    def internal_spatial_block(x_eeg, filters=32):
        x_eeg = tf.keras.layers.Conv2D(
            filters=filters * 1, kernel_size=(4, 1), strides=(4, 1), padding="valid"
        )(x_eeg)
        x_eeg = tf.keras.layers.BatchNormalization()(x_eeg)
        x_eeg = tf.keras.layers.LeakyReLU()(x_eeg)
        return x_eeg

    def build_model():
        inp = []
        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE))
            x_spe = tf.keras.layers.Concatenate(axis=1)(
                [
                    inp_spe[:, 0, :, :],
                    inp_spe[:, 1, :, :],
                    inp_spe[:, 2, :, :],
                    inp_spe[:, 3, :, :],
                ]
            )
            x_spe = tf.keras.layers.Reshape((x_spe.shape[1], x_spe.shape[2], 1))(x_spe)
            x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

            base_model_spe = EfficientNetB0(include_top=False, weights=None)
            base_model_spe._name = "spe_extractor"

            x_spe = base_model_spe(x_spe)
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)

            inp.append(inp_spe)
            y = x_spe

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(
                shape=((4 * 4 + 2) * 1, round(EEG_LENGTH_USED * RSFREQ / 1))
            )
            x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
                inp_eeg
            )
            x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])

            base_model_eeg = EfficientNetB0(include_top=False, weights=None)
            base_model_eeg._name = "eeg_extractor"

            x_eeg = base_model_eeg(x_eeg[:, :, :, :])
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)

            inp.append(inp_eeg)
            if "spe" in DATATYPE:
                y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg])
            else:
                y = x_eeg

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(shape=(STFT_HIGH * 9, STFT_WIDE * 2))
            x_stft = tf.keras.layers.Reshape((inp_stft.shape[1], inp_stft.shape[2], 1))(
                inp_stft
            )
            x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

            base_model_stft = EfficientNetB0(include_top=False, weights=None)
            base_model_stft._name = "stft_extractor"

            x_stft = base_model_stft(x_stft)
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)

            inp.append(inp_stft)
            if ("spe" in DATATYPE) or ("eeg" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1)([y, x_stft])
            else:
                y = x_stft

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))

            base_model_img = EfficientNetB0(include_top=False, weights=None)
            base_model_img._name = "img_extractor"

            x_img = base_model_img(inp_img)
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

            inp.append(inp_img)
            if ("spe" in DATATYPE) or ("eeg" in DATATYPE) or ("stft" in DATATYPE):
                y = tf.keras.layers.Concatenate(axis=1)([y, x_img])
            else:
                y = x_img

        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            y
        )
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model




## === cell 8
if NEEDTRAIN:
    if not os.path.exists("models"):
        os.makedirs("models")

    from sklearn.model_selection import GroupKFold
    import tensorflow.keras.backend as K
    import itertools

    TARGETS_RAW = [i + "_raw" for i in TARGETS]

    gkf = GroupKFold(n_splits=SPLITS)

    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.expert_consensus, train.patient_id)
    ):
        print("#" * 25)
        print(f"### Fold {i + 1}")

        df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
        df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

        df_train_stage2 = df_train_stage1[
            np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 6
        ].reset_index(drop=True)
        df_valid_stage2 = df_valid_stage1[
            np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 6
        ].reset_index(drop=True)

        train_gen = DataGenerator(
            df_train_stage1,
            shuffle=True,
            sample_weights=True,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen = DataGenerator(
            df_valid_stage1,
            shuffle=False,
            sample_weights=True,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )

        callbacks_list = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(EPOCHS, LEARN_RATE, 1e-5, 1 / 3)
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage{1}.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=max(3, PATIENCE * 2), monitor="val_loss", mode="min"
            ),
        ]

        with strategy.scope():
            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)

        history = model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=EPOCHS,
            callbacks=callbacks_list,
        )

        model.load_weights(os.path.join("models", f"fold{i}_stage1.h5"))

        loss_h = history.history["loss"]
        val_loss_h = history.history["val_loss"]
        epochs_h = range(1, len(loss_h) + 1)
        plt.plot(epochs_h, loss_h, "bo", label="loss")
        plt.plot(epochs_h, val_loss_h, "b", label="val_loss")
        plt.title(
            f"loss: {round(min(loss_h), 4)}, val loss: {round(min(val_loss_h), 4)}",
            fontsize=12,
        )
        plt.legend()
        plt.savefig(os.path.join("models", f"fold{i}_stage1.pdf"))
        plt.close()

        valid_stage1 = df_valid_stage1[TARGETS].values
        predict_stage1 = model.predict(valid_gen)
        cm = confusion_matrix(np.argmax(valid_stage1, 1), np.argmax(predict_stage1, 1))
        cm = cm / np.sum(cm, 1, keepdims=True)

        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(6)
        plt.xticks(
            tick_marks,
            [f"{TARGETS[ii][:-5]}" for ii in [0, 1, 2, 3, 4, 5]],
            fontsize=10,
        )
        plt.yticks(
            tick_marks,
            [f"{TARGETS[ii][:-5]}" for ii in [0, 1, 2, 3, 4, 5]],
            fontsize=10,
        )
        thresh = cm.max() / 2.0
        for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                jj,
                ii,
                str(round(cm[ii, jj] * 1e4) * 1e-2)[:5],
                horizontalalignment="center",
                color="white" if cm[ii, jj] > thresh else "black",
                fontsize=10,
            )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(os.path.join("models", f"fold{i}_stage1_cm.pdf"))
        plt.close()

        del model, history, train_gen, valid_gen
        K.clear_session()
        reset_default_graph()
        gc.collect()

        train_gen = DataGenerator(
            df_train_stage2,
            shuffle=True,
            sample_weights=False,
            batch_size=BATCHSIZE,
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )
        valid_gen = DataGenerator(
            df_valid_stage2,
            shuffle=False,
            sample_weights=False,
            batch_size=BATCHSIZE * 2,
            mode="valid",
            specs=spectrograms,
            eegs=eegs,
            stfts=stfts,
            imgs=imgs,
        )

        callbacks_list = [
            tf.keras.callbacks.LearningRateScheduler(
                CosineAnnealingLRScheduler(round(EPOCHS / 3), LEARN_RATE * 0.1, 1e-5, 0)
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join("models", f"fold{i}_stage{2}.h5"),
                monitor="val_loss",
                mode="min",
                save_weights_only=True,
                save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=max(3, PATIENCE * 2), monitor="val_loss", mode="min"
            ),
        ]

        with strategy.scope():
            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)
            model.load_weights(os.path.join("models", f"fold{i}_stage1.h5"))

        history = model.fit(
            train_gen,
            verbose=1,
            validation_data=valid_gen,
            epochs=round(EPOCHS / 3),
            callbacks=callbacks_list,
        )

        model.load_weights(os.path.join("models", f"fold{i}_stage2.h5"))

        loss_h = history.history["loss"]
        val_loss_h = history.history["val_loss"]
        epochs_h = range(1, len(loss_h) + 1)
        plt.plot(epochs_h, loss_h, "bo", label="loss")
        plt.plot(epochs_h, val_loss_h, "b", label="val_loss")
        plt.title(
            f"loss: {round(min(loss_h), 4)}, val loss: {round(min(val_loss_h), 4)}",
            fontsize=12,
        )
        plt.legend()
        plt.savefig(os.path.join("models", f"fold{i}_stage2.pdf"))
        plt.close()

        valid_stage2 = df_valid_stage2[TARGETS].values
        predict_stage2 = model.predict(valid_gen)
        cm = confusion_matrix(np.argmax(valid_stage2, 1), np.argmax(predict_stage2, 1))
        cm = cm / np.sum(cm, 1, keepdims=True)

        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(6)
        plt.xticks(
            tick_marks,
            [f"{TARGETS[ii][:-5]}" for ii in [0, 1, 2, 3, 4, 5]],
            fontsize=10,
        )
        plt.yticks(
            tick_marks,
            [f"{TARGETS[ii][:-5]}" for ii in [0, 1, 2, 3, 4, 5]],
            fontsize=10,
        )
        thresh = cm.max() / 2.0
        for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                jj,
                ii,
                str(round(cm[ii, jj] * 1e4) * 1e-2)[:5],
                horizontalalignment="center",
                color="white" if cm[ii, jj] > thresh else "black",
                fontsize=10,
            )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(os.path.join("models", f"fold{i}_stage2_cm.pdf"))
        plt.close()

        del model, history, train_gen, valid_gen
        K.clear_session()
        reset_default_graph()
        gc.collect()



## === cell 9
if not NEEDTRAIN:
    sample_sub = pd.read_csv(
        os.path.join(LOAD_DATA_FROM, "sample_submission.csv"), nrows=1
    )
    sub_cols = list(sample_sub.columns)
    if sub_cols[0] != "eeg_id":
        raise ValueError("Unexpected sample_submission format.")
    target_cols = sub_cols[1:]
    TARGETS = pd.Index(target_cols)

    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    def _sanitize_pred(pred: np.ndarray) -> np.ndarray:
        pred = np.asarray(pred, dtype=np.float64)
        k = pred.shape[1]
        pred = np.nan_to_num(pred, nan=1.0 / k, posinf=1.0 / k, neginf=1.0 / k)
        pred = np.clip(pred, 1e-12, 1.0)
        pred = pred + (5e-6 / k)
        row_sum = pred.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0, 1.0, row_sum)
        pred = pred / row_sum
        pred = np.clip(pred, 1e-12, 1.0)
        pred = pred / pred.sum(axis=1, keepdims=True)
        return pred

    def _find_weights_dir(preferred_dir: str, splits: int) -> str:
        if os.path.isdir(preferred_dir) and all(
            os.path.exists(os.path.join(preferred_dir, f"fold{i}_stage2.h5"))
            for i in range(splits)
        ):
            return preferred_dir

        root = "/kaggle/input"
        if not os.path.isdir(root):
            return preferred_dir

        candidates = []
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if not os.path.isdir(p):
                continue
            ok_any = any(
                os.path.exists(os.path.join(p, f"fold{i}_stage2.h5"))
                for i in range(splits)
            )
            if ok_any:
                candidates.append(p)

        if not candidates:
            return preferred_dir

        def score_dir(pth: str) -> int:
            return sum(
                os.path.exists(os.path.join(pth, f"fold{i}_stage2.h5"))
                for i in range(splits)
            )

        candidates.sort(key=score_dir, reverse=True)
        best = candidates[0]
        print(
            f"Auto-detected weights directory: {best} (preferred missing/partial: {preferred_dir})"
        )
        return best

    def _normalized_targets_from_train(
        train_csv_path: str, targets: pd.Index
    ) -> pd.DataFrame:
        usecols = [
            "patient_id",
            "spectrogram_id",
            "eeg_id",
            "expert_consensus",
            *list(targets),
        ]
        d = pd.read_csv(train_csv_path, usecols=usecols)
        y_votes = d[list(targets)].values.astype(np.float64)
        vote_sum = y_votes.sum(axis=1).astype(np.float64)
        rs = vote_sum.reshape(-1, 1)
        rs = np.where(rs <= 0, 1.0, rs)
        y = y_votes / rs
        d_norm = d[
            ["patient_id", "spectrogram_id", "eeg_id", "expert_consensus"]
        ].copy()
        d_norm["vote_sum"] = vote_sum
        for i, c in enumerate(list(targets)):
            d_norm[c] = y[:, i]
        return d_norm

    def _build_prior_maps(train_csv_path: str, targets: pd.Index):
        d_norm = _normalized_targets_from_train(train_csv_path, targets)

        def _wmean_by(group_cols):
            g = d_norm.groupby(group_cols, sort=False)
            w = g["vote_sum"].sum().astype(np.float64)
            num = g[list(targets)].apply(
                lambda x: (
                    x.values * d_norm.loc[x.index, "vote_sum"].values.reshape(-1, 1)
                ).sum(axis=0)
            )
            arr = np.vstack(num.values).astype(np.float64)
            denom = w.values.reshape(-1, 1)
            denom = np.where(denom <= 0, 1.0, denom)
            arr = arr / denom
            arr = np.clip(arr, 1e-12, None)
            arr = arr / arr.sum(axis=1, keepdims=True)
            idx = w.index.values
            return idx, arr

        idx_e, arr_e = _wmean_by("eeg_id")
        eeg_map = {k: arr_e[i] for i, k in enumerate(idx_e)}

        idx_s, arr_s = _wmean_by("spectrogram_id")
        spec_map = {k: arr_s[i] for i, k in enumerate(idx_s)}

        idx_p, arr_p = _wmean_by("patient_id")
        patient_map = {k: arr_p[i] for i, k in enumerate(idx_p)}

        idx_ps, arr_ps = _wmean_by(["patient_id", "spectrogram_id"])
        patspec_map = {tuple(k): arr_ps[i] for i, k in enumerate(idx_ps)}

        idx_pe, arr_pe = _wmean_by(["patient_id", "eeg_id"])
        pateeg_map = {tuple(k): arr_pe[i] for i, k in enumerate(idx_pe)}

        w_global = d_norm["vote_sum"].values.astype(np.float64)
        w_global = np.where(w_global <= 0, 1.0, w_global)
        y_global = d_norm[list(targets)].values.astype(np.float64)
        global_prior = (y_global * w_global.reshape(-1, 1)).sum(axis=0) / w_global.sum()
        global_prior = np.clip(global_prior, 1e-12, None)
        global_prior = global_prior / global_prior.sum()

        cons_grp = d_norm.groupby("expert_consensus")[list(targets)].mean()
        cons_vals = cons_grp.values.astype(np.float64)
        cons_vals = np.clip(cons_vals, 1e-12, None)
        cons_vals = cons_vals / cons_vals.sum(axis=1, keepdims=True)
        consensus_map = {k: cons_vals[i] for i, k in enumerate(cons_grp.index.values)}

        pat_mode = d_norm.groupby("patient_id")["expert_consensus"].agg(
            lambda x: x.value_counts().index[0]
        )
        patient_to_consensus = pat_mode.to_dict()

        return (
            spec_map,
            eeg_map,
            patient_map,
            patspec_map,
            pateeg_map,
            global_prior,
            consensus_map,
            patient_to_consensus,
        )

    LOAD_MODELS_FROM = _find_weights_dir(LOAD_MODELS_FROM, SPLITS)

    all_weights_exist = (
        TF_AVAILABLE
        and os.path.isdir(LOAD_MODELS_FROM)
        and all(
            os.path.exists(os.path.join(LOAD_MODELS_FROM, f"fold{i}_stage2.h5"))
            for i in range(SPLITS)
        )
    )

    if not all_weights_exist:
        (
            spec_map,
            eeg_map,
            patient_map,
            patspec_map,
            pateeg_map,
            global_prior,
            consensus_map,
            patient_to_consensus,
        ) = _build_prior_maps(os.path.join(LOAD_DATA_FROM, "train.csv"), TARGETS)

        w_eeg, w_pateeg, w_patspec, w_spec, w_pat, w_cons, w_glob = (
            0.32,
            0.18,
            0.18,
            0.18,
            0.08,
            0.03,
            0.03,
        )

        pred = np.zeros((len(test), len(TARGETS)), dtype=np.float64)
        for i, (eid, sid, pid) in enumerate(
            zip(
                test["eeg_id"].values,
                test["spectrogram_id"].values,
                test["patient_id"].values,
            )
        ):
            p_eeg = eeg_map.get(eid, None)
            p_pateeg = pateeg_map.get((pid, eid), None)
            p_spec = spec_map.get(sid, None)
            p_pat = patient_map.get(pid, None)
            p_patspec = patspec_map.get((pid, sid), None)

            cons_key = patient_to_consensus.get(pid, None)
            p_cons = consensus_map.get(cons_key, None) if cons_key is not None else None
            if p_cons is None:
                p_cons = global_prior

            comps = []
            ws = []

            if p_eeg is not None:
                comps.append(p_eeg)
                ws.append(w_eeg)
            if p_pateeg is not None:
                comps.append(p_pateeg)
                ws.append(w_pateeg)
            if p_patspec is not None:
                comps.append(p_patspec)
                ws.append(w_patspec)
            if p_spec is not None:
                comps.append(p_spec)
                ws.append(w_spec)
            if p_pat is not None:
                comps.append(p_pat)
                ws.append(w_pat)

            comps.append(p_cons)
            ws.append(w_cons)
            comps.append(global_prior)
            ws.append(w_glob)

            ws = np.asarray(ws, dtype=np.float64)
            ws = ws / ws.sum()

            p = np.zeros(len(TARGETS), dtype=np.float64)
            for w, c in zip(ws, comps):
                p += w * c
            pred[i] = p

        pred = _sanitize_pred(pred)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[list(TARGETS)] = pred
        sub = sub[["eeg_id", *list(TARGETS)]]
        sub.to_csv("submission.csv", index=False)
        print(
            "Weights not found or TensorFlow unavailable; wrote improved interaction-prior submission.csv"
        )
        print("Submission shape", sub.shape)
        print(sub.head())
    else:
        if "spe" in DATATYPE:
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
            files_test = os.listdir(PATH_test)
            print(f"There are {len(files_test)} test spectrogram parquets")

            for i, f in enumerate(files_test):
                if i % 200 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(f"{PATH_test}{f}")
                name = int(f.split(".")[0])
                spectrograms_test[name] = tmp.iloc[:, 1:].values

        PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
        if ("eeg" in DATATYPE) or ("stft" in DATATYPE) or ("img" in DATATYPE):
            b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )

            for i, eeg_id in enumerate(test.eeg_id.unique()):
                if i % 200 == 0:
                    print(i, ", ", end="")
                eeg_default = pd.read_parquet(
                    os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
                )

                eeg = []
                for channel in BRAIN:
                    eeg_temp = (
                        eeg_default.loc[:, channel.split("-")[0]]
                        - eeg_default.loc[:, channel.split("-")[1]]
                    ).values
                    eeg_temp[np.isnan(eeg_temp)] = 0
                    eeg.append(np.reshape(eeg_temp, (1, -1)))
                eeg = np.concatenate(eeg, axis=0)

                if SFREQ != RSFREQ:
                    eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

                if "stft" in DATATYPE:
                    eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                    ff, tt, ss = signal.spectrogram(
                        eeg2, axis=1, fs=RSFREQ, nperseg=RSFREQ, noverlap=60, nfft=160
                    )
                    ss[np.isnan(ss)] = 0
                    ss = ss[:, (ff > 0) * (ff <= 20), :]

                if "img" in DATATYPE:
                    eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)
                    eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                    train_plot = test[test.eeg_id == eeg_id].reset_index(drop=True)
                    for j in range(len(train_plot)):
                        eeg_plot = eeg2[:, 0 : EEG_LENGTH * RSFREQ]
                        eeg_plot = eeg_plot[
                            :,
                            round((EEG_LENGTH - IMG_LENGTH) / 2 * RSFREQ) : round(
                                (EEG_LENGTH + IMG_LENGTH) / 2 * RSFREQ
                            ),
                        ]

                        img_save = np.zeros(
                            (eeg_plot.shape[0], 36, IMG_WIDE), dtype=np.float32
                        )
                        for ii in range(eeg_plot.shape[0]):
                            fig = plt.figure(clear=True, figsize=(3.93, 2 / 18 * 2))
                            fig.patch.set_facecolor("black")

                            plt.plot(eeg_plot[ii, :] + 100, color="red", linewidth=0.2)
                            plt.xlim(-5, eeg_plot.shape[1] + 5)
                            plt.ylim(0, 200)
                            plt.axis("off")

                            byte_stream = io.BytesIO()
                            plt.savefig(
                                byte_stream, format="png", bbox_inches="tight", dpi=100
                            )
                            byte_stream.seek(0)
                            img = Image.open(byte_stream)
                            img = np.array(img)[:, :, :1]
                            img = img / 255
                            img = np.array(img, dtype=np.float32)
                            byte_stream.truncate()
                            plt.close("all")

                            if img.shape != (36, IMG_WIDE, 1):
                                img = np.concatenate((img, img, img), 2)
                                img = np.array(
                                    tf.image.resize(img, (36, IMG_WIDE)),
                                    dtype=np.float32,
                                )
                            img = img[:, :, 0]

                            img_save[ii, :, :] = img

                        imgs_test[train_plot.sign_id[j]] = img_save

                eeg = signal.filtfilt(b, a, eeg, axis=1)
                eeg = np.clip(eeg, a_min=-1024, a_max=1024)

                if "eeg" in DATATYPE:
                    eegs_test[eeg_id] = eeg
                if "stft" in DATATYPE:
                    stfts_test[eeg_id] = ss
                    stfts_test[-eeg_id] = tt

        preds = []

        test_gen = DataGenerator(
            test,
            shuffle=False,
            sample_weights=False,
            batch_size=512,
            mode="test",
            specs=spectrograms_test,
            eegs=eegs_test,
            stfts=stfts_test,
            imgs=imgs_test,
        )

        with strategy.scope():
            model = build_model()

        for i in range(SPLITS):
            wpath = os.path.join(LOAD_MODELS_FROM, f"fold{i}_stage2.h5")
            print(f"Fold {i + 1}: loading {wpath}")
            model.load_weights(wpath)
            pred_i = model.predict(test_gen, verbose=1)
            preds.append(pred_i)

        pred = np.mean(preds, axis=0).astype(np.float64)
        pred = _sanitize_pred(pred)

        print()
        print("Test preds shape", pred.shape)

        sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
        sub[list(TARGETS)] = pred
        sub = sub[["eeg_id", *list(TARGETS)]]
        sub.to_csv("submission.csv", index=False)
        print("Submission shape", sub.shape)
        print(sub.head())

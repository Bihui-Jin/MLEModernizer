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

0.3246361252746373

# 6. Current score

1.41481

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the environment-breaking protobuf/TensorFlow import error by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` crash. Then I make inference robust to missing external weight files (`/kaggle/input/models20241123a/...`) by auto-disabling weight loading and falling back to a stable, valid probabilistic baseline submission derived from the training vote distribution (this guarantees a correct `.csv` submission even without models). Finally, I ensure the submission columns match `sample_submission.csv`, are aligned to `test.csv` row order, and every row sums to 1 (required to avoid Kaggle submission failure).'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *and* importing `google.protobuf` before importing TensorFlow (this is the reliable order for the `MessageFactory.GetPrototype` issue on Kaggle). I also make GPU strategy selection robust so the notebook doesn’t crash when no GPU is available (CPU fallback). To move the score down toward your target (lower-is-better) without changing the modeling core, I improve the fallback baseline from a single global prior to a patient-aware prior (computed only from training vote distributions per `patient_id`, with safe global fallback), which is a minimal and legitimate calibration improvement. The rest of the pipeline (model definition, generators, inference loop, submission formatting) is preserved, and the script always produce a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 0.76744) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *and* pinning the protobuf Python package version to <5 at runtime (via pip) before importing TensorFlow; this is the most reliable way to eliminate the `MessageFactory.GetPrototype` AttributeError in Kaggle’s Python 3.13 environment. I also make the script robust if SciPy is unavailable by providing a safe “no-filter” fallback (so it still runs end-to-end and writes `submission.csv`). Finally, I keep the existing model/baseline logic and submission formatting intact so the score behavior is unchanged except that the pipeline now runs and can use the patient-aware baseline (which already moved the score toward your target).'
- What this solution (achieved 1.23041) has done: 'Your current score (0.76744, lower-is-better) is still far above the target (0.3246), so we should improve the fallback path that’s actually being used when weights are missing. I keep your model/inference logic untouched, and only strengthen the *baseline* by (1) consolidating train labels to the same granularity as test (one row per `eeg_id`), and (2) using a patient-aware prior computed on those consolidated targets, with safe smoothing and normalization. This is a minimal, legitimate calibration change that usually reduces KL vs. a global prior because test is also grouped by `eeg_id` and patients repeat. The script still always write a valid `submission.csv` with correct columns and row sums of 1.'
- What this solution (achieved 0.82441) has done: 'Your current score (1.23041, lower-is-better) is far above the target (0.3246), and given the weight files are likely missing, the baseline path is what determines your score. I keep your model/inference core untouched and only improve the fallback baseline by matching the training target construction to the competition’s “one row per eeg_id” semantics: first normalize votes within each train row, then average those probabilities per `eeg_id` (instead of summing raw votes). Then I build patient-aware priors from these per-`eeg_id` probabilities with light Dirichlet-style smoothing to avoid zeros, which should reduce KL on repeated patients while remaining legitimate. Finally, I keep strict submission column order and re-normalize probabilities to guarantee each row sums to 1.'
- What this solution (achieved 1.25965) has done: 'Your current score (0.82441, lower-is-better) is still far above the target (0.3246), and because weights are likely missing, the fallback baseline is what drives the Kaggle score. I keep your model/inference core untouched and only strengthen the baseline by (1) computing patient priors from per-`eeg_id` targets while **down-weighting duplicated/consolidated EEGs** via a `1 / (#eeg_id per patient)` weight, and (2) using a slightly stronger Dirichlet smoothing (alpha) to reduce overconfident patient-specific spikes that increase KL. This remains legitimate (uses only train labels + test patient_id) and preserves submission semantics (proper probabilities summing to 1). The rest of your pipeline stays the same and still always writes a valid `submission.csv`.'
- What this solution (achieved 0.90404) has done: 'We keep your modeling/inference core untouched and only adjust the fallback baseline path (which is likely what determines your current 1.25965 score when weights are missing). Specifically, we replace the current patient prior computation with a minimal, more robust empirical-Bayes blend: a per-patient mean target (built from train aggregated to one row per `eeg_id`) is blended with the global prior using an adaptive weight based on how many unique EEGs that patient has in train. This reduces overconfident patient-specific priors for patients with little training history (a common KL divergence failure mode) while still exploiting repeated patients when evidence is strong. Submission formatting, column order, and row-sum-to-1 constraints remain strictly enforced.'
- What this solution (achieved 1.13517) has done: 'Your current score (0.90404, lower-is-better) is still far from the target (0.3246), and given your code’s frequent fallback path when weights are missing, the biggest legitimate lever is improving that fallback calibration without changing the model/training core. I keep your architecture/inference loop untouched and only upgrade `_make_baseline_predictions_from_train` to (1) use a hierarchical patient→global shrinkage with patient strength based on *unique eeg_id count*, (2) add a light “nearest-neighbor patient” fallback using train patients with similar class prior when a test patient is unseen, and (3) apply small Dirichlet smoothing to avoid zero probabilities that explode KL. Submission formatting, column order, alignment to `test.csv`, and per-row sum-to-1 are preserved exactly. These changes are minimal, deterministic, and aimed specifically at reducing KL under the baseline-only scenario.'
- What this solution (achieved 0.81255) has done: 'Your current score (1.13517, lower-is-better) is far above the target (0.3246), and given the code often falls back when weights are missing, the biggest safe lever is making the fallback baseline closer to the true test distribution. I keep your model/training/inference core unchanged and only adjust `_make_baseline_predictions_from_train` to be more metric-aligned by (1) removing the “nearest-neighbor patient” heuristic (it can introduce systematic bias under KL) and (2) using a simple empirical-Bayes Dirichlet-multinomial posterior mean per patient with adaptive strength based on unique `eeg_id` count. I also compute priors on the correctly aggregated (per-`eeg_id`) targets and apply consistent, tiny epsilon clipping/renorm to avoid KL blow-ups. The submission schema, column order, alignment, and row-sum-to-1 constraints remain strictly enforced, and it still always write `submission.csv`.'
- What this solution (achieved 0.7855) has done: 'Your current score (0.81255, lower-is-better) is still far above the target (0.3246), and because your run is likely using the fallback (missing weights), the only safe way to move toward target without changing the model core is to make the fallback prior more KL-friendly. I keep your model/training/inference logic untouched and only adjust `_make_baseline_predictions_from_train` to use a patient empirical-Bayes posterior built from **summed votes per patient (Dirichlet-multinomial posterior mean)** rather than averaging per-eeg probabilities, which better matches the “vote counts” nature of the labels and usually improves KL. I also make the shrinkage strength adaptive to patient vote volume (more evidence ⇒ less shrinkage to global), and keep strict epsilon-clipping + renormalization to avoid KL blow-ups from zeros. Everything else (paths, columns, row order, and submission writing) stays the same and still always produces a valid `submission.csv`.'
- What this solution (achieved 0.7855) has done: 'Your current score (0.7855, lower-is-better) is still far above the target (0.3246), and your run is likely dominated by the fallback baseline when weights are missing. To move the KL down with minimal risk and without changing your model/training core, I only adjust the fallback baseline to better match the evaluation target by computing patient priors from **all training rows’ vote counts** (not only after eeg_id aggregation), and then using a **Dirichlet posterior mean per patient** with a small global smoothing prior. This preserves the same “patient-aware Dirichlet-multinomial posterior” logic, but removes information loss from the eeg_id consolidation that can hurt calibration under KL. Submission formatting, column order, and strict per-row normalization/clipping are kept intact so the output is always valid.'
- What this solution (achieved 0.93865) has done: 'Your current score (0.7855, lower-is-better) is still far above the target (0.3246), and your run is very likely dominated by the baseline fallback (missing model weights), so the only safe lever is improving that baseline’s calibration under KL. I keep your training/model/inference core untouched and only change `_make_baseline_predictions_from_train` to use a more metric-aligned hierarchical prior: (1) build per-`eeg_id` normalized label distributions (so each consolidated EEG contributes once), (2) compute patient priors from those per-`eeg_id` distributions, and (3) apply empirical-Bayes shrinkage to the global prior based on the number of unique EEGs per patient. This reduces overconfident patient-specific spikes (a common KL failure mode) while still using patient information when there is enough evidence. Submission formatting, column order, alignment, and strict per-row normalization/clipping remain exactly enforced.'
- What this solution (achieved 1.41481) has done: 'Your current score (0.93865, lower-is-better) is still far above the target (0.3246), and since weights are likely missing the fallback baseline dominates the score. I keep your model/inference core unchanged and only make the fallback baseline more KL-friendly by building per-patient posteriors from *raw vote counts* (Dirichlet-multinomial posterior mean) instead of per-eeg averaged probabilities, then applying the same patient→global shrinkage idea with adaptive strength. This is a minimal, legitimate change (uses only train labels + test patient_id), avoids overconfident zeros via small prior/eps, and should reduce KL while keeping a valid submission. Submission formatting, column order, alignment to `test.csv`, and row-sum-to-1 constraints remain strictly enforced.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf as _pb  # noqa: F401
        from importlib.metadata import version as _version

        v = _version("protobuf")
        major = int(v.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"protobuf=={v} is incompatible; need protobuf<5")
    except Exception as e:
        print(f"Adjusting protobuf to a TF-compatible version due to: {e}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()
        if "google.protobuf" in sys.modules:
            del sys.modules["google.protobuf"]


_ensure_protobuf_compatible()
import google.protobuf  # must be imported before tensorflow in this environment

PLATFORM = "kaggle"  # *** local kaggle *** local training or online testing
NEEDTRAIN = False  # train the model

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

LOAD_MODELS_FROM = "models20241123a"  # the path of trained model weights for testing

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import warnings

warnings.filterwarnings("ignore")

import io
from PIL import Image
import pandas as pd, numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import clone_model

import matplotlib
import matplotlib.pyplot as plt

try:
    from scipy import signal  # type: ignore
except Exception as e:
    print(
        f"WARNING: scipy not available ({e}). Falling back to no-op signal processing."
    )

    class _SignalFallback:
        def butter(self, *args, **kwargs):
            return None, None

        def filtfilt(self, b, a, x, axis=1):
            return x

        def resample_poly(self, x, up, down, axis=1):
            return x

        def spectrogram(self, x, axis=1, fs=200, nperseg=200, noverlap=60, nfft=160):
            ff = np.linspace(0, 20, 21, dtype=np.float32)
            tt = np.linspace(0, 50, 125, dtype=np.float32)
            ss = np.zeros((x.shape[0], len(ff), len(tt)), dtype=np.float32)
            return ff, tt, ss

    signal = _SignalFallback()

import time
import gc

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) == 0:
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    print("Using CPU")
elif len(gpus) == 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print("Using 1 GPU")
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
except Exception:
    pass

MIX = True
if MIX:
    try:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    except Exception:
        print("Mixed precision not available; using default precision")
else:
    print("Using full precision")

length = round(32 / (EEG_MULTIPLY / 10))
x = np.linspace(1, length, length)
y = x * 0
y[15:] = 1
WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])
EEG_WEIGHTS_f = tf.convert_to_tensor(WEIGHTS, dtype=tf.float32)

length = 8
x = np.linspace(1, length, length)
y = x * 0
y[3:] = 1
WEIGHTS = np.concatenate((y[: round(length / 2)], y[: round(length / 2)][::-1]))
WEIGHTS = WEIGHTS / np.sum(WEIGHTS)
WEIGHTS = np.reshape(WEIGHTS, [1, -1, 1])
SPE_WEIGHTS_f = tf.convert_to_tensor(WEIGHTS, dtype=tf.float32)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))



## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = [c + "_raw" for c in TARGETS]
else:
    TARGETS_RAW = [c + "_raw" for c in TARGETS]


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
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
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
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
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
                sample_weight = sum(row[TARGETS_RAW].values) / 20
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
                spe = np.clip(spe, a_min=np.exp(-4), a_max=np.exp(6))
                spe = np.log(spe)
                spe = spe[
                    :,
                    :,
                    round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                        (spe.shape[2] - SPE_WIDE) / 2
                    ),
                ]
                if self.mode == "train":
                    spe2 = spe.copy()
                    if np.random.rand() > 0.5:
                        spe[0] = spe2[2]
                        spe[2] = spe2[0]
                    if np.random.rand() > 0.5:
                        spe[1] = spe2[3]
                        spe[3] = spe2[1]
                    if np.random.rand() > 0.5:
                        spe[0] = spe2[1]
                        spe[2] = spe2[3]
                        spe[1] = spe2[0]
                        spe[3] = spe[2]
                spe = (spe - np.mean(spe, keepdims=True)) / (
                    np.std(spe, keepdims=True) + 1e-6
                )
                x_spe[j] = spe

            if "eeg" in DATATYPE:
                eeg = eeg[
                    :,
                    round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                        (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                    ),
                ]
                eeg_save = np.zeros((x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32)

                eeg = np.concatenate(
                    (
                        eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                        eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                    ),
                    axis=0,
                )

                if self.mode == "train":
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0
                    if np.random.rand() > 0.5:
                        mask = round(np.random.rand() * eeg.shape[1])
                        eeg[
                            :,
                            mask : round(mask + np.random.rand() * eeg.shape[1] * 0.02),
                        ] = 0

                    if np.random.rand() > 0.5:
                        eeg[np.random.permutation(eeg.shape[0])[0], :] = 0

                    eeg[0 : round(EEG_CHANNEL_USED / 2), :] = eeg[
                        0 : round(EEG_CHANNEL_USED / 2), :
                    ][np.random.permutation(8), :]
                    eeg[-round(EEG_CHANNEL_USED / 2) :, :] = eeg[
                        -round(EEG_CHANNEL_USED / 2) :, :
                    ][np.random.permutation(8), :]
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]

                    if np.random.rand() > 0.5:
                        eeg = eeg[::-1, :]

                    for ii in range(eeg_save.shape[0]):
                        eeg_save[ii, :] = eeg[
                            ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                        ]
                else:
                    eeg2 = eeg.copy()
                    eeg[4:8, :] = eeg2[12:16, :]
                    eeg[8:12, :] = eeg2[4:8, :]
                    eeg[12:16, :] = eeg2[8:12, :]
                    for ii in range(eeg_save.shape[0]):
                        eeg_save[ii, :] = eeg[
                            ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                        ]

                eeg = (eeg_save - np.mean(eeg_save, keepdims=True)) / (
                    np.std(eeg_save, keepdims=True) + 1e-6
                )
                x_eeg[j] = eeg

            if "stft" in DATATYPE:
                stft = np.clip(stft, a_min=1e-6, a_max=1e6)
                stft = np.log2(stft)
                if self.mode == "train":
                    stft[0:8, :, :] = stft[0:8, :, :][np.random.permutation(8), :, :]
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
                    img[10:18, :, :] = img[10:18, :, :][np.random.permutation(8), :, :]
                    if np.random.rand() > 0.5:
                        img = img[::-1, :, :]
                for ii in range(img.shape[0]):
                    axis_temp = img_save.shape[1] / img_save.shape[0] / 2 * (2 * ii + 1)
                    start_temp = round(
                        max(axis_temp - img_save.shape[1] / img_save.shape[0], 0)
                    )
                    end_temp = round(
                        min(
                            img_save.shape[0],
                            axis_temp + img_save.shape[1] / img_save.shape[0],
                        )
                    )
                    temp_temp = round(img.shape[1] / 2 - (axis_temp - start_temp))
                    img_save[start_temp:end_temp, :] = (
                        img_save[start_temp:end_temp, :]
                        + img[
                            ii, temp_temp : round(temp_temp + end_temp - start_temp), :
                        ]
                    )
                img_save = np.clip(img_save, a_min=0, a_max=1)
                img = np.reshape(img_save, (img_save.shape[0], img_save.shape[1], 1))
                img = np.concatenate((img, img, img), -1)
                img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                x_img[j] = img

            if self.mode != "test":
                y[j] = row[TARGETS].values / sum(row[TARGETS].values)
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




## === cell 2
class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
        super(CosineAnnealingLRScheduler, self).__init__()
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
                    (step - self.warm_step) / (self.total_step - self.warm_step) * np.pi
                )
            )
        return lr




## === cell 3
from tensorflow.keras.applications import EfficientNetB0


def build_model():
    inp = []
    y = None

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

        base_model_spe = EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model_spe._name = "spe_extractor"
        x_spe = base_model_spe(x_spe)

        x_spe = tf.multiply(x_spe, SPE_WEIGHTS_f)
        x_spe = tf.reduce_sum(x_spe, 2, keepdims=True)

        x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
        x_spe = tf.keras.layers.Dropout(0.2)(x_spe)

        inp.append(inp_spe)
        y = x_spe

    if "eeg" in DATATYPE:
        inp_eeg = tf.keras.Input(
            shape=(
                EEG_CHANNEL_USED * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            )
        )
        x_eeg = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )
        x_eeg = tf.keras.layers.Concatenate(axis=-1)([x_eeg, x_eeg, x_eeg])

        base_model_eeg_full = EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model_eeg = tf.keras.Model(
            inputs=base_model_eeg_full.input,
            outputs=base_model_eeg_full.get_layer("block5a_project_bn").output,
            name="eeg_extractor",
        )
        x_eeg = base_model_eeg(x_eeg)

        x_eeg = x_eeg[
            :,
            :,
            (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2] - (x_eeg.shape[2] - 1) // 2),
            :,
        ]
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.2)(x_eeg)

        inp.append(inp_eeg)
        y = tf.keras.layers.Concatenate(axis=1)([y, x_eeg]) if y is not None else x_eeg

    if "stft" in DATATYPE:
        inp_stft = tf.keras.Input(shape=(STFT_HIGH * 9, STFT_WIDE * 2))
        x_stft = tf.keras.layers.Reshape((inp_stft.shape[1], inp_stft.shape[2], 1))(
            inp_stft
        )
        x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

        base_model_stft = EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model_stft._name = "stft_extractor"
        x_stft = base_model_stft(x_stft)
        x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)

        inp.append(inp_stft)
        y = (
            tf.keras.layers.Concatenate(axis=1)([y, x_stft])
            if y is not None
            else x_stft
        )

    if "img" in DATATYPE:
        inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3))
        base_model_img = EfficientNetB0(
            include_top=False, weights=None, input_shape=None
        )
        base_model_img._name = "img_extractor"
        x_img = base_model_img(inp_img)
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)

        inp.append(inp_img)
        y = tf.keras.layers.Concatenate(axis=1)([y, x_img]) if y is not None else x_img

    y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(y)
    model = tf.keras.Model(inputs=inp, outputs=y)
    return model




## === cell 4
def _make_baseline_predictions_from_train(
    train_df: pd.DataFrame, test_df: pd.DataFrame, targets
):
    targets = list(targets)
    eps = 1e-9

    tmp = train_df[["patient_id"] + targets].copy()

    global_counts = tmp[targets].sum(axis=0).astype("float64").values

    pat_counts = tmp.groupby("patient_id", sort=False)[targets].sum().astype("float64")
    pat_total_votes = pat_counts.sum(axis=1).astype("float64")  # evidence strength

    alpha0 = 1.0
    alpha_vec = (alpha0 + global_counts).astype("float64")  # global-informed prior

    global_post = alpha_vec / np.sum(alpha_vec)
    global_post = np.clip(global_post, eps, 1.0)
    global_post = global_post / global_post.sum()

    base_m_votes = 400.0  # shrinkage in "vote count" units (small change lever, stable)
    preds = np.zeros((len(test_df), len(targets)), dtype="float64")
    for i, pid in enumerate(test_df["patient_id"].values):
        if pid in pat_counts.index:
            cnt = pat_counts.loc[pid].values.astype("float64")
            post = (alpha_vec + cnt) / np.sum(alpha_vec + cnt)

            n = float(pat_total_votes.loc[pid])
            w = n / (n + base_m_votes)  # in [0,1)
            p = (1.0 - w) * global_post + w * post
        else:
            p = global_post
        preds[i] = p

    preds = np.clip(preds, eps, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)
    return preds.astype("float32")


def _weights_available(load_dir: str, splits: int) -> bool:
    if not os.path.isdir(load_dir):
        return False
    for i in range(splits):
        if not os.path.exists(os.path.join(load_dir, f"fold{i}_stage2.h5")):
            return False
    return True


if not NEEDTRAIN:
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    can_load = _weights_available(LOAD_MODELS_FROM, SPLITS)
    if not can_load:
        print(f"WARNING: Model weights not found in: {LOAD_MODELS_FROM}")
        print(
            "Falling back to baseline prior predictions (Dirichlet vote-count patient→global shrinkage)."
        )
        preds_all = _make_baseline_predictions_from_train(df, test, TARGETS)
    else:
        preds_all = []
        models = []

        with strategy.scope():
            model_template = build_model()
        for model_i in range(SPLITS):
            print(f"Fold {model_i+1}")
            model = clone_model(model_template)
            model.load_weights(
                os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.h5")
            )
            models.append(model)

        if "spe" in DATATYPE:
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_spectrograms") + "/"
            files_test = os.listdir(PATH_test)
            print(f"There are {len(files_test)} test spectrogram parquets")

            for i, f in enumerate(files_test):
                if i % 100 == 0:
                    print(i, ", ", end="")
                tmp = pd.read_parquet(f"{PATH_test}{f}")
                name = int(f.split(".")[0])
                spectrograms_test[name] = tmp.iloc[:, 1:].values

        PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"
        if (
            ("spe" in DATATYPE)
            or ("eeg" in DATATYPE)
            or ("stft" in DATATYPE)
            or ("img" in DATATYPE)
        ):
            b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")
            b2, a2 = signal.butter(
                3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
            )

            for i, eeg_id in enumerate(test.eeg_id):
                if i % 100 == 0:
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

                if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                    preds = []
                    start = (
                        len(preds_all)
                        if isinstance(preds_all, list)
                        else preds_all.shape[0]
                    )
                    batch_df = test.iloc[start : i + 1, :].reset_index(drop=True)

                    test_gen = DataGenerator(
                        batch_df,
                        shuffle=False,
                        sample_weights=False,
                        batch_size=TEST_BATCHSIZE,
                        mode="test",
                        specs=spectrograms_test,
                        eegs=eegs_test,
                        stfts=stfts_test,
                        imgs=imgs_test,
                    )

                    for model_i in range(SPLITS):
                        pred = models[model_i].predict(test_gen, verbose=0)
                        preds.append(pred)
                    pred = np.mean(preds, axis=0)

                    eegs_test = {}
                    stfts_test = {}
                    imgs_test = {}
                    gc.collect()

                    if isinstance(preds_all, list) and len(preds_all) == 0:
                        preds_all = pred.copy()
                    else:
                        preds_all = np.concatenate((preds_all, pred), axis=0)

        if isinstance(preds_all, list):
            preds_all = np.asarray(preds_all, dtype=np.float32)

        assert (
            preds_all.shape[0] == test.shape[0]
        ), f"Prediction rows {preds_all.shape[0]} != test rows {test.shape[0]}"

        preds_all = np.clip(preds_all, 1e-9, 1.0)
        preds_all = preds_all / preds_all.sum(axis=1, keepdims=True)

    sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[list(TARGETS)] = preds_all.astype("float32")

    for c in sample_sub.columns:
        if c not in sub.columns:
            raise ValueError(f"Missing required submission column: {c}")
    sub = sub[sample_sub.columns]

    prob_cols = [c for c in sub.columns if c != "eeg_id"]
    arr = sub[prob_cols].values.astype("float64")
    arr = np.clip(arr, 1e-9, 1.0)
    arr = arr / arr.sum(axis=1, keepdims=True)
    sub[prob_cols] = arr.astype("float32")

    sub.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
    print("Submission shape", sub.shape)
    print(sub.head())
    print(
        "Row-sum stats:",
        float(sub[prob_cols].sum(axis=1).min()),
        float(sub[prob_cols].sum(axis=1).max()),
    )

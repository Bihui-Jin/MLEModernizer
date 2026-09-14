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

albumentations==2.0.8
geopandas==0.14.4
librosa==0.11.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyWavelets==1.8.0
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.6532515188163773

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first fix the execution blockers so you reliably get a valid `submission.csv`: the script currently has missing cell numbers, a Jupyter-only `%%time` magic, and an undefined `label_cols` inside the dataset class (even if not used for test). Then I fix a silent but important inference bug: inside the model-weights loop you create a new DataLoader named `train_loader` but still run inference on the old `test_loader`, which can cause mismatched state and makes the ensemble loop logically incorrect. Finally, I make the inference deterministic and robust (proper checkpoint loading on CPU/GPU, consistent loader variable, probability normalization to sum-to-1) without changing the model architecture, feature extraction, or loss semantics.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.65325), so we need a small but meaningful improvement without changing the model/training core. The biggest likely issue is that test-time feature construction is inconsistent with how the model was trained: in `mode="test"` you always crop the spectrogram at `r=0`, which is usually wrong because train uses an offset-derived crop; we instead take a centered crop for test to better match the intended 10-minute window and reduce KL. Additionally, we ensure deterministic ordering of checkpoint ensembling by sorting `model_weights` (to avoid run-to-run drift) and keep the same normalization/softmax behavior. These are minimal, metric-relevant changes that should improve predictions while preserving architecture and inference semantics and still produce a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is much worse than the target (0.65325), so we need a small but meaningful fix that improves correctness rather than tuning. The biggest silent issue is that test-time spectrogram cropping uses a center crop, while train-time uses an offset-derived crop (`r = (min+max)//4`), so inference is likely misaligned with what the model learned; we compute `min/max` for each `spectrogram_id` from `train.csv` and use the same offset-based crop at test time. This preserves the exact model/feature core, but makes test preprocessing consistent with training and usually reduces KL substantially. We also make the test spectrogram slice robust to bounds (avoid empty/shifted windows) and keep the probability normalization as-is to guarantee valid submissions.'
- What this solution (achieved 1.40995) has done: 'Your score is much worse than the target (KL 1.40995 vs 0.65325, lower is better), so the most likely gain comes from fixing a remaining preprocessing mismatch rather than changing the model. The current test-time crop uses a rough heuristic derived from train offsets (and uses `//4`), which can still misalign the 300-frame window the model expects; we instead use a more direct, robust crop center derived from the train offset distribution per `spectrogram_id` (median of `offset//2`, then center the 300 window) while keeping the exact same spectrogram normalization and model/inference logic. We also keep a safe fallback to center-crop within the test spectrogram length when no mapping exists. These are minimal changes focused on aligning test preprocessing to training semantics, which should reduce KL toward the target without touching architecture/loss.'
- What this solution (achieved 1.40995) has done: 'To move the KL score down toward the target, the most likely “minimal but meaningful” fix is to better align test-time spectrogram cropping with how labels were positioned during training. Your current `pos_map` uses a median offset, but this can be biased for spectrograms with multi-modal/shifted labeled regions; switching to a robust trimmed-mean of per-spectrogram offsets typically reduces mismatch without changing any model/training logic. I also fix a small region-name ordering bug in `plot_spectrogram` (not score-critical, but it is a correctness fix), and I keep the rest (architecture, reshape, softmax, ensembling) identical while ensuring the crop window never goes out of bounds. The script still run end-to-end and write a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.40995) has done: 'We make one metric-relevant, minimal change: ensemble in probability space using the **geometric mean** (average log-probabilities) instead of the arithmetic mean of probabilities, which is often better aligned with KL-divergence evaluation and typically reduces over-confident spikes that hurt KL. This keeps the exact same model, features, softmax, and checkpoints; it only changes how multiple model outputs are combined. We also keep the existing clip+row-normalization to guarantee valid submissions that sum to 1. The rest of the pipeline, paths, and core logic remain unchanged and it still write `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'To move the KL score down toward the target with minimal risk, I keep your model, features, and inference unchanged, and only fix one likely correctness mismatch in test preprocessing. In your dataset, test-time crop position is derived from a `spectrogram_id -> pos` map built from train offsets, but that mapping is for *train spectrograms*, while the test spectrograms are disjoint—so most test IDs fall back to a center heuristic, often misaligning the labeled 10s region the model learned. I instead build an `eeg_id -> pos` map from train and use it for test cropping (because train/test share patient/eeg characteristics more directly), with a safe fallback to center-crop if missing; this preserves the exact crop size and normalization and should reduce KL materially. I also ensure the crop index uses the same units as your current logic (frames after `//2`) and stays in-bounds.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.65325), so the most likely “minimal but meaningful” improvement is to fix a remaining preprocessing mismatch: the spectrogram crop position should be in the same time-index units as the spectrogram array (frames), but the current `pos = offset_seconds // 2` is not converted into frames, which can shift the 300-frame crop far from the labeled region. I convert label offsets (seconds) into spectrogram-frame indices using the known 600-row (10 min) spectrogram timeline (≈1 frame/sec), and use that mapping consistently for train-derived position maps and the test-time crop. This preserves your model, features, loss, and inference logic while making the central crop align much better with what the model expects. The submission writing and row-sum normalization remain unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.65325), so the most likely minimal improvement is correcting a remaining train/test preprocessing mismatch that can severely hurt KL: the spectrogram crop position units. I compute a per-`eeg_id` crop center in **spectrogram frame indices** from `train.csv` using the known ~1 frame/sec timeline (600 frames over 600 seconds), then apply that consistently at test time (with safe bounds and a center fallback). This keeps the model, input construction, softmax, and checkpoint ensembling unchanged, but makes test crops align better with the labeled region the model learned. The script still runs end-to-end and writes a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.40995) has done: 'The score gap is large (1.40995 vs target 0.65325, lower is better), so the most likely “minimal but meaningful” improvement is correcting a remaining train/test preprocessing mismatch rather than changing the model. Your dataset currently slices **300 time-rows** from the spectrogram but then assigns `img[:, 22:-22]` (width 56) into a 256-wide target, which silently mis-shapes/should crash or, if it runs due to unexpected shapes, severely corrupts inputs; we fix this to use the full 256-width consistently (same core feature logic, just correct alignment). We also avoid the huge memory/time hit of loading all test spectrograms into RAM by lazily reading per-sample parquet (same data, same transformation, just I/O strategy) to keep runtime within limits and reduce OOM-induced instability that can harm inference. Finally, we keep your geometric-mean ensembling and probability normalization unchanged so submission semantics stay identical while inputs become correct and consistent.'
- What this solution (achieved 1.40995) has done: 'We make two minimal, metric-relevant fixes to move KL down toward the target without changing the model architecture or inference semantics. First, your spectrogram crop is currently using `pos` in 0..600 “time frames” while the spectrogram parquet has ~300 “time rows”, so the crop often clamps to the end and becomes misaligned; we correctly convert label offsets to the spectrogram’s row index using the actual per-file time length (number of rows). Second, we remove the hard-coded `300` time window mismatch by using the available spectrogram time length to center a fixed window safely, avoiding systematic edge-cropping. Everything else (EEG feature extraction, ResNeXt, softmax, geometric-mean ensemble, submission normalization) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your KL gap to the target is large (1.40995 vs 0.65325, lower is better), so the most impactful minimal fix is to correct a likely input corruption in the spectrogram branch: the code currently writes a 100×256 slice into a 100×300 target area, which should error or silently distort inputs depending on shapes, severely hurting predictions. I keep the exact same crop height (300), normalization, model, and ensembling, and only make the spectrogram assignment shape-consistent by resizing the time dimension to 256 (like the target width) so the model sees stable, intended geometry. I also fix the hard-coded `TRAIN_SPEC_TIME_ROWS=300` to be derived from the actual train spectrogram parquet row count (still same units/logic, but correct), so the offset-to-row conversion and test rescaling are consistent with the real data. These changes preserve core logic but should materially reduce KL by making test preprocessing match what the model expects.'

# 9. Code solution

## === cell 0
import albumentations as A
import gc
import librosa
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import pywt
import random
import time
import timm
import torch
import torch.nn as nn

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config_resnext:
    CADINALITY = 32
    CONFIG_NAME = 50
    INFER_RESNEXT = True




## === cell 2
class config:
    BATCH_SIZE = 64
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 0  # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False


class paths:
    MODEL_WEIGHTS = "/kaggle/input/hms-multi-class-image-classification-train/tf_efficientnet_b0_epoch_3.pth"
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    TRAIN_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )


if config_resnext.INFER_RESNEXT:
    model_weights = sorted(
        [x for x in glob("/kaggle/input/hms-pytorch-resnext-cnn/*.pth")]
    )
else:
    model_weights = []

print("Found", len(model_weights), "ResNeXt weight files")
model_weights[:5]



## === cell 3
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
label_cols = TARGETS  # needed by dataset even if test doesn't use labels


def maddest(d, axis: int = None):
    """
    Denoise function.
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")  # multilevel 1D DWT.
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    output = pywt.waverec(coeff, wavelet, mode="per")
    return output


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 256,
                n_fft=1024,
                n_mels=128,
                fmin=0,
                fmax=20,
                win_length=128,
            )

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]

            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect="auto", origin="lower")
            plt.title(f"Spectrogram {NAMES[k]}")

    if display:
        plt.show()
        plt.figure(figsize=(10, 5))
        offset = 0
        for k in range(4):
            if k > 0:
                offset -= signals[3 - k].min()
            plt.plot(range(10_000), signals[k] + offset, label=NAMES[3 - k])
            offset += signals[3 - k].max()
        plt.legend()
        plt.title("EEG Signals")
        plt.show()
        print()
        print("#" * 25)
        print()

    return img


def plot_spectrogram(spectrogram_path: str):
    """
    Visualize spectrogram recordings from a parquet file.
    """
    sample_spect = pd.read_parquet(spectrogram_path)

    split_spect = {
        "LL": sample_spect.filter(regex="^LL", axis=1),
        "RL": sample_spect.filter(regex="^RL", axis=1),
        "LP": sample_spect.filter(regex="^LP", axis=1),
        "RP": sample_spect.filter(regex="^RP", axis=1),
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()
    label_interval = 5
    for i, split_name in enumerate(split_spect.keys()):
        ax = axes[i]
        img = ax.imshow(
            np.log(split_spect[split_name]).T,
            cmap="viridis",
            aspect="auto",
            origin="lower",
        )
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Log(Value)")
        ax.set_title(split_name)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")

        ax.set_yticks(np.arange(len(split_spect[split_name].columns)))
        ax.set_yticklabels(
            [column_name[3:] for column_name in split_spect[split_name].columns]
        )
        frequencies = [
            column_name[3:] for column_name in split_spect[split_name].columns
        ]
        ax.set_yticks(
            np.arange(0, len(split_spect[split_name].columns), label_interval)
        )
        ax.set_yticklabels(frequencies[::label_interval])
    plt.tight_layout()
    plt.show()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sep():
    print("-" * 100)


label_to_num = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
num_to_label = {v: k for k, v in label_to_num.items()}
seed_everything(config.SEED)



## === cell 4
TRAIN_SPEC_TOTAL_SECONDS = 600.0

train_df_for_crop = pd.read_csv(
    paths.TRAIN_CSV,
    usecols=["eeg_id", "spectrogram_id", "spectrogram_label_offset_seconds"],
)

_train_spec_files = glob(os.path.join(paths.TRAIN_SPECTROGRAMS, "*.parquet"))
if len(_train_spec_files) > 0:
    _tmp = pd.read_parquet(_train_spec_files[0])
    TRAIN_SPEC_TIME_ROWS = int(_tmp.shape[0])
    del _tmp
else:
    TRAIN_SPEC_TIME_ROWS = 300  # safe fallback if path missing

sec = (
    train_df_for_crop["spectrogram_label_offset_seconds"].astype(np.float32).fillna(0.0)
)
pos_rows = np.rint(sec * (TRAIN_SPEC_TIME_ROWS / TRAIN_SPEC_TOTAL_SECONDS)).astype(
    np.int32
)
train_df_for_crop["pos"] = np.clip(pos_rows, 0, TRAIN_SPEC_TIME_ROWS - 1).astype(
    np.int32
)


def _trimmed_mean_int(x: pd.Series, trim_q: float = 0.1) -> int:
    x = x.dropna().astype(np.float32).values
    if x.size == 0:
        return 0
    lo = np.quantile(x, trim_q)
    hi = np.quantile(x, 1.0 - trim_q)
    x = np.clip(x, lo, hi)
    return int(np.round(x.mean()))


spec_pos_map = (
    train_df_for_crop.groupby("spectrogram_id")["pos"]
    .apply(_trimmed_mean_int)
    .astype(np.int32)
    .to_dict()
)

eeg_pos_map = (
    train_df_for_crop.groupby("eeg_id")["pos"]
    .apply(_trimmed_mean_int)
    .astype(np.int32)
    .to_dict()
)

del train_df_for_crop, sec, pos_rows, _train_spec_files
gc.collect()

print("TRAIN_SPEC_TIME_ROWS =", TRAIN_SPEC_TIME_ROWS)
print("Built position map for", len(spec_pos_map), "spectrogram_ids from train.csv")
print("Built position map for", len(eeg_pos_map), "eeg_ids from train.csv")



## === cell 5
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 6
test_spec_files = glob(paths.TEST_SPECTROGRAMS + "*.parquet")
print(f"There are {len(test_spec_files)} spectrogram parquets (will be read lazily)")

if config.VISUALIZE and len(test_spec_files) > 0:
    idx = np.random.randint(0, len(test_spec_files))
    spectrogram_path = test_spec_files[idx]
    plot_spectrogram(spectrogram_path)



## === cell 7
t0 = time.time()

paths_eegs = glob(paths.TEST_EEGS + "*.parquet")
print(f"There are {len(paths_eegs)} EEG parquets")
all_eegs = {}
counter = 0

for file_path in tqdm(paths_eegs):
    eeg_id = file_path.split("/")[-1].split(".")[0]
    eeg_spectrogram = spectrogram_from_eeg(file_path, counter < 1)
    all_eegs[int(eeg_id)] = eeg_spectrogram
    counter += 1

print(f"Loaded test EEG-derived spectrograms in {time.time() - t0:.1f}s")




## === cell 8
class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        augment: bool = False,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = None,
        eeg_specs: Dict[int, np.ndarray] = None,
        pos_map: Dict[int, int] = None,
        eeg_pos_map_in: Dict[int, int] = None,
    ):
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode

        self.spectrograms = specs
        self.eeg_spectrograms = all_eegs if eeg_specs is None else eeg_specs
        self.pos_map = spec_pos_map if pos_map is None else pos_map
        self.eeg_pos_map = eeg_pos_map if eeg_pos_map_in is None else eeg_pos_map_in

        self._spec_cache: Dict[int, np.ndarray] = {}
        self._spec_cache_order: List[int] = []
        self._spec_cache_max = 64

    def __len__(self):
        return len(self.df)

    def _get_spec(self, spectrogram_id: int) -> np.ndarray:
        if self.spectrograms is not None and spectrogram_id in self.spectrograms:
            return self.spectrograms[spectrogram_id]

        if spectrogram_id in self._spec_cache:
            return self._spec_cache[spectrogram_id]

        fp = os.path.join(paths.TEST_SPECTROGRAMS, f"{spectrogram_id}.parquet")
        aux = pd.read_parquet(fp)
        spec = aux.iloc[:, 1:].values  # drop time column
        del aux

        self._spec_cache[spectrogram_id] = spec
        self._spec_cache_order.append(spectrogram_id)
        if len(self._spec_cache_order) > self._spec_cache_max:
            old = self._spec_cache_order.pop(0)
            if old in self._spec_cache:
                del self._spec_cache[old]
        return spec

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def __data_generation(self, index):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.df.iloc[index]

        spectrogram_id = int(row.spectrogram_id)
        spec = self._get_spec(spectrogram_id)

        crop_h = 300
        spec_h = int(spec.shape[0])
        max_r = max(0, spec_h - crop_h)

        if self.mode == "test":
            if int(row.eeg_id) in self.eeg_pos_map:
                pos = int(self.eeg_pos_map[int(row.eeg_id)])
            elif spectrogram_id in self.pos_map:
                pos = int(self.pos_map[spectrogram_id])
            else:
                pos = int(spec_h // 2)

            if spec_h > 0 and spec_h != TRAIN_SPEC_TIME_ROWS:
                pos = int(np.rint(pos * (spec_h / float(TRAIN_SPEC_TIME_ROWS))))
            r = int(pos - crop_h // 2)
        else:
            if "spectrogram_label_offset_seconds" in row.index:
                pos = int(
                    np.rint(
                        float(row["spectrogram_label_offset_seconds"])
                        * (TRAIN_SPEC_TIME_ROWS / TRAIN_SPEC_TOTAL_SECONDS)
                    )
                )
                if spec_h > 0 and spec_h != TRAIN_SPEC_TIME_ROWS:
                    pos = int(np.rint(pos * (spec_h / float(TRAIN_SPEC_TIME_ROWS))))
                r = int(pos - crop_h // 2)
            else:
                r = 0

        r = int(np.clip(r, 0, max_r))

        for region in range(4):
            img = spec[r : r + crop_h, region * 100 : (region + 1) * 100].T  # (100,300)

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            if img.shape[1] != 256:
                old_t = img.shape[1]
                new_t = 256
                old_idx = np.arange(old_t, dtype=np.float32)
                new_idx = np.linspace(0, old_t - 1, new_t, dtype=np.float32)
                img_resized = np.empty((img.shape[0], new_t), dtype=np.float32)
                for rr in range(img.shape[0]):
                    img_resized[rr] = np.interp(new_idx, old_idx, img[rr]).astype(
                        np.float32
                    )
                img = img_resized

            X[14:-14, :, region] = img / 2.0

        img_eeg = self.eeg_spectrograms[int(row.eeg_id)]
        X[:, :, 4:] = img_eeg

        if self.mode != "test":
            y = row[label_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose([A.HorizontalFlip(p=0.5)])
        return transforms(image=img)["image"]




## === cell 9
test_dataset = CustomDataset(test_df, config, mode="test")
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    drop_last=False,
)
X, y = test_dataset[0]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")




## === cell 10
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.c = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.c(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, first=False, cardinatlity=32):
        super().__init__()
        self.C = cardinatlity
        self.downsample = stride == 2 or first
        res_channels = out_channels // 2
        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0)
        self.c2 = ConvBlock(res_channels, res_channels, 3, stride, 1, self.C)
        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)

        self.relu = nn.ReLU()

        if self.downsample:
            self.p = ConvBlock(in_channels, out_channels, 1, stride, 0)

    def forward(self, x):
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)

        if self.downsample:
            x = self.p(x)

        h = self.relu(torch.add(f, x))
        return h


class ResNeXt(nn.Module):
    def __init__(
        self, config_name: int, in_channels: int = 3, classes: int = 1000, C: int = 32
    ):
        super().__init__()

        configurations = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
        no_blocks = configurations[config_name]

        out_features = [256, 512, 1024, 2048]
        self.blocks = nn.ModuleList([ResidualBlock(64, 256, 1, True, cardinatlity=32)])

        for i in range(len(out_features)):
            if i > 0:
                self.blocks.append(
                    ResidualBlock(
                        out_features[i - 1], out_features[i], 2, cardinatlity=C
                    )
                )
            for _ in range(no_blocks[i] - 1):
                self.blocks.append(
                    ResidualBlock(out_features[i], out_features[i], 1, cardinatlity=C)
                )

        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, classes)

        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)




## === cell 11
def __reshape_input(x):
    """
    Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
    """
    spectograms = [x[:, :, :, i : i + 1] for i in range(4)]
    spectograms = torch.cat(spectograms, dim=1)

    eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
    eegs = torch.cat(eegs, dim=1)

    x = torch.cat([spectograms, eegs], dim=2)
    x = torch.cat([x, x, x], dim=3)
    x = x.permute(0, 3, 1, 2)
    return x




## === cell 12
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for step, (X, y) in enumerate(tqdm_test_loader):
            X = __reshape_input(X)
            input_size = X.size(0)
            X = X.view(input_size, 3, 512, 512)

            X = X.to(device, non_blocking=True)
            with torch.no_grad():
                y_preds = model(X)
            y_preds = softmax(y_preds)
            preds.append(y_preds.detach().cpu().numpy())

    prediction_dict = {"predictions": np.concatenate(preds, axis=0)}
    return prediction_dict




## === cell 13
predictions = []

if len(model_weights) == 0:
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
else:
    for model_weight in model_weights:
        test_dataset = CustomDataset(test_df, config, mode="test", augment=False)
        infer_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        model = ResNeXt(
            config_resnext.CONFIG_NAME,
            in_channels=3,
            classes=6,
            C=config_resnext.CADINALITY,
        )

        checkpoint = torch.load(model_weight, map_location="cpu")
        state = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state, strict=True)
        model.to(device)

        prediction_dict = inference_function(infer_loader, model, device)

        p = np.clip(prediction_dict["predictions"].astype(np.float32), 1e-8, 1.0)
        predictions.append(p)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    predictions = np.stack(predictions, axis=0)  # (n_models, n_samples, 6)

    logp = np.log(predictions)
    mean_logp = logp.mean(axis=0)
    predictions = np.exp(mean_logp)

predictions = np.clip(predictions, 1e-8, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

print("Predictions shape:", predictions.shape)



## === cell 14
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions.astype(np.float32)

row_sums = sub[TARGETS].sum(axis=1).values
print(
    "Row sum min/mean/max:",
    float(row_sums.min()),
    float(row_sums.mean()),
    float(row_sums.max()),
)

sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
sub.head()

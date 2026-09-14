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

0.5269789944558243

# 6. Current score

1.41378

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime blockers by removing dependencies on non-existent Kaggle input model directories and by adding a safe fallback to generate valid probabilities when weights are unavailable. I also fix dataset issues that would prevent inference from running (undefined `targets`, incorrect row indexing for test mode, and excessive `print` statements that can cause timeouts). To keep core logic intact, the models and inference functions remain unchanged; we only adjust how checkpoints are discovered/loaded and ensure predictions are always a proper `(N,6)` probability array summing to 1. Finally, the script always write a valid `submission.csv` with the correct columns.'
- What this solution (achieved 1.40995) has done: 'Your current score is far above the target (lower is better), and the main reason is that the script usually falls back to uniform probabilities because it can’t find/load the intended checkpoints (so the model is effectively not used). I make a minimal change to (1) correctly use GPU when available (your config currently forces CPU), and (2) robustly locate and load checkpoints if they exist in common Kaggle input locations (including the competition dataset directory you already have mounted), while keeping the same models and inference. I also add a tiny compatibility layer so `load_state_dict` works even if the checkpoint keys are prefixed with `module.` (common from DDP), which otherwise silently prevents using good weights. These changes should substantially reduce KL toward your target without altering the core architecture, features, or inference semantics.'
- What this solution (achieved 1.40995) has done: 'I make two minimal fixes that are directly tied to improving the KL score while preserving your exact model/inference logic: (1) correct the test-time spectrogram window selection so it uses the full available spectrogram height (your current `_safe_r_for_spec` always chooses ~the middle of the whole 10-min spec, which is misaligned with how these models were trained to look at labeled windows), and (2) ensure the submission rows are in the exact same order as `sample_submission.csv` (your current merge can reorder/duplicate and silently misalign predictions, which typically explodes KL). I also add a very small, safe probability floor (`1e-4`) before normalization to reduce KL penalties from overconfident near-zeros without changing core semantics. The script still run end-to-end and always write a valid `submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your KL is far above target (lower is better), so the smallest meaningful improvement is to reduce test-time feature mismatch and ensure we are actually using model outputs consistently. I (1) fix test-time spectrogram crop selection to use the same 300-row windowing idea as training (the current center-crop on the full 10-min spectrogram is often misaligned), (2) normalize the target votes to probabilities at train-time only if ever used (keeps semantics consistent), and (3) make inference deterministic and slightly safer by enforcing eval+no_grad already present and adding a tiny probability floor + renorm (keeps submission valid and avoids KL blow-ups from near-zeros). Core model architectures, feature construction, and inference averaging remain unchanged; these are narrowly-scoped alignment/calibration fixes aimed at moving KL down toward ~0.53 without changing the overall approach.'
- What this solution (achieved 1.40995) has done: 'I make a minimal set of changes aimed at lowering KL (your current 1.40995 is far worse than the 0.5269 target), without changing the model architectures or inference logic. The main fix is to remove the train-only spectrogram crop logic that currently references non-existent `min/max` columns (which causes silent feature mismatch/garbage if ever used) and instead use a safe, deterministic crop for both train/test that matches the expected 300-row window. I also make inference DataLoaders deterministic and consistent (no shuffle, proper `num_workers/pin_memory`) and ensure we always load checkpoints as effectively as possible by handling common key prefixes (`model.` as well as `module.`). Finally, I keep your probability flooring/renorm (good for KL stability) and preserve submission row order via `sample_submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I make two minimal, score-relevant fixes to move KL down toward your target without changing the model architectures or inference logic. First, I correct the test-time spectrogram crop: for test there is no labeled offset, so we should consistently take the centered 300-row window (your current helper does this, but it assumes `spec_arr.shape[0]` is already the time axis; in these parquets it’s safer to explicitly handle both possible orientations). Second, I ensure prediction-to-submission alignment is strictly by `eeg_id` using `sample_submission.csv` order (keeping your approach) but also aggregate multiple rows per `eeg_id` safely if they ever appear, preventing accidental row-level mismatch that can blow up KL. Everything else (models, softmax, checkpoint loading, averaging, probability floor/renorm, paths) stays the same.'
- What this solution (achieved 1.40995) has done: 'I fix the broadcasting crash in `CustomDataset.__getitem__` by making the spectrogram slice robust to varying spectrogram widths/heights and by resizing the extracted patch to the exact `(100, 256)` target before assignment, without changing the model/inference logic. This keeps your existing “300 time rows × 4 regions” crop idea intact but prevents shape mismatches like `(16,256)` caused by different parquet layouts. I also add a small safety pad when a spectrogram is too short for a 300-row window, so inference never fails. These changes are purely correctness/stability fixes that should also improve KL by ensuring the model actually sees meaningful spectrogram content instead of crashing/falling back.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.52698), and the biggest likely cause is that your script often can’t actually load compatible checkpoints, so it falls back to near-uniform predictions. I make a minimal, score-relevant fix: improve checkpoint loading by (1) handling nested prefixes like `model.module.` and `module.model.` and (2) automatically selecting the checkpoint file that best matches this model by highest key-overlap, so you use real learned weights instead of a partial/empty load. I also keep your exact model/inference logic but add a tiny temperature smoothing at the very end (before the existing clip/renorm) to reduce overconfident zeros that can spike KL, which typically moves KL down without changing semantics. The script still run end-to-end and write a valid `submission.csv` with correct column order and row-wise probabilities summing to 1.'
- What this solution (achieved 1.41378) has done: 'I make two minimal, score-relevant fixes that keep your architecture/inference intact but should materially reduce KL toward the 0.5269 target. First, I avoid the current “uniform fallback” when no external checkpoints exist by fitting a simple per-class prior from `train.csv` (label distribution) and using that as the fallback probabilities (this is legitimate and usually much better than uniform for KL). Second, I make the checkpoint key-normalization slightly more robust for common nested prefixes so that if any compatible weights are present, they actually load (reducing the chance you silently run near-random models). Everything else (feature creation, models, softmax inference, ensembling, submission alignment, probability flooring/renorm) stays the same and the script still write a valid `submission.csv`.'
- What this solution (achieved 1.41378) has done: 'I make the smallest score-relevant change: ensure the model actually uses the official **provided spectrograms** (which contain the 10-minute context these HMS models are typically trained on) instead of accidentally feeding the EEG-derived mel images into the “spectrogram channels”. Right now `CustomDataset` uses `self.specs` for the first 4 channels, but you are passing `all_spectrograms` that you loaded from `test_spectrograms` parquets **without** converting them into the expected 4×(100×256) region format, which makes those channels essentially garbage and keeps KL high. I fix this by precomputing `all_spectrograms` into the exact same 4-region `(128,256,4)` layout the dataset expects (matching your existing `_extract_region_img_100x256` pipeline), while keeping your models, softmax, ensembling, and submission logic unchanged. This should materially reduce KL toward the 0.5269 target while remaining within your “core logic unchanged” constraint.'
- What this solution (achieved 1.41378) has done: 'Your KL is far above the target (lower is better), so the smallest likely win is to remove the biggest train/test mismatch without changing any model code: right now you build EEG-derived mel spectrograms from the *middle* 50s of each EEG, but the test EEG files are already exactly 50s long, so your current crop often takes the wrong segment (and can even produce empty/near-empty signals), which makes predictions drift toward the prior/uniform and inflates KL. I make `spectrogram_from_eeg()` use the full 50s when the file is already 10,000 samples, only center-cropping when the recording is longer, preserving all feature extraction settings. I also keep your exact post-processing, but add a tiny safety clamp after softmax during inference (still probabilities) to avoid any near-zero values that can spike KL in rare cases. Everything else (models, checkpoints, ensembling, submission alignment) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import math
import time
import random
import multiprocessing
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import librosa
import pywt

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 1
class config:
    model1 = "resnet50d"
    model2 = "vit_base_patch16_224"
    epoch = 10
    lr = 1e-3
    batchsize = 32
    splits = 5
    momentum = 0.9
    MAX_GRAD_NORM = 1e7
    WEIGHT_DECAY = 0.01
    device = str(device)
    FOLDS = 5
    AMP = True


class paths:
    preloadedeeg = "/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"
    train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs"
    train_spec_dir = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms"
    )
    train_csv = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    test_csv = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    test_eeg = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
    test_spec = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms"
    )
    out = "/kaggle/working/"


TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 2
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def maddest(d, axis: int = None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    output = pywt.waverec(coeff, wavelet, mode="per")
    return output


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)

    if len(eeg) > 10_000:
        middle = (len(eeg) - 10_000) // 2
        eeg = eeg.iloc[middle : middle + 10_000]
    else:
        eeg = eeg.iloc[:10_000]

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

    return img




## === cell 3
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 4
all_eegs = {}
eeg_files = sorted(os.listdir(paths.test_eeg))
for fn in tqdm(eeg_files, desc="Building EEG-derived spectrograms"):
    sp = spectrogram_from_eeg(os.path.join(paths.test_eeg, fn))
    name = int(fn.split(".")[0])
    all_eegs[name] = np.array(sp, dtype=np.float32)

print("Built all_eegs:", len(all_eegs))



## === cell 5
print("Example EEG key:", next(iter(all_eegs.keys())) if len(all_eegs) else None)



## === cell 6
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()




## === cell 7
def _safe_r_for_spec(spec_arr: np.ndarray) -> int:
    if spec_arr is None:
        return 0
    h, w = int(spec_arr.shape[0]), int(spec_arr.shape[1])
    time_len = h if h >= w else w
    max_r = int(max(0, time_len - 300))
    return int(max_r // 2)


def _extract_region_img_100x256(
    spec_use: np.ndarray, r: int, region: int
) -> np.ndarray:
    time_len, feat_len = spec_use.shape[0], spec_use.shape[1]
    if time_len < 300:
        pad = 300 - time_len
        spec_use = np.pad(spec_use, ((0, pad), (0, 0)), mode="edge")
        time_len = spec_use.shape[0]
        r = 0
    r = int(np.clip(r, 0, max(0, time_len - 300)))

    start = int(region * 100)
    end = int(min((region + 1) * 100, feat_len))
    if start >= feat_len:
        base = np.zeros((100, 300), dtype=np.float32)
    else:
        patch = spec_use[r : r + 300, start:end].T  # (freq_like, time=300)
        if patch.shape[0] < 100:
            pad_f = 100 - patch.shape[0]
            patch = np.pad(patch, ((0, pad_f), (0, 0)), mode="edge")
        elif patch.shape[0] > 100:
            patch = patch[:100, :]
        base = patch.astype(np.float32, copy=False)

    if base.shape[1] < 256:
        pad_t = 256 - base.shape[1]
        left = pad_t // 2
        right = pad_t - left
        base = np.pad(base, ((0, 0), (left, right)), mode="edge")
    elif base.shape[1] > 256:
        mid = base.shape[1] // 2
        half = 256 // 2
        base = base[:, mid - half : mid - half + 256]

    return base  # (100, 256)


def _spec_parquet_to_img128x256x4(spec_arr: np.ndarray) -> np.ndarray:
    if spec_arr is None:
        return np.zeros((128, 256, 4), dtype=np.float32)

    if spec_arr.ndim != 2:
        spec_arr = np.asarray(spec_arr).reshape(spec_arr.shape[0], -1)

    if spec_arr.shape[0] < spec_arr.shape[1]:
        spec_use = spec_arr.T
    else:
        spec_use = spec_arr

    r = _safe_r_for_spec(spec_use)
    out = np.zeros((128, 256, 4), dtype=np.float32)
    for region in range(4):
        img = _extract_region_img_100x256(spec_use, r, region)  # (100,256)
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        ep = 1e-6
        mu = np.nanmean(img)
        std = np.nanstd(img)
        img = (img - mu) / (std + ep)
        img = np.nan_to_num(img, nan=0.0)

        out[14:-14, :, region] = img / 2.0
    return out




## === cell 8
all_spectrograms = {}
spec_files = sorted(os.listdir(paths.test_spec))
for fn in tqdm(spec_files, desc="Loading+preprocessing test spectrogram parquet"):
    sp = pd.read_parquet(os.path.join(paths.test_spec, fn))
    name = int(fn.split(".")[0])
    sp_arr = np.array(sp, dtype=np.float32)

    all_spectrograms[name] = _spec_parquet_to_img128x256x4(sp_arr)

print("Loaded all_spectrograms:", len(all_spectrograms))



## === cell 9
print(
    "Example spectrogram key:",
    next(iter(all_spectrograms.keys())) if len(all_spectrograms) else None,
)




## === cell 10
class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = all_spectrograms,
        eegs: Dict[int, np.ndarray] = all_eegs,
    ):
        self.traindf = traindf
        self.specs = specs
        self.eeg = eegs
        self.mode = mode

    def __len__(self):
        return len(self.traindf)

    def __getitem__(self, idx):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")

        row = self.traindf.iloc[idx]

        spec_arr = self.specs.get(int(row.spectrogram_id), None)
        eeg_img = self.eeg.get(int(row.eeg_id), None)

        if spec_arr is None or eeg_img is None:
            x = torch.tensor(X)
            spectograms = [x[:, :, i : i + 1] for i in range(4)]
            spectograms = torch.cat(spectograms, dim=0)
            eegs = [x[:, :, i : i + 1] for i in range(4, 8)]
            eegs = torch.cat(eegs, dim=0)
            x = torch.cat([spectograms, eegs], dim=1)
            x = torch.cat([x, x, x], dim=2)
            x = x.permute(2, 0, 1)
            return {"data": x, "target": y}

        X[:, :, :4] = spec_arr
        X[:, :, 4:] = eeg_img

        X = torch.tensor(X)
        spectograms = [X[:, :, i : i + 1] for i in range(4)]
        spectograms = torch.cat(spectograms, dim=0)

        eegs = [X[:, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=0)

        x = torch.cat([spectograms, eegs], dim=1)
        x = torch.cat([x, x, x], dim=2)
        x = x.permute(2, 0, 1)

        if self.mode != "test":
            y = row[TARGETS].values.astype(np.float32)
            s = float(y.sum())
            if s > 0:
                y = y / s

        return {"data": x, "target": y}




## === cell 11
customdataset = CustomDataset(test_df, config, mode="test")
print("Dataset length:", len(customdataset))



## === cell 12
sample = customdataset[0]
print(
    "Sample data shape:", sample["data"].shape, "target shape:", sample["target"].shape
)



## === cell 13
print("First spectrogram_id:", test_df.iloc[0].spectrogram_id)



## === cell 14
num_workers = min(2, multiprocessing.cpu_count())
test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
X = customdataset[0]["data"]
y = customdataset[0]["target"]
print("Single item:", X.shape, y)




## === cell 15
class Custommodel(nn.Module):
    def __init__(self, config, numclass: int = 6):
        super(Custommodel, self).__init__()
        self.model = timm.create_model(
            config.model1,
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.2,
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.customlayer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, numclass),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.customlayer(x)
        return x




## === cell 16
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    for batch in tqdm(test_loader, desc="Inference", leave=False):
        x = batch["data"].to(device, non_blocking=True)
        with torch.no_grad():
            ypred = model(x)
        ypred = softmax(ypred)

        ypred = torch.clamp(ypred, 1e-8, 1.0)
        ypred = ypred / ypred.sum(dim=1, keepdim=True)

        preds.append(ypred.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 17
def _load_ckpt_paths(maybe_dir: str) -> List[str]:
    if maybe_dir is None or (not os.path.isdir(maybe_dir)):
        return []
    files = []
    for ext in ("*.pth", "*.pt", "*.bin"):
        files.extend(glob(os.path.join(maybe_dir, ext)))
    if len(files) == 0:
        files = [os.path.join(maybe_dir, f) for f in os.listdir(maybe_dir)]
    return sorted([f for f in files if os.path.isfile(f)])


def _find_ckpt_files_by_pattern(patterns: List[str]) -> List[str]:
    found = []
    for base in ("/kaggle/input", "/kaggle/working"):
        if not os.path.isdir(base):
            continue
        for pat in patterns:
            found.extend(glob(os.path.join(base, "**", pat), recursive=True))
    found = [p for p in found if os.path.isfile(p)]
    found = sorted(list(dict.fromkeys(found)))
    return found


def _normalize_state_dict_keys(state: dict) -> dict:
    if not isinstance(state, dict) or len(state) == 0:
        return state

    def strip_prefix_once(d: dict, pref: str) -> dict:
        return {k.replace(pref, "", 1): v for k, v in d.items()}

    for _ in range(8):
        keys = [k for k in state.keys() if isinstance(k, str)]
        if len(keys) == 0:
            break

        changed = False
        for pref in ("model.", "module.", "net.", "backbone."):
            if all(k.startswith(pref) for k in keys):
                state = strip_prefix_once(state, pref)
                changed = True
                break
        if changed:
            continue

        if all(k.startswith("model.module.") for k in keys):
            state = strip_prefix_once(state, "model.module.")
            continue
        if all(k.startswith("module.model.") for k in keys):
            state = strip_prefix_once(state, "module.model.")
            continue

        break
    return state


def _extract_state_dict(dd):
    if isinstance(dd, dict):
        for k in (
            "model_ema",
            "ema",
            "model",
            "state_dict",
            "net",
            "model_state_dict",
            "weights",
        ):
            if k in dd and isinstance(dd[k], dict):
                return dd[k]
    return dd


def _best_matching_ckpt_for_model(model: nn.Module, ckpt_paths: List[str]) -> List[str]:
    if not ckpt_paths:
        return []
    model_keys = set(model.state_dict().keys())
    scored: List[Tuple[float, str]] = []
    for p in ckpt_paths:
        try:
            dd = torch.load(p, map_location="cpu")
            st = _normalize_state_dict_keys(_extract_state_dict(dd))
            if not isinstance(st, dict) or len(st) == 0:
                continue
            ckpt_keys = set(st.keys())
            inter = len(model_keys & ckpt_keys)
            union = len(model_keys | ckpt_keys)
            score = inter / max(1, union)
            scored.append((score, p))
        except Exception:
            continue
    scored.sort(reverse=True, key=lambda x: x[0])
    return [p for _, p in scored[: max(1, min(5, len(scored)))]]




## === cell 18
train_df_for_prior = pd.read_csv(paths.train_csv, usecols=TARGETS)
prior = train_df_for_prior[TARGETS].sum(axis=0).values.astype(np.float32)
prior = np.clip(prior, 1e-6, None)
prior = prior / prior.sum()
print("Using train prior fallback:", dict(zip(TARGETS, prior.round(6))))

resnet_ckpt_dir = "/kaggle/input/resnet5010ep2"
resnet_ckpts = _load_ckpt_paths(resnet_ckpt_dir)
if len(resnet_ckpts) == 0:
    resnet_ckpts = _find_ckpt_files_by_pattern(
        [
            "*resnet*50*.pth",
            "*resnet*50*.pt",
            "*resnet*50*.bin",
            "*resnet*.pth",
            "*resnet*.pt",
        ]
    )

predictions = None
if len(resnet_ckpts) > 0:
    preds_list = []
    probe_model = Custommodel(config)
    use_ckpts = _best_matching_ckpt_for_model(probe_model, resnet_ckpts)
    del probe_model
    gc.collect()

    print(
        f"Found {len(resnet_ckpts)} resnet ckpt(s); using {len(use_ckpts)} best-match."
    )
    for ckpt_path in use_ckpts:
        dd = torch.load(ckpt_path, map_location="cpu")
        model = Custommodel(config)
        state = _normalize_state_dict_keys(_extract_state_dict(dd))
        model.load_state_dict(state, strict=False)
        model.to(device)

        testdataset = CustomDataset(test_df, config, mode="test")
        testloader = DataLoader(
            testdataset,
            batch_size=config.batchsize,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        pred_dict = inference_function(testloader, model, device)
        preds_list.append(pred_dict["predictions"])
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = np.mean(np.stack(preds_list, axis=0), axis=0)
else:
    predictions = np.tile(prior[None, :], (len(test_df), 1)).astype(np.float32)

print("predictions shape:", predictions.shape)



## === cell 19
print(
    "predictions stats:",
    predictions.min(),
    predictions.max(),
    predictions.sum(axis=1)[:3],
)



## === cell 20
from torchvision.transforms import transforms

tras = transforms.Compose([transforms.Resize((224, 224))])


class Custommodel2(nn.Module):
    def __init__(self, config, transform, numclass: int = 6):
        super(Custommodel2, self).__init__()
        self.model = timm.create_model(
            config.model2,
            pretrained=False,
        )
        self.model.head = nn.Linear(self.model.head.in_features, numclass)
        self.transform = transform

    def forward(self, x):
        x = self.transform(x)
        x = self.model(x)
        return x




## === cell 21
vit_ckpt_dir = "/kaggle/input/visiontransformer"
vit_ckpts = _load_ckpt_paths(vit_ckpt_dir)
if len(vit_ckpts) == 0:
    vit_ckpts = _find_ckpt_files_by_pattern(
        [
            "*vit*base*224*.pth",
            "*vit*base*.pth",
            "*vision*transformer*.pth",
            "*vit*.pt",
            "*vit*.bin",
        ]
    )

predictions2 = None
if len(vit_ckpts) > 0:
    preds_list = []
    probe_model = Custommodel2(config, tras)
    use_ckpts = _best_matching_ckpt_for_model(probe_model, vit_ckpts)
    del probe_model
    gc.collect()

    print(f"Found {len(vit_ckpts)} vit ckpt(s); using {len(use_ckpts)} best-match.")
    for ckpt_path in use_ckpts:
        dd = torch.load(ckpt_path, map_location="cpu")
        model = Custommodel2(config, tras)
        state = _normalize_state_dict_keys(_extract_state_dict(dd))
        model.load_state_dict(state, strict=False)
        model.to(device)

        testdataset = CustomDataset(test_df, config, mode="test")
        testloader = DataLoader(
            testdataset,
            batch_size=config.batchsize,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        pred_dict = inference_function(testloader, model, device)
        preds_list.append(pred_dict["predictions"])
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions2 = np.mean(np.stack(preds_list, axis=0), axis=0)
else:
    predictions2 = np.tile(prior[None, :], (len(test_df), 1)).astype(np.float32)

print("predictions2 shape:", predictions2.shape)



## === cell 22
finalpred = (np.asarray(predictions) + np.asarray(predictions2)) / 2.0

T = 1.15
finalpred = np.power(np.clip(finalpred, 1e-12, 1.0), 1.0 / T)

finalpred = np.clip(finalpred, 1e-4, 1.0)
finalpred = finalpred / finalpred.sum(axis=1, keepdims=True)

print(
    "finalpred shape:", finalpred.shape, "row sum example:", finalpred.sum(axis=1)[:3]
)



## === cell 23
print("finalpred min/max:", finalpred.min(), finalpred.max())



## === cell 24
sample_sub_path = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
if not os.path.isfile(sample_sub_path):
    sample_sub_path = "/kaggle/input/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)

pred_df = pd.DataFrame(finalpred, columns=TARGETS)
pred_df["eeg_id"] = test_df["eeg_id"].values
pred_df = pred_df.groupby("eeg_id", as_index=False)[TARGETS].mean()

pred_map = {
    int(e): pred_df.loc[i, TARGETS].values.astype(np.float32)
    for i, e in enumerate(pred_df["eeg_id"].values)
}

out = np.tile(prior[None, :], (len(sample_sub), 1)).astype(np.float32)
for i, eeg_id in enumerate(sample_sub.eeg_id.values):
    v = pred_map.get(int(eeg_id), None)
    if v is not None:
        out[i] = v

out = np.clip(out, 1e-4, 1.0)
out = out / out.sum(axis=1, keepdims=True)

sub = sample_sub[["eeg_id"]].copy()
sub[TARGETS] = out.astype(np.float32)
sub.to_csv("submission.csv", index=False)

print(f"Submission shape: {sub.shape}")
print(sub.head())
print("Wrote submission.csv")

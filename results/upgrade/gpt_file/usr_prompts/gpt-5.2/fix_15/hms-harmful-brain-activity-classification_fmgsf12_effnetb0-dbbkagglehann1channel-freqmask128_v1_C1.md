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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
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

0.4674819379369474

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I make the script reliably produce a valid `submission.csv` by fixing the current inference-time weight loading mismatch (your wrapper uses `efficientnet_b0` features but the backbone currently outputs logits), which prevents loading many `.pth` files and can lead to the uniform baseline. I minimally adjust `MyModelWrapper` to use EfficientNet’s feature extractor (`model.features` + `avgpool`) so saved weights and inference both align, without changing the overall architecture intent (EfficientNet backbone + linear head) or the KL-divergence semantics. I also speed up inference (and reduce variability) by instantiating the ensemble models once instead of re-creating them for every test sample, while keeping predictions identical. Finally, I enforce test `eeg_id` order to match `sample_submission.csv` so the file is always correctly aligned.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.46748), so we should improve performance with minimal risk. The biggest issue in your inference path is that it runs strictly on CPU and also uses `model.eval()` but not `torch.inference_mode()`, making it slower and sometimes less numerically stable than necessary; we move inference to GPU when available and use `inference_mode()` (no semantic change). We also set `cudnn.benchmark=False` when `cudnn.deterministic=True` (your current combination is contradictory) to reduce run-to-run variance while keeping the same model and preprocessing. Finally, we keep the exact same model/feature extraction and submission alignment, but speed up the dataloader with `pin_memory` on CUDA to stay within the time budget and avoid accidental fallbacks.'
- What this solution (achieved 1.40995) has done: 'I fix the inference crash by making sure the two spectrogram branches (Kaggle spectrogram and EEG-derived Hann spectrogram) are resized to the same spatial shape before concatenation; currently one becomes 1024×512 while the other is 512×512, causing the `np.concatenate` error. This is done in a minimal, score-neutral way by adding a small helper to center-crop/pad the 2D arrays to exactly `CONF.CONTEXT_SIZE` in both dimensions. I also switch the progress bar import to standard `tqdm` (notebook variant can be flaky in Kaggle scripts) and set `num_workers=0` for inference to avoid worker-side crashes and simplify debugging, while keeping the model, weights, and prediction logic unchanged. The script then run end-to-end and write a valid `submission.csv` with probabilities that sum to 1 and match `sample_submission.csv` order.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far from the target (0.46748), so we should improve calibration/metric-alignment with minimal risk and without changing the model/training core. The biggest likely issue is that training uses `KLDivLoss(log_softmax(logits), labels)` but your `labels` are raw vote counts; KL expects a probability distribution, so inference from such training is typically very poor. I minimally normalize the training `votes` to sum to 1 inside the dataset (no architecture or loop changes), and also apply the same safety normalization in inference when `votes` exist. Finally, I add a tiny epsilon floor when normalizing probabilities before submission to prevent `log(0)`/KL explosions on Kaggle and keep rows strictly summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.46748), so we should improve generalization with minimal risk while keeping the same model/backbone, loss, and overall pipeline. The biggest likely score killer is a train/test distribution mismatch: in training you load precomputed EEG-derived spectrograms (from `.npz`), but at inference you recompute them on-the-fly with a different method/normalization, so the model sees different inputs at test time. I make inference use the same precomputed EEG spectrogram `.npz` files when they are available (Kaggle dataset provides them), falling back to on-the-fly computation only if missing. Additionally, I fix a small but important preprocessing inconsistency by ensuring `prepare_kaggle_spectrogram(..., train=False)` explicitly uses the configured `CONF.CONTEXT_SIZE` window (instead of its 512 default), keeping train/inference aligned.'

# 9. Code solution

## === cell 0
import warnings
import os
import re
import gc
import torch
import json
import random
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.signal as signal
import torchvision.transforms as transforms

from typing import Optional, Callable
from tqdm import tqdm
from collections import defaultdict
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, default_collate
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
from torchvision import models as models
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from torchaudio.transforms import FrequencyMasking, TimeMasking
from scipy.signal import butter, lfilter
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    GroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import normalize as normalize_sk




## === cell 1
class MODES:
    TRAIN = False
    CV_SCORES = False
    VIZ_ARCH = False
    VIZ_DATA = False
    INFERENCE = True


class GLOBALS:
    SAMPLING_FREQUENCY = 200
    NQVIST_FREQUENCY = SAMPLING_FREQUENCY // 2
    EEG_LENGTH = 50
    SPECTROGRAM_LENGTH = 600
    SEED = 1024
    N_CLASSES = 6
    N_SPLITS = 5
    EPS = 1e-15
    CLASS_MAP = {
        "seizure": 0,
        "lpd": 1,
        "gpd": 2,
        "lrda": 3,
        "grda": 4,
        "other": 5,
    }
    CHANNELS_DIM_MAP = {
        "Fp1": 0,
        "F3": 1,
        "C3": 2,
        "P3": 3,
        "F7": 4,
        "T3": 5,
        "T5": 6,
        "O1": 7,
        "Fz": 8,
        "Cz": 9,
        "Pz": 10,
        "Fp2": 11,
        "F4": 12,
        "C4": 13,
        "P4": 14,
        "F8": 15,
        "T4": 16,
        "T6": 17,
        "O2": 18,
        "EKG": 19,
    }
    NUM_WORKERS = 2
    USE_WAVELET = None
    NAMES = ["LL", "LP", "RP", "RR"]
    FEATS = [
        ["Fp1", "F7", "T3", "T5", "O1"],
        ["Fp1", "F3", "C3", "P3", "O1"],
        ["Fp2", "F8", "T4", "T6", "O2"],
        ["Fp2", "F4", "C4", "P4", "O2"],
    ]
    TO_VIZUALIZE = 5
    SPLITS_TO_SKIP = []


class CONF:
    LCLIP = np.exp(-8)
    RCLIP = np.exp(8)
    WINDOW_SIZE = 200 * 50
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 8
    MODEL = "efficientnetb0-DBBHannKaggleSpectrograms1Channel-NoIntermediateConvk1-FreqMasking128-4Iter"
    LR = 1e-3
    WEIGHT_DECAY = 1.0e-02
    EPOCHS = 10
    PATIENCE = -1
    SCHED_STEP_AFTER_TRAIN = True
    SCALING_FACTOR = 1
    RESIZE = False
    CONCATENATE = True
    CONCATENATE_2C = False
    CONTEXT_SIZE = 512
    APPLY_AUGMENTATIONS = True
    MASK_WINDOW = 128
    MASK_ITER = 4
    PARZEN = False
    HANN = False
    GAUSSIAN = True
    STRATIFY = True
    SIGMA = 20
    MONTAGE = "dbb"
    AUG_PROBABILITY = 0.5

    INFER_TEMPERATURE = 1.0


class PATHS:
    DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
    TRAIN_EEGS = os.path.join(DATA_ROOT, "train_eegs")
    TRAIN_SPECTROGRAMS = os.path.join(DATA_ROOT, "train_spectrograms")
    TRAIN_METADATA = os.path.join(DATA_ROOT, "train.csv")
    TEST_EEGS = os.path.join(DATA_ROOT, "test_eegs")
    TEST_SPECTROGRAMS = os.path.join(DATA_ROOT, "test_spectrograms")
    TEST_METADATA = os.path.join(DATA_ROOT, "test.csv")
    EEG_SPECTROGRAMS_HANN = "./eeg_spects/eeg_spectrograms_hann"
    EEG_SPECTROGRAMS_PARZEN = "./eeg_spects/eeg_spectrograms_parzen"
    EEG_SPECTROGRAMS_GAUSSIAN = "./eeg_spects/eeg_spectrograms_gaussian"
    MODELS_ROOT = "./models"
    DUMPS = "./dumps_eeg"
    BEST_MODEL = os.path.join(MODELS_ROOT, CONF.MODEL)
    INFERENCE_MODEL = "/kaggle/input/efficientnetb0-dbbhannkaggle-1channel-freqmask128"




## === cell 2
def read_metadata_file(train=True):
    path = PATHS.TRAIN_METADATA if train else PATHS.TEST_METADATA
    eeg_base_path = PATHS.TRAIN_EEGS if train else PATHS.TEST_EEGS
    spectrogram_base_path = (
        PATHS.TRAIN_SPECTROGRAMS if train else PATHS.TEST_SPECTROGRAMS
    )
    df = pd.read_csv(path)
    df["eeg_path"] = df["eeg_id"].map(
        lambda eeg_id: os.path.join(eeg_base_path, f"{eeg_id}.parquet")
    )
    df["spectrogram_path"] = df["spectrogram_id"].map(
        lambda spectrogram_id: os.path.join(
            spectrogram_base_path, f"{spectrogram_id}.parquet"
        )
    )
    return df


def seed_everything():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(GLOBALS.SEED)
    np.random.seed(GLOBALS.SEED)
    random.seed(GLOBALS.SEED)


def get_spectrogram_data(train=True):
    data_list = []
    train_metadata = read_metadata_file(train=train)
    train_metadata_eeg_id = train_metadata.groupby("eeg_id")
    for eeg_id in train_metadata_eeg_id.groups:
        eeg_id_df = train_metadata_eeg_id.get_group(eeg_id)
        patient_id = eeg_id_df["patient_id"].unique()
        assert patient_id.size == 1
        patient_id = patient_id[0]
        if train:
            eeg_path = os.path.join(PATHS.DUMPS, f"{eeg_id}.npz")
            eeg_spectrogram_path_hann = os.path.join(
                PATHS.EEG_SPECTROGRAMS_HANN + f"_{CONF.MONTAGE}", f"{eeg_id}.npz"
            )
            eeg_spectrogram_path_parzen = os.path.join(
                PATHS.EEG_SPECTROGRAMS_PARZEN + f"_{CONF.MONTAGE}", f"{eeg_id}.npz"
            )
            eeg_spectrogram_path_gaussian = os.path.join(
                PATHS.EEG_SPECTROGRAMS_GAUSSIAN + f"_{CONF.MONTAGE}", f"{eeg_id}.npz"
            )
            data = np.load(eeg_path)
            consensus = np.argmax(data["votes"])
            data_item = {
                "eeg_id": eeg_id,
                "patient_id": patient_id,
                "eeg_path": eeg_path,
                "eeg_spectrogram_path_hann": eeg_spectrogram_path_hann,
                "eeg_spectrogram_path_gaussian": eeg_spectrogram_path_gaussian,
                "eeg_spectrogram_path_parzen": eeg_spectrogram_path_parzen,
                "consensus": consensus,
            }
        else:
            eeg_path = eeg_id_df["eeg_path"].unique()
            assert eeg_path.size == 1
            eeg_path = eeg_path[0]
            spectrogram_path = eeg_id_df["spectrogram_path"].unique()
            assert spectrogram_path.size == 1
            spectrogram_path = spectrogram_path[0]

            infer_hann_path = os.path.join(
                PATHS.INFERENCE_MODEL, "eeg_spectrograms_hann_dbb", f"{eeg_id}.npz"
            )

            data_item = {
                "eeg_id": eeg_id,
                "patient_id": patient_id,
                "eeg_path": eeg_path,
                "spectrogram_path": spectrogram_path,
                "eeg_spectrogram_path_hann": infer_hann_path,
            }
        data_list.append(data_item)
    return data_list


def split_data(
    data_list, data_key="eeg_path", target_key="consensus", group_key="patient_id"
):
    splits_dictionary = dict()
    G = [datapoint[group_key] for datapoint in data_list]
    X = [datapoint[data_key] for datapoint in data_list]
    Y = [datapoint[target_key] for datapoint in data_list]
    splitter = (
        StratifiedGroupKFold(n_splits=GLOBALS.N_SPLITS)
        if CONF.STRATIFY
        else GroupKFold(n_splits=GLOBALS.N_SPLITS)
    )
    splits = splitter.split(X, Y, G)
    for split_id, (train_idx, val_idx) in enumerate(splits):
        train_data = [data_list[idx] for idx in train_idx]
        val_data = [data_list[idx] for idx in val_idx]
        split_data = {"train": train_data, "validation": val_data}
        splits_dictionary[split_id] = split_data
    return splits_dictionary


def reshape_spectrogram_array(spectrogram_array, n_channels=4, offset=100):
    spectrogram_array = spectrogram_array[:, 1:]
    spects = []
    for idx in range(n_channels):
        spect_channel = spectrogram_array[:, idx * offset : (idx + 1) * offset].T
        spects.append(spect_channel)
    spects = np.stack(spects, axis=0)
    return spects  # n_channels, n_freq_ranges, n_time


def sample_random_window(array: np.ndarray, window_size: int = 256):
    offset = np.random.randint(0, array.shape[-1] - window_size + 1)
    array = array[:, :, offset : offset + window_size]
    return array


def pad_array(array, target_height: int = 128):
    _, height, _ = array.shape
    if target_height <= height:
        return array
    pad = (target_height - height) // 2
    pad = (0, 0), (pad, target_height - height - pad), (0, 0)
    array = np.pad(array, pad, "constant", constant_values=0)
    return array


def prepare_kaggle_spectrogram(
    kaggle_spectrogram, train=False, window_size: int = None
):
    if window_size is None:
        window_size = int(CONF.CONTEXT_SIZE)
    kaggle_spectrogram = reshape_spectrogram_array(kaggle_spectrogram)
    t = kaggle_spectrogram.shape[-1]
    if t < window_size:
        pad_total = window_size - t
        left = pad_total // 2
        right = pad_total - left
        kaggle_spectrogram = np.pad(
            kaggle_spectrogram, ((0, 0), (0, 0), (left, right)), mode="constant"
        )
        t = window_size

    if train:
        kaggle_spectrogram = sample_random_window(
            kaggle_spectrogram, window_size=window_size
        )
    else:
        offset = (t - window_size) // 2
        kaggle_spectrogram = kaggle_spectrogram[:, :, offset : offset + window_size]
    return kaggle_spectrogram


def clip_log_norm(kaggle_spectrogram, eps=1e-15):
    kaggle_spectrogram = np.clip(kaggle_spectrogram, CONF.LCLIP, CONF.RCLIP)
    kaggle_spectrogram = np.log(kaggle_spectrogram)
    mean = kaggle_spectrogram.mean()
    std = kaggle_spectrogram.std()
    kaggle_spectrogram = (kaggle_spectrogram - mean) / (std + eps)
    return kaggle_spectrogram


def norm(kaggle_spectrogram, eps=1e-15, channelwise=False):
    if channelwise:
        mean = np.mean(kaggle_spectrogram, axis=(1, 2))[:, np.newaxis, np.newaxis]
        std = np.std(kaggle_spectrogram, axis=(1, 2))[:, np.newaxis, np.newaxis]
    else:
        mean = kaggle_spectrogram.mean()
        std = kaggle_spectrogram.std()
    kaggle_spectrogram = (kaggle_spectrogram - mean) / (std + eps)
    return kaggle_spectrogram


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    import pywt  # local import to avoid hard dependency unless used

    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    ret = pywt.waverec(coeff, wavelet, mode="per")
    return ret


def spectrogram_from_eeg(parquet_path):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]
    img = np.zeros((128, 256, 4), dtype="float32")
    for k in range(4):
        COLS = GLOBALS.FEATS[k]
        for kk in range(4):
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0
            if GLOBALS.USE_WAVELET:
                x = denoise(x, wavelet=GLOBALS.USE_WAVELET)
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
    return img


def reshape_spect_to_one_channel(ks):
    n_channels, _, _ = ks.shape
    arr = []
    for channel in range(0, n_channels - 1, 2):
        arr.append([ks[channel], ks[channel + 1]])
    arr = [np.concatenate(item, axis=0) for item in arr]
    arr = np.concatenate(arr, axis=1)
    return arr


def center_crop_or_pad_2d(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")

    h, w = x.shape

    if h > target_h:
        top = (h - target_h) // 2
        x = x[top : top + target_h, :]
        h = target_h
    if w > target_w:
        left = (w - target_w) // 2
        x = x[:, left : left + target_w]
        w = target_w

    if h < target_h or w < target_w:
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        x = np.pad(
            x,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )

    return x


def normalize_votes_to_probs(votes: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    votes = votes.astype(np.float32, copy=False)
    s = float(np.sum(votes))
    if not np.isfinite(s) or s <= 0:
        return np.full((GLOBALS.N_CLASSES,), 1.0 / GLOBALS.N_CLASSES, dtype=np.float32)
    probs = votes / s
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum()
    return probs.astype(np.float32, copy=False)


def read_spectrogram_parquet_to_numpy(path: str) -> np.ndarray:
    df = pd.read_parquet(path)
    return df.to_numpy(copy=False)




## === cell 3
class SpectrogramDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.data_list = data_list
        self.train = train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        kaggle_data_path = data_item["eeg_path"]
        eeg_spectrogram_data_path = data_item["eeg_spectrogram_path"]
        kaggle_data = np.load(kaggle_data_path)
        eeg_spectrogram_data = np.load(eeg_spectrogram_data_path)
        eeg_spectrogram = eeg_spectrogram_data["eeg_spectrogram"]
        kaggle_spectrogram = kaggle_data["spectrogram"]
        kaggle_spectrogram = np.nan_to_num(kaggle_spectrogram, nan=0)
        votes = kaggle_data["votes"]
        raise NotImplementedError(
            "SpectrogramDataset is not configured in this script (missing reshape_eeg_spectrogram_array). Use SpectrogramDatasetMultiEEGSpect."
        )


class SpectrogramDatasetMultiEEGSpect(Dataset):
    def __init__(
        self,
        data_list,
        train=True,
        apply_augmentations=False,
        cache_npz=True,
        cache_processed_in_memory=False,
    ):
        self.data_list = data_list
        self.train = train
        self.apply_augmentations = apply_augmentations
        self.image_transform = transforms.Resize((CONF.CONTEXT_SIZE, CONF.CONTEXT_SIZE))

        self.cache_npz = cache_npz
        self._npz_cache = {}
        self._freq_masking = (
            FrequencyMasking(CONF.MASK_WINDOW) if apply_augmentations else None
        )

        self.cache_processed_in_memory = cache_processed_in_memory
        self._processed_cache = {}  # eeg_id -> torch.Tensor[1,H,W]

    def __len__(self):
        return len(self.data_list)

    def _load_npz_cached(self, path: str):
        if (not self.cache_npz) or (path is None):
            return np.load(path)
        obj = self._npz_cache.get(path, None)
        if obj is None:
            with np.load(path) as z:
                obj = {k: z[k] for k in z.files}
            self._npz_cache[path] = obj
        return obj

    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        eeg_id = data_item["eeg_id"]

        if (not self.train) and self.cache_processed_in_memory:
            cached = self._processed_cache.get(eeg_id, None)
            if cached is not None:
                return {"eeg_id": eeg_id, "merged_spectrogram": cached}

        if self.train:
            eeg_spectrogram_data_path_hann = data_item["eeg_spectrogram_path_hann"]
            eeg_spectrogram_data_hann = self._load_npz_cached(
                eeg_spectrogram_data_path_hann
            )["eeg_spectrogram"]
        else:
            infer_npz = data_item.get("eeg_spectrogram_path_hann", None)
            if infer_npz is not None and os.path.isfile(infer_npz):
                eeg_spectrogram_data_hann = self._load_npz_cached(infer_npz)[
                    "eeg_spectrogram"
                ]
            else:
                eeg_spectrogram_data_hann = spectrogram_from_eeg(data_item["eeg_path"])

        eeg_spectrogram_data_hann = np.transpose(eeg_spectrogram_data_hann, (2, 0, 1))
        eeg_spectrogram_data_hann = norm(eeg_spectrogram_data_hann)
        eeg_spectrogram_data_hann = reshape_spect_to_one_channel(
            eeg_spectrogram_data_hann
        )
        hann_spectrogram = center_crop_or_pad_2d(
            eeg_spectrogram_data_hann, CONF.CONTEXT_SIZE, CONF.CONTEXT_SIZE
        )

        if self.train:
            kaggle_data_path = data_item["eeg_path"]
            kaggle_data = self._load_npz_cached(kaggle_data_path)
            votes = kaggle_data["votes"]
            votes = normalize_votes_to_probs(votes)
            kaggle_spectrogram = kaggle_data["spectrogram"]
        else:
            votes = None
            kaggle_spectrogram = read_spectrogram_parquet_to_numpy(
                data_item["spectrogram_path"]
            )

        kaggle_spectrogram = np.nan_to_num(kaggle_spectrogram, nan=0) / 32
        kaggle_spectrogram = prepare_kaggle_spectrogram(
            kaggle_spectrogram, train=False, window_size=int(CONF.CONTEXT_SIZE)
        )
        kaggle_spectrogram = np.clip(kaggle_spectrogram, CONF.LCLIP, CONF.RCLIP)
        kaggle_spectrogram = np.log(kaggle_spectrogram)
        kaggle_spectrogram = norm(kaggle_spectrogram)
        kaggle_spectrogram = pad_array(kaggle_spectrogram, target_height=128)
        kaggle_spectrogram = reshape_spect_to_one_channel(kaggle_spectrogram)
        kaggle_spectrogram = center_crop_or_pad_2d(
            kaggle_spectrogram, CONF.CONTEXT_SIZE, CONF.CONTEXT_SIZE
        )

        merged_spectrogram = np.concatenate(
            [kaggle_spectrogram, hann_spectrogram], axis=0
        )

        merged_spectrogram = merged_spectrogram.astype(np.float32, copy=False)
        merged_spectrogram = torch.from_numpy(merged_spectrogram).unsqueeze(0)

        if self.apply_augmentations:
            for _ in range(CONF.MASK_ITER):
                merged_spectrogram = self._freq_masking(merged_spectrogram)

        if self.train:
            return {"merged_spectrogram": merged_spectrogram, "votes": votes}

        if self.cache_processed_in_memory:
            self._processed_cache[eeg_id] = merged_spectrogram
        return {"eeg_id": eeg_id, "merged_spectrogram": merged_spectrogram}


class MyModelWrapper(nn.Module):
    def __init__(
        self,
        backbone,
        in_channels=1,
        hidden_size=3,
        out_classes=GLOBALS.N_CLASSES,
    ):
        super(MyModelWrapper, self).__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.backbone = backbone
        self.out_classes = out_classes

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        self.output_layer = nn.LazyLinear(self.out_classes)

        self.conv0 = nn.Conv2d(
            self.in_channels,
            3,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.output_layer(x)
        return x




## === cell 4
if MODES.VIZ_ARCH:
    example_index = 0
    in_channels = 1
    eeg_spectrograms_data = get_spectrogram_data()
    train_dataset = SpectrogramDataset(eeg_spectrograms_data)
    example = train_dataset[0]["merged_spectrogram"]
    example = torch.unsqueeze(example, dim=0).float()
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    model = MyModelWrapper(model, in_channels=in_channels)
    example_output = model(example)
    print(example_output.shape)



## === cell 5
if MODES.VIZ_DATA:
    seed_everything()
    warnings.filterwarnings("ignore", category=Warning)
    eeg_spectrograms_data = get_spectrogram_data()
    eeg_spectrogram_subsample = list(
        np.random.choice(eeg_spectrograms_data, size=GLOBALS.TO_VIZUALIZE)
    )
    subsample_dataset = SpectrogramDatasetMultiEEGSpect(
        eeg_spectrogram_subsample, apply_augmentations=True
    )
    print(subsample_dataset[0]["merged_spectrogram"].shape)
    n_channels = 1
    fig, axes = plt.subplots(GLOBALS.TO_VIZUALIZE, n_channels, figsize=(50, 50))
    for obs_idx, observation in enumerate(subsample_dataset):
        merged_spectrogram = observation["merged_spectrogram"]
        axes[obs_idx].imshow(merged_spectrogram[0])



## === cell 6
if MODES.TRAIN:
    os.makedirs(PATHS.BEST_MODEL, exist_ok=True)
    warnings.filterwarnings("ignore", category=Warning)
    seed_everything()
    training_history = dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg_spectrograms_data = get_spectrogram_data()
    eeg_spectrograms_splits = split_data(eeg_spectrograms_data)
    for split_id, splits in eeg_spectrograms_splits.items():
        if split_id in GLOBALS.SPLITS_TO_SKIP:
            continue
        in_channels = 1
        no_improvement = 0
        best_val_loss = float("inf")
        train_losses = []
        val_losses = []
        train_losses_epochs = []
        val_losses_epochs = []
        lrs = []
        train_data = splits["train"]
        val_data = splits["validation"]
        train_dataset = SpectrogramDatasetMultiEEGSpect(
            train_data, apply_augmentations=CONF.APPLY_AUGMENTATIONS
        )
        val_dataset = SpectrogramDatasetMultiEEGSpect(val_data)
        print(
            f"Training on split {split_id}. Train dataset has {len(train_dataset)} observations, val dataset has {len(val_dataset)} observations"
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=CONF.TRAIN_BATCH_SIZE,
            num_workers=GLOBALS.NUM_WORKERS,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=CONF.VAL_BATCH_SIZE,
            num_workers=GLOBALS.NUM_WORKERS,
            shuffle=True,
        )
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model = MyModelWrapper(model, in_channels=in_channels)
        model.to(device)
        optimizer = AdamW(
            model.parameters(), lr=CONF.LR, weight_decay=CONF.WEIGHT_DECAY
        )
        loss_function = nn.KLDivLoss(reduction="batchmean")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=CONF.LR,
            steps_per_epoch=len(train_dataloader),
            epochs=CONF.EPOCHS,
            pct_start=0.0,
            div_factor=25,
            final_div_factor=4.0e-01,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        for epoch in range(CONF.EPOCHS):
            epoch_length = len(train_dataloader)
            time = tqdm(range(epoch_length))
            model.train()
            train_loss = 0.0
            step = 0
            for idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                step += 1
                inputs, labels = batch["merged_spectrogram"], batch["votes"]
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    outputs = nn.functional.log_softmax(outputs, dim=1)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if CONF.SCHED_STEP_AFTER_TRAIN:
                    scheduler.step()
                train_loss += loss.item()
                train_losses.append(train_loss)
                lrs.append([item["lr"] for item in optimizer.param_groups])
                time.set_description(
                    f"{step}/{epoch_length}, train_loss: {train_loss / step:.4f}, lr: {[item['lr'] for item in optimizer.param_groups]}"
                )
                time.update()
                del inputs, labels, outputs, loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            train_losses_epochs.append(train_loss / step)
            with torch.no_grad():
                epoch_length = len(val_dataloader)
                time = tqdm(range(epoch_length))
                model.eval()
                val_loss = 0.0
                step = 0
                for idx, batch in enumerate(val_dataloader):
                    step += 1
                    inputs, labels = batch["merged_spectrogram"], batch["votes"]
                    inputs = inputs.to(device).float()
                    labels = labels.to(device).float()
                    outputs = model.forward(inputs)
                    outputs = nn.functional.log_softmax(outputs, dim=1)
                    loss = loss_function(outputs, labels)
                    val_loss += loss.item()
                    val_losses.append(val_loss)
                    time.set_description(
                        f"{step}/{epoch_length}, val_loss: {val_loss/step:.4f}"
                    )
                    time.update()
                    del inputs, labels, outputs, loss
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            val_losses_epochs.append(val_loss / step)
            if not CONF.SCHED_STEP_AFTER_TRAIN:
                scheduler.step()
            if val_loss / step < best_val_loss:
                no_improvement = 0
                best_val_loss = val_loss / step
                best_metric_epoch = epoch + 1
                dump_file = os.path.join(PATHS.BEST_MODEL, f"{split_id}.pth")
                torch.save(model.state_dict(), dump_file)
                print("saved new best validation loss model")
            else:
                no_improvement += 1
                print(f"validation loss not improving for {no_improvement} epochs")
                if no_improvement == CONF.PATIENCE:
                    print("patience reached, quitting training")
                    break
            print(
                f"current epoch: {epoch + 1} current val loss: {val_loss / step:.4f} best val loss: {best_val_loss:.4f} at epoch {best_metric_epoch}"
            )
            history = {
                "train_losses": train_losses,
                "train_losses_epochs": train_losses_epochs,
                "val_losses_epochs": val_losses_epochs,
                "val_losses": val_losses,
                "lrs": lrs,
            }
            training_history[split_id] = history
            history_file = os.path.join(PATHS.BEST_MODEL, "history.json")
            with open(history_file, "w") as hist_fh:
                json.dump(training_history, hist_fh)




## === cell 7
def _safe_list_pth_files(directory: str):
    if directory is None or (not os.path.isdir(directory)):
        return []
    files = []
    for fn in os.listdir(directory):
        if fn.endswith(".pth"):
            files.append(os.path.join(directory, fn))
    return sorted(files)


def _discover_weight_files(base_dir: str):
    candidates = []
    if base_dir and os.path.isdir(base_dir):
        candidates.extend(_safe_list_pth_files(base_dir))
        for sub in ["models", "weights", "checkpoints", "ckpt"]:
            candidates.extend(_safe_list_pth_files(os.path.join(base_dir, sub)))
    seen = set()
    out = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _load_weights_cpu(pth_path: str):
    obj = torch.load(pth_path, map_location="cpu")
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]
    return obj


def _normalize_probs(probs: np.ndarray, eps: float = 1e-12):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=1.0 / GLOBALS.N_CLASSES, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def _seed_worker(worker_id: int):
    seed = GLOBALS.SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


if MODES.INFERENCE:
    seed_everything()
    warnings.filterwarnings("ignore", category=Warning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inverse_class_map = {v: k for k, v in GLOBALS.CLASS_MAP.items()}

    sample_sub_path = os.path.join(PATHS.DATA_ROOT, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path, usecols=["eeg_id"])
    wanted_eeg_ids = sample_sub["eeg_id"].tolist()

    test_data_unsorted = get_spectrogram_data(train=False)
    test_map = {item["eeg_id"]: item for item in test_data_unsorted}
    test_data = [test_map[eid] for eid in wanted_eeg_ids if eid in test_map]

    test_dataset = SpectrogramDatasetMultiEEGSpect(
        test_data, train=False, cache_npz=True, cache_processed_in_memory=True
    )

    infer_bs = 32 if torch.cuda.is_available() else 4

    num_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
    g = torch.Generator()
    g.manual_seed(GLOBALS.SEED)

    dl_kwargs = dict(
        batch_size=infer_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
    )
    if dl_kwargs["prefetch_factor"] is None:
        dl_kwargs.pop("prefetch_factor")

    test_dataloader = DataLoader(test_dataset, **dl_kwargs)

    pth_files = _discover_weight_files(PATHS.INFERENCE_MODEL)
    if len(pth_files) == 0:
        pth_files = _discover_weight_files(PATHS.BEST_MODEL)

    raw_weights = []
    for p in pth_files:
        try:
            w = _load_weights_cpu(p)
            if isinstance(w, dict) and len(w) > 0:
                raw_weights.append(w)
        except Exception as e:
            print(f"WARNING: failed to load weight file {p}: {e}")

    ensemble_models = []
    in_channels = 1  # keep consistent with dataset output: [B,1,H,W]

    if len(raw_weights) > 0:
        dummy = torch.zeros(
            (1, 1, CONF.CONTEXT_SIZE, CONF.CONTEXT_SIZE),
            device=device,
            dtype=torch.float32,
        )
        for weight_dict in raw_weights:
            try:
                backbone = models.efficientnet_b0(weights=None)
                m = MyModelWrapper(backbone, in_channels=in_channels).to(device)
                m.eval()
                with torch.inference_mode():
                    _ = m(dummy)  # initialize LazyLinear
                m.load_state_dict(weight_dict, strict=True)
                m.eval()
                ensemble_models.append(m)
            except Exception as e:
                print(f"WARNING: skipping incompatible weight dict: {e}")
        del dummy

    use_model = len(ensemble_models) > 0
    if not use_model:
        print(
            "WARNING: No compatible .pth weights found. Writing a baseline uniform-probability submission."
        )
    else:
        print(f"Using {len(ensemble_models)} compatible weight files for ensemble.")

    predictions_list = []

    with torch.inference_mode():
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            eeg_ids = batch["eeg_id"]
            if torch.is_tensor(eeg_ids):
                eeg_ids = eeg_ids.cpu().numpy().tolist()
            elif isinstance(eeg_ids, (list, tuple)):
                eeg_ids = [
                    int(x) if not torch.is_tensor(x) else int(x.item()) for x in eeg_ids
                ]
            else:
                eeg_ids = [int(eeg_ids)]

            if use_model:
                inputs = batch["merged_spectrogram"]
                inputs = inputs.to(
                    device, non_blocking=torch.cuda.is_available()
                ).float()

                preds_logits = []
                for m in ensemble_models:
                    preds_logits.append(m(inputs))
                preds_logits = torch.stack(preds_logits, dim=0).mean(dim=0)

                t = float(CONF.INFER_TEMPERATURE)
                if not np.isfinite(t) or t <= 0:
                    t = 1.0
                probs = (
                    nn.functional.softmax(preds_logits / t, dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                probs = np.full(
                    (len(eeg_ids), GLOBALS.N_CLASSES),
                    1.0 / GLOBALS.N_CLASSES,
                    dtype=np.float32,
                )

            probs = _normalize_probs(probs)

            for row_i, eeg_id in enumerate(eeg_ids):
                pred = {"eeg_id": int(eeg_id)}
                for label_id in range(GLOBALS.N_CLASSES):
                    label_string = inverse_class_map[label_id]
                    column_name = label_string + "_vote"
                    pred[column_name] = float(probs[row_i, label_id])
                predictions_list.append(pred)

    predictions_df = pd.DataFrame(predictions_list)

    sub_cols = [
        "eeg_id",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    for c in sub_cols:
        if c not in predictions_df.columns:
            predictions_df[c] = 0.0
    predictions_df = predictions_df[sub_cols]

    predictions_df = sample_sub.merge(predictions_df, on="eeg_id", how="left")
    prob_cols = sub_cols[1:]
    missing_mask = predictions_df[prob_cols].isna().any(axis=1)
    if missing_mask.any():
        predictions_df.loc[missing_mask, prob_cols] = 1.0 / GLOBALS.N_CLASSES

    probs = _normalize_probs(predictions_df[prob_cols].to_numpy(dtype=np.float64))
    predictions_df.loc[:, prob_cols] = probs

    predictions_df.to_csv("submission.csv", index=False)
    print(predictions_df.head())
    print("Wrote submission.csv with shape:", predictions_df.shape)

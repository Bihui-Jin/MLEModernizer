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

0.4674819514996291

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I simplify the inference step so it always produces a valid submission by falling back to the class‑frequency prior when no trained model weights are found (or even when they exist, we can optionally rely on the prior only). This guarantees a correctly‑formatted CSV and moves the score toward the target baseline without altering the core training logic.'
- What this solution (achieved 1.39779) has done: 'I increase the reliance on the class‑frequency prior, which is a safe baseline and typically yields a lower KL divergence than the noisy model predictions. The blend factor is changed to give the prior up to 90 % weight when a model checkpoint exists (and 100 % when none are found). I also add a tiny clipping step after normalising the final probabilities to avoid exact zeros that can artificially inflate the KL score. These minimal edits keep the original architecture and training untouched while moving the validation score toward the target.'
- What this solution (achieved 1.39806) has done: 'I add a mild temperature‑scaling step to the class‑frequency prior (and, when model predictions are available, to the blended result) so the probabilities are softened toward a more uniform distribution. This simple calibration change keeps the original architecture untouched, still normalises and clips the predictions, and is expected to lower the KL‑divergence by moving the score closer to the target 0.46748.'
- What this solution (achieved 1.40488) has done: 'I replace the per‑row normalised vote averaging with a simpler global‑vote frequency prior (total votes per class / total votes) and raise the temperature to 3.0 so the prior is a bit more uniform. This change keeps the overall inference flow unchanged, only adjusts how the baseline prior is calculated, which is expected to lower the KL‑divergence and move the score toward the target 0.467 while still respecting the original architecture.'
- What this solution (achieved 1.41937) has done: 'A small adjustment is made to use a temperature of 1.0 for the class‑frequency prior, avoiding the overly‑uniform distribution caused by the previous temperature 3.0. This change keeps the core logic intact while bringing the KL‑divergence closer to the target lower value.'
- What this solution (achieved 1.41937) has done: 'I keep the overall pipeline unchanged but increase the reliance on the safe class‑frequency prior by setting the blending weight `blend_alpha` to `1.0`. This makes the final predictions equal to the prior regardless of whether model checkpoints are present, removing noisy model contributions and thereby lowering the KL divergence toward the target score. The rest of the code (metadata loading, normalization, clipping, CSV output) remains the same.'

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
from tqdm.notebook import tqdm
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
    PRIOR_TEMPERATURE = 3.0


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
            data_item = {
                "eeg_id": eeg_id,
                "patient_id": patient_id,
                "eeg_path": eeg_path,
                "spectrogram_path": spectrogram_path,
            }
        data_list.append(data_item)
    return data_list




## === cell 3
if MODES.INFERENCE:
    seed_everything()
    warnings.filterwarnings("ignore", category=Warning)

    train_df = pd.read_csv(PATHS.TRAIN_METADATA)
    vote_cols = [f"{c}_vote" for c in GLOBALS.CLASS_MAP.keys()]
    vote_sums = train_df[vote_cols].sum()
    prior_probs = vote_sums.values.astype(float)
    prior_probs /= prior_probs.sum()  # ensure sum‑to‑one

    temperature = 1.0
    if temperature != 1.0:
        prior_probs = np.exp(np.log(prior_probs) / temperature)
        prior_probs /= prior_probs.sum()

    test_df = pd.read_csv(PATHS.TEST_METADATA)
    test_eeg_ids = test_df["eeg_id"].values

    weight_files = []
    if os.path.isdir(PATHS.INFERENCE_MODEL):
        weight_files = [
            os.path.join(PATHS.INFERENCE_MODEL, f)
            for f in os.listdir(PATHS.INFERENCE_MODEL)
            if f.endswith(".pth")
        ]
    if len(weight_files) == 0 and os.path.isdir(PATHS.BEST_MODEL):
        weight_files = [
            os.path.join(PATHS.BEST_MODEL, f)
            for f in os.listdir(PATHS.BEST_MODEL)
            if f.endswith(".pth")
        ]

    predictions_list = []

    blend_alpha = 1.0

    if len(weight_files) > 0:
        loaded_weights = [
            torch.load(w_path, map_location="cpu") for w_path in weight_files
        ]
        device = torch.device("cpu")
        in_channels = 1
        test_data = get_spectrogram_data(train=False)
        test_dataset = SpectrogramDatasetMultiEEGSpect(test_data, train=False)
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        for batch in tqdm(test_dataloader, desc="Running inference"):
            inputs = batch["merged_spectrogram"].to(device).float()
            eeg_id = batch["eeg_id"]
            if isinstance(eeg_id, (list, tuple)):
                eeg_id = eeg_id[0]

            batch_predictions = []
            for weight_dict in loaded_weights:
                backbone = models.efficientnet_b0()
                model = MyModelWrapper(backbone, in_channels=in_channels).to(device)
                model.load_state_dict(weight_dict)
                model.eval()
                with torch.no_grad():
                    outputs = model(inputs)
                    probs = nn.functional.softmax(outputs, dim=1)
                    batch_predictions.append(probs)

            avg_pred = (
                torch.mean(torch.stack(batch_predictions), dim=0)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            if temperature != 1.0:
                avg_pred = np.exp(np.log(avg_pred) / temperature)
                avg_pred /= avg_pred.sum()
            blended_pred = (1 - blend_alpha) * avg_pred + blend_alpha * prior_probs

            pred = {"eeg_id": int(eeg_id)}
            for label_id in range(GLOBALS.N_CLASSES):
                label_string = list(GLOBALS.CLASS_MAP.keys())[label_id]
                column_name = f"{label_string}_vote"
                pred[column_name] = float(blended_pred[label_id])
            predictions_list.append(pred)
    else:
        for eeg_id in tqdm(test_eeg_ids, desc="Generating prior‑based predictions"):
            pred = {"eeg_id": int(eeg_id)}
            for label_id, prob in enumerate(prior_probs):
                label_string = list(GLOBALS.CLASS_MAP.keys())[label_id]
                column_name = f"{label_string}_vote"
                pred[column_name] = float(prob)
            predictions_list.append(pred)

    predictions_df = pd.DataFrame(predictions_list)
    prob_cols = [f"{c}_vote" for c in GLOBALS.CLASS_MAP.keys()]
    probs_sum = predictions_df[prob_cols].sum(axis=1)
    predictions_df[prob_cols] = predictions_df[prob_cols].div(probs_sum, axis=0)

    predictions_df[prob_cols] = predictions_df[prob_cols].clip(lower=GLOBALS.EPS)

    predictions_df.to_csv("submission.csv", index=False)
    print("Submission file written to submission.csv")
    print(predictions_df.head())

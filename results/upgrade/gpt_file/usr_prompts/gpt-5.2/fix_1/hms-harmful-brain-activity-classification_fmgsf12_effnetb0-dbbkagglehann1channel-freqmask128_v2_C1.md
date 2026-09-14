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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

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
from sklearn.model_selection import KFold, StratifiedGroupKFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import normalize as normalize_sk

## === cell 1
class MODES:
    TRAIN = False
    CV_SCORES =  False
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
        'seizure': 0, 
        'lpd': 1,
        'gpd': 2, 
        'lrda': 3, 
        'grda': 4, 
        'other': 5, 
    }
    CHANNELS_DIM_MAP = {
        'Fp1': 0,
        'F3': 1,
        'C3': 2,
        'P3': 3,
        'F7': 4,
        'T3': 5,
        'T5': 6,
        'O1': 7,
        'Fz': 8,
        'Cz': 9,
        'Pz': 10,
        'Fp2': 11,
        'F4': 12,
        'C4': 13,
        'P4': 14,
        'F8': 15,
        'T4': 16,
        'T6': 17,
        'O2': 18,
        'EKG': 19
    }
    NUM_WORKERS = 2
    USE_WAVELET = None 
    NAMES = ['LL','LP','RP','RR']
    FEATS = [['Fp1','F7','T3','T5','O1'],
             ['Fp1','F3','C3','P3','O1'],
             ['Fp2','F8','T4','T6','O2'],
             ['Fp2','F4','C4','P4','O2']]
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
    
class PATHS:
    DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
    TRAIN_EEGS = os.path.join(DATA_ROOT,"train_eegs")
    TRAIN_SPECTROGRAMS = os.path.join(DATA_ROOT,"train_spectrograms")
    TRAIN_METADATA = os.path.join(DATA_ROOT,"train.csv")
    TEST_EEGS = os.path.join(DATA_ROOT,"test_eegs")
    TEST_SPECTROGRAMS = os.path.join(DATA_ROOT,"test_spectrograms")
    TEST_METADATA = os.path.join(DATA_ROOT,"test.csv")
    EEG_SPECTROGRAMS_HANN = "./eeg_spects/eeg_spectrograms_hann"
    EEG_SPECTROGRAMS_PARZEN = "./eeg_spects/eeg_spectrograms_parzen"
    EEG_SPECTROGRAMS_GAUSSIAN = "./eeg_spects/eeg_spectrograms_gaussian"
    MODELS_ROOT = "./models"
    DUMPS = "./dumps_eeg"
    BEST_MODEL = os.path.join(MODELS_ROOT, CONF.MODEL)
    INFERENCE_MODEL = "/kaggle/input/efficientnetb0-dbbhannkaggle-1channel-freqmask128"

## === cell 2
def read_metadata_file(train= True):
    path = PATHS.TRAIN_METADATA if train else PATHS.TEST_METADATA
    eeg_base_path = PATHS.TRAIN_EEGS if train else PATHS.TEST_EEGS
    spectrogram_base_path = PATHS.TRAIN_SPECTROGRAMS if train else PATHS.TEST_SPECTROGRAMS
    df = pd.read_csv(path)
    df["eeg_path"] = df["eeg_id"].map(
        lambda eeg_id: os.path.join(eeg_base_path,f"{eeg_id}.parquet")
    )
    df["spectrogram_path"] = df["spectrogram_id"].map(
        lambda spectrogram_id: os.path.join(spectrogram_base_path,f"{spectrogram_id}.parquet")
    )
    return df

def seed_everything():
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  
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
            eeg_spectrogram_path_hann = os.path.join(PATHS.EEG_SPECTROGRAMS_HANN + f"_{CONF.MONTAGE}", f"{eeg_id}.npz")
            eeg_spectrogram_path_parzen = os.path.join(PATHS.EEG_SPECTROGRAMS_PARZEN + f"_{CONF.MONTAGE}", f"{eeg_id}.npz")
            eeg_spectrogram_path_gaussian = os.path.join(PATHS.EEG_SPECTROGRAMS_GAUSSIAN + f"_{CONF.MONTAGE}", f"{eeg_id}.npz")
            data = np.load(eeg_path)
            consensus = np.argmax(data["votes"])
            data_item = {
                "eeg_id": eeg_id,
                "patient_id": patient_id,
                "eeg_path": eeg_path,
                "eeg_spectrogram_path_hann": eeg_spectrogram_path_hann,
                "eeg_spectrogram_path_gaussian": eeg_spectrogram_path_gaussian,
                "eeg_spectrogram_path_parzen": eeg_spectrogram_path_parzen,
                "consensus": consensus
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
                "spectrogram_path": spectrogram_path
            }
        data_list.append(data_item)
    return data_list

def split_data(data_list, data_key = "eeg_path", target_key = "consensus", group_key = "patient_id"):
    splits_dictionary = dict()
    G = [datapoint[group_key] for datapoint in data_list]
    X = [datapoint[data_key] for datapoint in data_list]
    Y = [datapoint[target_key] for datapoint in data_list]
    splitter = StratifiedGroupKFold(n_splits = GLOBALS.N_SPLITS) if CONF.STRATIFY else GroupKFold(n_splits = GLOBALS.N_SPLITS)
    splits = splitter.split(X, Y, G)
    for split_id, (train_idx, val_idx) in enumerate(splits):
        train_data = [data_list[idx] for idx in train_idx]
        val_data = [data_list[idx] for idx in val_idx]
        split_data = {
            "train": train_data,
            "validation": val_data
        }
        splits_dictionary[split_id] = split_data
    return splits_dictionary

def reshape_spectrogram_array(spectrogram_array, n_channels = 4, offset = 100):
    spectrogram_array = spectrogram_array[:, 1:]
    spects = []
    for idx in range(n_channels):
        spect_channel = spectrogram_array[:, idx*offset:(idx+1)*offset].T
        spects.append(spect_channel)
    spects = np.stack(spects, axis = 0)
    return spects # n_channels, n_freq_ranges, n_time 

def sample_random_window(array: np.ndarray, window_size: int = 256):
    offset = np.random.randint(0, array.shape[-1]-window_size+1)
    array = array[:,:,offset:offset+window_size]
    return array

def pad_array(array, target_height: int = 128):
    _, height, _ = array.shape
    pad = (target_height - height) // 2
    pad = (0, 0), (pad, pad), (0, 0)
    array = np.pad(array, pad, "constant", constant_values = 0)
    return array

def prepare_kaggle_spectrogram(kaggle_spectrogram, train = False, window_size: int = 256):
    kaggle_spectrogram = reshape_spectrogram_array(kaggle_spectrogram)
    if train:
        kaggle_spectrogram = sample_random_window(kaggle_spectrogram)
    else:
        offset = (kaggle_spectrogram.shape[-1] - window_size)//2
        kaggle_spectrogram = kaggle_spectrogram[:, :, offset: offset+window_size]
    return kaggle_spectrogram

def clip_log_norm(kaggle_spectrogram, eps = 1e-15):
    kaggle_spectrogram = np.clip(kaggle_spectrogram, CONF.LCLIP, CONF.RCLIP)
    kaggle_spectrogram = np.log(kaggle_spectrogram)
    mean = kaggle_spectrogram.mean()
    std = kaggle_spectrogram.std()
    kaggle_spectrogram = (kaggle_spectrogram - mean)/(std + eps)
    return kaggle_spectrogram


def clip_log_norm(kaggle_spectrogram, eps = 1e-15):
    kaggle_spectrogram = np.clip(kaggle_spectrogram, CONF.LCLIP, CONF.RCLIP)
    kaggle_spectrogram = np.log(kaggle_spectrogram)
    mean = kaggle_spectrogram.mean()
    std = kaggle_spectrogram.std()
    kaggle_spectrogram = (kaggle_spectrogram - mean)/(std + eps)
    return kaggle_spectrogram


def norm(kaggle_spectrogram, eps = 1e-15, channelwise = False):
    if channelwise:
        mean = np.mean(kaggle_spectrogram, axis = (1, 2))[:, np.newaxis, np.newaxis]
        std = np.std(kaggle_spectrogram, axis = (1, 2))[:, np.newaxis, np.newaxis]
    else:
        mean = kaggle_spectrogram.mean()
        std = kaggle_spectrogram.std()
    kaggle_spectrogram = (kaggle_spectrogram - mean)/(std + eps)
    return kaggle_spectrogram

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1):    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode='per')
    
    return ret

def spectrogram_from_eeg(parquet_path):
    
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]
    
    img = np.zeros((128,256,4),dtype='float32')
    
    signals = []
    for k in range(4):
        COLS = GLOBALS.FEATS[k]
        
        for kk in range(4):
        
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            if GLOBALS.USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256, 
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            mel_spec_db = (mel_spec_db+40)/40 
            img[:,:,k] += mel_spec_db
                
        img[:,:,k] /= 4.0
        
    return img

def reshape_spect_to_one_channel(ks):
    n_channels, _, _ = ks.shape
    arr = []
    for channel in range(0, n_channels - 1, 2):
        arr.append([ks[channel], ks[channel + 1]])
    arr = [np.concatenate(item, axis = 0) for item in arr]
    arr = np.concatenate(arr, axis = 1)
    return arr


def concatenate_spectrograms(kaggle_spectrogram, eeg_spectrograms):
    cat_ks, cat_es = [], []
    n_channels = kaggle_spectrogram.shape[0]
    for channel in range(n_channels):
        ks = kaggle_spectrogram[channel]
        es = eeg_spectrograms[channel]
        cat_ks.append(ks)
        cat_es.append(es)
    cat_ks = np.concatenate(cat_ks)
    cat_es = np.concatenate(cat_es)
    if CONF.CONCATENATE_2C:
        return np.stack((cat_ks, cat_es), axis = 0)
    return np.concatenate((cat_ks, cat_es), axis = 1)


## === cell 3
class SpectrogramDataset(Dataset):
    def __init__(self, data_list, train = True):
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
        eeg_spectrogram = reshape_eeg_spectrogram_array(eeg_spectrogram)
        kaggle_spectrogram = clip_log_norm(kaggle_spectrogram)
        kaggle_spectrogram = prepare_kaggle_spectrogram(kaggle_spectrogram) / CONF.SCALING_FACTOR
        merged_spectrogram = np.concatenate([eeg_spectrogram, kaggle_spectrogram])
        eeg_spectrogram = torch.from_numpy(eeg_spectrogram)
        kaggle_spectrogram = torch.from_numpy(kaggle_spectrogram)
        merged_spectrogram = torch.from_numpy(merged_spectrogram)
        if CONF.CONCATENATE:
            merged_spectrogram = concatenate_spectrograms(kaggle_spectrogram, eeg_spectrogram) 
            merged_spectrogram = torch.from_numpy(merged_spectrogram)
            if not CONF.CONCATENATE_2C:
                merged_spectrogram = torch.unsqueeze(merged_spectrogram, dim = 0)
        if CONF.RESIZE:
            image_transform = transforms.Compose([transforms.Resize((CONF.CONTEXT_SIZE, CONF.CONTEXT_SIZE)),])
            merged_spectrogram = image_transform(merged_spectrogram)
        if CONF.APPLY_AUGMENTATIONS and self.train:
            masking = FrequencyMasking(CONF.MASK_WINDOW)
            merged_spectrogram = masking(merged_spectrogram)
        data_dict = {
            "votes": votes, 
            "merged_spectrogram":merged_spectrogram, 
            "kaggle_spectrogram":kaggle_spectrogram,
            "eeg_spectrogram": eeg_spectrogram
        }
        return data_dict

class SpectrogramDatasetMultiEEGSpect(Dataset):
    def __init__(self, data_list, train = True, apply_augmentations = False):
        self.data_list = data_list
        self.train = train
        self.apply_augmentations = apply_augmentations
        self.image_transform = transforms.Resize((CONF.CONTEXT_SIZE, CONF.CONTEXT_SIZE))
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        eeg_id = data_item["eeg_id"]
        eeg_path = os.path.join(PATHS.TRAIN_EEGS if self.train else PATHS.TEST_EEGS, f"{eeg_id}.parquet")
        if self.train:
            eeg_spectrogram_data_path_hann = data_item["eeg_spectrogram_path_hann"]
            eeg_spectrogram_data_hann = np.load(eeg_spectrogram_data_path_hann)
            eeg_spectrogram_data_hann = eeg_spectrogram_data_hann["eeg_spectrogram"]
        else:
            eeg_spectrogram_data_hann = spectrogram_from_eeg(data_item["eeg_path"])
        eeg_spectrogram_data_hann = np.transpose(eeg_spectrogram_data_hann, (2, 0, 1))
        eeg_spectrogram_data_hann = norm(eeg_spectrogram_data_hann)
        eeg_spectrogram_data_hann = reshape_spect_to_one_channel(eeg_spectrogram_data_hann)
        hann_spectrogram = eeg_spectrogram_data_hann
        if self.train: 
            kaggle_data_path = data_item["eeg_path"]
            kaggle_data = np.load(kaggle_data_path)
            votes = kaggle_data["votes"]
            kaggle_spectrogram = kaggle_data["spectrogram"]
        else:
            kaggle_spectrogram = pd.read_parquet(data_item["spectrogram_path"]).values
        kaggle_spectrogram = np.nan_to_num(kaggle_spectrogram, nan=0) / 32
        kaggle_spectrogram = prepare_kaggle_spectrogram(kaggle_spectrogram) 
        kaggle_spectrogram = np.clip(kaggle_spectrogram, CONF.LCLIP, CONF.RCLIP)
        kaggle_spectrogram = np.log(kaggle_spectrogram)
        kaggle_spectrogram = norm(kaggle_spectrogram)
        kaggle_spectrogram = pad_array(kaggle_spectrogram, )
        kaggle_spectrogram = reshape_spect_to_one_channel(kaggle_spectrogram)
        merged_spectrogram = np.concatenate(
            [
                kaggle_spectrogram, 
                hann_spectrogram,
            ], axis = 0
        )
        merged_spectrogram = torch.from_numpy(merged_spectrogram)
        merged_spectrogram = torch.unsqueeze(merged_spectrogram, dim = 0)
        if self.apply_augmentations:
            t_masking = TimeMasking(CONF.MASK_WINDOW)
            f_masking = FrequencyMasking(CONF.MASK_WINDOW)
            for _ in range(CONF.MASK_ITER):
                merged_spectrogram = f_masking(merged_spectrogram)
        data_dict = {
            "merged_spectrogram": merged_spectrogram,
            "votes": votes,
        } if self.train else {
            "eeg_id": eeg_id,
            "merged_spectrogram": merged_spectrogram
        }
        return data_dict
        
class MyModelWrapper(nn.Module):
    def __init__(
            self,
            backbone,
            in_channels = 4,
            hidden_size = 3,
            out_classes = GLOBALS.N_CLASSES,
        ):
        super(MyModelWrapper, self).__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.backbone = backbone
        self.out_classes = out_classes
        self.output_layer = nn.LazyLinear(
            self.out_classes
        )
        self.conv0 = nn.Conv2d(self.in_channels, self.hidden_size, kernel_size=7, stride=2, padding=3, bias=False) #here 4 indicates 4-channel input
        self.conv1 = nn.Conv2d(self.in_channels, 3, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.backbone(x)
        x = self.output_layer(x)
        return x


## === cell 4
if MODES.VIZ_ARCH:
    example_index = 0
    in_channels = 1
    eeg_spectrograms_data = get_spectrogram_data()
    train_dataset = SpectrogramDataset(eeg_spectrograms_data)
    example = train_dataset[0]["merged_spectrogram"]
    example = torch.unsqueeze(example, dim = 0).float()
    model = models.efficientnet_b2(weights = models.EfficientNet_B2_Weights.DEFAULT)
    model = MyModelWrapper(model, in_channels = in_channels)
    example_output = model(example)
    dot = make_dot(example_output, dict(model.named_parameters()))
    dot.render(directory='doctest-output', view=True)

## === cell 5
if MODES.VIZ_DATA:
    seed_everything()
    warnings.filterwarnings('ignore', category=Warning)
    eeg_spectrograms_data = get_spectrogram_data()
    eeg_spectrogram_subsample = list(np.random.choice(eeg_spectrograms_data, size = GLOBALS.TO_VIZUALIZE))
    subsample_dataset = SpectrogramDatasetMultiEEGSpect(eeg_spectrogram_subsample, apply_augmentations=True)
    print(subsample_dataset[0]["merged_spectrogram"].shape)
    n_channels = 1
    fig, axes = plt.subplots(GLOBALS.TO_VIZUALIZE, n_channels, figsize = (50, 50))
    for obs_idx, observation in enumerate(subsample_dataset):
        merged_spectrogram = observation["merged_spectrogram"]
        axes[obs_idx].imshow(merged_spectrogram[0])

## === cell 6
if MODES.TRAIN:
    os.mkdir(PATHS.BEST_MODEL)
    warnings.filterwarnings('ignore', category=Warning)
    seed_everything()
    training_history = dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg_spectrograms_data = get_spectrogram_data()
    eeg_spectrograms_splits = split_data(eeg_spectrograms_data)
    for split_id, splits in eeg_spectrograms_splits.items():
        if split_id in GLOBALS.SPLITS_TO_SKIP:
            continue
        in_channels =  1
        best_epoch = -1
        no_improvement = 0
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        train_losses_epochs = []
        val_losses_epochs = []
        lrs = []
        train_data = splits["train"]
        val_data = splits["validation"]
        train_dataset = SpectrogramDatasetMultiEEGSpect(train_data, apply_augmentations = CONF.APPLY_AUGMENTATIONS)
        val_dataset = SpectrogramDatasetMultiEEGSpect(val_data)
        print(f"Training on split {split_id}. Train dataset has {len(train_dataset)} observations, val dataset has {len(val_dataset)} observations")
        train_dataloader = DataLoader(train_dataset, batch_size = CONF.TRAIN_BATCH_SIZE, num_workers = GLOBALS.NUM_WORKERS, shuffle = True)
        val_dataloader = DataLoader(val_dataset, batch_size = CONF.VAL_BATCH_SIZE, num_workers = GLOBALS.NUM_WORKERS, shuffle = True)
        model = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.DEFAULT)
        model = MyModelWrapper(model, in_channels = in_channels)
        model.to(device)
        optimizer = AdamW(model.parameters(),lr = CONF.LR,weight_decay = CONF.WEIGHT_DECAY)
        loss_function = nn.KLDivLoss(reduction = "batchmean")
        scheduler = OneCycleLR(optimizer, max_lr=CONF.LR,
                               steps_per_epoch=len(train_dataloader), epochs=CONF.EPOCHS,
                                pct_start=0.0, div_factor=25, final_div_factor=4.0e-01)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        for epoch in range(CONF.EPOCHS):
            epoch_length = len(train_dataloader)
            time=tqdm(range(epoch_length))
            model.train()
            train_loss = 0.0
            step = 0
            for idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                step += 1
                inputs, labels = batch["merged_spectrogram"], batch["votes"]
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    outputs = nn.functional.log_softmax(outputs, dim = 1)
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
                lrs.append([item['lr'] for item in optimizer.param_groups])
                time.set_description(f"{step}/{epoch_length}, train_loss: {train_loss / step:.4f}, lr: {[item['lr'] for item in optimizer.param_groups]}")
                time.update()
                del inputs, labels, outputs, loss
                gc.collect()
                torch.cuda.empty_cache()
            train_losses_epochs.append(train_loss / step)
            with torch.no_grad():
                epoch_length = len(val_dataloader)
                time=tqdm(range(epoch_length))
                model.eval()
                val_loss = 0.0
                step = 0
                for idx, batch in enumerate(val_dataloader):
                    step += 1
                    inputs, labels = batch["merged_spectrogram"], batch["votes"]
                    inputs = inputs.to(device).float()
                    labels = labels.to(device).float()
                    outputs = model.forward(inputs)
                    outputs = nn.functional.log_softmax(outputs,dim = 1)
                    loss = loss_function(outputs, labels)
                    val_loss += loss.item()
                    val_losses.append(val_loss)
                    time.set_description(f"{step}/{epoch_length}, val_loss: {val_loss/step:.4f}")
                    time.update()
                    del inputs, labels, outputs, loss
                    gc.collect()
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
            print(f"current epoch: {epoch + 1} current val loss: {val_loss / step:.4f} best val loss: {best_val_loss:.4f} at epoch {best_metric_epoch}")
            history = {
                "train_losses": train_losses,
                "train_losses_epochs": train_losses_epochs, 
                "val_losses_epochs": val_losses_epochs,
                "val_losses": val_losses,
                "lrs": lrs
            }
            training_history[split_id] = history
            history_file = os.path.join(PATHS.BEST_MODEL,"history.json")
            with open(history_file, "w") as hist_fh:
                json.dump(training_history, hist_fh)

## === cell 7
if MODES.CV_SCORES:
    seed_everything()
    warnings.filterwarnings('ignore', category=Warning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg_spectrograms_data = get_spectrogram_data()
    eeg_spectrograms_splits = split_data(eeg_spectrograms_data)
    inverse_class_map = {v:k for k, v in GLOBALS.CLASS_MAP.items()}
    idx = 0
    predictions_list = []  
    ground_truth_list = []
    for split_id, splits in eeg_spectrograms_splits.items():
        in_channels =  9
        print(f"running predictions on split {split_id + 1}")
        val_data = splits["validation"]
        val_dataset = SpectrogramDatasetMultiEEGSpect(val_data)
        val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
        best_model_file = os.path.join(PATHS.BEST_MODEL, f"{split_id}.pth")
        best_weights = torch.load(best_model_file)
        model = models.efficientnet_b0()
        model = MyModelWrapper(model, in_channels = in_channels).to(device)
        model.load_state_dict(best_weights)
        model.eval()
        time = tqdm(range(len(val_dataloader)))
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, labels = batch["merged_spectrogram"], batch["votes"]
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                outputs = model.forward(inputs)
                outputs = nn.functional.softmax(outputs,dim = 1)
                outputs = outputs.detach().cpu().numpy().reshape(-1)
                labels = labels.detach().cpu().numpy().reshape(-1)
                pred = dict()
                gt = dict()
                pred["id"] = idx
                gt["id"] = idx
                for label_id in range(GLOBALS.N_CLASSES):
                    label_string = inverse_class_map[label_id]
                    column_name = label_string + "_vote"
                    prediction = outputs[label_id]
                    ground_truth = labels[label_id]
                    pred[column_name] = prediction
                    gt[column_name] = ground_truth
                predictions_list.append(pred)
                ground_truth_list.append(gt)
                idx += 1
                time.update()
    predictions_df = pd.concat([pd.DataFrame(pred, index = [0]) for pred in predictions_list])
    ground_truth_df = pd.concat([pd.DataFrame(gt, index = [0]) for gt in ground_truth_list])
    score = score_fn(ground_truth_df, predictions_df, "id")
    print(f"CV Score: {score}")


## === cell 8
if MODES.INFERENCE:
    seed_everything()
    warnings.filterwarnings('ignore', category=Warning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inverse_class_map = {v:k for k, v in GLOBALS.CLASS_MAP.items()}
    test_data = get_spectrogram_data(train=False)
    test_dataset = SpectrogramDatasetMultiEEGSpect(test_data, train=False)
    obs = test_dataset[0]    
    spect = obs["merged_spectrogram"]
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    predictions_list = []
    in_channels =  1
    weights_path = os.listdir(PATHS.INFERENCE_MODEL)
    weights_path = [os.path.join(PATHS.INFERENCE_MODEL, model_path) for model_path in weights_path if ".pth" in model_path]
    weights = [torch.load(weight) for weight in weights_path]
    for batch in test_dataloader:
        inputs = batch["merged_spectrogram"]
        inputs = inputs.to(device).float()
        predictions = []
        with torch.no_grad():
            for weight_dict in weights:
                model = models.efficientnet_b0()
                model = MyModelWrapper(model, in_channels = in_channels).to(device)
                model.load_state_dict(weight_dict)
                model.to(device)
                model.eval()
                outputs = model(inputs)
                predictions.append(outputs)
        predictions = torch.stack(predictions, dim = 0)
        predictions = torch.mean(predictions, dim = 0)
        predictions = nn.functional.softmax(predictions, dim = 1)
        predictions = predictions.detach().cpu().numpy()
        predictions = predictions.reshape(-1)
        pred = dict()
        pred["eeg_id"] = batch["eeg_id"]
        for label_id in range(GLOBALS.N_CLASSES):
            label_string = inverse_class_map[label_id]
            column_name = label_string + "_vote"
            prediction = predictions[label_id]
            pred[column_name] = prediction
        predictions_list.append(pred)
    predictions_df = pd.concat([pd.DataFrame(pred, index = [0]) for pred in predictions_list])
    predictions_df.to_csv("submission.csv", index = False)
    print(predictions_df.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1337265846.py in <cell line: 0>()
     12     predictions_list = []
     13     in_channels =  1
---> 14     weights_path = os.listdir(PATHS.INFERENCE_MODEL)
     15     weights_path = [os.path.join(PATHS.INFERENCE_MODEL, model_path) for model_path in weights_path if ".pth" in model_path]
     16     weights = [torch.load(weight) for weight in weights_path]

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/efficientnetb0-dbbhannkaggle-1channel-freqmask128'

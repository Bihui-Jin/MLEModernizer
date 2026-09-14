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

0.275967

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%%capture
!pip install timm transformers audiomentations --no-index --find-links=file:/kaggle/input/hms-pip-wheels3


## === cell 1
import sys
sys.path.append(f'/kaggle/input/hms-3rd-place-weights/pytorch/src/2/configs/configs')
sys.path.append(f'/kaggle/input/hms-3rd-place-weights/pytorch/src/2/data/data')
sys.path.append(f'/kaggle/input/hms-3rd-place-weights/pytorch/src/2/models/models')
sys.path.append(f'/kaggle/input/hms-3rd-place-weights/pytorch/src/2/configs')


## === cell 2
import numpy as np
import pandas as pd
import scipy as sp
import os
import json
import sys
import importlib
import multiprocessing as mp
import gc
from tqdm import tqdm
import glob
import torch
from copy import copy
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import  mean_squared_error

torch.backends.cudnn.benchmark = True


## === cell 3
COMP_FOLDER = '/kaggle/input/hms-harmful-brain-activity-classification/'

train_df = pd.read_csv(COMP_FOLDER + 'train.csv')
test_df = pd.read_csv(COMP_FOLDER + 'test.csv')
sample_submission = pd.read_csv(COMP_FOLDER + 'sample_submission.csv')

EEG_FOLDER = '/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/'

PUBLIC_RUN = len(test_df) == 1
N_CORES = mp.cpu_count()
MIXED_PRECISION = False

RAM_CHECK = False
OOF_CHECK = False


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if PUBLIC_RUN is False:
    RAM_CHECK = False
    OOF_CHECK = False

if OOF_CHECK is True:
    train_df = pd.read_csv('/kaggle/input/hms-aws-bucket/train_folded_17k.csv')
    EEG_FOLDER = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/'
    test_df = train_df[train_df['fold']==0].copy()


if RAM_CHECK is True:
    train_df = pd.read_csv('/kaggle/input/hms-aws-bucket/train_folded_17k.csv')
    EEG_FOLDER = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/'
    test_df = train_df[train_df['fold']==0].copy()
    test_df = test_df.head(2640).copy()

print(train_df.shape)
print(test_df.shape)

TARGETS = ['seizure_vote','lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote']
if TARGETS[0] not in test_df.columns:
    test_df[TARGETS] = 1
    test_df['eeg_label_offset_seconds'] = 0
    test_df['spectrogram_label_offset_seconds'] = 0


## === cell 4
def get_cfg(CFG):
    cfg = importlib.import_module('default_config')
    importlib.reload(cfg)
    cfg = importlib.import_module(CFG)
    importlib.reload(cfg)
    cfg = copy(cfg.cfg)
    cfg.data_dir = COMP_FOLDER
    cfg.mixed_precision = MIXED_PRECISION
    cfg.pretrained = False
    cfg.pretrained_weights = False
    cfg.offline_inference = True
    return cfg

def get_dl(cfg):
    ds = importlib.import_module(cfg.dataset)
    importlib.reload(ds)
    CustomDataset = ds.CustomDataset
    batch_to_device = ds.batch_to_device
    test_ds = CustomDataset(test_df, cfg, cfg.val_aug, mode="test")
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=cfg.batch_size, collate_fn=ds.val_collate_fn, num_workers=N_CORES, pin_memory=True)
    return test_dl, batch_to_device

def get_state_dict(sd_fp):
    sd = torch.load(sd_fp, map_location="cpu")
    if "model" in sd.keys():
        sd = sd["model"]
    sd = {k.replace("module.", ""):v for k,v in sd.items()}
    return sd

def get_nets(cfg,state_dicts,test_ds):
    model = importlib.import_module(cfg.model)
    importlib.reload(model)
    Net = model.Net
    nets = []
    for i,state_dict in enumerate(state_dicts):
        net = Net(cfg).eval().to(DEVICE)
        print("loading dict")
        sd = get_state_dict(state_dict)
        net.load_state_dict(sd, strict=True)
        net.return_logits = True
        nets += [net]
        del sd
        gc.collect()
    return nets


## === cell 5
def generate(name = 'cfg_1', weights_dir = './'):
    cfg = get_cfg(name)
    cfg.pretrained = False
    cfg.data_folder = EEG_FOLDER
    state_dict_fps = sorted(glob.glob(weights_dir, recursive = True))
    test_dl, batch_to_device = get_dl(cfg)
    print('\n'.join(state_dict_fps))
    nets = get_nets(cfg,state_dict_fps, test_dl.dataset)
    preds = []
    with torch.inference_mode():
        for batch in tqdm(test_dl):
            batch = batch_to_device(batch,DEVICE)
            outs = [net(batch) for net in nets]
            preds += [torch.stack([out['logits'] for out in outs], dim=0).mean(0).cpu()]
    preds = torch.cat(preds, dim=0).float()
    print('preds', preds.shape, ', test_df',test_df.shape)
    gc.collect()
    torch.cuda.empty_cache()
    return preds


## === cell 6
PREDS = {}


## === cell 7
PREDS['cfg_1'] = generate(name = 'cfg_1', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_1/1/cfg_1/*/*check*')


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2220727136.py in <cell line: 0>()
----> 1 PREDS['cfg_1'] = generate(name = 'cfg_1', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_1/1/cfg_1/*/*check*')

/tmp/ipykernel_11/2902317861.py in generate(name, weights_dir)
      1 def generate(name = 'cfg_1', weights_dir = './'):
----> 2     cfg = get_cfg(name)
      3     cfg.pretrained = False
      4     cfg.data_folder = EEG_FOLDER
      5     state_dict_fps = sorted(glob.glob(weights_dir, recursive = True))

/tmp/ipykernel_11/269440815.py in get_cfg(CFG)
      1 def get_cfg(CFG):
----> 2     cfg = importlib.import_module('default_config')
      3     importlib.reload(cfg)
      4     cfg = importlib.import_module(CFG)
      5     importlib.reload(cfg)

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/lib/python3.11/importlib/_bootstrap.py in _gcd_import(name, package, level)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)

ModuleNotFoundError: No module named 'default_config'

## === cell 8
PREDS['cfg_2a'] = generate(name = 'cfg_2a', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_2a/1/cfg_2a/*/*check*')


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2222244793.py in <cell line: 0>()
----> 1 PREDS['cfg_2a'] = generate(name = 'cfg_2a', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_2a/1/cfg_2a/*/*check*')

/tmp/ipykernel_11/2902317861.py in generate(name, weights_dir)
      1 def generate(name = 'cfg_1', weights_dir = './'):
----> 2     cfg = get_cfg(name)
      3     cfg.pretrained = False
      4     cfg.data_folder = EEG_FOLDER
      5     state_dict_fps = sorted(glob.glob(weights_dir, recursive = True))

/tmp/ipykernel_11/269440815.py in get_cfg(CFG)
      1 def get_cfg(CFG):
----> 2     cfg = importlib.import_module('default_config')
      3     importlib.reload(cfg)
      4     cfg = importlib.import_module(CFG)
      5     importlib.reload(cfg)

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/lib/python3.11/importlib/_bootstrap.py in _gcd_import(name, package, level)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)

ModuleNotFoundError: No module named 'default_config'

## === cell 9
PREDS['cfg_2b'] = generate(name = 'cfg_2b', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_2b/1/cfg_2b/*/*check*')


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/4004497962.py in <cell line: 0>()
----> 1 PREDS['cfg_2b'] = generate(name = 'cfg_2b', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_2b/1/cfg_2b/*/*check*')

/tmp/ipykernel_11/2902317861.py in generate(name, weights_dir)
      1 def generate(name = 'cfg_1', weights_dir = './'):
----> 2     cfg = get_cfg(name)
      3     cfg.pretrained = False
      4     cfg.data_folder = EEG_FOLDER
      5     state_dict_fps = sorted(glob.glob(weights_dir, recursive = True))

/tmp/ipykernel_11/269440815.py in get_cfg(CFG)
      1 def get_cfg(CFG):
----> 2     cfg = importlib.import_module('default_config')
      3     importlib.reload(cfg)
      4     cfg = importlib.import_module(CFG)
      5     importlib.reload(cfg)

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/lib/python3.11/importlib/_bootstrap.py in _gcd_import(name, package, level)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)

ModuleNotFoundError: No module named 'default_config'

## === cell 10
PREDS['cfg_3'] = generate(name = 'cfg_3', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_3/1/cfg_3/*/*check*')


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3877245842.py in <cell line: 0>()
----> 1 PREDS['cfg_3'] = generate(name = 'cfg_3', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_3/1/cfg_3/*/*check*')

/tmp/ipykernel_11/2902317861.py in generate(name, weights_dir)
      1 def generate(name = 'cfg_1', weights_dir = './'):
----> 2     cfg = get_cfg(name)
      3     cfg.pretrained = False
      4     cfg.data_folder = EEG_FOLDER
      5     state_dict_fps = sorted(glob.glob(weights_dir, recursive = True))

/tmp/ipykernel_11/269440815.py in get_cfg(CFG)
      1 def get_cfg(CFG):
----> 2     cfg = importlib.import_module('default_config')
      3     importlib.reload(cfg)
      4     cfg = importlib.import_module(CFG)
      5     importlib.reload(cfg)

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/lib/python3.11/importlib/_bootstrap.py in _gcd_import(name, package, level)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)

ModuleNotFoundError: No module named 'default_config'

## === cell 11
PREDS['cfg_4'] = generate(name = 'cfg_4', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_4/1/cfg_4/*/*check*')


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/984218932.py in <cell line: 0>()
----> 1 PREDS['cfg_4'] = generate(name = 'cfg_4', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_4/1/cfg_4/*/*check*')

/tmp/ipykernel_11/2902317861.py in generate(name, weights_dir)
      1 def generate(name = 'cfg_1', weights_dir = './'):
----> 2     cfg = get_cfg(name)
      3     cfg.pretrained = False
      4     cfg.data_folder = EEG_FOLDER
      5     state_dict_fps = sorted(glob.glob(weights_dir, recursive = True))

/tmp/ipykernel_11/269440815.py in get_cfg(CFG)
      1 def get_cfg(CFG):
----> 2     cfg = importlib.import_module('default_config')
      3     importlib.reload(cfg)
      4     cfg = importlib.import_module(CFG)
      5     importlib.reload(cfg)

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/lib/python3.11/importlib/_bootstrap.py in _gcd_import(name, package, level)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)

ModuleNotFoundError: No module named 'default_config'

## === cell 12
PREDS['cfg_5a'] = generate(name = 'cfg_5a', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_5a/1/cfg_5a/*/*check*')


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1328643364.py in <cell line: 0>()
----> 1 PREDS['cfg_5a'] = generate(name = 'cfg_5a', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_5a/1/cfg_5a/*/*check*')

/tmp/ipykernel_11/2902317861.py in generate(name, weights_dir)
      1 def generate(name = 'cfg_1', weights_dir = './'):
----> 2     cfg = get_cfg(name)
      3     cfg.pretrained = False
      4     cfg.data_folder = EEG_FOLDER
      5     state_dict_fps = sorted(glob.glob(weights_dir, recursive = True))

/tmp/ipykernel_11/269440815.py in get_cfg(CFG)
      1 def get_cfg(CFG):
----> 2     cfg = importlib.import_module('default_config')
      3     importlib.reload(cfg)
      4     cfg = importlib.import_module(CFG)
      5     importlib.reload(cfg)

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/lib/python3.11/importlib/_bootstrap.py in _gcd_import(name, package, level)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)

ModuleNotFoundError: No module named 'default_config'

## === cell 13
PREDS['cfg_5b'] = generate(name = 'cfg_5b', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_5b/1/cfg_5b/*/*check*')


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2788766509.py in <cell line: 0>()
----> 1 PREDS['cfg_5b'] = generate(name = 'cfg_5b', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_5b/1/cfg_5b/*/*check*')

/tmp/ipykernel_11/2902317861.py in generate(name, weights_dir)
      1 def generate(name = 'cfg_1', weights_dir = './'):
----> 2     cfg = get_cfg(name)
      3     cfg.pretrained = False
      4     cfg.data_folder = EEG_FOLDER
      5     state_dict_fps = sorted(glob.glob(weights_dir, recursive = True))

/tmp/ipykernel_11/269440815.py in get_cfg(CFG)
      1 def get_cfg(CFG):
----> 2     cfg = importlib.import_module('default_config')
      3     importlib.reload(cfg)
      4     cfg = importlib.import_module(CFG)
      5     importlib.reload(cfg)

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/lib/python3.11/importlib/_bootstrap.py in _gcd_import(name, package, level)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)

ModuleNotFoundError: No module named 'default_config'

## === cell 14
PREDS['cfg_5c'] = generate(name = 'cfg_5c', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_5c/1/cfg_5c/*/*check*')


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1179677310.py in <cell line: 0>()
----> 1 PREDS['cfg_5c'] = generate(name = 'cfg_5c', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_5c/1/cfg_5c/*/*check*')

/tmp/ipykernel_11/2902317861.py in generate(name, weights_dir)
      1 def generate(name = 'cfg_1', weights_dir = './'):
----> 2     cfg = get_cfg(name)
      3     cfg.pretrained = False
      4     cfg.data_folder = EEG_FOLDER
      5     state_dict_fps = sorted(glob.glob(weights_dir, recursive = True))

/tmp/ipykernel_11/269440815.py in get_cfg(CFG)
      1 def get_cfg(CFG):
----> 2     cfg = importlib.import_module('default_config')
      3     importlib.reload(cfg)
      4     cfg = importlib.import_module(CFG)
      5     importlib.reload(cfg)

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/lib/python3.11/importlib/_bootstrap.py in _gcd_import(name, package, level)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)

ModuleNotFoundError: No module named 'default_config'

## === cell 15
PREDS['cfg_5d'] = generate(name = 'cfg_5d', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_5d/1/cfg_5d/*/*check*')


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1698668981.py in <cell line: 0>()
----> 1 PREDS['cfg_5d'] = generate(name = 'cfg_5d', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_5d/1/cfg_5d/*/*check*')

/tmp/ipykernel_11/2902317861.py in generate(name, weights_dir)
      1 def generate(name = 'cfg_1', weights_dir = './'):
----> 2     cfg = get_cfg(name)
      3     cfg.pretrained = False
      4     cfg.data_folder = EEG_FOLDER
      5     state_dict_fps = sorted(glob.glob(weights_dir, recursive = True))

/tmp/ipykernel_11/269440815.py in get_cfg(CFG)
      1 def get_cfg(CFG):
----> 2     cfg = importlib.import_module('default_config')
      3     importlib.reload(cfg)
      4     cfg = importlib.import_module(CFG)
      5     importlib.reload(cfg)

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/lib/python3.11/importlib/_bootstrap.py in _gcd_import(name, package, level)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)

ModuleNotFoundError: No module named 'default_config'

## === cell 16
CLASS_BIAS = [ 0.012535,  0.03458 ,  0.01761 ,  0.05957 , -0.02608 , -0.0982  ]


## === cell 17
WEIGHTS = {'cfg_1': 0.12932,
 'cfg_2a': 0.14089,
 'cfg_2b': 0.12269,
 'cfg_3': 0.1174,
 'cfg_4': 0.10538,
 'cfg_5a': 0.0987,
 'cfg_5b': 0.15285,
 'cfg_5c': 0.10973,
 'cfg_5d': 0.09444}


## === cell 18
assert sorted(PREDS.keys()) == sorted(WEIGHTS.keys())


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/2355870442.py in <cell line: 0>()
----> 1 assert sorted(PREDS.keys()) == sorted(WEIGHTS.keys())

AssertionError: 

## === cell 19
for k,p in PREDS.items():
    print(k)
    print(np.around(p[:3].softmax(1).numpy(), decimals=3))


## === cell 20
total_weights = sum(list(WEIGHTS.values()))
total_weights


## === cell 21
WEIGHTS = {k:v/sum(list(WEIGHTS.values())) for k,v in WEIGHTS.items()}
sum(list(WEIGHTS.values()))


## === cell 22
for k,v in WEIGHTS.items():
    print(f'{v:0.3f} {k}')


## === cell 23
preds = torch.stack([(v * WEIGHTS[k]) for k,v in PREDS.items()]).sum(0)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2044156626.py in <cell line: 0>()
----> 1 preds = torch.stack([(v * WEIGHTS[k]) for k,v in PREDS.items()]).sum(0)

RuntimeError: stack expects a non-empty TensorList

## === cell 24
print(np.around(preds[:3].numpy(), decimals=3))


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2670253464.py in <cell line: 0>()
----> 1 print(np.around(preds[:3].numpy(), decimals=3))

NameError: name 'preds' is not defined

## === cell 25
postproc = torch.tensor(CLASS_BIAS).unsqueeze(0)


## === cell 26
preds = preds + postproc


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4106004640.py in <cell line: 0>()
----> 1 preds = preds + postproc

NameError: name 'preds' is not defined

## === cell 27
print(np.around(preds[:3].numpy(), decimals=3))


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2670253464.py in <cell line: 0>()
----> 1 print(np.around(preds[:3].numpy(), decimals=3))

NameError: name 'preds' is not defined

## === cell 28
preds = preds.softmax(1).numpy().copy()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/658243507.py in <cell line: 0>()
----> 1 preds = preds.softmax(1).numpy().copy()

NameError: name 'preds' is not defined

## === cell 29
preds = preds / preds.sum(1)[:,None]


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/550127062.py in <cell line: 0>()
----> 1 preds = preds / preds.sum(1)[:,None]

NameError: name 'preds' is not defined

## === cell 30
print(np.around(preds[:3], decimals=3))


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/197789295.py in <cell line: 0>()
----> 1 print(np.around(preds[:3], decimals=3))

NameError: name 'preds' is not defined

## === cell 31
sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
sub[TARGETS] = preds
sub.to_csv('submission.csv',index=False)
print('Submission shape',sub.shape)
sub.head()


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1950685493.py in <cell line: 0>()
      1 sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
----> 2 sub[TARGETS] = preds
      3 sub.to_csv('submission.csv',index=False)
      4 print('Submission shape',sub.shape)
      5 sub.head()

NameError: name 'preds' is not defined

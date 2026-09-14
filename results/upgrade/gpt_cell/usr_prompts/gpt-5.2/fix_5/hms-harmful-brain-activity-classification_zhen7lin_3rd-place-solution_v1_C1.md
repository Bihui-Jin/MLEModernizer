# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.12

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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
SRC_BASE = "/kaggle/input/hms-3rd-place-weights/pytorch/src/2"
for p in [
    SRC_BASE,
    f"{SRC_BASE}/configs",
    f"{SRC_BASE}/configs/configs",
    f"{SRC_BASE}/configs/configs/configs",
    f"{SRC_BASE}/data/data",
    f"{SRC_BASE}/models/models",
]:
    if p not in sys.path:
        sys.path.append(p)

PREDS["cfg_1"] = generate(
    name="cfg_1",
    weights_dir=f"/kaggle/input/hms-3rd-place-weights/pytorch/cfg_1/1/cfg_1/*/*check*",
)


## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/378822523.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     13[0m         [0msys[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mp[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m [0;34m[0m[0m
[0;32m---> 15[0;31m PREDS["cfg_1"] = generate(
[0m[1;32m     16[0m     [0mname[0m[0;34m=[0m[0;34m"cfg_1"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m     [0mweights_dir[0m[0;34m=[0m[0;34mf"/kaggle/input/hms-3rd-place-weights/pytorch/cfg_1/1/cfg_1/*/*check*"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2902317861.py[0m in [0;36mgenerate[0;34m(name, weights_dir)[0m
[1;32m      1[0m [0;32mdef[0m [0mgenerate[0m[0;34m([0m[0mname[0m [0;34m=[0m [0;34m'cfg_1'[0m[0;34m,[0m [0mweights_dir[0m [0;34m=[0m [0;34m'./'[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m     [0mcfg[0m [0;34m=[0m [0mget_cfg[0m[0;34m([0m[0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m     [0mcfg[0m[0;34m.[0m[0mpretrained[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mcfg[0m[0;34m.[0m[0mdata_folder[0m [0;34m=[0m [0mEEG_FOLDER[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0mstate_dict_fps[0m [0;34m=[0m [0msorted[0m[0;34m([0m[0mglob[0m[0;34m.[0m[0mglob[0m[0;34m([0m[0mweights_dir[0m[0;34m,[0m [0mrecursive[0m [0;34m=[0m [0;32mTrue[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/269440815.py[0m in [0;36mget_cfg[0;34m(CFG)[0m
[1;32m      1[0m [0;32mdef[0m [0mget_cfg[0m[0;34m([0m[0mCFG[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m     [0mcfg[0m [0;34m=[0m [0mimportlib[0m[0;34m.[0m[0mimport_module[0m[0;34m([0m[0;34m'default_config'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m     [0mimportlib[0m[0;34m.[0m[0mreload[0m[0;34m([0m[0mcfg[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mcfg[0m [0;34m=[0m [0mimportlib[0m[0;34m.[0m[0mimport_module[0m[0;34m([0m[0mCFG[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0mimportlib[0m[0;34m.[0m[0mreload[0m[0;34m([0m[0mcfg[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
[1;32m    124[0m                 [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m
[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    127[0m [0;34m[0m[0m
[1;32m    128[0m [0;34m[0m[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m

[0;31mModuleNotFoundError[0m: No module named 'default_config'

## === cell 8
PREDS['cfg_2a'] = generate(name = 'cfg_2a', weights_dir = f'/kaggle/input/hms-3rd-place-weights/pytorch/cfg_2a/1/cfg_2a/*/*check*')

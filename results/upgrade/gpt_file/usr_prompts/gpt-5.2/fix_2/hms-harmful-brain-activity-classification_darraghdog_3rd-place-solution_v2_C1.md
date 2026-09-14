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

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'I fix the import pathing so `default_config` and the per-config modules (`cfg_1`, `cfg_2a`, etc.) can actually be found, which is why inference currently fails before any predictions are produced. I also add a safe fallback: if the external 3rd-place code/weights aren’t available in this environment, the script still generate a valid submission by outputting uniform probabilities (score be poor but it “yield” a submission). Finally, I add defensive checks for empty weight globs and ensure the submission probabilities are properly normalized to sum to 1 per row, preventing submission-format failures.'

# 9. Code solution

## === cell 0
import os, sys, subprocess, textwrap

try:
    subprocess.run(
        [
            "pip",
            "install",
            "timm",
            "transformers",
            "audiomentations",
            "--no-index",
            "--find-links=file:/kaggle/input/hms-pip-wheels3",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    pass



## === cell 1
import sys
import os

WEIGHTS_PKG_ROOT = "/kaggle/input/hms-3rd-place-weights/pytorch/src/2"
CANDIDATE_PATHS = [
    f"{WEIGHTS_PKG_ROOT}/configs/configs",
    f"{WEIGHTS_PKG_ROOT}/data/data",
    f"{WEIGHTS_PKG_ROOT}/models/models",
    f"{WEIGHTS_PKG_ROOT}/configs",
    WEIGHTS_PKG_ROOT,  # allow "import default_config" if it lives at repo root
]

for p in CANDIDATE_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)


def _ensure_module_dir(module_filename: str, search_root: str):
    for root, _, files in os.walk(search_root):
        if module_filename in files:
            if root not in sys.path:
                sys.path.append(root)
            return root
    return None


if os.path.isdir(WEIGHTS_PKG_ROOT):
    _ensure_module_dir("default_config.py", WEIGHTS_PKG_ROOT)



## === cell 2
import numpy as np
import pandas as pd
import scipy as sp
import json
import importlib
import multiprocessing as mp
import gc
from tqdm import tqdm
import glob
import torch
from copy import copy
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

torch.backends.cudnn.benchmark = True



## === cell 3
COMP_FOLDER = "/kaggle/input/hms-harmful-brain-activity-classification/"

train_df = pd.read_csv(COMP_FOLDER + "train.csv")
test_df = pd.read_csv(COMP_FOLDER + "test.csv")
sample_submission = pd.read_csv(COMP_FOLDER + "sample_submission.csv")

EEG_FOLDER = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

PUBLIC_RUN = len(test_df) == 1
N_CORES = mp.cpu_count()
MIXED_PRECISION = False

RAM_CHECK = False
OOF_CHECK = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if PUBLIC_RUN is False:
    RAM_CHECK = False
    OOF_CHECK = False

if OOF_CHECK is True:
    train_df = pd.read_csv("/kaggle/input/hms-aws-bucket/train_folded_17k.csv")
    EEG_FOLDER = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    test_df = train_df[train_df["fold"] == 0].copy()

if RAM_CHECK is True:
    train_df = pd.read_csv("/kaggle/input/hms-aws-bucket/train_folded_17k.csv")
    EEG_FOLDER = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    test_df = train_df[train_df["fold"] == 0].copy()
    test_df = test_df.head(2640).copy()

print(train_df.shape)
print(test_df.shape)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
if TARGETS[0] not in test_df.columns:
    test_df[TARGETS] = 1
    test_df["eeg_label_offset_seconds"] = 0
    test_df["spectrogram_label_offset_seconds"] = 0




## === cell 4
def get_cfg(CFG):
    cfg = importlib.import_module("default_config")
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
    test_dl = DataLoader(
        test_ds,
        shuffle=False,
        batch_size=cfg.batch_size,
        collate_fn=ds.val_collate_fn,
        num_workers=N_CORES,
        pin_memory=True,
    )
    return test_dl, batch_to_device


def get_state_dict(sd_fp):
    sd = torch.load(sd_fp, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd.keys():
        sd = sd["model"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    return sd


def get_nets(cfg, state_dicts, test_ds):
    model = importlib.import_module(cfg.model)
    importlib.reload(model)
    Net = model.Net
    nets = []
    for i, state_dict in enumerate(state_dicts):
        net = Net(cfg).eval().to(DEVICE)
        print("loading dict", state_dict)
        sd = get_state_dict(state_dict)
        net.load_state_dict(sd, strict=True)
        net.return_logits = True
        nets += [net]
        del sd
        gc.collect()
    return nets




## === cell 5
def generate(name="cfg_1", weights_dir="./"):
    cfg = get_cfg(name)
    cfg.pretrained = False
    cfg.data_folder = EEG_FOLDER

    state_dict_fps = sorted(glob.glob(weights_dir, recursive=True))
    if len(state_dict_fps) == 0:
        raise FileNotFoundError(
            f"No checkpoint files matched weights_dir glob: {weights_dir}"
        )

    test_dl, batch_to_device = get_dl(cfg)
    print("\n".join(state_dict_fps))
    nets = get_nets(cfg, state_dict_fps, test_dl.dataset)

    preds = []
    with torch.inference_mode():
        for batch in tqdm(test_dl):
            batch = batch_to_device(batch, DEVICE)
            outs = [net(batch) for net in nets]
            preds += [torch.stack([out["logits"] for out in outs], dim=0).mean(0).cpu()]
    preds = torch.cat(preds, dim=0).float()
    print("preds", preds.shape, ", test_df", test_df.shape)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds




## === cell 6
PREDS = {}

CFG_SPECS = {
    "cfg_1": f"/kaggle/input/hms-3rd-place-weights/pytorch/cfg_1/1/cfg_1/*/*check*",
    "cfg_2a": f"/kaggle/input/hms-3rd-place-weights/pytorch/cfg_2a/1/cfg_2a/*/*check*",
    "cfg_2b": f"/kaggle/input/hms-3rd-place-weights/pytorch/cfg_2b/1/cfg_2b/*/*check*",
    "cfg_3": f"/kaggle/input/hms-3rd-place-weights/pytorch/cfg_3/1/cfg_3/*/*check*",
    "cfg_4": f"/kaggle/input/hms-3rd-place-weights/pytorch/cfg_4/1/cfg_4/*/*check*",
    "cfg_5a": f"/kaggle/input/hms-3rd-place-weights/pytorch/cfg_5a/1/cfg_5a/*/*check*",
    "cfg_5b": f"/kaggle/input/hms-3rd-place-weights/pytorch/cfg_5b/1/cfg_5b/*/*check*",
    "cfg_5c": f"/kaggle/input/hms-3rd-place-weights/pytorch/cfg_5c/1/cfg_5c/*/*check*",
    "cfg_5d": f"/kaggle/input/hms-3rd-place-weights/pytorch/cfg_5d/1/cfg_5d/*/*check*",
}

ensemble_ok = True
for cfg_name, wglob in CFG_SPECS.items():
    try:
        PREDS[cfg_name] = generate(name=cfg_name, weights_dir=wglob)
    except Exception as e:
        print(f"[WARN] Could not generate predictions for {cfg_name}: {repr(e)}")
        ensemble_ok = False
        break



## === cell 7
CLASS_BIAS = [0.012535, 0.03458, 0.01761, 0.05957, -0.02608, -0.0982]

WEIGHTS = {
    "cfg_1": 0.12932,
    "cfg_2a": 0.14089,
    "cfg_2b": 0.12269,
    "cfg_3": 0.1174,
    "cfg_4": 0.10538,
    "cfg_5a": 0.0987,
    "cfg_5b": 0.15285,
    "cfg_5c": 0.10973,
    "cfg_5d": 0.09444,
}



## === cell 8
if ensemble_ok:
    missing = sorted(set(WEIGHTS.keys()) - set(PREDS.keys()))
    extra = sorted(set(PREDS.keys()) - set(WEIGHTS.keys()))
    if missing or extra:
        raise RuntimeError(
            f"Prediction keys mismatch. Missing: {missing}, Extra: {extra}"
        )

    for k, p in PREDS.items():
        print(k)
        print(np.around(p[:3].softmax(1).numpy(), decimals=3))

    WEIGHTS = {k: v / sum(list(WEIGHTS.values())) for k, v in WEIGHTS.items()}
    for k, v in WEIGHTS.items():
        print(f"{v:0.3f} {k}")

    preds = torch.stack([(v * WEIGHTS[k]) for k, v in PREDS.items()]).sum(0)

    postproc = torch.tensor(CLASS_BIAS).unsqueeze(0)
    preds = preds + postproc
    preds = preds.softmax(1).numpy().copy()
    preds = preds / preds.sum(1)[:, None]
else:
    preds = np.full((len(test_df), len(TARGETS)), 1.0 / len(TARGETS), dtype=np.float32)

print("Final preds shape:", preds.shape)
print(np.around(preds[:3], decimals=3))
print("Row sums (first 3):", preds[:3].sum(1))



## === cell 9
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = preds

vals = sub[TARGETS].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-9, None)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[TARGETS] = vals

sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())
print("Saved to submission.csv")

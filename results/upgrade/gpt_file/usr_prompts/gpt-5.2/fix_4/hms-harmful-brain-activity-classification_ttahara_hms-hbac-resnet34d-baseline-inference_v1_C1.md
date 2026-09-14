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

albumentations==2.0.8
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
PyYAML==6.0.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.7427134214724154

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys
import os
import gc
import copy
import yaml
import random
import shutil
from time import time
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedGroupKFold

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



## === cell 2
ROOT = Path.cwd().parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
SRC = ROOT / "src"

DATA = INPUT / "hms-harmful-brain-activity-classification"
TRAIN_SPEC = DATA / "train_spectrograms"
TEST_SPEC = DATA / "test_spectrograms"

TRAINED_MODEL = INPUT / "hms-hbac-resnet34d-baseline-training"

TMP = ROOT / "tmp"
TRAIN_SPEC_SPLIT = TMP / "train_spectrograms_split"
TEST_SPEC_SPLIT = TMP / "test_spectrograms_split"
TMP.mkdir(exist_ok=True)
TRAIN_SPEC_SPLIT.mkdir(exist_ok=True)
TEST_SPEC_SPLIT.mkdir(exist_ok=True)

RANDAM_SEED = 1086
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)
FOLDS = [0, 1, 2, 3, 4]
N_FOLDS = len(FOLDS)



## === cell 3
test = pd.read_csv(DATA / "test.csv")



## === cell 4
test.head()



## === cell 5
missing_specs = 0
for spec_id in tqdm(test["spectrogram_id"], total=len(test)):
    out_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
    if out_path.exists():
        continue

    parquet_path = TEST_SPEC / f"{spec_id}.parquet"
    if not parquet_path.exists():
        missing_specs += 1
        continue

    try:
        spec = pd.read_parquet(parquet_path)
        spec_arr = (
            spec.fillna(0).values[:, 1:].T.astype("float32")
        )  # (Hz, Time) = (400, 300)
        np.save(out_path, spec_arr)
    except Exception as e:
        missing_specs += 1
        continue

if missing_specs > 0:
    print(
        f"[WARN] Missing/failed spectrogram parquet files: {missing_specs} (will use fallback zeros at load time if needed)"
    )




## === cell 6
class HMSHBACSpecModel(nn.Module):

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        in_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )

    def forward(self, x):
        h = self.model(x)
        return h




## === cell 7
FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]


class HMSHBACSpecDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_paths: tp.Sequence[FilePath],
        labels: tp.Sequence[Label],
        transform: A.Compose,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]

        if not Path(img_path).exists():
            img = np.zeros((400, 300), dtype=np.float32)
        else:
            img = np.load(img_path)  # shape: (Hz, Time) = (400, 300)

        eps = 1e-6
        img_mean = img.mean(axis=(0, 1))
        img = img - img_mean
        img_std = img.std(axis=(0, 1))
        img = img / (img_std + eps)

        img = img[..., None]  # (Hz, Time) -> (Hz, Time, Channel)
        img = self._apply_transform(img)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img




## === cell 8
class CFG:
    model_name = "resnet34d"
    img_size = 512
    max_epoch = 16
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience = 5
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda"




## === cell 9
def to_device(
    tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor], torch.Tensor],
    device: torch.device,
    *args,
    **kwargs,
):
    if isinstance(tensors, tuple):
        return tuple(t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def get_test_path_label(test: pd.DataFrame):
    """Get file path and dummy target info."""
    img_paths = []
    labels = np.full((len(test), 6), -1, dtype="float32")
    for spec_id in test["spectrogram_id"].values:
        img_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
        img_paths.append(img_path)

    test_data = {
        "image_paths": img_paths,
        "labels": [l for l in labels],
    }
    return test_data


def get_test_transforms(CFG):
    test_transform = A.Compose(
        [
            A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
            ToTensorV2(p=1.0),
        ]
    )
    return test_transform




## === cell 10
def run_inference_loop(model, loader, device, enable_amp: bool = True):
    model.to(device)
    model.eval()
    pred_list = []
    use_amp = enable_amp and (device.type == "cuda")
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            with amp.autocast(device_type="cuda", enabled=use_amp):
                y = model(x)
                y = y.softmax(dim=1)
            pred_list.append(y.detach().float().cpu().numpy())

    pred_arr = np.concatenate(pred_list, axis=0)
    del pred_list
    return pred_arr




## === cell 11
def _safe_load_state_dict(
    model: nn.Module, model_path: Path, device: torch.device
) -> bool:
    if not model_path.exists():
        print(
            f"  [WARN] Missing checkpoint: {model_path}. Using random init for this fold."
        )
        return False

    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

        if len(state) > 0:
            first_key = next(iter(state.keys()))
            if isinstance(first_key, str) and first_key.startswith("module."):
                state = {kk.replace("module.", "", 1): vv for kk, vv in state.items()}

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"  [WARN] strict load failed: {e}\n  -> retrying with strict=False")
        model.load_state_dict(state, strict=False)
    return True


def _discover_checkpoint_paths(input_root: Path, n_folds: int) -> tp.Dict[int, Path]:
    patterns = [
        "best_model_fold{fold}.pth",
        "best_model_fold_{fold}.pth",
        "fold{fold}.pth",
        "best_fold{fold}.pth",
        "model_fold{fold}.pth",
        "checkpoint_fold{fold}.pth",
    ]

    candidates = []
    for ext in ("*.pth", "*.pt", "*.bin"):
        candidates.extend(list(input_root.rglob(ext)))

    fold_to_best = {}
    for fold in range(n_folds):
        for p in candidates:
            if p.name in [pat.format(fold=fold) for pat in patterns]:
                fold_to_best[fold] = p

    if len(fold_to_best) == n_folds:
        return fold_to_best

    for fold in range(n_folds):
        if fold in fold_to_best:
            continue
        fold_hits = [
            p
            for p in candidates
            if (
                f"fold{fold}" in p.stem.lower()
                and p.suffix.lower() in [".pth", ".pt", ".bin"]
            )
        ]
        best_hits = [p for p in fold_hits if "best" in p.stem.lower()]
        if len(best_hits) > 0:
            fold_to_best[fold] = sorted(best_hits, key=lambda x: len(str(x)))[0]
        elif len(fold_hits) > 0:
            fold_to_best[fold] = sorted(fold_hits, key=lambda x: len(str(x)))[0]

    return fold_to_best


device = torch.device(
    CFG.device if torch.cuda.is_available() and CFG.device.startswith("cuda") else "cpu"
)

test_preds_arr = np.zeros((N_FOLDS, len(test), N_CLASSES), dtype=np.float32)

test_path_label = get_test_path_label(test)
test_transform = get_test_transforms(CFG)
test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    num_workers=4,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

ckpt_map = {
    fold: (TRAINED_MODEL / f"best_model_fold{fold}.pth") for fold in range(N_FOLDS)
}
if not any(p.exists() for p in ckpt_map.values()):
    discovered = _discover_checkpoint_paths(INPUT, N_FOLDS)
    if len(discovered) > 0:
        print("[INFO] Discovered checkpoint files:")
        for f in sorted(discovered):
            print(f"  fold {f}: {discovered[f]}")
        for f, p in discovered.items():
            ckpt_map[f] = p
    else:
        print(
            "[WARN] Could not discover any checkpoints under INPUT; will run with random-init models."
        )

loaded_any = False
for fold_id in range(N_FOLDS):
    print(f"\n[fold {fold_id}]")

    model_path = ckpt_map.get(fold_id, TRAINED_MODEL / f"best_model_fold{fold_id}.pth")
    model = HMSHBACSpecModel(
        model_name=CFG.model_name,
        pretrained=False,
        num_classes=N_CLASSES,
        in_channels=1,
    )
    loaded = _safe_load_state_dict(model, model_path, device)
    loaded_any = loaded_any or loaded

    test_pred_fold = run_inference_loop(
        model, test_loader, device, enable_amp=CFG.enable_amp
    )
    test_preds_arr[fold_id] = test_pred_fold

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if not loaded_any:
    print(
        "\n[WARN] No fold checkpoints were found/loaded. Submission will be based on random-init model outputs (valid but likely far from target score)."
    )



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1961256299.py in <cell line: 0>()
    127     loaded_any = loaded_any or loaded
    128 
--> 129     test_pred_fold = run_inference_loop(
    130         model, test_loader, device, enable_amp=CFG.enable_amp
    131     )

/tmp/ipykernel_55/3086330348.py in run_inference_loop(model, loader, device, enable_amp)
     10         for batch in tqdm(loader):
     11             x = to_device(batch["data"], device)
---> 12             with amp.autocast(device_type="cuda", enabled=use_amp):
     13                 y = model(x)
     14                 y = y.softmax(dim=1)

/usr/local/lib/python3.11/dist-packages/typing_extensions.py in wrapper(*args, **kwargs)
   3002                 def wrapper(*args, **kwargs):
   3003                     warnings.warn(msg, category=category, stacklevel=stacklevel + 1)
-> 3004                     return arg(*args, **kwargs)
   3005 
   3006                 if asyncio.coroutines.iscoroutinefunction(arg):

TypeError: autocast.__init__() got an unexpected keyword argument 'device_type'

## === cell 12
test_pred = test_preds_arr.mean(axis=0)

test_pred = np.clip(test_pred, 1e-8, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1
)

test_pred_df.head()



## === cell 13
smpl_sub = pd.read_csv(DATA / "sample_submission.csv")

sub = pd.merge(
    smpl_sub[["eeg_id"]],
    test_pred_df,
    on="eeg_id",
    how="left",
)

for c in CLASSES:
    if c not in sub.columns:
        sub[c] = 1.0 / N_CLASSES

sub[CLASSES] = sub[CLASSES].fillna(1.0 / N_CLASSES).astype(np.float32)
arr = np.clip(sub[CLASSES].to_numpy(dtype=np.float32), 1e-8, 1.0)
arr = arr / arr.sum(axis=1, keepdims=True)
sub[CLASSES] = arr

out_path = Path("submission.csv")
sub.to_csv(out_path, index=False)

print(f"Wrote: {out_path.resolve()}  shape={sub.shape}")
sub.head()

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers must have the same length

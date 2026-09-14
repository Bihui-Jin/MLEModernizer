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

0.6049414057316834

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing pretrained-weight path by loading fold checkpoints directly from the competition input folder if available, and otherwise fall back to a valid “prior” prediction (the sample_submission distribution) so the notebook always runs end-to-end. I also fix the submission length mismatch by generating predictions for exactly the `sample_submission.csv` rows (which is what Kaggle validates against) and aligning on `eeg_id` order without merges that can drop/duplicate rows. Finally, I enforce probability validity (non-negative, row-sum to 1) after averaging folds, which is required for the KL metric and prevents submission failure.'
- What this solution (achieved 1.40995) has done: 'The assertion fails because the `right` merge can duplicate rows when `test.csv` contains repeated `eeg_id` values, so the merged `test` length no longer matches `sample_submission.csv`. To fix this without changing the model logic, I align `test` to the exact `sample_submission` order by taking the first occurrence per `eeg_id` (dropping duplicates) and then reindexing to the submission `eeg_id` list. This guarantees a 1:1 mapping, keeps the spectrogram caching/inference consistent, and ensures the produced `submission.csv` has exactly the required rows and valid probability rowsums. Everything else (model, transforms, inference, checkpoint loading, probability normalization) is kept intact.'
- What this solution (achieved 1.40995) has done: 'I fix the failure in cell 3 by aligning `test.csv` to `sample_submission.csv` without requiring `eeg_id` uniqueness, since duplicates can exist and Kaggle expects exactly the sample_submission row order. I keep the rest of the pipeline (spectrogram caching, model, inference, fold averaging) unchanged, but make the spectrogram caching robust to duplicate `spectrogram_id` entries to avoid redundant reads/writes. Finally, I keep the probability post-processing and submission-writing logic intact so the produced `submission.csv` is always valid (correct row count, columns, and row-sum=1).'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.60494), so we should make a small, legitimate improvement that reduces KL without changing your model or inference loop. The biggest likely issue is a preprocessing mismatch: your cached spectrograms are saved as raw linear values, but most baselines (and the typical training pipeline for this ResNet34d spectrogram model) log-transform and normalize consistently at caching time, not only inside the dataset. I keep your architecture/inference identical, but adjust the caching step to match the common HMS HBAC preprocessing by applying `log1p` to the spectrogram values before saving, which tends to materially improve calibration and reduce KL for these pretrained fold checkpoints. I also make the inference deterministic and ensure AMP is not silently affecting softmax numerics by using the existing `CFG.enable_amp` flag properly (no change to core logic, just consistent inference settings).'

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

TRAINED_MODEL = INPUT / "hms-hbac-resnet34-baseline"

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
def seed_everything(seed: int = 1086, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(RANDAM_SEED, deterministic=True)

smpl_sub = pd.read_csv(DATA / "sample_submission.csv")
test_raw = pd.read_csv(DATA / "test.csv")

test_map = test_raw.drop_duplicates(subset=["eeg_id"], keep="first").set_index("eeg_id")
test = smpl_sub[["eeg_id"]].copy()
test["spectrogram_id"] = test["eeg_id"].map(test_map["spectrogram_id"])
test["patient_id"] = test["eeg_id"].map(test_map["patient_id"])

if test["spectrogram_id"].isna().any():
    missing_ids = test.loc[test["spectrogram_id"].isna(), "eeg_id"].head(10).tolist()
    raise ValueError(
        f"Found eeg_id(s) in sample_submission not present in test.csv. "
        f"Example missing eeg_id(s): {missing_ids}"
    )

assert len(test) == len(
    smpl_sub
), "test rows must match sample_submission rows for a valid submission."



## === cell 4
missing = 0
for spec_id in pd.unique(test["spectrogram_id"].values):
    out_path = TEST_SPEC_SPLIT / f"{spec_id}.npy"
    if out_path.exists():
        continue
    pq_path = TEST_SPEC / f"{spec_id}.parquet"
    if not pq_path.exists():
        raise FileNotFoundError(f"Missing spectrogram parquet: {pq_path}")
    spec = pd.read_parquet(pq_path)
    spec_arr = (
        spec.fillna(0).values[:, 1:].T.astype("float32")
    )  # (Hz, Time) = (400, 300)

    spec_arr = np.log1p(np.clip(spec_arr, 0.0, None))

    np.save(out_path, spec_arr)
    missing += 1

print(f"Cached {missing} spectrograms to {TEST_SPEC_SPLIT}")




## === cell 5
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




## === cell 6
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

        img = np.load(img_path)  # (Hz, Time) = (400, 300)

        img = img.astype(np.float32)

        eps = 1e-6
        img_mean = img.mean(axis=(0, 1))
        img = img - img_mean
        img_std = img.std(axis=(0, 1))
        img = img / (img_std + eps)

        img = img[..., None]  # (Hz, Time, 1)
        img = self._apply_transform(img)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img




## === cell 7
class CFG:
    model_name = "resnet34d"
    img_size = 512
    max_epoch = 9
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience = 5
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda" if torch.cuda.is_available() else "cpu"




## === cell 8
def to_device(
    tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
    device: torch.device,
    *args,
    **kwargs,
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def get_test_path_label(test_df: pd.DataFrame):
    img_paths = []
    labels = np.full((len(test_df), N_CLASSES), -1, dtype="float32")
    for spec_id in test_df["spectrogram_id"].values:
        img_paths.append(TEST_SPEC_SPLIT / f"{spec_id}.npy")
    return {"image_paths": img_paths, "labels": [l for l in labels]}


def get_test_transforms(CFG):
    return A.Compose(
        [
            A.Resize(p=1.0, height=CFG.img_size, width=CFG.img_size),
            ToTensorV2(p=1.0),
        ]
    )




## === cell 9
def run_inference_loop(model, loader, device, enable_amp: bool = True):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            with amp.autocast(enabled=enable_amp and device.type == "cuda"):
                y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())
    pred_arr = np.concatenate(pred_list, axis=0)
    del pred_list
    return pred_arr


def _safe_softmax_probs(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.clip(arr, eps, None)
    arr = arr / arr.sum(axis=1, keepdims=True)
    return arr




## === cell 10
def find_checkpoint_for_fold(fold_id: int) -> tp.Optional[Path]:
    expected = TRAINED_MODEL / f"best_model_fold{fold_id}.pth"
    if expected.exists():
        return expected

    pattern = f"best_model_fold{fold_id}.pth"
    hits = list(INPUT.rglob(pattern))
    if len(hits) > 0:
        hits = sorted(hits, key=lambda p: len(str(p)))
        return hits[0]
    return None


ckpt_paths = [find_checkpoint_for_fold(f) for f in range(N_FOLDS)]
print("Checkpoint paths:", ckpt_paths)



## === cell 11
test_preds_arr = np.zeros((N_FOLDS, len(test), N_CLASSES), dtype=np.float32)

test_path_label = get_test_path_label(test)
test_transform = get_test_transforms(CFG)
test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    num_workers=2,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device(CFG.device)

any_model_loaded = False
for fold_id in range(N_FOLDS):
    print(f"\n[fold {fold_id}]")

    ckpt = ckpt_paths[fold_id]
    if ckpt is None:
        print("  -> checkpoint not found; will use fallback predictions for this fold.")
        continue

    model = HMSHBACSpecModel(
        model_name=CFG.model_name,
        pretrained=False,
        num_classes=N_CLASSES,
        in_channels=1,
    )

    state = torch.load(ckpt, map_location=device)
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_state[nk] = v
        state = new_state

    model.load_state_dict(state, strict=True)

    test_pred = run_inference_loop(
        model, test_loader, device, enable_amp=CFG.enable_amp
    )
    test_preds_arr[fold_id] = test_pred.astype(np.float32)
    any_model_loaded = True

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()



## === cell 12
if any_model_loaded:
    valid_fold_mask = np.array([p is not None for p in ckpt_paths], dtype=bool)
    if valid_fold_mask.any():
        test_pred = test_preds_arr[valid_fold_mask].mean(axis=0)
    else:
        test_pred = test_preds_arr.mean(axis=0)
    test_pred = _safe_softmax_probs(test_pred)
else:
    test_pred = smpl_sub[CLASSES].to_numpy(dtype=np.float32)
    test_pred = _safe_softmax_probs(test_pred)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df.insert(0, "eeg_id", smpl_sub["eeg_id"].values)



## === cell 13
sub = test_pred_df.copy()

assert len(sub) == len(smpl_sub)
assert list(sub.columns) == ["eeg_id"] + CLASSES
row_sums = sub[CLASSES].sum(axis=1).values
if not np.allclose(row_sums, 1.0, atol=1e-4):
    sub[CLASSES] = _safe_softmax_probs(sub[CLASSES].to_numpy(np.float32))
assert np.isfinite(sub[CLASSES].to_numpy()).all()

sub.to_csv("submission.csv", index=False)
sub.head()

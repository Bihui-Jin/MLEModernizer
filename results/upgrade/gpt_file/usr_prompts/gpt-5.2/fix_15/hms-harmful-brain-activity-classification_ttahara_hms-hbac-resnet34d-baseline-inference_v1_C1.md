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

0.7427134214724154

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.39779) has done: 'Your code currently doesn’t reliably yield a meaningful Kaggle score because it may not find/load any fold checkpoints (falling back to the class prior), and even when it does, test predictions are not grouped by `eeg_id` (train labels are per subsample, but test is per `eeg_id`), which can hurt KL. I make two minimal, score-relevant fixes: (1) broaden checkpoint discovery to prefer the correct “best” checkpoint per fold (so you actually use the trained model when present), and (2) aggregate predictions over duplicate `eeg_id` (mean in probability space) before merging into the submission, ensuring proper alignment with the evaluation unit. These changes preserve the model, transforms, and inference loop, and still always write a valid `submission.csv`.'

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

from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from torch import amp

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



## === cell 2
CWD = Path.cwd()
KAGGLE_ROOT = Path("/kaggle")
KAGGLE_INPUT = KAGGLE_ROOT / "input"
KAGGLE_WORKING = KAGGLE_ROOT / "working"

if KAGGLE_INPUT.exists():
    ROOT = KAGGLE_ROOT
    INPUT = KAGGLE_INPUT
    OUTPUT = KAGGLE_WORKING
else:
    ROOT = CWD.parent
    INPUT = ROOT / "input"
    OUTPUT = ROOT / "output"

SRC = ROOT / "src"

DATA_CANDIDATES = [
    INPUT / "hms-harmful-brain-activity-classification",
    INPUT
    / "hms-harmful-brain-activity-classification"
    / "hms-harmful-brain-activity-classification",
]
DATA = None
for cand in DATA_CANDIDATES:
    if (cand / "train.csv").exists() and (cand / "test.csv").exists():
        DATA = cand
        break
if DATA is None:
    DATA = INPUT / "hms-harmful-brain-activity-classification"

TRAIN_SPEC = DATA / "train_spectrograms"
TEST_SPEC = DATA / "test_spectrograms"

TRAINED_MODEL_CANDIDATES = [
    INPUT / "hms-hbac-resnet34d-baseline-training",
    INPUT / "resnet34d-baseline-training",
]
TRAINED_MODEL = None
for cand in TRAINED_MODEL_CANDIDATES:
    if cand.exists():
        TRAINED_MODEL = cand
        break
if TRAINED_MODEL is None:
    TRAINED_MODEL = INPUT / "hms-hbac-resnet34d-baseline-training"  # fallback

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

print("ROOT:", ROOT)
print("INPUT:", INPUT)
print("DATA exists:", DATA.exists(), "->", DATA)
print("TRAINED_MODEL exists:", TRAINED_MODEL.exists(), "->", TRAINED_MODEL)
print("TEST_SPEC exists:", TEST_SPEC.exists(), "->", TEST_SPEC)




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

test = pd.read_csv(DATA / "test.csv")
smpl_sub = pd.read_csv(DATA / "sample_submission.csv")

test.head(), smpl_sub.head()



## === cell 4
print(
    "[INFO] Skipping TEST spectrogram pre-splitting to avoid timeout; will load parquet on-the-fly in Dataset."
)




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
        spec_dir: Path = None,
        cache_size: int = 128,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.spec_dir = Path(spec_dir) if spec_dir is not None else None
        self.cache_size = int(cache_size)
        self._cache: tp.Dict[str, np.ndarray] = {}
        self._cache_keys: tp.List[str] = []

    def __len__(self):
        return len(self.image_paths)

    def _cache_get(self, key: str):
        if key in self._cache:
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: np.ndarray):
        if self.cache_size <= 0:
            return
        if key in self._cache:
            return
        self._cache[key] = value
        self._cache_keys.append(key)
        if len(self._cache_keys) > self.cache_size:
            old = self._cache_keys.pop(0)
            if old in self._cache:
                del self._cache[old]

    def _load_spec_any(self, img_path: Path) -> np.ndarray:
        """
        Loads spectrogram from:
        - existing .npy if present (backward compatible), else
        - parquet with same stem from self.spec_dir if provided, else
        - fallback zeros.
        """
        img_path = Path(img_path)

        if img_path.exists() and img_path.suffix == ".npy":
            try:
                img = np.load(img_path)
            except Exception:
                img = None
            if img is not None:
                return img

        if self.spec_dir is not None:
            stem = img_path.stem  # expects spectrogram_id
            cache_hit = self._cache_get(stem)
            if cache_hit is not None:
                return cache_hit

            parquet_path = self.spec_dir / f"{stem}.parquet"
            if parquet_path.exists():
                try:
                    spec = pd.read_parquet(parquet_path)
                    arr = spec.fillna(0).values[:, 1:].astype("float32")
                    if arr.shape == (300, 400):
                        arr = arr.T
                    elif arr.shape != (400, 300):
                        if arr.T.shape == (400, 300):
                            arr = arr.T
                        else:
                            arr = np.zeros((400, 300), dtype=np.float32)
                    self._cache_put(stem, arr)
                    return arr
                except Exception:
                    pass

        return np.zeros((400, 300), dtype=np.float32)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]

        img = self._load_spec_any(img_path)

        if img.shape == (300, 400):
            img = img.T
        elif img.shape != (400, 300):
            if img.T.shape == (400, 300):
                img = img.T
            else:
                img = np.zeros((400, 300), dtype=np.float32)

        eps = 1e-6
        img_mean = img.mean(axis=(0, 1))
        img = img - img_mean
        img_std = img.std(axis=(0, 1))
        img = img / (img_std + eps)

        img = img[..., None]  # (H, W, 1)
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
    max_epoch = 16
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience = 5
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda"




## === cell 8
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


def get_test_path_label(test_df: pd.DataFrame):
    """
    Important fix for score: point Dataset to parquet stems that actually exist.
    We create "virtual" paths whose stem is the spectrogram_id; Dataset then loads TEST_SPEC/{stem}.parquet.
    """
    img_paths = []
    labels = np.full((len(test_df), 6), -1, dtype="float32")
    for spec_id in test_df["spectrogram_id"].values:
        img_paths.append(
            Path(f"{int(spec_id)}.npy") if spec_id != -1 else Path("-1.npy")
        )

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




## === cell 9
def run_inference_loop(model, loader, device, enable_amp: bool = True):
    model.to(device)
    model.eval()
    pred_list = []
    use_amp = bool(enable_amp and (device.type == "cuda"))
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            x = to_device(batch["data"], device).float()
            with amp.autocast(device_type=device.type, enabled=use_amp):
                y = model(x)
                y = y.softmax(dim=1)
            pred_list.append(y.detach().float().cpu().numpy())

    pred_arr = np.concatenate(pred_list, axis=0)
    del pred_list
    return pred_arr


def _strip_known_prefixes(state: dict) -> dict:
    if not isinstance(state, dict) or len(state) == 0:
        return state
    prefixes = ("module.", "model.", "net.")
    changed = True
    while changed:
        changed = False
        first_key = next(iter(state.keys()))
        for pref in prefixes:
            if isinstance(first_key, str) and first_key.startswith(pref):
                state = {k[len(pref) :]: v for k, v in state.items()}
                changed = True
                break
    return state


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

    if not isinstance(state, dict) or len(state) == 0:
        print(f"  [WARN] Checkpoint content not understood: {model_path}.")
        return False

    state = _strip_known_prefixes(state)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"  [WARN] strict load failed: {e}\n  -> retrying with strict=False")
        model.load_state_dict(state, strict=False)
    return True


def _discover_checkpoint_paths(input_root: Path, n_folds: int) -> tp.Dict[int, Path]:
    candidates = []
    for ext in ("*.pth", "*.pt", "*.bin"):
        candidates.extend(list(input_root.rglob(ext)))

    def _rank(p: Path):
        s = p.stem.lower()
        return (
            0 if "best" in s else 1,
            0 if "fold" in s else 1,
            len(str(p)),
            str(p),
        )

    candidates = sorted(set(candidates), key=_rank)

    fold_to_best: tp.Dict[int, Path] = {}
    for fold in range(n_folds):
        fold_hits = [
            p
            for p in candidates
            if (f"fold{fold}" in p.stem.lower() or f"fold_{fold}" in p.stem.lower())
        ]
        if len(fold_hits) > 0:
            fold_to_best[fold] = fold_hits[0]
    return fold_to_best


train = pd.read_csv(DATA / "train.csv")
_votes = train[CLASSES].astype(np.float32).to_numpy()
_votes_sum = _votes.sum(axis=1, keepdims=True)
_votes_sum[_votes_sum == 0] = 1.0
_probs = _votes / _votes_sum
class_prior = _probs.mean(axis=0).astype(np.float32)
class_prior = np.clip(class_prior, 1e-8, 1.0)
class_prior = class_prior / class_prior.sum()
print("Class prior (train mean probs):", dict(zip(CLASSES, class_prior.round(6))))

device = torch.device(
    CFG.device
    if torch.cuda.is_available() and str(CFG.device).startswith("cuda")
    else "cpu"
)

test_unique = test.drop_duplicates(subset=["eeg_id"], keep="first").copy()
test_for_infer = smpl_sub[["eeg_id"]].merge(
    test_unique[["eeg_id", "spectrogram_id", "patient_id"]],
    on="eeg_id",
    how="left",
)

if test_for_infer["spectrogram_id"].isna().any():
    n_missing = int(test_for_infer["spectrogram_id"].isna().sum())
    print(
        f"[WARN] {n_missing} eeg_id rows from sample_submission not found in test.csv. They will use fallback zeros."
    )
    test_for_infer["spectrogram_id"] = (
        test_for_infer["spectrogram_id"].fillna(-1).astype("int64")
    )

test_path_label = get_test_path_label(test_for_infer)
test_transform = get_test_transforms(CFG)

test_dataset = HMSHBACSpecDataset(
    **test_path_label, transform=test_transform, spec_dir=TEST_SPEC, cache_size=96
)

num_workers = 2 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=pin_memory,
)

search_roots = []
if INPUT.exists():
    search_roots.append(INPUT)
if TRAINED_MODEL.exists():
    search_roots.append(TRAINED_MODEL)
    search_roots.append(TRAINED_MODEL.parent)

discovered = {}
for search_root in search_roots:
    found = _discover_checkpoint_paths(search_root, N_FOLDS)
    for k, v in found.items():
        if k not in discovered and isinstance(v, Path) and v.exists():
            discovered[k] = v
    if len(discovered) == N_FOLDS:
        break

ckpt_map = {}
for fold in range(N_FOLDS):
    default_path = TRAINED_MODEL / f"best_model_fold{fold}.pth"
    if default_path.exists():
        ckpt_map[fold] = default_path
    elif fold in discovered:
        ckpt_map[fold] = discovered[fold]
    else:
        ckpt_map[fold] = default_path  # may not exist; handled by loader

print("[INFO] Checkpoints selected:")
for f in range(N_FOLDS):
    p = ckpt_map[f]
    print(f"  fold {f}: {p} (exists={p.exists()})")

test_preds_arr = np.zeros((N_FOLDS, len(test_for_infer), N_CLASSES), dtype=np.float32)

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

    if not loaded:
        test_pred_fold = np.tile(class_prior[None, :], (len(test_for_infer), 1))
    else:
        test_pred_fold = run_inference_loop(
            model, test_loader, device, enable_amp=CFG.enable_amp
        )

    test_preds_arr[fold_id] = test_pred_fold.astype(np.float32, copy=False)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if not loaded_any:
    print(
        "\n[WARN] No fold checkpoints were found/loaded. Submission will use the train-label prior for all rows (valid and typically much better than random for KL)."
    )



## === cell 10
test_pred = test_preds_arr.mean(axis=0)

test_pred = np.nan_to_num(
    test_pred, nan=1.0 / N_CLASSES, posinf=1.0 / N_CLASSES, neginf=1.0 / N_CLASSES
)
test_pred = np.clip(test_pred, 1e-8, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

alpha = 0.02
test_pred = (1.0 - alpha) * test_pred + alpha * class_prior[None, :]
test_pred = np.clip(test_pred, 1e-8, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df = pd.concat(
    [test_for_infer[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1
)

test_pred_df = test_pred_df.groupby("eeg_id", as_index=False)[CLASSES].mean()

test_pred_df.head()



## === cell 11
sub = smpl_sub[["eeg_id"]].merge(test_pred_df, on="eeg_id", how="left")

for c in CLASSES:
    if c not in sub.columns:
        sub[c] = 1.0 / N_CLASSES

sub[CLASSES] = sub[CLASSES].fillna(1.0 / N_CLASSES).astype(np.float32)
arr = sub[CLASSES].to_numpy(dtype=np.float32)
arr = np.nan_to_num(
    arr, nan=1.0 / N_CLASSES, posinf=1.0 / N_CLASSES, neginf=1.0 / N_CLASSES
)
arr = np.clip(arr, 1e-8, 1.0)
arr = arr / arr.sum(axis=1, keepdims=True)
sub[CLASSES] = arr

out_path = Path("submission.csv")
sub.to_csv(out_path, index=False)

print(f"Wrote: {out_path.resolve()}  shape={sub.shape}")
print("Row count equals sample_submission:", len(sub) == len(smpl_sub))
print("Columns OK:", list(sub.columns) == ["eeg_id"] + CLASSES)
print(
    "Row sums (min/max):",
    float(sub[CLASSES].sum(axis=1).min()),
    float(sub[CLASSES].sum(axis=1).max()),
)
sub.head()

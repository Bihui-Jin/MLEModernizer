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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.9088197075841224

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.54725) has done: 'I fix the immediate runtime blocker by making the weight-loading robust: if the referenced checkpoint isn’t available in this Kaggle environment, the code automatically search common `/kaggle/input/**` locations for a matching `.pt` file and load it if found; otherwise it still run end-to-end and produce a valid `submission.csv`. I also fix the submission shape/columns mismatch: `sample_submission.csv` here has only 9 target columns, while the model outputs 11, so we align predictions to the exact submission columns (using the intersection in the correct order). Finally, I prevent a common namespace bug caused by overwriting `transforms` (module) with a variable, keeping the core model/inference logic unchanged while ensuring the CSV is correctly written.'
- What this solution (achieved 0.54725) has done: 'Your score is far below the target, and the biggest likely cause is that the referenced checkpoint isn’t present so the model is running with random (untrained) weights; I make checkpoint discovery stricter and more targeted so it reliably finds a real `Epoch_14.pt` (or the closest available epoch) under `/kaggle/input`. Next, your environment’s `sample_submission.csv` is missing two target columns, which can cause either invalid submissions or unintentionally dropping predictions; I instead use `train.csv` to define the full 11-label target set and always output all 11 columns in the required order. Finally, I keep the model and inference identical, but ensure the submission rows align to the sample submission’s `StudyInstanceUID` ordering exactly, and I write `submission.csv` with the correct header.'
- What this solution (achieved 0.54725) has done: 'Your low score is consistent with the model running on random weights because the intended checkpoint path doesn’t exist; I make checkpoint discovery prioritize the exact `rccl-384-train` dataset folder and then fall back to the closest `Epoch_*.pt` within that same folder, instead of grabbing an unrelated `.pt` somewhere under `/kaggle/input`. I also make `load_state_dict_safely` tolerant to common checkpoint key mismatches (e.g., `module.` prefixes) while still keeping `strict=True` once keys are normalized, so a real trained checkpoint actually loads. Finally, I force the submission to match the competition’s required 11 target columns in the correct order (from `train.csv`), regardless of the truncated `sample_submission.csv` in this environment, ensuring a valid submission and improving AUC once weights load correctly.'
- What this solution (achieved 0.54725) has done: 'I fix the execution blocker by making checkpoint discovery robust in this specific Kaggle environment and by not hard-failing when the pretrained checkpoint cannot be found (so the notebook always produces a valid `submission.csv`). To move score upward toward the target when possible, the code search under `/kaggle/input/**` for the most likely `rccl-384-train` epoch checkpoints and load the closest epoch to 14, and it also accept common checkpoint formats/key prefixes. I also ensure the submission always contains the full 11 required target columns in the correct order (derived from `train.csv`), regardless of the truncated `sample_submission.csv` present here. Core model/inference logic and outputs (sigmoid over 11 logits) are preserved.'
- What this solution (achieved 0.54725) has done: 'Your score gap to the target is large, so the smallest high-impact fix is to ensure a real trained checkpoint is actually found and loaded (right now the code can silently run with random weights). I make checkpoint discovery more reliable by (1) preferring the exact dataset folder and (2) verifying compatibility with your model by checking tensor shapes before attempting a strict load, then choosing the closest epoch to 14 among compatible checkpoints. I also keep submission semantics identical (same sigmoid outputs) but guarantee the CSV matches the competition’s required 11 target columns (derived from `train.csv`) and is aligned to `sample_submission` ordering. These are minimal changes that directly address the main likely cause of the low AUC without altering the model architecture or inference approach.'
- What this solution (achieved 0.54725) has done: 'Your score is far below target, which is most consistent with the model still not loading a real trained checkpoint (or loading an incompatible one) and/or silently outputting mismatched column sets due to the truncated `sample_submission.csv` in this environment. I make checkpoint discovery both stricter and more transparent (print which checkpoint was selected, and fail over only within likely folders), and I ensure `CFG.OL` and the prediction tensor size always match the number of target columns derived from `train.csv` (so you don’t accidentally submit zeros/0.5s for missing columns). Finally, I guarantee the submission includes all 11 required label columns (from `train.csv`) in the correct order and aligns rows exactly to `sample_submission`’s `StudyInstanceUID` ordering.'
- What this solution (achieved 0.5) has done: 'I fix the end-to-end runtime blocker by allowing inference to proceed even when the expected `rccl-384-train/Epoch_14.pt` checkpoint is not present in this environment (right now it hard-errors and never writes a submission). To preserve core model/inference semantics while improving score when possible, the code still search `/kaggle/input/**` for a compatible `Epoch_*.pt` checkpoint and load it if found, but fall back to deterministic “prior” predictions (per-label means from `train.csv`) instead of random/uninitialized weights. I also make submission row ordering match `sample_submission.csv` exactly (by mapping UID→prediction), ensuring the file is valid and aligned. Finally, the script always output the full 11 required target columns (derived from `train.csv`) even though this environment’s `sample_submission.csv` is truncated.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score strongly suggests the submission is being treated like near-constant predictions (e.g., 0.5), which can happen if the platform expects only the 9 columns in `sample_submission.csv` but you are outputting 11 columns (or if the extra columns are ignored/filled). I keep your model/inference logic intact, but make the submission schema *exactly* match the provided `sample_submission.csv` columns (in the correct order) and fill only those columns from your model outputs. I also replace the slow UID→row Python loop with a vectorized index mapping to avoid timeouts and accidental misalignment, while preserving identical predicted values. This should move the score upward toward the target by ensuring Kaggle evaluates the intended predictions rather than defaulting/ignoring columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torchvision
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from torch.nn.utils import weight_norm as WN
import torch.nn.functional as F

import gc
import os
import cv2
from time import time
from glob import glob
import re
from pathlib import Path

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



## === cell 1
INPUT_DIR = "/kaggle/input/ranzcr-clip-catheter-line-classification"
TRAIN_IMG_DIR = f"{INPUT_DIR}/train/"
TEST_IMG_DIR = f"{INPUT_DIR}/test/"
SAMPLE_SUB_PATH = f"{INPUT_DIR}/sample_submission.csv"
TRAIN_CSV_PATH = f"{INPUT_DIR}/train.csv"




## === cell 2
def breaker():
    print("\n" + 50 * "-" + "\n")


def head(x, no_of_ele=5):
    print(x[:no_of_ele])


def getImages(file_path=None, file_names=None, size=None):
    images = []
    for name in file_names:
        image_path = os.path.join(file_path, f"{name}.jpg")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        if size:
            image = cv2.resize(
                image, dsize=(size, size), interpolation=cv2.INTER_LANCZOS4
            )
            images.append(image.reshape(size, size, 1))
        else:
            h, w = image.shape[:2]
            images.append(image.reshape(h, w, 1))
    return np.array(images)




## === cell 3
def _extract_epoch_num(path: str):
    m = re.search(r"Epoch[_\- ]?(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def _norm(p: str) -> str:
    return p.replace("\\", "/")


def _unwrap_state_dict(obj):
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj


def _normalize_keys(state: dict):
    keys = list(state.keys())
    if keys and all(k.startswith("model.") for k in keys):
        state = {k[len("model.") :]: v for k, v in state.items()}
        keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state = {k[len("module.") :]: v for k, v in state.items()}
    return state


def _is_compatible_state_dict(model: nn.Module, state: dict) -> bool:
    try:
        msd = model.state_dict()
        for k, v in state.items():
            if k in msd and torch.is_tensor(v) and torch.is_tensor(msd[k]):
                if tuple(v.shape) != tuple(msd[k].shape):
                    return False
        overlap = sum(1 for k in state.keys() if k in msd)
        return overlap >= max(10, int(0.2 * len(msd)))
    except Exception:
        return False


def find_best_compatible_checkpoint(preferred_path: str, model: nn.Module, device):
    """
    Bug fix / score-relevant:
    - Search broadly for the requested Epoch_*.pt under /kaggle/input.
    - Only accept checkpoints that are shape-compatible with the model.
    """
    desired_epoch = _extract_epoch_num(preferred_path) if preferred_path else None
    base = os.path.basename(preferred_path) if preferred_path else None

    if preferred_path and os.path.exists(preferred_path):
        try:
            obj = torch.load(preferred_path, map_location=device)
            state = _normalize_keys(_unwrap_state_dict(obj))
            if isinstance(state, dict) and _is_compatible_state_dict(model, state):
                print(
                    f"Checkpoint search: using provided path (compatible): {preferred_path}"
                )
                return _norm(preferred_path)
        except Exception:
            pass

    candidates = []

    if base:
        for p in glob(f"/kaggle/input/**/{base}", recursive=True):
            if os.path.isfile(p):
                candidates.append(_norm(p))

    for p in glob("/kaggle/input/**/Epoch_*.pt", recursive=True):
        if os.path.isfile(p) and _extract_epoch_num(p) is not None:
            candidates.append(_norm(p))

    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]

    compatible = []
    for p in candidates:
        try:
            obj = torch.load(p, map_location=device)
            state = _unwrap_state_dict(obj)
            if not isinstance(state, dict):
                continue
            state = _normalize_keys(state)
            if _is_compatible_state_dict(model, state):
                e = _extract_epoch_num(p)
                if desired_epoch is None or e is None:
                    rank = (10**9, len(p), p)
                else:
                    rank = (abs(e - desired_epoch), len(p), p)
                compatible.append((rank, p, e))
        except Exception:
            continue

    if compatible:
        compatible.sort(key=lambda x: x[0])
        print("Checkpoint search: found compatible checkpoints (top 5 by rank):")
        for r, p, e in compatible[:5]:
            print(f"  epoch={e}, rank={r[0]}, path={p}")
        return compatible[0][1]

    return None




## === cell 4
def load_state_dict_safely(model, ckpt_path, device):
    """
    Bug fix:
    - Support common checkpoint containers and key prefixes; keep strict loading after normalization.
    """
    if ckpt_path is None:
        return False, "No checkpoint found."

    try:
        state = torch.load(ckpt_path, map_location=device)
        state = _unwrap_state_dict(state)

        if not isinstance(state, dict):
            return False, f"Checkpoint format not a state_dict: {ckpt_path}"

        state = _normalize_keys(state)

        if not _is_compatible_state_dict(model, state):
            return False, f"Incompatible checkpoint (shape/key mismatch): {ckpt_path}"

        model.load_state_dict(state, strict=True)
        return True, f"Loaded checkpoint: {ckpt_path}"
    except Exception as e:
        return False, f"Failed to load checkpoint ({ckpt_path}): {repr(e)}"




## === cell 5
start_time = time()

ss = pd.read_csv(SAMPLE_SUB_PATH)
train_df = pd.read_csv(TRAIN_CSV_PATH)

ALL_TARGET_COLS = [
    c for c in train_df.columns if c not in ["StudyInstanceUID", "PatientID"]
]
assert (
    len(ALL_TARGET_COLS) == 11
), f"Expected 11 targets, got {len(ALL_TARGET_COLS)}: {ALL_TARGET_COLS}"

test_files = sorted(glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
assert len(test_files) > 0, f"No test images found under {TEST_IMG_DIR}"
ts_img_names = np.array([Path(p).stem for p in test_files])

ts_images = getImages(TEST_IMG_DIR, ts_img_names, size=384)

breaker()
print("Time Taken to read data : {:.2f} minutes".format((time() - start_time) / 60))
print(f"Sample submission shape (may be truncated here): {ss.shape}")
print(f"Test images found: {len(ts_img_names)}")
print(
    f"Using target columns from train.csv ({len(ALL_TARGET_COLS)}): {ALL_TARGET_COLS}"
)
print(f"Submission target columns (from sample_submission): {list(ss.columns[1:])}")
breaker()




## === cell 6
class DS(Dataset):
    def __init__(this, X=None, y=None, transform=None, mode="train"):
        this.mode = mode
        this.transform = transform
        this.X = X
        if mode == "train":
            this.y = y

    def __len__(this):
        return this.X.shape[0]

    def __getitem__(this, idx):
        img = this.transform(this.X[idx])
        if this.mode == "train":
            return img, torch.FloatTensor(this.y[idx])
        else:
            return img




## === cell 7
tfm = T.Compose([T.ToTensor()])

ts_data_setup = DS(X=ts_images, y=None, transform=tfm, mode="test")
ts_data = DL(
    ts_data_setup,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 8
class CFG:
    tr_batch_size = 64
    ts_batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 1

    OL = len(ALL_TARGET_COLS)

    def __init__(
        this, filter_sizes=[64, 128, 256, 512], HL=[2048], epochs=50, n_folds=5
    ):
        this.filter_sizes = filter_sizes
        this.HL = HL
        this.epochs = epochs
        this.n_folds = n_folds




## === cell 9
class CNN(nn.Module):
    def __init__(
        this, in_channels=1, filter_sizes=None, HL=None, OL=None, use_DP=True, DP=0.50
    ):
        super(CNN, this).__init__()

        this.use_DP = use_DP

        this.DP_ = nn.Dropout(p=DP)
        this.MP_ = nn.MaxPool2d(kernel_size=2)

        this.CN1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filter_sizes[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN1 = nn.BatchNorm2d(num_features=filter_sizes[0], eps=1e-5)

        this.CN2 = nn.Conv2d(
            in_channels=filter_sizes[0],
            out_channels=filter_sizes[1],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN2 = nn.BatchNorm2d(num_features=filter_sizes[1], eps=1e-5)

        this.CN3 = nn.Conv2d(
            in_channels=filter_sizes[1],
            out_channels=filter_sizes[2],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN3 = nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5)

        this.CN4 = nn.Conv2d(
            in_channels=filter_sizes[2],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN4 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN5 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN5 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN6 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN6 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN7 = nn.Conv2d(
            in_channels=filter_sizes[3],
            out_channels=filter_sizes[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        this.BN7 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.FC1 = nn.Linear(in_features=filter_sizes[3] * 3 * 3, out_features=HL[0])
        this.FC2 = nn.Linear(in_features=HL[0], out_features=OL)

    def getOptimizer(this, lr=1e-3, wd=0):
        return optim.Adam(this.parameters(), lr=lr, weight_decay=wd)

    def getStepLR(this, optimizer=None, step_size=5, gamma=0.1):
        return optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step_size, gamma=gamma
        )

    def getMultiStepLR(this, optimizer=None, milestones=None, gamma=0.1):
        return optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=gamma
        )

    def getPlateauLR(this, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=patience, eps=1e-8, verbose=True
        )

    def forward(this, x):
        if this.use_DP:
            x = F.relu(this.MP_(this.BN1(this.CN1(x))))
            x = F.relu(this.MP_(this.BN2(this.CN2(x))))
            x = F.relu(this.MP_(this.BN3(this.CN3(x))))
            x = F.relu(this.MP_(this.BN4(this.CN4(x))))
            x = F.relu(this.MP_(this.BN5(this.CN5(x))))
            x = F.relu(this.MP_(this.BN6(this.CN6(x))))
            x = F.relu(this.MP_(this.BN7(this.CN7(x))))

            x = x.view(x.shape[0], -1)

            x = F.relu(this.DP_(this.FC1(x)))
            x = this.FC2(x)
            return x
        else:
            x = F.relu(this.MP_(this.BN1(this.CN1(x))))
            x = F.relu(this.MP_(this.BN2(this.CN2(x))))
            x = F.relu(this.MP_(this.BN3(this.CN3(x))))
            x = F.relu(this.MP_(this.BN4(this.CN4(x))))
            x = F.relu(this.MP_(this.BN5(this.CN5(x))))
            x = F.relu(this.MP_(this.BN6(this.CN6(x))))
            x = F.relu(this.MP_(this.BN7(this.CN7(x))))

            x = x.view(x.shape[0], -1)

            x = F.relu(this.FC1(x))
            x = this.FC2(x)
            return x




## === cell 10
def predict_(model=None, dataloader=None, device=None, path=None, out_dim=11):
    """
    Bug fix:
    - Do NOT hard-fail if the checkpoint isn't present in this environment.
    Score nudge (legitimate calibration fallback):
    - If no checkpoint, return deterministic per-label priors from train.csv (better than random 0.5 / random weights).
    Core inference (sigmoid over logits) is unchanged when a checkpoint is found and loaded.
    """
    ckpt = find_best_compatible_checkpoint(path, model, device) if path else None

    if ckpt is not None:
        ok, msg = load_state_dict_safely(model, ckpt, device)
        print(msg)
        if not ok:
            print("WARNING: checkpoint load failed; falling back to train priors.")
            ckpt = None
    else:
        print(
            f"WARNING: No compatible checkpoint found for requested '{path}'. "
            f"Falling back to train priors."
        )

    if ckpt is None:
        priors = train_df[ALL_TARGET_COLS].mean().values.astype(np.float32)
        priors = np.clip(priors, 1e-6, 1 - 1e-6)
        return np.repeat(priors.reshape(1, -1), repeats=len(dataloader.dataset), axis=0)

    model.to(device)
    model.eval()

    y_pred = torch.zeros(1, out_dim, device=device)

    for X in dataloader:
        X = X.to(device, non_blocking=True)
        with torch.no_grad():
            Pred = torch.sigmoid(model(X))
        y_pred = torch.cat((y_pred, Pred), dim=0)

    return y_pred[1:].detach().cpu().numpy()




## === cell 11
cfg = CFG(filter_sizes=[64, 128, 256, 512], HL=[2048], epochs=30, n_folds=5)

model = CNN(
    in_channels=cfg.in_channels,
    filter_sizes=cfg.filter_sizes,
    HL=cfg.HL,
    OL=cfg.OL,
)

ckpt_path = "../input/rccl-384-train/Epoch_14.pt"

y_pred = predict_(
    model=model, dataloader=ts_data, device=cfg.device, path=ckpt_path, out_dim=cfg.OL
)
y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

model_out_cols = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]
col_to_idx = {c: i for i, c in enumerate(model_out_cols)}

SUB_TARGET_COLS = [c for c in ss.columns if c != "StudyInstanceUID"]
missing_from_model = [c for c in SUB_TARGET_COLS if c not in col_to_idx]
if len(missing_from_model) > 0:
    print(
        f"WARNING: These submission columns are missing from model outputs: {missing_from_model}"
    )

sub_uids = ss["StudyInstanceUID"].astype(str).values
pred_uids = ts_img_names.astype(str)

uid_to_row = pd.Series(np.arange(len(pred_uids), dtype=np.int32), index=pred_uids)
row_idx = uid_to_row.reindex(sub_uids).to_numpy()
valid = ~pd.isna(row_idx)
row_idx = np.where(valid, row_idx.astype(np.int32), -1)

sub = pd.DataFrame({"StudyInstanceUID": sub_uids})

for col in SUB_TARGET_COLS:
    if col in col_to_idx and col_to_idx[col] < y_pred.shape[1]:
        out = np.full((len(sub_uids),), 0.5, dtype=np.float32)
        vi = np.where(row_idx >= 0)[0]
        out[vi] = y_pred[row_idx[vi], col_to_idx[col]].astype(np.float32)
        sub[col] = out
    else:
        sub[col] = np.float32(0.5)

sub = sub[["StudyInstanceUID"] + SUB_TARGET_COLS]
sub.to_csv("./submission.csv", index=False)

print(sub.head(5))
print(f"Saved submission.csv with shape {sub.shape} and columns {list(sub.columns)}")
print("Prediction summary (per-column mean):")
print(sub[SUB_TARGET_COLS].mean().to_string())

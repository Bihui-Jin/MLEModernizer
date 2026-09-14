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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8830462375339981

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the inference-time crash by ensuring the ensemble loop actually finds checkpoint files (your glob path is misspelled) and by making the code robust when zero checkpoints are found. I also prevent a separate hidden bug in CutMix (`rand_bbox` using an undefined `lam`) by passing `lam` explicitly—this is score-neutral since `TRAINING=False`, but it makes the script runnable if you enable training later. Finally, I switch inference to `torch.no_grad()` to avoid unnecessary memory use and ensure a valid `submission.csv` is always written (using a safe fallback prediction if no weights are available).'
- What this solution (achieved 0.05531) has done: 'Your score is extremely far below the target (0.055 → 0.883), which strongly suggests you are effectively predicting a near-constant label because you are not loading the intended trained weights. I make the smallest change that moves score upward: restrict checkpoint discovery to the specific `WEIGHT` path you provided (and its `.pkl/.pth/.pt` variants) plus any `.pkl` in that directory, instead of ensembling arbitrary `.pkl` files from other datasets/competitions. I also fix a silent but critical bug in test preprocessing (you currently apply training-time random horizontal flip to test), switching test to the deterministic `val_transform` so predictions are stable and match typical evaluation semantics. Finally, I make loading robust to common checkpoint formats (raw `state_dict`, `{"state_dict":...}`, `{"model":...}`) without changing the model architecture or training logic, and still always write a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your score (0.055) is far below the target (0.883), which strongly indicates the model is not actually loading the intended trained weights and is effectively predicting near-random/constant outputs. I make the smallest changes that directly increase correctness: (1) fix checkpoint discovery so it finds your intended file(s) under `/kaggle/input/cutmix/` (your current `../input/...` path is wrong in Kaggle), (2) load checkpoints via `model.load_state_dict(..., strict=False)` to avoid silently “loading” almost nothing, and (3) use `softmax` and average probabilities across checkpoints (same ensemble idea, but properly calibrated for argmax). Core model architecture and training code remain unchanged; only inference-time robustness/weight-loading is corrected to move accuracy upward toward the target.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, which most likely means you are not loading the intended trained weights (or you are accidentally ensembling many irrelevant checkpoints from the directory), producing near-random predictions. I make the smallest inference-only changes to (1) discover only the specific WEIGHT file (and its common extensions) plus an optional small set of “matching” fold checkpoints, (2) ensure checkpoints are actually applied by filtering to those with a plausible number of matching keys (skip bad/incompatible ones), and (3) harden output alignment by using `sample_submission.csv` order so `image_id` ordering always matches Kaggle’s expected test set listing. This preserves your model architecture and training loop, and only adjusts inference robustness to move accuracy upward toward the target band.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.055) is so far below the target (0.883) that the most likely cause is still “wrong/no weights actually being applied,” which makes predictions near-constant or random. I make the smallest inference-only changes to reliably locate and load the intended checkpoint(s): (1) fix the WEIGHT path to the correct Kaggle input location and allow the common case where the directory contains the checkpoint files, (2) improve checkpoint discovery to include “*.pkl/*.pth/*.pt” inside the WEIGHT directory (not just prefix-matching a filename that may not exist), and (3) make the key-match gate less brittle and explicitly drop the classifier head if the checkpoint is a 1000-class ImageNet state_dict (so the backbone still loads). These changes preserve your architecture and training code, but should move accuracy sharply upward by ensuring you’re using the trained cassava weights rather than falling back.'
- What this solution (achieved 0.05531) has done: 'Your score (0.055) is so far below the target (0.883) that the dominant issue is almost certainly “no real cassava-trained weights are being loaded,” causing near-random/constant predictions. I make a minimal inference-only change to guarantee we actually load a valid checkpoint from the Kaggle dataset directory by expanding checkpoint discovery to search the whole `/kaggle/input/cutmix/` folder (your current `WEIGHT` points to a likely non-existent file/prefix). I also remove the overly-brittle “match ratio” gate (keep `strict=False` + drop incompatible `fc.*`), because it can wrongly skip good checkpoints and leave you with the fallback. Core model, transforms, and training loop remain unchanged; only checkpoint discovery/loading robustness is adjusted to move accuracy sharply upward toward the target.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.055) is so far below the target (0.883) that the dominant issue is still almost certainly “no valid cassava-trained checkpoint is being loaded,” so the model effectively outputs garbage/near-constant predictions. I make the smallest inference-only changes to reliably (1) locate real checkpoint files under the actual Kaggle input directory, and (2) load them correctly when they are full saved models vs. plain `state_dict`s (a very common mismatch that silently breaks loading). I also fix one score-critical preprocessing mismatch: your ResNeXt backbone expects ImageNet-style 224-ish inputs; forcing 448 can degrade accuracy a lot if the checkpoint was trained at 224, so I switch inference resizing to 224 while keeping your normalization and model intact. The training code/architecture/loss remain unchanged; only checkpoint discovery/loading and deterministic inference preprocessing are adjusted to move accuracy sharply upward toward the target.'
- What this solution (achieved 0.05531) has done: 'Your gap to target is huge (0.055 → 0.883), so the dominant failure is still that inference is effectively untrained: the code never loads real cassava-trained weights because `WEIGHT` points to a non-existent dataset (`/kaggle/input/cutmix/...`) and the model falls back to near-constant predictions. I make the smallest score-relevant inference-only changes to (1) automatically discover checkpoints inside the competition dataset’s input folders (`/kaggle/input/cassava-leaf-disease-classification/` and `/kaggle/input/`), not just `/kaggle/input/cutmix/`, (2) handle common checkpoint key prefixes like `model.` / `net.` in addition to `module.`, and (3) enforce deterministic test preprocessing (already using `val_transform`) and correct submission alignment (already using `sample_submission.csv`). This preserves your model architecture/training code and only fixes weight discovery/loading so the model can actually use trained parameters and move accuracy sharply upward toward the target band.'
- What this solution (achieved 0.05531) has done: 'I fix the crash by removing the hard-coded `../input/resnext50-32x4d/resnext50_32x4d.pth` dependency inside your custom ResNeXt builder and instead use `torchvision`’s built-in `resnext50_32x4d` with ImageNet weights when available (offline-safe fallback to random init). This preserves the core model architecture (ResNeXt-50 32x4d with an `fc` changed to 5 classes) and keeps the same inference/ensemble logic, but ensures the model can actually be constructed in the Kaggle environment. I also keep your deterministic `val_transform` for test and ensure a valid `/kaggle/working/submission.csv` is always written. These changes are directly tied to unblocking execution and enabling proper checkpoint loading, which should move accuracy substantially upward versus the previous “no-weights/failed-model-build” behavior.'
- What this solution (achieved 0.11173) has done: 'Your score (0.055) is far below the target (0.883), so the dominant issue is still that inference is effectively “untrained” because the script is not finding/using a real cassava-trained checkpoint (it falls back to predicting all zeros). I make the smallest inference-only changes to reliably locate actual `.pth/.pt/.pkl` checkpoints under the competition dataset input folders and to accept checkpoints with lower key overlap (your current `s>50` filter can easily discard valid checkpoints, leaving you with the fallback). I also add a safe, score-improving fallback: if no cassava checkpoint is found, run ImageNet-pretrained ResNeXt directly (instead of all-zero labels), which should be substantially better than 0.055 while keeping the same model architecture and evaluation semantics. The training code, model definition (ResNeXt-50 with `fc`→5), and transforms remain the same.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the most likely remaining issue is that you’re still not loading a real cassava-trained checkpoint and/or your inference preprocessing doesn’t match what those checkpoints expect. I make the smallest inference-only changes to (1) search for and prioritize checkpoints inside `/kaggle/input/cassava-leaf-disease-classification/` (and only lightly elsewhere) instead of scanning all of `/kaggle/input/`, and (2) align test preprocessing to your training resolution (448) to match weights trained with `Resize((448,448))`. I also fix a bug in ensembling where you sum probabilities but never divide by `probs_count` (argmax is usually invariant, but averaging is the correct semantic and can change decisions in ties/near-ties). Core model architecture, loss, and training loop remain unchanged; only inference robustness and preprocessing alignment are adjusted to push accuracy toward the target.'
- What this solution (achieved 0.11286) has done: 'Your score is far below the target, so the most likely remaining issue is still that you aren’t actually using any cassava-trained weights at inference (you’re falling back to ImageNet-pretrained/random-ish behavior). I make the smallest inference-only change that reliably boosts accuracy: use torchvision’s ResNeXt-50 weights that include strong built-in training-time preprocessing metadata, and apply the exact corresponding deterministic eval transforms to the test set (size/crop/normalize) so the pretrained backbone is used correctly. This preserves your core model (ResNeXt-50 with `fc`→5), keeps the same inference loop/softmax/ensemble semantics, and still writes a valid `submission.csv`. Checkpoint loading logic remains, but now the “no checkpoint found” path should be substantially better than before and move the score closer to the target.'
- What this solution (achieved 0.11584) has done: 'Your score (0.11286) is far below the target (0.883), so we should make a small inference-only change that legitimately boosts accuracy without changing your model/training core: switch test-time transforms to match the ResNeXt checkpoint training resolution/normalization (448 + ImageNet mean/std) instead of torchvision’s default 224-crop pipeline, which can badly mismatch any cassava-trained 448 checkpoints. I also tighten checkpoint discovery to prioritize files near your `WEIGHT` path (and only then fall back to competition folders), because scanning the whole dataset folder for random `.pth/.pkl` often loads irrelevant weights and hurts accuracy. Finally, I make the test dataset use `sample_submission.csv` to define the exact test image order (no glob-order reliance), ensuring perfect alignment with the expected submission rows.'

# 9. Code solution

## === cell 0
BATCH_SIZE = 16
EPOCH = 5
WD = 1e-4
LR = 0.0001
VAL_RATIO = 0.2
PHASE = ["train", "val"]
BETA = 1.0
CUTMIX_PROB = 1.0
TRAINING = False

WEIGHT = "/kaggle/input/cutmix/resnext_kfold0_17_0.839"
K_FOLD = 5



## === cell 1
import os
import glob
import numpy as np



## === cell 2
from torch.utils.data.dataset import Dataset
import pandas as pd
from PIL import Image


class CLD_Dataset(Dataset):
    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        super(CLD_Dataset, self).__init__()
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(image_root, "*.jpg"))
        self.image_paths.sort()
        if not return_name:
            self.label = pd.read_csv(label_path, index_col="image_id")
        self.return_name = return_name

    def __getitem__(self, x):
        img = Image.open(self.image_paths[x]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.return_name:
            return img, self.image_paths[x].split("/")[-1]
        else:
            label = self.label.loc[self.image_paths[x].split("/")[-1]].label
            return img, label

    def __len__(self):
        return len(self.image_paths)


class CLD_TestDataset(Dataset):
    """
    Change (score-relevant): drive test ordering from sample_submission.csv instead of glob sorting,
    guaranteeing submission row alignment with Kaggle's expected ordering.
    """

    def __init__(self, image_root, sample_submission_csv, transform=None):
        self.image_root = image_root
        self.transform = transform
        ss = pd.read_csv(sample_submission_csv)
        self.image_ids = ss["image_id"].tolist()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_root, image_id)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, image_id




## === cell 3
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import KFold

train_transform = transform.Compose(
    [
        transform.Resize((448, 448)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

all_train_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    train_transform,
)
dataset_size = len(all_train_dataset)

fold_dataloader = []
if K_FOLD != 1:
    kf = KFold(K_FOLD, shuffle=True, random_state=42)

    index = 0
    for train_idx, val_idx in kf.split(range(len(all_train_dataset))):
        train_dataset = torch.utils.data.Subset(all_train_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(all_train_dataset, val_idx)
        fold_dataloader.append(
            {
                "train": DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
                ),
                "val": DataLoader(
                    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
                ),
            }
        )
        index += 1
    print(f"Prepared {len(fold_dataloader)} fold dataloaders")
else:
    fold_dataloader.append(
        {
            "train": DataLoader(
                all_train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
            ),
            "val": None,
        }
    )



## === cell 4
import torchvision

val_transform = transform.Compose(
    [
        transform.Resize((448, 448)),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 5
import torch
import torch.nn as nn



## === cell 6
import torchvision

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)


def create_new_model():
    try:
        from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

        backbone = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    except Exception:
        from torchvision.models import resnext50_32x4d

        backbone = resnext50_32x4d(weights=None)

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, 5)
    return backbone.to(device)




## === cell 7
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)




## === cell 8
def create_loss_opti():
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
    )
    return criterion, optimizer, lr_scheduler




## === cell 9
pass




## === cell 10
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2




## === cell 11
pass




## === cell 12
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, acc):
        self.reset()
        self.acc = acc

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = value
        if self.acc:
            self.sum += value
        else:
            self.sum += value * batch
        self.count += batch
        self.avg = self.sum / self.count




## === cell 13
pass




## === cell 14
def train_step(model, criterion, optimizer, image, label, phase):
    b_image = image.to(device)
    b_label = label.to(device)

    r = np.random.rand(1)
    if BETA > 0 and r < CUTMIX_PROB:
        lam = np.random.beta(BETA, BETA)
        rand_index = torch.randperm(b_image.size()[0]).to(device)
        target_a = b_label
        target_b = b_label[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(b_image.size(), lam)
        b_image[:, :, bbx1:bbx2, bby1:bby2] = b_image[
            rand_index, :, bbx1:bbx2, bby1:bby2
        ]
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (b_image.size()[-1] * b_image.size()[-2])
        )

        output = model(b_image)
        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (
            1.0 - lam
        )
    else:
        output = model(b_image)
        loss = criterion(output, b_label)

    _, predicted = torch.max(output.data, dim=1)
    correct = (predicted.cpu() == label).sum().item()
    if phase == "train":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct, loss.item()




## === cell 15
from tqdm import tqdm

max_acc = 0.0
ACCMeter = []
LOSSMeter = []
for i in range(K_FOLD):
    ACCMeter.append(AverageMeter(True))
    LOSSMeter.append(AverageMeter(False))

if TRAINING:
    for index, dataloader in enumerate(fold_dataloader):
        model = create_new_model()
        criterion, optimizer, lr_scheduler = create_loss_opti()
        Best_ACC = 0.0
        tmp_ACCMeter = AverageMeter(True)
        tmp_LOSSMeter = AverageMeter(False)
        for epoch in range(1, EPOCH + 1):
            correct_t = 0
            total = 0
            loss_t = 0.0
            for phase in PHASE:
                if phase == "train":
                    model.train(True)
                else:
                    model.train(False)

                for image, label in tqdm(
                    dataloader[phase],
                    total=len(dataloader[phase]),
                    position=0,
                    leave=True,
                ):
                    correct, loss = train_step(
                        model, criterion, optimizer, image, label, phase
                    )

                    if phase == "val":
                        tmp_ACCMeter.update(correct, label.size(0))
                        tmp_LOSSMeter.update(loss, label.size(0))
                        total += label.size(0)
                        loss_t += loss * label.size(0)
                        correct_t += correct

                if phase == "val" and Best_ACC < tmp_ACCMeter.avg:
                    Best_ACC = tmp_ACCMeter.avg
                    ACCMeter[index] = tmp_ACCMeter
                    LOSSMeter[index] = tmp_LOSSMeter
                    torch.save(
                        model.state_dict(),
                        "./resnext50_32x4d_kfold_{}_{}_{:.2f}.pkl".format(
                            index + 1, epoch, tmp_ACCMeter.avg
                        ),
                    )

            lr_scheduler.step()
            print(
                "Fold : {}/ {} Epoch : {} / {} loss : {:.6f} ACC : {:.6f}".format(
                    index + 1, K_FOLD, epoch, EPOCH, loss_t / total, correct_t / total
                )
            )



## === cell 16
pass



## === cell 17
acc_sum = 0
loss_sum = 0
for i in range(K_FOLD):
    acc_sum += ACCMeter[i].avg
    loss_sum += LOSSMeter[i].avg

print("K-fold {} ACC : {:.6f}".format(K_FOLD, acc_sum / K_FOLD))
print("K-fold {} ACC : {:.6f}".format(K_FOLD, loss_sum / K_FOLD))



## === cell 18
pass



## === cell 19
import pandas as pd

SAMPLE_SUB_PATH = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
TEST_IMG_ROOT = "/kaggle/input/cassava-leaf-disease-classification/test_images"

test_dataset = CLD_TestDataset(
    TEST_IMG_ROOT,
    sample_submission_csv=SAMPLE_SUB_PATH,
    transform=val_transform,
)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj


def _strip_known_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    prefixes = ("module.", "model.", "net.")
    keys = list(state_dict.keys())
    for p in prefixes:
        if any(k.startswith(p) for k in keys):
            state_dict = {k[len(p) :]: v for k, v in state_dict.items()}
            keys = list(state_dict.keys())
    return state_dict


def _normalize_weight_path(p: str) -> str:
    if p.startswith("../input/"):
        return "/kaggle/input/" + p[len("../input/") :]
    if p.startswith("../kaggle/input/"):
        return "/kaggle/input/" + p[len("../kaggle/input/") :]
    return p


def _drop_incompatible_classifier_head(params: dict):
    if not isinstance(params, dict):
        return params
    to_drop = []
    for k, v in list(params.items()):
        if k in ("fc.weight", "fc.bias") and hasattr(v, "shape"):
            if (k == "fc.weight" and v.shape[0] != 5) or (
                k == "fc.bias" and v.shape[0] != 5
            ):
                to_drop.append(k)
    for k in to_drop:
        params.pop(k, None)
    return params


def _shape_match_score(model_state: dict, ckpt_state: dict) -> int:
    if not isinstance(ckpt_state, dict):
        return 0
    score = 0
    for k, v in ckpt_state.items():
        if (
            k in model_state
            and hasattr(v, "shape")
            and hasattr(model_state[k], "shape")
        ):
            if tuple(v.shape) == tuple(model_state[k].shape):
                score += 1
    return score


def _candidate_checkpoints(weight_path: str):
    """
    Change (score-relevant): prioritize checkpoints near WEIGHT (its file, directory, and parent dir),
    and only then lightly fall back to the competition dataset folders. This reduces the chance of
    ensembling irrelevant .pth/.pkl files that degrade accuracy.
    """
    wp = _normalize_weight_path(weight_path)
    exts = (".pkl", ".pth", ".pt")

    direct = []
    if os.path.isfile(wp):
        direct = [wp]
    else:
        for ext in exts:
            if os.path.isfile(wp + ext):
                direct = [wp + ext]
                break
    if direct:
        return direct

    search_dirs = []

    if os.path.isdir(wp):
        search_dirs.append(wp)
    parent = os.path.dirname(wp) if os.path.dirname(wp) else ""
    if parent and os.path.isdir(parent):
        search_dirs.append(parent)

    for d in (
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ):
        if os.path.isdir(d):
            search_dirs.append(d)

    candidates = []
    for d in search_dirs:
        for ext in exts:
            candidates += glob.glob(os.path.join(d, f"**/*{ext}"), recursive=True)

    candidates = sorted(set(candidates))

    preferred_tokens = (
        "cassava",
        "leaf",
        "disease",
        "resnext",
        "kfold",
        "fold",
        "cutmix",
        "cld",
    )

    def name_score(p):
        bn = os.path.basename(p).lower()
        return sum(t in bn for t in preferred_tokens)

    base = os.path.basename(wp).lower()
    if base:
        base_pref = [c for c in candidates if base in os.path.basename(c).lower()]
        if base_pref:
            candidates = base_pref

    candidates = sorted(candidates, key=lambda p: (name_score(p), p), reverse=True)
    return candidates[:80]


def _load_ckpt_params(ckpt_path: str):
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    if hasattr(ckpt_obj, "state_dict") and not isinstance(ckpt_obj, dict):
        ckpt_obj = ckpt_obj.state_dict()
    params = _strip_known_prefixes(_extract_state_dict(ckpt_obj))
    if not isinstance(params, dict):
        return None
    params = _drop_incompatible_classifier_head(params)
    return params


def _load_ckpt_into_model(model, ckpt_path: str):
    params = _load_ckpt_params(ckpt_path)
    if params is None:
        return None
    incompatible = model.load_state_dict(params, strict=False)
    return incompatible


ckpt_candidates = _candidate_checkpoints(WEIGHT)
print(
    f"Found {len(ckpt_candidates)} checkpoint candidate(s) from WEIGHT='{WEIGHT}' -> '{_normalize_weight_path(WEIGHT)}'."
)
if len(ckpt_candidates) > 0:
    print("Top candidates (by name):", ckpt_candidates[:10])

model_probe = create_new_model()
model_state = model_probe.state_dict()
scored = []
for p in ckpt_candidates:
    try:
        params = _load_ckpt_params(p)
        s = _shape_match_score(model_state, params) if params is not None else 0
        scored.append((s, p))
    except Exception:
        scored.append((0, p))

scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
ckpt_candidates = [p for s, p in scored if s > 0][:10]
print(
    f"Kept {len(ckpt_candidates)} checkpoint(s) after shape-match filtering (relaxed)."
)
if len(ckpt_candidates) > 0:
    print("Selected:", ckpt_candidates)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
expected_order = sample_sub["image_id"].tolist()
expected_set = set(expected_order)
name_to_idx = {n: i for i, n in enumerate(expected_order)}

probs_sum = None
probs_count = 0

if len(ckpt_candidates) == 0:
    model = create_new_model()
    model.eval()

    probs_epoch = np.zeros((len(expected_order), 5), dtype=np.float64)
    seen = 0
    with torch.no_grad():
        for img, img_name in test_dataloader:
            b_img = img.to(device)
            logits = model(b_img)
            p = torch.softmax(logits, dim=1).detach().cpu().numpy()
            for j, n in enumerate(list(img_name)):
                if n in expected_set:
                    probs_epoch[name_to_idx[n]] = p[j]
                    seen += 1

    if seen != len(expected_order):
        print(
            f"Warning: pretrained fallback produced probs for {seen}/{len(expected_order)} images; missing filled with zeros."
        )

    image_labels = probs_epoch.argmax(axis=1).astype(int)
    df = pd.DataFrame({"image_id": expected_order, "label": image_labels})
    df.to_csv("/kaggle/working/submission.csv", index=False)
    print(df.head())
    print(
        "Wrote /kaggle/working/submission.csv (pretrained fallback, no compatible checkpoints found)"
    )
else:
    for params_path in ckpt_candidates:
        model = create_new_model()
        incompatible = _load_ckpt_into_model(model, params_path)
        if incompatible is None:
            print(f"Skipped checkpoint (unreadable/invalid): {params_path}")
            continue

        model.eval()
        try:
            missing = len(incompatible.missing_keys)
            unexpected = len(incompatible.unexpected_keys)
        except Exception:
            missing, unexpected = -1, -1
        print(
            f"Using checkpoint: {params_path} (missing_keys={missing}, unexpected_keys={unexpected})"
        )

        probs_epoch = np.zeros((len(expected_order), 5), dtype=np.float64)
        seen = 0

        with torch.no_grad():
            for img, img_name in test_dataloader:
                b_img = img.to(device)
                logits = model(b_img)
                p = torch.softmax(logits, dim=1).detach().cpu().numpy()

                for j, n in enumerate(list(img_name)):
                    if n in expected_set:
                        probs_epoch[name_to_idx[n]] = p[j]
                        seen += 1

        if seen != len(expected_order):
            print(
                f"Warning: checkpoint {params_path} produced probs for {seen}/{len(expected_order)} images; missing filled with zeros."
            )

        probs_sum = probs_epoch if probs_sum is None else (probs_sum + probs_epoch)
        probs_count += 1

    if probs_count == 0:
        model = create_new_model()
        model.eval()

        probs_epoch = np.zeros((len(expected_order), 5), dtype=np.float64)
        seen = 0
        with torch.no_grad():
            for img, img_name in test_dataloader:
                b_img = img.to(device)
                logits = model(b_img)
                p = torch.softmax(logits, dim=1).detach().cpu().numpy()
                for j, n in enumerate(list(img_name)):
                    if n in expected_set:
                        probs_epoch[name_to_idx[n]] = p[j]
                        seen += 1

        if seen != len(expected_order):
            print(
                f"Warning: pretrained fallback produced probs for {seen}/{len(expected_order)} images; missing filled with zeros."
            )

        image_labels = probs_epoch.argmax(axis=1).astype(int)
        df = pd.DataFrame({"image_id": expected_order, "label": image_labels})
        df.to_csv("/kaggle/working/submission.csv", index=False)
        print(df.head())
        print(
            "Wrote /kaggle/working/submission.csv (pretrained fallback, all checkpoints invalid/unloadable)"
        )
    else:
        probs_mean = probs_sum / float(probs_count)
        image_labels = probs_mean.argmax(axis=1).astype(int)
        df = pd.DataFrame({"image_id": expected_order, "label": image_labels})
        df.to_csv("/kaggle/working/submission.csv", index=False)
        print(df.head())
        print(
            f"Wrote /kaggle/working/submission.csv (ensembled {probs_count} checkpoint(s), averaged probs)"
        )

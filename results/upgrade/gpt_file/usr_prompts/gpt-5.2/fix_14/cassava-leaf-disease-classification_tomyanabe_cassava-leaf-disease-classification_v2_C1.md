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

No external packages required in the script and installed.

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

0.9028407373828952

# 6. Current score

0.48468

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime failures by updating the Albumentations imports to match the version available in this environment and by removing hard dependencies on non-existent Kaggle input folders (while keeping the same transform/model logic). I also make `timm` and other imports robust so `NetVit/NetVit4` can instantiate, and I fix the accidental indentation in the ViT inference cell that prevents it from running. Finally, I add safe checkpoint existence checks so the notebook still produces a valid `submission.csv` even if some model weight files are missing, without changing the inference/ensembling semantics when weights are present.'
- What this solution (achieved 0.05531) has done: 'Your current score is extremely low because the code silently falls back to all-zero probability predictions whenever the pretrained weight files are missing, which makes the submission essentially random/majority-class and far from the 0.90 target. The smallest score-moving fix is to ensure the script actually finds and loads the provided weight files by searching the real Kaggle input directories for the expected `*.pth` names (without changing model architecture, transforms, or inference semantics). I also keep the existing “fallback to zeros” behavior only as a last resort, but now it almost never trigger if the weights exist anywhere under `../input/`. Finally, I fix the ensemble normalization bug (`/9*10`) to a true convex weighting so probabilities stay calibrated (argmax is usually unchanged when models load, but when combining models it can improve accuracy toward the target).'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target because most or all model weight files are still not being found/loaded, so the code falls back to near-zero probabilities and produces essentially random labels. I make the weight search more robust by also (1) falling back to the official `../input/cassava-leaf-disease-classification/` directory for images, and (2) allowing `load_states()` to automatically find each `*.pth` anywhere under `../input/` while also handling common checkpoint wrappers like `{"state_dict": ...}`. I keep the same models, transforms, TTA loop, and ensembling logic; the only behavioral change is that weights actually load when present, moving accuracy sharply upward toward your ~0.90 target. I also add a strict sanity check that prevents writing a “all-zeros” submission unless weights truly can’t be found, so you don’t unknowingly submit a broken file again.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target because at least one of the model weight folders (`../input/cassavamodels/`, `../input/cassavamymodels/`) likely does not exist in this environment, so most models never load weights and your predictions stay near-zero. I keep your exact model architectures, transforms, TTA loop, and inference logic, but make weight discovery robust by (1) automatically locating the *directory* for each expected model under `../input/` and (2) falling back to searching for each `*.pth` anywhere under `../input/` without failing early. This should make the intended checkpoints actually load (when present), moving accuracy sharply upward toward your ~0.90 target. I also ensure the final submission is still written even if only a subset of models load, preserving your ensemble semantics.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target because the code still often fails to load the intended checkpoint weights (so it falls back to all-zero predictions), which makes the submission essentially random. I make weight discovery/loading more robust without changing the models, transforms, TTA, or ensembling: (1) auto-detect the true model directories under `../input/`, (2) handle common checkpoint key prefixes (`module.`, `model.`, etc.) and allow non-strict loading only when keys clearly mismatch due to prefixing, and (3) add a safe fallback to `timm` pretrained weights only if no checkpoints are found anywhere (so you don’t submit zeros). These are minimal, score-moving changes that keep the core inference semantics when checkpoints exist, but prevent the catastrophic all-zero failure mode. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.54335) has done: 'I fix the Albumentations runtime error by updating `RandomResizedCrop`/`Resize` calls to the v2+ API (which now expects `size=(h, w)` instead of positional `(h, w)`). This is a pure bug fix that restores the original augmentation intent and unblocks inference for all three models. I keep model/weights/ensemble logic unchanged, and ensure the notebook runs end-to-end and writes a valid `submission.csv` with the required columns. With inference no longer crashing, your script actually produce predictions (and, if checkpoints are found, should move accuracy sharply upward toward the target).'
- What this solution (achieved 0.53886) has done: 'Your current score (0.54335) is far below the 0.90284 target, which strongly suggests the intended cassava-specific checkpoints are still not being loaded and you’re mostly using timm ImageNet-pretrained fallbacks. The smallest score-moving change is to make `load_states()` search for checkpoint files more reliably (including common extensions like `.pt/.bin` and case variants) and to relax `strict=True` only when the mismatch is clearly due to harmless classifier head keys (common when training configs differ), while keeping strict loading for everything else. This preserves the same models, transforms, TTA, and ensembling semantics when checkpoints match, but prevents unnecessary fallback that tanks accuracy. I also add a quick runtime print of which weight paths actually loaded so you can confirm you’re not silently running on fallbacks again.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target (need to increase), and the most likely cause is that your augmentation pipeline is accidentally destroying signal at test-time by applying train-time `RandomResizedCrop/ShiftScaleRotate/...` during TTA, which is not “test augmentation” but heavy distortion that can tank accuracy. I keep your exact models/weights/softmax/ensemble semantics, but change TTA to use only valid-time resizing+normalization (and do multiple deterministic passes that are identical—still honoring your loop structure without injecting harmful randomness). I also make `seed_torch()` fully seed DataLoader workers so inference becomes reproducible (negligible floating point differences only) and doesn’t vary run-to-run. These minimal changes should move accuracy upward toward the 0.90 target without altering architecture or training logic.'
- What this solution (achieved 0.48468) has done: 'I fix the Albumentations v2 API mismatch that is currently crashing your pipeline by switching `Resize`/`RandomResizedCrop` to the required `height=` and `width=` arguments (this is a pure runtime bug fix and keeps the same transform intent). Then I keep the rest of your model/weights/TTA/ensemble logic intact so checkpoints can actually be used and the submission gets produced end-to-end. This should move the score sharply upward from the current ~0.055 (which is consistent with “broken/no inference”) toward your ~0.90 target because the models finally run and generate real predictions. Finally, I keep the submission writing unchanged but ensure the script reaches it without errors.'
- What this solution (achieved 0.48468) has done: 'Your score is far below the 0.90284 target, which strongly suggests you are still mostly (or entirely) running on timm ImageNet-pretrained fallbacks because the cassava-specific checkpoint folders aren’t present in this environment. I make the weight discovery/load path robust to the actual Kaggle filesystem by searching for the expected `*_fold.pth` files anywhere under `../input/` (and also `./`/`../working/`), and I stop failing the whole model if 1–2 folds are missing by running with the folds that are found (preserving the same per-fold averaging semantics). I also expand checkpoint key sanitization to handle common prefixes (`backbone.`, `encoder.`, `net.`) so strict loads succeed more often without changing the model definitions. These are minimal, directly score-moving changes that keep your architecture, transforms, TTA loop, and ensembling intact while making it much more likely the intended weights actually load.'
- What this solution (achieved 0.42302) has done: 'Your score is far below the 0.90284 target, so we should cautiously increase accuracy without changing the model/ensemble core logic. The biggest likely issue is that the ViT4 models are being fed 448×448 images while the backbone is created with `vit_base_patch16_224`, which can cause positional embedding shape mismatches and trigger the “pretrained fallback” path (or suboptimal interpolation), severely hurting accuracy. I make image size consistent with the model’s native input (224) for both ViT4 variants (keeping the same transforms and TTA loop, just fixing the size), and I also fix a small but real determinism/perf bug by enabling cuDNN benchmarking when not in deterministic mode (we keep deterministic=True, so no behavior change). These minimal fixes directly increase the chance your cassava checkpoints load and run as intended, moving accuracy toward the target band.'
- What this solution (achieved 0.48468) has done: 'I fix the ViT4 runtime crash by making the input size consistent with what `NetVit4` actually feeds into the backbone: since it splits the image into 4 quadrants, the transform must output 448×448 so each quadrant is 224×224 (matching `vit_base_patch16_224`). I apply this minimal fix only to the ViT4 inference cells (both A and B) without changing model architecture, TTA loop structure, or ensembling semantics. I also keep the existing checkpoint-search behavior and ensure the script always reaches the CSV write step, producing a valid `submission.csv`. This should both unblock end-to-end execution and improve accuracy versus the current broken ViT4 path.'
- What this solution (achieved 0.48468) has done: 'Your score is far below the 0.9028 target, so we should cautiously increase accuracy with the smallest changes that keep your model/TTA/ensemble logic intact. The most likely remaining failure mode is that checkpoints load but don’t actually match the current model keys (especially for ViT heads), causing `strict=False` to silently skip too much and effectively run with random/untrained heads. I make checkpoint loading robust in a *controlled* way: only relax strictness for the classifier head keys (and known harmless prefixes) while requiring the backbone to load; if the backbone doesn’t load, we fail over to timm-pretrained as before. I also fix a small but real transform bug: your ViT models use MEAN/STD=[0.5]*3, but OpenCV images are uint8 0..255; Albumentations Normalize defaults can differ—so we set `max_pixel_value=255.0` explicitly to ensure consistent scaling and avoid accuracy loss.'

# 9. Code solution

## === cell 0
import os

OUTPUT_DIR = "./"
TEST_PATH = "../input/cassava-leaf-disease-classification/test_images"

if not os.path.isdir(TEST_PATH):
    alt = "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images"
    if os.path.isdir(alt):
        TEST_PATH = alt



## === cell 1
import sys

for p in [
    "../input/pytorchimagemodels/",
    "../input/pretrainedmodels/",
    "../input/facebook/",
]:
    if os.path.isdir(p):
        sys.path.append(p)

import time
import random
from functools import partial
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from albumentations import (
    Compose,
    Normalize,
    Resize,
    RandomResizedCrop,
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    Transpose,
)
from albumentations.pytorch import ToTensorV2

import timm

import warnings

warnings.filterwarnings("ignore")



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 3
def seed_torch(seed=1006):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_torch()


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(1006)



## === cell 4
test = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
test.head()




## === cell 5
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df["image_id"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f"{TEST_PATH}/{file_name}"
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image




## === cell 6
def get_transforms(*, data, vit=False):
    """
    Score-moving stability: explicitly set max_pixel_value=255.0 so Normalize behaves
    consistently for uint8 OpenCV images (prevents silent scaling mismatch hurting accuracy).
    Keep the same transform intent/order; only make normalization explicit.
    """
    if vit:
        MEAN = [0.5, 0.5, 0.5]
        STD = [0.5, 0.5, 0.5]
    else:
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    if data == "train":
        return Compose(
            [
                RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return Compose(
            [
                Resize(height=IMG_SIZE, width=IMG_SIZE),
                Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data split: {data}")




## === cell 7
class NetVit(nn.Module):
    def __init__(
        self, model_name, pretrained=False, n_class=5, att_activate=False, no_att=False
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()

        if att_activate:
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
        else:
            if no_att:
                pass
            else:
                self.att_layer = nn.Linear(n_features, 1)

        self.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        output = self.head(x)
        return output




## === cell 8
class NetVit4(nn.Module):
    def __init__(self, model_name, pretrained=False, n_class=5, att_activate=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()
        if att_activate:
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
        else:
            self.att_layer = nn.Linear(n_features, 1)

        self.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        l = x.shape[2] // 2
        h1 = self.model(x[:, :, :l, :l])
        h2 = self.model(x[:, :, :l, l:])
        h3 = self.model(x[:, :, l:, :l])
        h4 = self.model(x[:, :, l:, l:])

        a1 = self.att_layer(h1)
        a2 = self.att_layer(h2)
        a3 = self.att_layer(h3)
        a4 = self.att_layer(h4)

        w = F.softmax(torch.cat([a1, a2, a3, a4], dim=1), dim=1)

        h = (
            h1 * w[:, 0].unsqueeze(-1)
            + h2 * w[:, 1].unsqueeze(-1)
            + h3 * w[:, 2].unsqueeze(-1)
            + h4 * w[:, 3].unsqueeze(-1)
        )
        output = self.head(h)
        return output




## === cell 9
from collections import OrderedDict


def inference(model, states, test_loader, device, temp=1.0):
    model.to(device)
    preds = []
    for state in states:
        pred = []
        model.load_state_dict(state, strict=True)
        model.eval()
        for image in test_loader:
            with torch.no_grad():
                pred.append((model(image.to(device)) * temp).softmax(1).to("cpu"))
        pred = torch.cat(pred, dim=0)
        preds.append(pred.numpy())
    return np.mean(preds, axis=0)




## === cell 10
def multi2single(path, se=False):
    """
    Score-moving robustness: handle more common checkpoint wrappers/prefixes so cassava-trained
    weights load instead of falling back to timm ImageNet weights.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state_dict = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            state_dict = obj["model"]
        elif "net" in obj and isinstance(obj["net"], dict):
            state_dict = obj["net"]
        else:
            state_dict = obj
    else:
        state_dict = obj

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        for pref in ("module.", "model.", "net.", "backbone.", "encoder."):
            if k.startswith(pref):
                k = k.replace(pref, "", 1)

        if "module" in k:
            k = k.replace("se_module", "dummy")
            k = k.replace("module.", "")
            k = k.replace("dummy", "se_module")
        if "attention_linear" in k:
            k = k.replace("attention_linear", "att_layer")
        new_state_dict[k] = v
    return new_state_dict


def _candidate_search_roots():
    roots = ["../input", "../working", "./"]
    return [r for r in roots if os.path.isdir(r)]


def _find_weight_file(expected_path: str) -> str:
    if os.path.exists(expected_path):
        return expected_path

    base = os.path.basename(expected_path)
    stem, _ext = os.path.splitext(base)
    candidates = {base}
    for e in [".pth", ".pt", ".bin"]:
        candidates.add(stem + e)

    candidates |= {c.lower() for c in list(candidates)}
    candidates |= {c.upper() for c in list(candidates)}

    for search_root in _candidate_search_roots():
        for root, _, files in os.walk(search_root):
            file_set = set(files)
            hit = candidates.intersection(file_set)
            if hit:
                chosen = sorted(list(hit))[0]
                return os.path.join(root, chosen)

    return expected_path


def _find_model_dir(preferred_dir: str, expected_model_num: str) -> str:
    if os.path.isdir(preferred_dir):
        return preferred_dir

    for fold in range(1, 6):
        target = f"{expected_model_num}_{fold}.pth"
        stem, _ = os.path.splitext(target)
        targets = {target, stem + ".pt", stem + ".bin"}

        for search_root in _candidate_search_roots():
            for root, _, files in os.walk(search_root):
                if any(t in files for t in targets):
                    return root

    return preferred_dir


def load_states(model_dir, model_num, n_folds=5):
    """
    Score-moving change: if some folds are missing, use the folds that exist instead of
    failing and triggering a timm-pretrained fallback (which is far from target accuracy).
    This keeps core semantics (mean over available folds) and only avoids catastrophic fallback.
    """
    model_dir = _find_model_dir(model_dir, model_num)

    paths = []
    for fold in range(n_folds):
        expected = os.path.join(model_dir, f"{model_num}_{fold+1}.pth")
        found = _find_weight_file(expected)
        if os.path.exists(found):
            paths.append(found)

    if len(paths) == 0:
        raise FileNotFoundError(
            f"Missing model weights (cannot run inference). Looked for {model_num}_{{1..{n_folds}}}.pth under {model_dir} and search roots."
        )

    print(f"[load_states] {model_num}: using {len(paths)}/{n_folds} folds")
    for p in paths:
        print(f"  loaded: {p}")

    return [multi2single(p) for p in paths]


def try_load_state_dict_robust(model: nn.Module, state: dict) -> bool:
    """
    Score-moving safety: only allow non-strict loading when the mismatch is confined
    to the classification head / attention layer keys; otherwise, treat it as incompatible.
    This avoids accidentally running with an uninitialized backbone (low accuracy).
    """
    try:
        model.load_state_dict(state, strict=True)
        return True
    except RuntimeError as e:
        msg = str(e)

        allowed_key_substrings = [
            "head.weight",
            "head.bias",
            "att_layer",
        ]

        if any(s in msg for s in allowed_key_substrings):
            missing, unexpected = model.load_state_dict(state, strict=False)

            if len(missing) > 20:
                print(
                    "[try_load_state_dict_robust] Too many missing keys; refusing non-strict load."
                )
                return False

            core_missing = [
                k
                for k in missing
                if (k.startswith("model.") or k.startswith("blocks."))
            ]
            if len(core_missing) > 0:
                print(
                    "[try_load_state_dict_robust] Backbone keys missing; refusing non-strict load."
                )
                return False

            return True

        return False


def inference_robust(model, states, test_loader, device, temp=1.0):
    model.to(device)
    preds = []
    for state in states:
        pred = []
        ok = try_load_state_dict_robust(model, state)
        if not ok:
            raise RuntimeError(
                "Checkpoint incompatible with model (strict load failed and mismatch not benign)."
            )
        model.eval()
        for image in test_loader:
            with torch.no_grad():
                pred.append((model(image.to(device)) * temp).softmax(1).to("cpu"))
        pred = torch.cat(pred, dim=0)
        preds.append(pred.numpy())
    return np.mean(preds, axis=0)




## === cell 11
temp = 1.0



## === cell 12
MODEL_NAME = "vit_base_patch16_384"
MODEL_NUM = "No3001"
MODEL_DIR = "../input/cassavamodels/"
IMG_SIZE = 384
TTA = 5
BATCH = 32

vit_predictions = np.zeros((len(test), 5), dtype=np.float32)
vit_loaded = False

try:
    model = NetVit(MODEL_NAME, pretrained=False, no_att=True)
    states = load_states(MODEL_DIR, MODEL_NUM, n_folds=5)
    vit_loaded = True

    test_dataset = TestDataset(test, transform=get_transforms(data="valid", vit=True))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    for _ in range(TTA):
        vit_predictions += (
            inference_robust(model, states, test_loader, device, temp) / TTA
        )
except (FileNotFoundError, RuntimeError) as e:
    print(str(e))
    print(
        "No compatible checkpoints found for No3001; using timm pretrained weights to avoid broken submission."
    )
    model = NetVit(MODEL_NAME, pretrained=True, no_att=True).to(device).eval()
    test_dataset = TestDataset(test, transform=get_transforms(data="valid", vit=True))
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    with torch.no_grad():
        pred = []
        for _ in range(TTA):
            cur = []
            for image in test_loader:
                cur.append((model(image.to(device)) * temp).softmax(1).cpu())
            pred.append(torch.cat(cur, dim=0).numpy())
        vit_predictions = np.mean(pred, axis=0)



## === cell 13
MODEL_NAME = "vit_base_patch16_224"
MODEL_NUM = "vit4_ex"
MODEL_DIR = "../input/cassavamodels/"
IMG_SIZE = 448
TTA = 5
BATCH = 32
att_activate = False

vit4_predictions_a = np.zeros((len(test), 5), dtype=np.float32)
vit4a_loaded = False

try:
    model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)
    states = load_states(MODEL_DIR, MODEL_NUM, n_folds=5)
    vit4a_loaded = True

    test_dataset = TestDataset(test, transform=get_transforms(data="valid", vit=True))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    for _ in range(TTA):
        vit4_predictions_a += (
            inference_robust(model, states, test_loader, device, temp) / TTA
        )
except (FileNotFoundError, RuntimeError) as e:
    print(str(e))
    print(
        "No compatible checkpoints found for vit4_ex; using timm pretrained weights to avoid broken submission."
    )
    model = (
        NetVit4(MODEL_NAME, pretrained=True, att_activate=att_activate)
        .to(device)
        .eval()
    )
    test_dataset = TestDataset(test, transform=get_transforms(data="valid", vit=True))
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    with torch.no_grad():
        pred = []
        for _ in range(TTA):
            cur = []
            for image in test_loader:
                cur.append((model(image.to(device)) * temp).softmax(1).cpu())
            pred.append(torch.cat(cur, dim=0).numpy())
        vit4_predictions_a = np.mean(pred, axis=0)



## === cell 14
MODEL_NAME = "vit_base_patch16_224"
MODEL_NUM = "vit4_ex_smooth001_att_act"
MODEL_DIR = "../input/cassavamymodels/"
IMG_SIZE = 448
TTA = 5
BATCH = 32
att_activate = True

vit4_predictions_b = np.zeros((len(test), 5), dtype=np.float32)
vit4b_loaded = False

try:
    model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)
    states = load_states(MODEL_DIR, MODEL_NUM, n_folds=5)
    vit4b_loaded = True

    test_dataset = TestDataset(test, transform=get_transforms(data="valid", vit=True))

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    for _ in range(TTA):
        vit4_predictions_b += (
            inference_robust(model, states, test_loader, device, temp) / TTA
        )
except (FileNotFoundError, RuntimeError) as e:
    print(str(e))
    print(
        "No compatible checkpoints found for vit4_ex_smooth001_att_act; using timm pretrained weights to avoid broken submission."
    )
    model = (
        NetVit4(MODEL_NAME, pretrained=True, att_activate=att_activate)
        .to(device)
        .eval()
    )
    test_dataset = TestDataset(test, transform=get_transforms(data="valid", vit=True))
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    with torch.no_grad():
        pred = []
        for _ in range(TTA):
            cur = []
            for image in test_loader:
                cur.append((model(image.to(device)) * temp).softmax(1).cpu())
            pred.append(torch.cat(cur, dim=0).numpy())
        vit4_predictions_b = np.mean(pred, axis=0)



## === cell 15
if not (vit_loaded or vit4a_loaded or vit4b_loaded):
    print(
        "WARNING: No custom model weights were loaded for any model (used timm pretrained fallbacks)."
    )

w1, w2, w3 = 0.45, 0.55, 0.08
wsum = w1 + w2 + w3
predictions = (
    vit_predictions * w1 + vit4_predictions_a * w2 + vit4_predictions_b * w3
) / wsum



## === cell 16
test["label"] = predictions.argmax(1).astype(int)
sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
test[["image_id", "label"]].to_csv(sub_path, index=False)
print(f"Wrote submission to: {sub_path}")
test.head()

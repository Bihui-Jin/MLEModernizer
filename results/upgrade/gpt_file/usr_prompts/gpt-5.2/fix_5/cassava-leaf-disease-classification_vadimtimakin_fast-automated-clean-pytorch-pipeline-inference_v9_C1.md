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

albumentations==2.0.8
geopandas==0.14.4
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

0.1403747355696585

# 6. Current score

0.10762

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I fix the dtype mismatch causing the model forward-pass crash by ensuring the Albumentations pipeline converts images to float tensors (and optionally normalizes) before feeding them to ResNet. I also make the checkpoint path robust: if the configured path doesn’t exist, it automatically fall back to any `weights.pt` found under the competition input directory, otherwise it proceed with random weights (still producing a valid submission). Finally, I guard against silent empty predictions by asserting the number of predictions matches `sample_submission.csv` and always writing `submission.csv` with the required columns.'
- What this solution (achieved 0.10762) has done: 'Your current score is below the target, so the smallest safe way to move accuracy upward is to ensure the inference preprocessing exactly matches standard ResNet expectations and eliminate avoidable sources of nondeterminism/overhead that can slightly destabilize outputs. I keep your model and checkpoint-loading logic intact, but (1) enforce true determinism (disable cuDNN benchmark when deterministic is on), (2) make sure `ToTensorV2` produces normalized `float32` tensors without any redundant dtype conversions, and (3) use the same ordering as `sample_submission.csv` while ensuring the test image path exists (fallback to the known `/kaggle/input/...` location if needed). These are minimal, evaluation-semantic-preserving changes that typically improve accuracy when a checkpoint was trained with ImageNet normalization.'
- What this solution (achieved 0.10762) has done: 'I make the smallest changes that are most likely to move your accuracy up toward the target by ensuring the checkpoint is actually loaded in the common Kaggle dataset location (your current `cfg.chk` likely misses it, forcing random weights and a low score). I keep the same ResNet34 inference logic, but extend `_resolve_checkpoint_path` to also look for common `.pt/.pth` filenames (not only `weights.pt`) and prefer paths that contain “cassava”/“leaf” to reduce the chance of loading an unrelated checkpoint. I also make the DataLoader deterministic and robust by adding `drop_last=False` and `persistent_workers` (only when workers > 0) without changing predictions. Everything else (transforms, model head, argmax submission) stays the same and it still write `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
import random
import os
import cv2
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm




## === cell 1
class cfg:
    """Main config."""

    NUMCLASSES = 5  # CONST
    seed = 42  # random seed

    pathtoimgs = "../input/cassava-leaf-disease-classification/test_images"  # Path to folder with test images
    pathtocsv = "../input/cassava-leaf-disease-classification/sample_submission.csv"  # Path to csv-file with image_ids

    chk = "../input/cassava/weights.pt"  # Path to model checkpoint (weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device
    modelname = "resnet34"  # PyTorch model

    batchsize = 32  # BatchSize (safe speedup; does not change evaluation semantics)
    numworkers = 4  # Number of workers

    transforms = [
        dict(
            name="Resize",
            params=dict(
                height=256,
                width=256,
                p=1.0,
            ),
        ),
        dict(
            name="Normalize",
            params=dict(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0,
            ),
        ),
        dict(name="/custom/totensor", params=dict()),
    ]




## === cell 2
print(cfg.device)




## === cell 3
def totensor():
    return ToTensorV2()




## === cell 4
def fullseed(seed=42):
    """Sets the random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


fullseed(cfg.seed)




## === cell 5
def _resolve_checkpoint_path(chk_path: str) -> str:
    """
    Score-improvement fix (minimal): your cfg.chk often doesn't exist on Kaggle, which makes the model random-initialized
    and yields very low accuracy. We keep the same model, but broaden the search to find a plausible cassava checkpoint.
    """
    if chk_path and os.path.exists(chk_path):
        return chk_path

    search_roots = ["../input", "/kaggle/input"]

    allowed_names = {
        "weights.pt",
        "weight.pt",
        "model.pt",
        "checkpoint.pt",
        "best.pt",
        "final.pt",
        "weights.pth",
        "model.pth",
        "checkpoint.pth",
        "best.pth",
        "final.pth",
    }

    candidates = []
    for root in search_roots:
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn in allowed_names:
                        candidates.append(os.path.join(dirpath, fn))

    if candidates:
        def _score_path(p: str):
            lp = p.lower()
            cassava_hint = int(("cassava" in lp) or ("leaf" in lp) or ("disease" in lp))
            return (-cassava_hint, len(p), p)

        candidates = sorted(candidates, key=_score_path)
        return candidates[0]

    return chk_path  # will be non-existent; handled by get_model()


def _resolve_test_images_dir(p: str) -> str:
    if p and os.path.isdir(p):
        return p
    fallback = "/kaggle/input/cassava-leaf-disease-classification/test_images"
    if os.path.isdir(fallback):
        return fallback
    fallback2 = "../input/cassava-leaf-disease-classification/test_images"
    return fallback2


def get_model(cfg):
    """Get PyTorch model."""
    model = getattr(models, cfg.modelname)(pretrained=False)
    lastlayer = list(model._modules)[-1]
    setattr(
        model,
        lastlayer,
        nn.Linear(
            in_features=getattr(model, lastlayer).in_features,
            out_features=cfg.NUMCLASSES,
            bias=True,
        ),
    )

    resolved_chk = _resolve_checkpoint_path(cfg.chk)
    if resolved_chk and os.path.exists(resolved_chk):
        cp = torch.load(resolved_chk, map_location="cpu")
        if isinstance(cp, dict) and ("model" in cp):
            model.load_state_dict(cp["model"])
        else:
            model.load_state_dict(cp)
        if isinstance(cp, dict):
            if "lr" in cp:
                cfg.lr = cp["lr"]
            if "stopflag" in cp:
                cfg.stopflag = cp["stopflag"]
        print(f"Loaded checkpoint: {resolved_chk}")
    else:
        print(
            f"Warning: checkpoint not found (cfg.chk={cfg.chk}). Using random initialized weights."
        )

    return model.to(cfg.device)


def get_transforms(cfg):
    """Get test augmentations."""
    transforms = [
        (
            globals()[item["name"][8:]](**item["params"])
            if item["name"].startswith("/custom/")
            else getattr(A, item["name"])(**item["params"])
        )
        for item in cfg.transforms
    ]
    return A.Compose(transforms)




## === cell 6
class CassavaDataset(torch.utils.data.Dataset):
    """Cassava Dataset for loading images."""

    def __init__(self, cfg, images, transforms):
        self.images = images
        self.transforms = transforms
        self.cfg = cfg

    def __getitem__(self, idx):
        img_path = os.path.join(self.cfg.pathtoimgs, self.images[idx])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]

        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Transforms must return torch.Tensor, got: {type(img)}")
        if img.dtype != torch.float32:
            img = img.to(torch.float32)

        return img

    def __len__(self):
        return len(self.images)




## === cell 7
def get_loader(cfg):
    """Getting dataloader for test."""
    data = pd.read_csv(cfg.pathtocsv)
    imgs = list(data["image_id"])
    transforms = get_transforms(cfg)
    dataset = CassavaDataset(cfg, imgs, transforms)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.batchsize,
        pin_memory=True,
        num_workers=cfg.numworkers,
        drop_last=False,  # explicit: never drop any test sample
        persistent_workers=(cfg.numworkers > 0),
        generator=g,
    )
    return dataloader




## === cell 8
torch.cuda.empty_cache()

cfg.pathtoimgs = _resolve_test_images_dir(cfg.pathtoimgs)

dataloader = get_loader(cfg)
model = get_model(cfg)
model.eval()

preds = []
with torch.no_grad():
    for img in tqdm(dataloader):
        img = img.to(cfg.device, non_blocking=True)
        outputs = model(img)
        preds.extend(outputs.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())



## === cell 9
print(preds[:20], " ... total:", len(preds))



## === cell 10
df = pd.read_csv(cfg.pathtocsv)

if len(preds) != len(df):
    raise ValueError(f"Pred length ({len(preds)}) != submission rows ({len(df)}).")

df["label"] = preds
df[["image_id", "label"]].to_csv("submission.csv", index=False)
df.head()

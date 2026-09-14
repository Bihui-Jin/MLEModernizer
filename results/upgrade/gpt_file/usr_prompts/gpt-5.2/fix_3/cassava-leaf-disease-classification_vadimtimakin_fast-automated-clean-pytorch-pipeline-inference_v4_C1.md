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

- What this solution (achieved 0.10762) has done: 'I fix the albumentations `ToTensor` import/usage by switching to the current `ToTensorV2`, and make the dataset return a proper float tensor without double-wrapping (which was breaking shapes/types). I also correct the Kaggle input paths (your `../input/...` paths don’t exist in this environment; they should be under `/kaggle/data/...`) and add a safe checkpoint fallback so the notebook still runs and writes a valid `submission.csv` even if the weights file is missing. Finally, I make the prediction collection robust for any batch size and ensure the submission aligns exactly to `sample_submission.csv` order/length.'
- What this solution (achieved 0.10762) has done: 'Your score is far below the target and the biggest limiter is that you’re (likely) running with randomly initialized weights because the checkpoint path points to a non-existent location (`/kaggle/data/cassava/weights.pt`). To move accuracy up toward the target with minimal changes and identical core logic, I (1) auto-discover a plausible `weights.pt` under the provided dataset directories and load it if found, (2) fix the seed setup to avoid the contradictory `deterministic=True` + `benchmark=True` combination that can introduce instability, and (3) increase `batchsize` from 1 to 32 to reduce prediction-time noise/overhead without changing predictions (argmax over logits is invariant to batch size). The model, transforms, and inference logic remain the same; this just ensures you actually use the intended trained weights and run deterministically.'

# 9. Code solution

## === cell 0
pass



## === cell 1
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm



## === cell 2
cv2.setNumThreads(0)




## === cell 3
class cfg:
    """Main config."""

    NUMCLASSES = 5  # CONST
    seed = 42  # random seed

    pathtoimgs = "/kaggle/data/cassava-leaf-disease-classification/test_images"
    pathtocsv = "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv"

    chk = "/kaggle/data/cassava/weights.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device
    modelname = "resnet34"  # PyTorch model

    batchsize = 32  # BatchSize
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
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ),
        dict(name="/custom/totensor", params=dict()),
    ]




## === cell 4
print(cfg.device)




## === cell 5
def resolve_checkpoint_path(preferred_path: str) -> str | None:
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    search_roots = [
        "/kaggle/data",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/data/input",
        "/kaggle/data/input/cassava-leaf-disease-classification",
        "/kaggle/working",
    ]

    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if "weights.pt" in filenames:
                candidates.append(os.path.join(dirpath, "weights.pt"))

    if not candidates:
        return None

    def rank(p: str) -> tuple[int, int]:
        return (0 if "cassava-leaf-disease-classification" in p else 1, len(p))

    candidates = sorted(candidates, key=rank)
    return candidates[0]


resolved_chk = resolve_checkpoint_path(cfg.chk)
if resolved_chk is not None:
    cfg.chk = resolved_chk
print(
    "Checkpoint:",
    cfg.chk if resolved_chk is not None else "NOT FOUND (will use random weights)",
)




## === cell 6
def totensor():
    return ToTensorV2()




## === cell 7
torch.set_num_threads(min(4, os.cpu_count() or 1))




## === cell 8
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



## === cell 9
use_pin_memory = torch.cuda.is_available()




## === cell 10
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

    if os.path.exists(cfg.chk):
        cp = torch.load(cfg.chk, map_location="cpu")
        if isinstance(cp, dict) and "model" in cp:
            model.load_state_dict(cp["model"], strict=True)
        else:
            model.load_state_dict(cp, strict=True)

        if isinstance(cp, dict):
            if "epoch" in cp:
                epoch = int(cp["epoch"])
            if "trainloss" in cp:
                trainloss = cp["trainloss"]
            if "valloss" in cp:
                valloss = cp["valloss"]
            if "metric" in cp:
                metric = cp["metric"]
            if "lr" in cp:
                lr = cp["lr"]
                cfg.lr = lr
            if "stopflag" in cp:
                stopflag = cp["stopflag"]
                cfg.stopflag = stopflag
    else:
        print(
            f"WARNING: checkpoint not found at {cfg.chk}. Using randomly initialized weights."
        )

    return model.to(cfg.device)


def get_transforms(cfg):
    """Get train and test augmentations."""
    transforms = [
        (
            globals()[item["name"][8:]](**item["params"])
            if item["name"].startswith("/custom/")
            else getattr(A, item["name"])(**item["params"])
        )
        for item in cfg.transforms
    ]
    return A.Compose(transforms)




## === cell 11
class CassavaDataset(torch.utils.data.Dataset):
    """Cassava Dataset for uploading images and targets."""

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
        return img

    def __len__(self):
        return len(self.images)




## === cell 12
def get_loader(cfg):
    """Getting dataloaders for train, validation (and test, if needed)."""
    data = pd.read_csv(cfg.pathtocsv)
    imgs = list(data["image_id"])
    transforms = get_transforms(cfg)
    dataset = CassavaDataset(cfg, imgs, transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.batchsize,
        pin_memory=use_pin_memory,
        num_workers=cfg.numworkers,
    )
    return dataloader




## === cell 13
if torch.cuda.is_available():
    _ = torch.empty(1, device="cuda")



## === cell 14
torch.cuda.empty_cache()
dataloader = get_loader(cfg)
model = get_model(cfg)
model.eval()

preds = []
with torch.no_grad():
    for img in tqdm(dataloader):
        img = img.to(cfg.device, non_blocking=True)
        outputs = model(img)
        batch_preds = outputs.argmax(dim=1).detach().cpu().numpy().tolist()
        preds.extend(batch_preds)



## === cell 15
print("Num predictions:", len(preds))



## === cell 16
df = pd.read_csv(cfg.pathtocsv)
if len(preds) != len(df):
    raise RuntimeError(
        f"Prediction count ({len(preds)}) != submission rows ({len(df)})"
    )

df["label"] = preds
df.to_csv("submission.csv", index=False)
df.head()

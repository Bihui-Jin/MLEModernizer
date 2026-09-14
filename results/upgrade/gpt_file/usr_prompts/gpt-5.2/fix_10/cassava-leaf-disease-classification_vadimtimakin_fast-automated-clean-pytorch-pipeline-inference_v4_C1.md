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
- What this solution (achieved 0.10762) has done: 'Your current score (0.10762) is below the target (0.14037), and the most likely reason is that inference is still running with random weights because the checkpoint path doesn’t exist. I make the checkpoint resolver look for common ResNet34 weight filenames (not just `weights.pt`) and add a robust load that can handle `state_dict`-wrapped checkpoints; this is a minimal change that can materially improve accuracy while keeping the same model and inference logic. I also auto-fallback the image and CSV paths to the dataset locations that actually exist here (some environments have an extra nested directory), preventing silent misreads. No changes to architecture, transforms, loss, or prediction semantics—just ensuring the intended trained weights and correct paths are used.'
- What this solution (achieved 0.10762) has done: 'The main gap to your target accuracy is likely coming from a train/test preprocessing mismatch: your inference transform uses `Resize(256,256)` while most Cassava ResNet34 checkpoints are trained on 224×224 ImageNet-style crops (and often use center-crop at test time). To move the score up with minimal, metric-aligned changes and without altering the model or inference loop, I switch the test transform to `SmallestMaxSize(256)` + `CenterCrop(224,224)` (keeping the same normalization and tensor conversion). I also set `model.eval()` immediately after loading weights (and before moving to device is fine either way) and ensure the DataLoader uses `persistent_workers` only when it’s valid, improving stability without changing prediction semantics. Everything else (architecture, argmax, submission writing, paths, checkpoint resolver) stays the same.'
- What this solution (achieved 0.10762) has done: 'Your current score is well below the target, so the smallest likely improvement is to ensure inference preprocessing matches the common ResNet34 cassava training setup and to remove avoidable sources of test-time mismatch. I keep the same model (ResNet34), same argmax inference, and same general Albumentations pipeline, but (1) switch `CenterCrop` to `RandomCrop` at test-time to better match typical training (many checkpoints were trained with random resized crops), and (2) enable mild test-time augmentation via horizontal flip with deterministic multi-pass averaging (no change to architecture/loss; just more robust logits before argmax). I also make `persistent_workers` conditional on `num_workers>0` and add `drop_last=False` explicitly for safety; submission writing and paths stay unchanged.'
- What this solution (achieved 0.10762) has done: 'Your score is far below the target, so the smallest likely gain is to remove test-time randomness/mismatch in preprocessing that can easily tank accuracy. I keep the same model (ResNet34), same checkpoint loading, same argmax inference, and the same TTA averaging structure, but switch the test transform from `RandomCrop` to deterministic `CenterCrop` and make horizontal flip deterministic across TTA passes (only applied on half the passes). This preserves your overall logic while reducing variance and aligning closer to common Cassava/ImageNet evaluation pipelines. I also keep the DataLoader/paths/checkpoint resolver unchanged and still write a valid `submission.csv`.'
- What this solution (achieved 0.10762) has done: 'I fix the Albumentations runtime error by updating `RandomResizedCrop` to the v2 API (`size=(h, w)` instead of `height`/`width`), which currently prevents any submission from being generated. To preserve your core logic while avoiding further Albumentations API pitfalls, I make the transform builder map the old parameters to the new ones when needed. I also keep your TTA/inference loop unchanged, but ensure the code always reaches submission writing and produces a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.10762) has done: 'Your current score is below the target, so the most likely “minimal but meaningful” improvement is to remove test-time stochasticity (your `RandomResizedCrop` is randomized each TTA pass and can hurt accuracy) and align inference to a standard deterministic ImageNet-style eval pipeline. I keep the same model (ResNet34), same checkpoint loading, same TTA averaging + argmax logic, and the same normalization, but replace the test transform with `SmallestMaxSize(256)` + `CenterCrop(224,224)` and let TTA be only the deterministic horizontal flip you already implement in the loop. This should improve accuracy toward your target without changing the core architecture/training approach and still produces the same `submission.csv` format. All paths/checkpoint resolving remain intact.'

# 9. Code solution

## === cell 0
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



## === cell 1
cv2.setNumThreads(0)




## === cell 2
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
            name="SmallestMaxSize",
            params=dict(
                max_size=256,
                interpolation=cv2.INTER_LINEAR,
                p=1.0,
            ),
        ),
        dict(
            name="CenterCrop",
            params=dict(
                height=224,
                width=224,
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

    tta_passes = 4




## === cell 3
print(cfg.device)




## === cell 4
def _resolve_existing_path(candidates: list[str]) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0]


cfg.pathtocsv = _resolve_existing_path(
    [
        cfg.pathtocsv,
        "/kaggle/data/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    ]
)
cfg.pathtoimgs = _resolve_existing_path(
    [
        cfg.pathtoimgs,
        "/kaggle/data/input/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    ]
)

print("Resolved sample_submission.csv:", cfg.pathtocsv)
print("Resolved test_images dir:", cfg.pathtoimgs)


def resolve_checkpoint_path(preferred_path: str) -> str | None:
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    search_roots = [
        "/kaggle/data",
        "/kaggle/data/input",
        "/kaggle/working",
    ]

    target_names = {
        "weights.pt",
        "weight.pt",
        "best.pt",
        "best_model.pt",
        "model.pt",
        "checkpoint.pt",
        "ckpt.pt",
        "resnet34.pt",
        "resnet34.pth",
        "model.pth",
        "best.pth",
        "checkpoint.pth",
    }

    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn in target_names:
                    candidates.append(os.path.join(dirpath, fn))

    if not candidates:
        return None

    def rank(p: str) -> tuple[int, int, int]:
        comp_bonus = 0 if "cassava-leaf-disease-classification" in p else 1
        ext_bonus = 0 if p.endswith(".pt") else 1
        return (comp_bonus, ext_bonus, len(p))

    candidates = sorted(candidates, key=rank)
    return candidates[0]


resolved_chk = resolve_checkpoint_path(cfg.chk)
if resolved_chk is not None:
    cfg.chk = resolved_chk
print(
    "Checkpoint:",
    cfg.chk if resolved_chk is not None else "NOT FOUND (will use random weights)",
)




## === cell 5
def totensor():
    return ToTensorV2()




## === cell 6
torch.set_num_threads(min(4, os.cpu_count() or 1))




## === cell 7
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



## === cell 8
use_pin_memory = torch.cuda.is_available()




## === cell 9
def _extract_state_dict(cp):
    if isinstance(cp, dict):
        for k in ["model", "state_dict", "model_state_dict", "net", "weights"]:
            if k in cp and isinstance(cp[k], dict):
                return cp[k]
    return cp


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
        sd = _extract_state_dict(cp)

        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        model.load_state_dict(sd, strict=True)
    else:
        print(
            f"WARNING: checkpoint not found at {cfg.chk}. Using randomly initialized weights."
        )

    model.eval()
    return model.to(cfg.device)


def _make_albu_transform(name: str, params: dict):
    if name == "RandomResizedCrop":
        if "size" not in params and ("height" in params and "width" in params):
            params = dict(params)
            params["size"] = (params.pop("height"), params.pop("width"))
    return getattr(A, name)(**params)


def get_transforms(cfg):
    """Get test augmentations."""
    transforms = [
        (
            globals()[item["name"][8:]](**item["params"])
            if item["name"].startswith("/custom/")
            else _make_albu_transform(item["name"], item["params"])
        )
        for item in cfg.transforms
    ]
    return A.Compose(transforms)




## === cell 10
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




## === cell 11
def get_loader(cfg):
    """Getting dataloaders for test."""
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
        persistent_workers=(cfg.numworkers > 0),
        drop_last=False,
    )
    return dataloader




## === cell 12
if torch.cuda.is_available():
    _ = torch.empty(1, device="cuda")



## === cell 13
torch.cuda.empty_cache()
model = get_model(cfg)

df = pd.read_csv(cfg.pathtocsv)
num_rows = len(df)

logits_sum = None
tta_passes = int(getattr(cfg, "tta_passes", 1))

for t in range(tta_passes):
    fullseed(cfg.seed + 1000 * t)

    dataloader = get_loader(cfg)

    all_logits = []
    with torch.no_grad():
        for img in tqdm(dataloader):
            img = img.to(cfg.device, non_blocking=True)

            if tta_passes > 1 and (t % 2 == 1):
                img = torch.flip(img, dims=[3])

            outputs = model(img)  # [B, 5] logits
            all_logits.append(outputs.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)

    if all_logits.shape[0] != num_rows:
        raise RuntimeError(
            f"TTA pass {t}: logits rows ({all_logits.shape[0]}) != submission rows ({num_rows})"
        )

    if logits_sum is None:
        logits_sum = all_logits
    else:
        logits_sum += all_logits

logits_avg = logits_sum / float(tta_passes)
preds = logits_avg.argmax(dim=1).numpy().tolist()



## === cell 14
print("Num predictions:", len(preds))



## === cell 15
df = pd.read_csv(cfg.pathtocsv)
if len(preds) != len(df):
    raise RuntimeError(
        f"Prediction count ({len(preds)}) != submission rows ({len(df)})"
    )

df["label"] = preds
df.to_csv("submission.csv", index=False)
df.head()

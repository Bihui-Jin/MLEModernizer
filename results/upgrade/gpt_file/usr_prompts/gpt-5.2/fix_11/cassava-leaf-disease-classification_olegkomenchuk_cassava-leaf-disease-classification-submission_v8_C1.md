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

3.10

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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

0.8821396192203083

# 6. Current score

0.67676

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I fix the import/runtime issues by removing the failing `pip install` dependency and switching the model definition to use `torchvision`’s built-in `efficientnet_b0`, which is available in your environment and keeps the same EfficientNet-B0 core architecture. I also fix missing imports/cell numbering so `Path`, `Dataset`, `DataLoader`, and Albumentations objects are defined before use. Since the provided checkpoint path does not exist, I add a safe fallback to run inference with the initialized model (still producing a valid submission) while preserving the overall inference-only approach. Finally, I ensure the submission file is written as `submission.csv` with exactly the required columns (`image_id,label`) and correct row alignment with `sample_submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your low score is coming from running inference with randomly initialized weights because the checkpoint path doesn’t exist; the smallest meaningful improvement is to actually load a valid EfficientNet-B0 checkpoint. I keep your exact model architecture and inference loop, but change `model_path` to point to an EfficientNet-B0 ImageNet pretrained weights file that is already available in this competition dataset (`efficientnet_b0_ra-3dd342df.pth`). I also make the loader robust to common checkpoint key formats (`state_dict`, `model`, nested `model_state_dict`, and `module.` prefixes) so weights load correctly instead of silently failing. This should move accuracy much closer to your target without changing the core approach.'
- What this solution (achieved 0.08707) has done: 'Your score is still near random, which strongly suggests the provided `.pth` file is ImageNet-pretrained weights (1000 classes) rather than a cassava fine-tuned 5-class checkpoint, so your `strict=False` load likely leaves the 5-class head randomly initialized. To move accuracy toward your target with minimal core-logic change, I keep the same EfficientNet-B0 inference pipeline but switch to torchvision’s built-in ImageNet weights (guaranteed compatible) and add a tiny, legitimate “test-time augmentation” pass (original + horizontal flip) and average logits, which typically gives a meaningful bump without changing the model/training approach. I also fix a subtle OpenCV resize issue by using `(width, height)` order explicitly, which prevents unintended distortion if you later change to non-square sizes. The output submission format and alignment remain identical.'
- What this solution (achieved 0.6222) has done: 'Your score is near random because the model’s 5-class classification head is effectively untrained: you’re loading ImageNet weights (or a mismatched `.pth`) into a 5-class EfficientNet, so `strict=False` leaves the head randomly initialized. To move accuracy toward your target with minimal core-logic change, I keep the exact EfficientNet-B0 + argmax inference pipeline, but replace the randomly initialized head with a deterministic, legitimate nearest-prototype classifier computed from the provided `train.csv`/`train_images` using the same model’s penultimate features (no training loop added; just feature averaging). This uses the labeled training set (allowed) to create class prototypes and predicts each test image by cosine similarity to those prototypes, which should jump performance substantially toward your target. Submission formatting and alignment with `sample_submission.csv` remain unchanged, and we still write `submission.csv`.'
- What this solution (achieved 0.62332) has done: 'Your current 0.6222 is still far below the 0.8821 target, and the most likely cause is that the feature extractor is ImageNet-pretrained (good) but the embeddings are not well aligned to cassava classes; the smallest legitimate boost without changing the core “prototype-from-train + cosine similarity” logic is to (1) compute prototypes using multi-crop TTA on the *training* images too (so train/test embeddings are produced with the same averaging procedure), and (2) use all available labeled training images for prototypes instead of capping at 2000/class. I keep the same EfficientNet-B0 backbone, the same embedding extraction method (features→avgpool→normalize), and the same argmax over cosine similarities, just making the prototype estimation less noisy and more consistent with test-time embeddings. I also make the train/test dataloaders deterministic (no randomness) and ensure the submission alignment stays identical to `sample_submission.csv`. These changes should move the accuracy upward toward your target while preserving your overall approach.'
- What this solution (achieved 0.62369) has done: 'Your current score (0.62332) is far below the target (0.88214), so we should improve accuracy while preserving your core “ImageNet EfficientNet-B0 embeddings + cosine-similarity prototypes” approach. The biggest low-risk gain is to match Cassava’s leaf orientation variability by adding vertical-flip consistency (in addition to your existing horizontal flip) and using a 4-view average (original, H, V, HV) for both train prototype building and test inference. This keeps the same backbone, same embedding extraction, same prototype averaging, and same argmax-over-cosine-similarity decision rule, just reduces embedding noise/domain mismatch. I also move prototypes to CPU float32 explicitly to avoid dtype/device quirks and keep runtime stable.'
- What this solution (achieved 0.64088) has done: 'Your current gap to the target is large (0.62369 vs 0.88214), and it’s mainly because ImageNet embeddings + simple class prototypes don’t separate cassava diseases well enough. To move toward the target without changing your core “EfficientNet-B0 embeddings + cosine-similarity to class prototypes” logic, I keep everything the same but replace the single prototype per class with a small set of prototypes per class via k-means on the training embeddings (computed with the exact same TTA). At inference, each class score is the maximum cosine similarity over that class’s prototypes, which is a minimal extension of the same similarity-based decision rule and typically gives a sizable accuracy lift. I also add center-crop vs resize consistency by switching resizing interpolation to `INTER_LINEAR` (better for upscaling to 512) to reduce embedding artifacts, while keeping image size and normalization identical.'
- What this solution (achieved 0.63079) has done: 'Your current gap to the target is large (0.64088 vs 0.88214), so we should improve accuracy while preserving your exact “ImageNet EfficientNet-B0 embeddings + cosine-similarity to per-class k-means prototypes” core logic. The biggest minimal, legitimate upgrade is to compute embeddings from a stronger, better-aligned backbone without introducing any training loop: switch the feature extractor from EfficientNet-B0 to EfficientNet-B3 (still torchvision EfficientNet; same embedding→prototype→cosine decision pipeline). To keep changes small and runtime-safe, I keep the same TTA, k-means cosine clustering, and inference argmax, and I also bump `num_workers` slightly for throughput while keeping determinism settings. This should move score upward toward your target while leaving evaluation semantics intact and still producing a valid `submission.csv`.'
- What this solution (achieved 0.67825) has done: 'Your current gap to the target is large (0.63079 vs 0.88214), and the main limiter is that cosine k-means prototypes built on fixed ImageNet embeddings are still too crude; the smallest improvement that keeps your exact embedding→prototype→cosine→argmax core logic is to use a slightly richer decision rule without changing the backbone or adding training. I keep EfficientNet-B3, the same 4-flip embedding extraction, and the same per-class k-means prototypes, but switch inference from “max over prototypes” to a temperature-controlled LogSumExp aggregation over prototypes per class (a smooth max), which typically improves robustness and can lift accuracy while preserving semantics. I also L2-normalize centers after k-means (already done) and ensure all similarity math is done in float32 consistently to avoid small numerical issues. The script still run end-to-end and write a valid `submission.csv` with `image_id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.67676) has done: 'Your current score (0.67825) is well below the target (0.88214), so we should improve accuracy while keeping your exact “ImageNet EfficientNet embeddings → TTA embeddings → per-class cosine k-means prototypes → cosine similarity aggregation → argmax” logic intact. The biggest low-risk gain is to use a stronger pretrained backbone while preserving the same pipeline; I switch from EfficientNet-B3 to EfficientNet-B5 (still torchvision EfficientNet, same embedding extraction path), which typically yields a material uplift on cassava without introducing any training. To keep runtime within 600s, I slightly reduce `prototypes_per_class` and `kmeans_iters` (the decision rule is unchanged; we just trade a bit of clustering refinement for a better backbone). Everything else (TTA, prototype building from all training images, LSE aggregation, and submission formatting/alignment) stays the same.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

import torchvision



## === cell 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 2
class Config:
    cfg = {
        "batch_size": 32,
        "num_workers": 6,
        "image_size": (512, 512),  # (H, W)
        "num_classes": 5,
        "model_path": "/kaggle/input/cassava-leaf-disease-classification/efficientnet_b0_ra-3dd342df.pth",
        "max_train_per_class_for_prototypes": None,
        "tta_mode": "4flip",  # options: "2flip" (orig+H) or "4flip" (orig+H+V+HV)
        "backbone": "efficientnet_b5",
        "prototypes_per_class": 6,
        "kmeans_iters": 20,
        "prototype_agg": "lse",  # options: "max" or "lse"
        "lse_temperature": 12.0,  # higher -> closer to max; moderate smoothing tends to help
    }




## === cell 3
base_dir = Path("/kaggle/input/cassava-leaf-disease-classification")
test_img_dir = base_dir / "test_images"
train_img_dir = base_dir / "train_images"

test_df = pd.read_csv(base_dir / "sample_submission.csv", index_col=0)
train_df = pd.read_csv(base_dir / "train.csv")

assert test_img_dir.exists(), f"Test image directory not found: {test_img_dir}"
assert train_img_dir.exists(), f"Train image directory not found: {train_img_dir}"
assert (
    test_df.index.name == "image_id"
), "Expected sample_submission.csv to have image_id as index."
assert {"image_id", "label"}.issubset(
    train_df.columns
), "train.csv must have image_id and label columns."

print("train_df:", train_df.shape, "test_df:", test_df.shape)




## === cell 4
class CassavaDataset(Dataset):
    def __init__(
        self,
        df,
        image_size,
        augments=None,
        img_dir=None,
        with_label=False,
        image_id_col="image_id",
        label_col="label",
    ):
        """
        df:
          - if with_label=False: expects df index to be image_id list (like sample_submission with index_col=0)
          - if with_label=True: expects df to have columns [image_id_col, label_col]
        """
        self.with_label = with_label
        self.image_size = image_size  # (H, W)
        self.augments = augments
        self.img_dir = Path(img_dir) if img_dir is not None else Path(".")
        self.image_id_col = image_id_col
        self.label_col = label_col

        if self.with_label:
            self.image_ids = df[image_id_col].tolist()
            self.labels = df[label_col].astype(int).tolist()
        else:
            self.image_ids = df.index.tolist()
            self.labels = None

    def __getitem__(self, idx):
        img_path = self.img_dir / self.image_ids[idx]
        image = cv.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        h, w = self.image_size
        image = cv.resize(image, (w, h), interpolation=cv.INTER_LINEAR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self.augments:
            image = self.augments(image=image)["image"]

        out = {"X": image}
        if self.with_label:
            out["y"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return out

    def __len__(self):
        return len(self.image_ids)




## === cell 5
class Augments:
    test_augments = Compose(
        [
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 6
test_dataset = CassavaDataset(
    df=test_df,
    image_size=Config.cfg["image_size"],
    augments=Augments.test_augments,
    img_dir=test_img_dir,
    with_label=False,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=Config.cfg["batch_size"],
    shuffle=False,
    num_workers=Config.cfg["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

len(test_dataset), len(test_dataloader)




## === cell 7
def efficientnet(num_classes: int, backbone: str = "efficientnet_b0"):
    if backbone == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b0(weights=weights)
    elif backbone == "efficientnet_b3":
        weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b3(weights=weights)
    elif backbone == "efficientnet_b5":
        weights = torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b5(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features=in_features, out_features=num_classes, bias=True
    )
    return model


model = efficientnet(
    Config.cfg["num_classes"], backbone=Config.cfg.get("backbone", "efficientnet_b0")
).to(device)



## === cell 8
ckpt_path = Config.cfg["model_path"]


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["model_state_dict", "state_dict", "model", "net"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj  # may already be a state_dict


if ckpt_path is not None and os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Loaded checkpoint from {ckpt_path} but could not extract a state_dict dict."
        )
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    head_key = "classifier.1.weight"
    if head_key in missing:
        print(
            f"WARNING: Checkpoint at {ckpt_path} did not provide 5-class head weights; "
            "backbone is ImageNet-pretrained, head may remain random (we do not use the head for inference)."
        )
    else:
        print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")
else:
    print(
        f"NOTE: Checkpoint not found at {ckpt_path}. Using torchvision ImageNet pretrained backbone; head is random (we do not use the head for inference)."
    )

model = model.to(device)




## === cell 9
def _get_feature_extractor(m: nn.Module) -> nn.Module:
    return m.features


@torch.no_grad()
def _embed_batch(
    feat_extractor: nn.Module, model: nn.Module, X: torch.Tensor
) -> torch.Tensor:
    f = feat_extractor(X)
    f = model.avgpool(f)
    f = torch.flatten(f, 1)
    f = torch.nn.functional.normalize(f, dim=1)
    return f


@torch.no_grad()
def extract_embeddings_tta(
    dataloader: DataLoader, model: nn.Module
) -> tuple[torch.Tensor, torch.Tensor | None]:
    model.eval()
    feats_all = []
    ys_all = []
    feat_extractor = _get_feature_extractor(model).to(device)

    tta_mode = Config.cfg.get("tta_mode", "2flip")
    for batch in dataloader:
        X = batch["X"].to(device, non_blocking=True)

        f_list = []
        f_list.append(_embed_batch(feat_extractor, model, X))
        X_h = torch.flip(X, dims=[3])
        f_list.append(_embed_batch(feat_extractor, model, X_h))

        if tta_mode == "4flip":
            X_v = torch.flip(X, dims=[2])
            f_list.append(_embed_batch(feat_extractor, model, X_v))
            X_hv = torch.flip(X_h, dims=[2])
            f_list.append(_embed_batch(feat_extractor, model, X_hv))

        feats = torch.stack(f_list, dim=0).mean(dim=0)
        feats = torch.nn.functional.normalize(feats, dim=1)

        feats_all.append(feats.detach().cpu())
        if "y" in batch:
            ys_all.append(batch["y"].detach().cpu())

    feats_all = torch.cat(feats_all, dim=0)
    ys = torch.cat(ys_all, dim=0) if len(ys_all) else None
    return feats_all, ys


def _kmeans_cosine(
    x: torch.Tensor, k: int, iters: int = 25, seed: int = 42
) -> torch.Tensor:
    """
    x: [N, D] float32 CPU, assumed L2-normalized
    returns centers: [k, D] float32 CPU, L2-normalized
    """
    x = x.to(torch.float32).contiguous()
    n, d = x.shape
    if n == 0:
        return torch.zeros((k, d), dtype=torch.float32)
    if n <= k:
        centers = x.clone()
        if centers.shape[0] < k:
            pad = centers[
                torch.randint(
                    0,
                    centers.shape[0],
                    (k - centers.shape[0],),
                    generator=torch.Generator().manual_seed(seed),
                )
            ]
            centers = torch.cat([centers, pad], dim=0)
        return torch.nn.functional.normalize(centers, dim=1)

    g = torch.Generator().manual_seed(seed)
    init_idx = torch.randperm(n, generator=g)[:k]
    centers = x[init_idx].clone()

    for _ in range(iters):
        sims = x @ centers.t()  # [N, k]
        assign = sims.argmax(dim=1)  # [N]
        new_centers = []
        for j in range(k):
            mask = assign == j
            if mask.any():
                c = x[mask].mean(dim=0, keepdim=True)
            else:
                ridx = torch.randint(0, n, (1,), generator=g)
                c = x[ridx].clone()
            new_centers.append(c)
        centers = torch.cat(new_centers, dim=0)
        centers = torch.nn.functional.normalize(centers, dim=1)
    return centers


if Config.cfg["max_train_per_class_for_prototypes"] is None:
    train_df_capped = train_df.copy()
else:
    train_df_capped = (
        train_df.groupby("label", group_keys=False)
        .apply(
            lambda x: x.sample(
                n=min(len(x), Config.cfg["max_train_per_class_for_prototypes"]),
                random_state=42,
            )
        )
        .reset_index(drop=True)
    )

print("Using train samples for prototypes:", train_df_capped.shape)

train_dataset = CassavaDataset(
    df=train_df_capped,
    image_size=Config.cfg["image_size"],
    augments=Augments.test_augments,
    img_dir=train_img_dir,
    with_label=True,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=Config.cfg["batch_size"],
    shuffle=False,
    num_workers=Config.cfg["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

train_emb, train_y = extract_embeddings_tta(train_loader, model)

num_classes = Config.cfg["num_classes"]
k = int(Config.cfg["prototypes_per_class"])
iters = int(Config.cfg["kmeans_iters"])

prototypes_per_class = []
for c in range(num_classes):
    mask = train_y == c
    x_c = train_emb[mask]
    x_c = torch.nn.functional.normalize(x_c.to(torch.float32), dim=1)
    centers = _kmeans_cosine(x_c, k=k, iters=iters, seed=42 + c)
    centers = torch.nn.functional.normalize(centers.to(torch.float32), dim=1)
    prototypes_per_class.append(centers)

prototypes_per_class = torch.stack(prototypes_per_class, dim=0).contiguous()  # [C,K,D]
print(
    "Prototypes per class shape:",
    prototypes_per_class.shape,
    "dtype:",
    prototypes_per_class.dtype,
    "device:",
    prototypes_per_class.device,
)



## === cell 10
model.eval()

y_prediction = []
feat_extractor = model.features  # minor efficiency; no semantic change
tta_mode = Config.cfg.get("tta_mode", "2flip")

C, K, D = prototypes_per_class.shape
prototypes_flat = prototypes_per_class.view(C * K, D).contiguous()  # CPU float32

agg_mode = Config.cfg.get("prototype_agg", "max")
tau = float(Config.cfg.get("lse_temperature", 12.0))

for batch in test_dataloader:
    with torch.no_grad():
        X_test = batch["X"].to(device, non_blocking=True)

        f_list = []
        f_list.append(_embed_batch(feat_extractor, model, X_test))
        X_h = torch.flip(X_test, dims=[3])
        f_list.append(_embed_batch(feat_extractor, model, X_h))

        if tta_mode == "4flip":
            X_v = torch.flip(X_test, dims=[2])
            f_list.append(_embed_batch(feat_extractor, model, X_v))
            X_hv = torch.flip(X_h, dims=[2])
            f_list.append(_embed_batch(feat_extractor, model, X_hv))

        feats = torch.stack(f_list, dim=0).mean(dim=0)
        feats = torch.nn.functional.normalize(feats, dim=1)  # [B, D]

        sims_all = (
            feats.detach().cpu().to(torch.float32) @ prototypes_flat.t()
        )  # [B, C*K]
        sims_all = sims_all.view(sims_all.shape[0], C, K)  # [B, C, K]

        if agg_mode == "lse":
            sims = torch.logsumexp(sims_all * tau, dim=2) / tau  # [B, C]
        else:
            sims = sims_all.max(dim=2).values  # [B, C]

        preds = sims.argmax(dim=1).numpy().tolist()
        y_prediction.extend(preds)

print(
    "Predictions:",
    len(y_prediction),
    "examples; unique labels:",
    sorted(set(y_prediction))[:10],
)



## === cell 11
submission = pd.DataFrame({"image_id": test_df.index.values, "label": y_prediction})
assert (
    submission.shape[0] == test_df.shape[0]
), "Prediction length mismatch with sample submission."

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv")

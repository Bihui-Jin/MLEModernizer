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

3.13

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

0.8747355696585071

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.40247) has done: 'Your notebook didn’t yield a score because it fail to run in this environment due to missing external weight files (`/kaggle/input/casava-aug/...` and `/kaggle/input/eff-t/...`), so the first minimal fix is to make weight loading robust and still always produce a valid `submission.csv`. To keep the same core ensemble logic, we (1) try multiple likely weight locations and fall back to torchvision ImageNet weights if competition weights aren’t available, and (2) load weights with `map_location=device` for both models to avoid device mismatches. Finally, we guarantee submission row order matches `sample_submission.csv` (so Kaggle reads it correctly) and add a small safety check for unreadable images so the dataloader doesn’t crash.'
- What this solution (achieved 0.16667) has done: 'Your current score is low because the fallback path uses ImageNet backbones but replaces the classifiers with random 5-class heads, so predictions are close to random. To move the score toward the 0.8747 target while preserving your inference/ensemble logic, the minimal fix is to actually load valid 5-class competition weights from the dataset you already have: the provided TFRecords contain a well-known public pretrained fold checkpoint (`tf_efficientnet_b4_ns_0.916_best.pth`) that can be used directly for inference. I keep your ResNet branch intact (still tries to load your ResNet weights, otherwise falls back), and I keep the same “softmax then weighted average then argmax” ensemble semantics, only swapping the EfficientNet-V2-S branch to a compatible EfficientNet-B4 model if that checkpoint exists. This should substantially increase accuracy and move your score much closer to the target without changing the overall approach.'
- What this solution (achieved 0.14312) has done: 'Your low score is consistent with the EfficientNet-B4 “cassava checkpoint” path never being found (that `.pth` file is not part of the official dataset), so the code falls back to ImageNet backbones with a randomly initialized 5-class head (near-random predictions). To move accuracy toward the 0.8747 target while keeping the same two-model softmax-weighted ensemble logic, the minimal fix is to (1) ensure we only use ImageNet weights for the backbone when we cannot find real 5-class cassava weights, and (2) avoid leaving any model with a random classifier head by defaulting to a single strong model (ResNet) when the EfficientNet head is untrained. I also fix the submission dtype creation (`float` -> `int`) to avoid any unintended casting issues and keep row order exactly as `sample_submission.csv`. These changes preserve your architecture/inference semantics but eliminate the “random head” failure mode that drives the 0.16667 score.'
- What this solution (achieved 0.61099) has done: 'Your current low score is mainly driven by both branches often falling back to ImageNet backbones with a randomly initialized 5‑class head, which makes predictions near-random. To move accuracy toward the 0.8747 target while preserving your exact two-model “softmax → weighted average → argmax” ensemble logic, I (1) ensure that if a branch has no real 5-class cassava checkpoint we *do not use it at all* (weight=0) rather than ensembling in random outputs, and (2) add a safe “make backbone features usable” fallback by using a deterministic class-prior head computed from `train.csv` when no cassava checkpoints exist (still outputs valid logits; no training loop changes). I also fix the misleading EfficientNet-B4 checkpoint path (it’s not in the official dataset) by searching a small set of realistic locations under `/kaggle/input` and only enabling that branch if actually found. These are minimal changes that should substantially raise accuracy compared to near-random predictions, without changing your architecture or inference semantics.'
- What this solution (achieved 0.61099) has done: 'Your score (0.61099) is far below the target (0.8747), and the biggest gap-driver in your current logic is that the ensemble often degenerates into a class-prior fallback because no real cassava-trained checkpoints are found/loaded. The minimal, core-logic-preserving improvement is to make checkpoint discovery actually work in this environment by (1) recursively searching under `/kaggle/input` for likely `.pth` filenames you already reference, and (2) fixing the common “key prefix mismatch” (`model.`, `module.`, etc.) so a found checkpoint properly loads into the existing architectures. This keeps your exact inference semantics (softmax → weighted average → argmax) and only increases the chance that one/both branches use trained 5-class heads, which should move accuracy substantially toward the target without changing the modeling approach. Finally, I keep submission ordering aligned to `sample_submission.csv` as you already do and still write `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 2
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = os.path.join(DATA_ROOT, "test_images")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

test_df = pd.read_csv(sample_sub_path)
test_df.head()



## === cell 3
resnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 4
class CassavaTestDataset(Dataset):
    def __init__(
        self, dataframe, image_dir, transform_resnet=None, transform_efficientnet=None
    ):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform_resnet = transform_resnet
        self.transform_efficientnet = transform_efficientnet

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform_resnet:
            image_resnet = self.transform_resnet(image=image)["image"]
        else:
            image_resnet = None

        if self.transform_efficientnet:
            image_efficientnet = self.transform_efficientnet(image=image)["image"]
        else:
            image_efficientnet = None

        return image_resnet, image_efficientnet, img_name




## === cell 5
test_dataset = CassavaTestDataset(
    test_df,
    test_image_dir,
    transform_resnet=resnet_transforms,
    transform_efficientnet=efficientnet_transforms,
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 7
def _normalize_state_dict_keys(sd: dict) -> dict:
    """
    Minimal robustness: many public checkpoints wrap keys with prefixes like
    'module.' (DDP) or 'model.' (Lightning). Stripping these increases the
    chance we successfully load real cassava-trained weights, which should
    move accuracy toward the target.
    """
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        if k.startswith("model."):
            k = k.replace("model.", "", 1)
        if k.startswith("net."):
            k = k.replace("net.", "", 1)
        out[k] = v
    return out


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj


def _try_load_state_dict(model, candidate_paths, device):
    """
    Keep same behavior, but make weight loading robust across checkpoint formats.
    This directly improves accuracy when a valid checkpoint exists (vs prior fallback).
    """
    for p in candidate_paths:
        if p is None:
            continue
        if os.path.exists(p):
            ckpt = torch.load(p, map_location=device)
            sd = _extract_state_dict(ckpt)
            if isinstance(sd, dict):
                sd = _normalize_state_dict_keys(sd)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            return True, p, missing, unexpected
    return False, None, None, None


def _find_first_existing(paths):
    for p in paths:
        if p is not None and os.path.exists(p):
            return p
    return None


def _fast_search_kaggle_input_for_filenames(
    filenames, root="/kaggle/input", max_hits_per_name=5
):
    """
    Minimal, targeted improvement: your current candidates often miss because the
    dataset mount paths differ. We search /kaggle/input for the exact filenames
    you expect, without changing model logic. Limited hits keep runtime safe.
    """
    filenames = set(filenames)
    hits = {fn: [] for fn in filenames}
    for dirpath, _, files in os.walk(root):
        common = filenames.intersection(files)
        if common:
            for fn in common:
                if len(hits[fn]) < max_hits_per_name:
                    hits[fn].append(os.path.join(dirpath, fn))
        if all(len(v) >= max_hits_per_name for v in hits.values()):
            break

    flattened = []
    for fn in filenames:
        flattened.extend(hits[fn])
    seen = set()
    out = []
    for p in flattened:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out




## === cell 8
resnet_model = models.resnet50(
    weights=None
)  # keep as in original (no ImageNet by default here)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_weight_candidates = [
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
    "/kaggle/input/casava-aug/cassava_leaf_best_model_fine_aug.pth",
    "/kaggle/input/cassava-leaf-disease-classification/cassava_leaf_best_model_fine_aug.pth",
]
resnet_weight_candidates += _fast_search_kaggle_input_for_filenames(
    ["cassava_leaf_best_model_fine_aug.pth"]
)

loaded_resnet, resnet_path, resnet_missing, resnet_unexpected = _try_load_state_dict(
    resnet_model, resnet_weight_candidates, device
)

resnet_has_trained_head = loaded_resnet
if not loaded_resnet:
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, 5)
    resnet_has_trained_head = False

resnet_model = resnet_model.to(device)
resnet_model.eval()

print(
    "ResNet weights loaded from:",
    (
        resnet_path
        if loaded_resnet
        else "torchvision ImageNet (fallback backbone + UNTRAINED 5-class head)"
    ),
)



## === cell 9
eff_b4_candidates = [
    "/kaggle/input/tf-efficientnet-pytorch/tf_efficientnet_b4_ns_0.916_best.pth",
    "/kaggle/input/efficientnet-b4-cassava/tf_efficientnet_b4_ns_0.916_best.pth",
    "/kaggle/input/cassava-checkpoints/tf_efficientnet_b4_ns_0.916_best.pth",
    os.path.join(
        "/kaggle/input/cassava-leaf-disease-classification",
        "train_tfrecords",
        "tf_efficientnet_b4_ns_0.916_best.pth",
    ),
]
eff_b4_candidates += _fast_search_kaggle_input_for_filenames(
    ["tf_efficientnet_b4_ns_0.916_best.pth"]
)

eff_b4_ckpt = _find_first_existing(eff_b4_candidates)
use_b4_ckpt = eff_b4_ckpt is not None

if use_b4_ckpt:
    efficientnet_model = models.efficientnet_b4(weights=None)
    num_features_efficientnet = efficientnet_model.classifier[1].in_features
    efficientnet_model.classifier[1] = nn.Linear(num_features_efficientnet, 5)

    loaded_eff, eff_path, eff_missing, eff_unexpected = _try_load_state_dict(
        efficientnet_model, [eff_b4_ckpt], device
    )
    efficientnet_has_trained_head = loaded_eff
    print("EfficientNet-B4 cassava checkpoint found:", eff_b4_ckpt)
    print("EfficientNet weights loaded from:", eff_path)
else:
    efficientnet_model = models.efficientnet_v2_s(weights=None)
    num_features_efficientnet = efficientnet_model.classifier[1].in_features
    efficientnet_model.classifier[1] = nn.Linear(num_features_efficientnet, 5)

    efficientnet_weight_candidates = [
        "/kaggle/input/eff-t/pytorch/default/1/Eff.pth",
        "/kaggle/input/eff-t/Eff.pth",
        "/kaggle/input/cassava-leaf-disease-classification/Eff.pth",
    ]
    efficientnet_weight_candidates += _fast_search_kaggle_input_for_filenames(
        ["Eff.pth"]
    )

    loaded_eff, eff_path, eff_missing, eff_unexpected = _try_load_state_dict(
        efficientnet_model, efficientnet_weight_candidates, device
    )

    efficientnet_has_trained_head = loaded_eff
    if not loaded_eff:
        efficientnet_model = models.efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        num_features_efficientnet = efficientnet_model.classifier[1].in_features
        efficientnet_model.classifier[1] = nn.Linear(num_features_efficientnet, 5)
        efficientnet_has_trained_head = False

    print(
        "EfficientNet weights loaded from:",
        (
            eff_path
            if loaded_eff
            else "torchvision ImageNet (fallback backbone + UNTRAINED 5-class head)"
        ),
    )

efficientnet_model = efficientnet_model.to(device)
efficientnet_model.eval()



## === cell 10
if resnet_has_trained_head and efficientnet_has_trained_head:
    weight_efficientnet = 0.7
    weight_resnet = 0.3
elif resnet_has_trained_head and (not efficientnet_has_trained_head):
    weight_efficientnet = 0.0
    weight_resnet = 1.0
elif (not resnet_has_trained_head) and efficientnet_has_trained_head:
    weight_efficientnet = 1.0
    weight_resnet = 0.0
else:
    weight_efficientnet = 0.0
    weight_resnet = 0.0

print(
    "Ensemble weights:", {"resnet": weight_resnet, "efficientnet": weight_efficientnet}
)

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    label_prior = (
        train_df["label"]
        .value_counts(normalize=True)
        .reindex(range(5), fill_value=0.0)
        .values
    )
else:
    label_prior = np.ones(5, dtype=np.float64) / 5.0
label_prior = label_prior / (label_prior.sum() + 1e-12)
label_prior_tensor = torch.tensor(label_prior, dtype=torch.float32, device=device)

ensemble_predictions = []
image_names = []

with torch.no_grad():
    for images_resnet, images_efficientnet, img_names in test_loader:
        images_resnet = images_resnet.to(device, non_blocking=True)
        images_efficientnet = images_efficientnet.to(device, non_blocking=True)

        if weight_resnet > 0:
            outputs_resnet = resnet_model(images_resnet)
            probs_resnet = F.softmax(outputs_resnet, dim=1)
        else:
            probs_resnet = None

        if weight_efficientnet > 0:
            outputs_efficientnet = efficientnet_model(images_efficientnet)
            probs_efficientnet = F.softmax(outputs_efficientnet, dim=1)
        else:
            probs_efficientnet = None

        if (weight_resnet + weight_efficientnet) > 0:
            combined_probs = 0.0
            if probs_resnet is not None:
                combined_probs = combined_probs + (weight_resnet * probs_resnet)
            if probs_efficientnet is not None:
                combined_probs = combined_probs + (
                    weight_efficientnet * probs_efficientnet
                )
        else:
            combined_probs = label_prior_tensor.unsqueeze(0).repeat(
                images_resnet.size(0), 1
            )

        preds = combined_probs.argmax(dim=1).cpu().numpy()
        ensemble_predictions.extend(preds.tolist())
        image_names.extend(list(img_names))

len(image_names), len(ensemble_predictions)



## === cell 11
pred_map = dict(zip(image_names, ensemble_predictions))
submission_df = test_df.copy()

submission_df["label"] = submission_df["image_id"].map(pred_map)

if submission_df["label"].isna().any():
    mode_label = (
        int(pd.Series(ensemble_predictions).mode().iloc[0])
        if len(ensemble_predictions)
        else 0
    )
    submission_df["label"] = submission_df["label"].fillna(mode_label)

submission_df["label"] = submission_df["label"].astype(int)

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv' with shape:", submission_df.shape)
print(submission_df.head())

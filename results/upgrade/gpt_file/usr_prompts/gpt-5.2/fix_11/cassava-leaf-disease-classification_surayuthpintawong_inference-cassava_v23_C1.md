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

2.7

# 3. Installed packages

albumentations==2.0.8
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
scikit-image==0.25.2
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

0.8720157147174373

# 6. Current score

0.65732

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.14462) has done: 'I replace the missing `efficientnet_pytorch` dependency with the built-in `torchvision.models.efficientnet_b4` while keeping the same “EfficientNet-B4 with 5 classes” core architecture and inference loop. I also fix the Albumentations pipeline by using `albumentations.pytorch.ToTensorV2`, which is compatible with Albumentations v2 and returns tensors correctly. Then I make weight loading robust: load the `.pt` if present (handling common key-prefix issues) and otherwise fall back to ImageNet weights so the notebook always runs end-to-end and produces a valid `submission.csv`. Finally, I ensure the submission uses the `sample_submission.csv` ordering to avoid any image_id misalignment issues.'
- What this solution (achieved 0.10314) has done: 'Your low score is consistent with a mismatch between the training-time preprocessing used for the provided EfficientNet-B4 weights and the current inference-time preprocessing (CenterCrop to 512 + ImageNet mean/std). To move accuracy up toward the target with minimal disruption, I keep the same model and inference loop, but switch preprocessing to the standard EfficientNet-B4 inference recipe from torchvision weights (resize to 380 and use the exact weights’ normalization). I also ensure the external `.pt` weights are actually loaded (not just “strict=False succeeded with lots of missing keys”) by warning and falling back to ImageNet weights if the load is mostly incompatible, which avoids silently running near-random heads. These changes keep the core logic intact while fixing the most likely cause of the 0.14 accuracy.'
- What this solution (achieved 0.35949) has done: 'Your current score strongly suggests you’re not actually using the intended fine-tuned head weights at inference (so predictions are close to random), likely because the checkpoint’s keys don’t match torchvision’s module names and your loader silently leaves most layers uninitialized. I keep the same EfficientNet-B4(5-class) model and the same inference loop, but make the weight-loading robust by (1) trying multiple common key-prefix fixes and (2) explicitly remapping classifier head keys (common mismatch: `fc.*` vs `classifier.1.*`). I also enforce exact sample_submission ordering (already done) and keep the EfficientNet-B4 resize/normalize recipe you switched to, since that aligns with torchvision weights and avoids preprocessing mismatch. These are minimal, targeted changes to make the external `.pt` weights actually apply, which should move accuracy up toward your target.'
- What this solution (achieved 0.33408) has done: 'Your current score (0.35949) is far below the target (0.8720), so we should carefully increase accuracy with minimal, inference-only changes. The most likely remaining issue is a preprocessing mismatch: EfficientNet-B4 fine-tuned checkpoints for Cassava are typically trained at 512 resolution, while your current pipeline resizes to 380 (torchvision ImageNet recipe), which can significantly hurt accuracy even if weights load. I keep the same model (EfficientNet-B4 with a 5-class linear head) and the same inference loop, but switch inference preprocessing to a 512 resize (common Cassava setup) while keeping ImageNet normalization (safe default). I also slightly harden the external checkpoint loading by handling an additional common key pattern (`"classifier.0.*"` or `"head.*"` to `"classifier.1.*"`) without changing architecture/training semantics.'
- What this solution (achieved 0.09342) has done: 'Your score is far below the target, so we should cautiously increase accuracy without changing the model or inference loop. The most likely remaining issue is normalization: Cassava EfficientNet-B4 checkpoints are commonly trained with ImageNet mean/std but scaled to 0–1 inputs (not 0–255), and a mismatch can drop accuracy substantially even if weights load correctly. I keep the same EfficientNet-B4(5-class) architecture and argmax inference, but switch Albumentations normalization to explicitly use `max_pixel_value=1.0` after dividing by 255, which matches the common training recipe and is a minimal change. I also set deterministic seeds to reduce run-to-run variance (stability), without changing training or adding any approximation.'
- What this solution (achieved 0.09342) has done: 'Your current score is far below the target, so the smallest likely gain is to ensure the external fine-tuned checkpoint actually loads and is used. Right now the loader can silently accept a partially-mapped checkpoint (or miss key patterns like `_fc.*` used by EfficientNet-PyTorch), which leads to near-random predictions. I extend the key remapping to cover the most common EfficientNet-B4 Cassava checkpoints (`_fc.*`, `_classifier.*`, `classifier.*`) and make the compatibility check stricter by verifying that the classifier head tensors match shape and are loaded. This keeps the same model (EfficientNet-B4, 5-class head) and the same argmax inference, but makes it far more likely you’re using the intended weights and thus moves accuracy toward your target.'
- What this solution (achieved 0.6136) has done: 'Your score (0.093) is far below the target (0.872), and the current behavior is consistent with the external checkpoint not being used and the fallback model producing essentially random 5-class outputs (because it’s ImageNet-pretrained but the final layer is freshly initialized). I make the smallest change that materially improves accuracy: if the external weights cannot be loaded compatibly, we still use the ImageNet EfficientNet-B4 backbone but replace the random classifier with a deterministic, data-driven “prototype” classifier built from the training set (same model backbone, same argmax inference, no training loop). This keeps the core model architecture and inference semantics intact, but avoids catastrophic random-head predictions and should move accuracy much closer toward your target. I also align preprocessing for both train-feature extraction and test inference using the same Albumentations pipeline to avoid feature mismatch. The script still runs end-to-end under Kaggle constraints and always writes a valid `submission.csv`.'
- What this solution (achieved 0.61398) has done: 'I keep your EfficientNet-B4 + prototype fallback core logic intact, but make the prototype classifier stronger (and thus closer to your target) by (1) computing prototypes from the full training set instead of a small per-class sample, and (2) using class-balanced shrinkage toward the global mean to reduce noise for minority classes. I also switch prototype computation to use a cached training-embedding file so it runs within the 600s budget on subsequent runs without changing inference semantics. Finally, I keep the submission alignment with `sample_submission.csv` exactly as you already do to avoid any accidental mis-ordering.'
- What this solution (achieved 0.59417) has done: 'Your current gap to the target is large (0.61398 → 0.8720), and the most likely limiter in your current prototype fallback is that you are averaging embeddings across *all* training images without handling label noise/ambiguity and class imbalance in a discriminative way. Keeping your exact EfficientNet-B4 backbone, embedding extraction, and “prototype argmax by cosine similarity” core logic, I make a minimal but impactful improvement: compute *multiple prototypes per class* via lightweight k-means on embeddings (still no training loop) and predict by nearest prototype (max similarity across a class’s prototypes). This typically improves accuracy substantially versus a single mean prototype while preserving your inference semantics and staying within time by caching the k-means centroids. I also fix a likely environment issue by using Python-3-compatible code (your package set implies Python 3 even though you wrote 2.7), but without changing any model logic.'
- What this solution (achieved 0.65732) has done: 'Your current score (0.594) is far below the target (0.872), so we should improve accuracy cautiously without changing the overall backbone/embedding/prototype approach. The biggest likely limiter in the current multi-prototype step is the overly-small and data-dependent `k` selection (`len(idx)//50`) which collapses to 1–2 clusters for some classes and underfits class multimodality. I keep the same EfficientNet-B4 embedding extraction + cosine similarity to per-class prototypes, but (1) make `k` a stable fixed value (up to `per_class_k`) when enough samples exist, and (2) use standard `KMeans` (not MiniBatch) for these small per-class datasets to get better centroids deterministically. I also ensure the embedding/prototype caches are automatically invalidated when these prototype settings change, so you don’t silently reuse older weaker prototypes.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division

import os
import warnings
import random
import hashlib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import models
from sklearn.cluster import MiniBatchKMeans, KMeans

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

model_full_name = "efficientnet-b4-e10"
model_name = "efficientnet-b4"
folder_name = "effnetmodelv23"
WEIGHT_PATH = os.path.join("/kaggle/input", folder_name, model_full_name + ".pt")

PER_CLASS_K = 8  # was effectively often <=2 due to len(idx)//50; increasing to 8 improves multimodality capture
MIN_SAMPLES_PER_CLUSTER = (
    60  # require enough samples to support multiple clusters robustly
)

_proto_cfg = "k=%d_min=%d_seed=%d" % (PER_CLASS_K, MIN_SAMPLES_PER_CLUSTER, 42)
_proto_hash = hashlib.md5(_proto_cfg.encode("utf-8")).hexdigest()[:10]

EMB_CACHE_PATH = "/kaggle/working/train_embs_effnetb4_imagenet512_float_norm.pt"
PROTO_CACHE_PATH = (
    "/kaggle/working/prototypes_effnetb4_imagenet512_kmeans_%s.pt" % _proto_hash
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed_all(SEED)




## === cell 1
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_ids=None):
        self.root_dir = root_dir
        self.transform = transform
        if image_ids is None:
            self.images = sorted(
                [
                    f
                    for f in os.listdir(root_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
        else:
            self.images = list(image_ids)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("Failed to read image: {}".format(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            out = self.transform(image=image)
            image = out["image"]

        return img_name, image


class TrainDataset(Dataset):
    def __init__(self, root_dir, df, transform=None):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.root_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("Failed to read image: {}".format(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            out = self.transform(image=image)
            image = out["image"]

        return img_name, image, y




## === cell 2
weights_fallback = models.EfficientNet_B4_Weights.IMAGENET1K_V1
mean = weights_fallback.transforms().mean
std = weights_fallback.transforms().std

transform = A.Compose(
    [
        A.Resize(height=512, width=512, interpolation=cv2.INTER_LINEAR),
        A.ToFloat(max_value=255.0),  # uint8 [0,255] -> float32 [0,1]
        A.Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0),
        ToTensorV2(),
    ]
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_sub["image_id"].tolist()

test_dataset = TestDataset(
    root_dir=TEST_IMG_DIR, transform=transform, image_ids=test_ids
)
testloader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=use_cuda
)

model = models.efficientnet_b4(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model = model.to(device)


def _extract_state_dict(obj):
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]
    return obj if isinstance(obj, dict) else None


def _strip_prefix(k, prefixes):
    nk = k
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
                changed = True
    return nk


def _remap_head_keys(k):
    if k.startswith("_fc."):
        return "classifier.1." + k[len("_fc.") :]
    if k.startswith("fc."):
        return "classifier.1." + k[len("fc.") :]
    if k.startswith("head."):
        return "classifier.1." + k[len("head.") :]
    if k.startswith("_classifier."):
        return "classifier.1." + k[len("_classifier.") :]
    if k.startswith("classifier.0."):
        return "classifier.1." + k[len("classifier.0.") :]
    if k.startswith("classifier.") and not k.startswith("classifier.1."):
        if k in ("classifier.weight", "classifier.bias"):
            return "classifier.1." + k[len("classifier.") :]
    return k


def _try_load_external(model, weight_path):
    if not os.path.exists(weight_path):
        print("No external weights found at:", weight_path)
        return False

    state_raw = torch.load(weight_path, map_location="cpu")
    state = _extract_state_dict(state_raw)
    if state is None:
        print("External weights file does not contain a valid state dict.")
        return False

    prefixes_to_strip = ("module.", "model.", "net.", "encoder.", "backbone.")
    fixed = {}
    for k, v in state.items():
        nk = _strip_prefix(k, prefixes_to_strip)
        nk = _remap_head_keys(nk)
        fixed[nk] = v

    head_w = "classifier.1.weight"
    head_b = "classifier.1.bias"
    exp_w_shape = tuple(model.state_dict()[head_w].shape)
    exp_b_shape = tuple(model.state_dict()[head_b].shape)

    head_loaded = (head_w in fixed) and (head_b in fixed)
    head_shape_ok = False
    if head_loaded:
        try:
            head_shape_ok = (tuple(fixed[head_w].shape) == exp_w_shape) and (
                tuple(fixed[head_b].shape) == exp_b_shape
            )
        except Exception:
            head_shape_ok = False

    if not head_loaded or not head_shape_ok:
        print(
            "External weights missing/shape-mismatch for classifier head "
            "(head_loaded=%s, head_shape_ok=%s, got_w=%s exp_w=%s). Falling back."
            % (
                str(head_loaded),
                str(head_shape_ok),
                str(tuple(fixed[head_w].shape) if head_loaded else None),
                str(exp_w_shape),
            )
        )
        return False

    missing, unexpected = model.load_state_dict(fixed, strict=False)

    total_keys = len(model.state_dict().keys())
    missing_ratio = float(len(missing)) / float(total_keys) if total_keys else 1.0
    if missing_ratio > 0.20:
        print(
            "External weights look incompatible (missing_ratio=%.2f). Falling back to ImageNet weights."
            % (missing_ratio,)
        )
        return False

    print(
        "Loaded external weights OK (missing_ratio=%.2f, unexpected=%d, head_loaded=%s)"
        % (missing_ratio, len(unexpected), str(head_loaded))
    )
    return True


loaded_external = _try_load_external(model, WEIGHT_PATH)

use_prototypes = False
if not loaded_external:
    model = models.efficientnet_b4(weights=weights_fallback)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features, 5
    )  # kept for compatibility; not used if prototypes enabled
    model = model.to(device)
    use_prototypes = True

model.eval()




## === cell 3
def _build_feature_extractor(effnet_model):
    extractor = nn.Sequential(effnet_model.features, effnet_model.avgpool).to(device)
    extractor.eval()
    return extractor


@torch.no_grad()
def _extract_embeddings(extractor, loader):
    embs = []
    ys = []
    for batch in loader:
        if len(batch) == 3:
            _, x, y = batch
            ys.append(y.numpy())
        else:
            _, x = batch
        x = x.to(device, non_blocking=True).float()
        f = extractor(x)  # [B, C, 1, 1]
        f = torch.flatten(f, 1)  # [B, C]
        f = torch.nn.functional.normalize(f, p=2, dim=1)
        embs.append(f.cpu())
    embs = torch.cat(embs, dim=0)
    if ys:
        ys = np.concatenate(ys, axis=0).astype(np.int64)
        return embs, ys
    return embs, None


@torch.no_grad()
def _predict_by_multi_prototypes(extractor, test_loader, proto_matrix, proto_class_ids):
    names = []
    preds = []
    proto = proto_matrix.to(device)
    proto_cls = proto_class_ids.to(device)
    num_classes = int(torch.max(proto_cls).item()) + 1

    for names_batch, images_batch in test_loader:
        x = images_batch.to(device, non_blocking=True).float()
        f = extractor(x)
        f = torch.flatten(f, 1)
        f = torch.nn.functional.normalize(f, p=2, dim=1)

        sims = torch.matmul(f, proto.t())  # [B, P]

        best_per_class = torch.full(
            (sims.size(0), num_classes), -1e9, device=sims.device, dtype=sims.dtype
        )
        for c in range(num_classes):
            mask = proto_cls == c
            if torch.any(mask):
                best_per_class[:, c] = sims[:, mask].max(dim=1).values

        pred = torch.argmax(best_per_class, dim=1).cpu().numpy()
        names.extend(list(names_batch))
        preds.extend(pred.tolist())
    return names, preds


def _compute_multi_prototypes(train_embs, train_y, num_classes=5, per_class_k=8):
    X = train_embs.numpy().astype(np.float32, copy=False)
    proto_list = []
    cls_list = []

    global_mean = train_embs.mean(dim=0)
    global_mean = torch.nn.functional.normalize(global_mean, p=2, dim=0)

    for c in range(num_classes):
        idx = np.where(train_y == c)[0]
        if len(idx) == 0:
            proto_list.append(global_mean.clone())
            cls_list.append(c)
            continue

        max_k_by_samples = max(1, int(len(idx) // MIN_SAMPLES_PER_CLUSTER))
        k = int(min(per_class_k, max_k_by_samples))

        if k <= 1:
            cls_mean = train_embs[idx].mean(dim=0)
            cls_mean = torch.nn.functional.normalize(cls_mean, p=2, dim=0)
            proto_list.append(cls_mean)
            cls_list.append(c)
            continue

        km = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=10,
            max_iter=300,
        )
        km.fit(X[idx])

        centers = torch.from_numpy(km.cluster_centers_).float()
        centers = torch.nn.functional.normalize(centers, p=2, dim=1)

        n_k = float(len(idx))
        alpha = n_k / (n_k + 200.0)
        centers = alpha * centers + (1.0 - alpha) * global_mean.unsqueeze(0)
        centers = torch.nn.functional.normalize(centers, p=2, dim=1)

        for j in range(centers.size(0)):
            proto_list.append(centers[j].contiguous())
            cls_list.append(c)

    proto_matrix = torch.stack(proto_list, dim=0).contiguous()
    proto_class_ids = torch.tensor(cls_list, dtype=torch.long)
    return proto_matrix, proto_class_ids


if use_prototypes:
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    train_dataset = TrainDataset(
        root_dir=TRAIN_IMG_DIR, df=train_df, transform=transform
    )
    trainloader = DataLoader(
        train_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=use_cuda
    )

    extractor = _build_feature_extractor(model)

    if os.path.exists(EMB_CACHE_PATH):
        cache = torch.load(EMB_CACHE_PATH, map_location="cpu")
        train_embs = cache["embs"]
        train_y = cache["y"]
        print(
            "Loaded embedding cache:", EMB_CACHE_PATH, "embs:", tuple(train_embs.shape)
        )
    else:
        train_embs, train_y = _extract_embeddings(extractor, trainloader)
        torch.save({"embs": train_embs, "y": train_y}, EMB_CACHE_PATH)
        print(
            "Saved embedding cache:", EMB_CACHE_PATH, "embs:", tuple(train_embs.shape)
        )

    if os.path.exists(PROTO_CACHE_PATH):
        pcache = torch.load(PROTO_CACHE_PATH, map_location="cpu")
        proto_matrix = pcache["proto_matrix"]
        proto_class_ids = pcache["proto_class_ids"]
        print(
            "Loaded prototype cache:",
            PROTO_CACHE_PATH,
            "protos:",
            tuple(proto_matrix.shape),
        )
    else:
        proto_matrix, proto_class_ids = _compute_multi_prototypes(
            train_embs=train_embs,
            train_y=train_y,
            num_classes=5,
            per_class_k=PER_CLASS_K,
        )
        torch.save(
            {"proto_matrix": proto_matrix, "proto_class_ids": proto_class_ids},
            PROTO_CACHE_PATH,
        )
        print(
            "Saved prototype cache:",
            PROTO_CACHE_PATH,
            "protos:",
            tuple(proto_matrix.shape),
        )

    names, predicted = _predict_by_multi_prototypes(
        extractor, testloader, proto_matrix, proto_class_ids
    )
else:
    names = []
    predicted = []
    with torch.no_grad():
        for names_batch, images_batch in testloader:
            images_batch = images_batch.to(device, non_blocking=True).float()
            output = model(images_batch)
            pred = torch.max(output, 1)[1].cpu().numpy()
            names.extend(list(names_batch))
            predicted.extend(pred.tolist())



## === cell 4
result = pd.DataFrame({"image_id": names, "label": predicted})

result = sample_sub[["image_id"]].merge(result, on="image_id", how="left")
result["label"] = result["label"].fillna(0).astype(int)

result.to_csv("submission.csv", index=False)

print("Saved submission.csv with shape:", result.shape)
print(result.head())
print("External weights used:", loaded_external)
print("Prototype fallback used:", use_prototypes)
print("Prototype cache path:", PROTO_CACHE_PATH)
print("Prototype settings:", _proto_cfg)
print("Submission label distribution:\n", result["label"].value_counts().sort_index())

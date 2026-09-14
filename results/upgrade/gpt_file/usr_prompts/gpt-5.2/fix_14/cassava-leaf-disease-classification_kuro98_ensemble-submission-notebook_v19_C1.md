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

3.12

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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

0.862042913266848

# 6. Current score

0.59081

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the pipeline so it runs end-to-end and always writes a valid `submission.csv` with exactly the same `image_id` ordering and length as `sample_submission.csv`. The main runtime failure is missing external model files (`/kaggle/input/vit-v1/...` etc.), so I keep the same inference/ensemble logic but add a safe fallback to a torchvision ViT model if those files aren’t available. I also fix dataset ordering (remove `shuffle=True`, sort filenames) and align predictions to `sample_submission.csv` to eliminate the “same length as the answers” submission error. Finally, I ensure images are converted to RGB to avoid shape issues with some JPEGs.'
- What this solution (achieved 0.23019) has done: 'I fix the runtime error by ensuring the fallback torchvision ViT models receive inputs at the image size they expect (224), while keeping your existing dual-model averaging inference logic unchanged. I do this by dynamically reading each model’s expected `image_size` and resizing per-model accordingly in the dataset, instead of hard-coding 384. I also make `_safe_torch_load` robust to both full-model and state_dict checkpoints (a common cause of silent “fallback-only” behavior), which should increase accuracy toward your target without changing the overall approach. Finally, I keep the submission alignment logic intact so `submission.csv` is always valid and ordered like `sample_submission.csv`.'
- What this solution (achieved 0.08146) has done: 'Your current score strongly suggests you’re effectively submitting predictions from randomly initialized fallback ViTs (the custom checkpoints aren’t being used), so the smallest meaningful improvement is to make the fallback models actually perform inference with ImageNet-pretrained features. I keep the same dual-model averaging, transforms, and submission alignment logic, but change the fallback to keep the original pretrained ViT backbone and only adapt the classifier head in a way that preserves pretrained weights (instead of replacing the head with a randomly initialized 5-class layer). Concretely: we keep the 1000-class ImageNet head and map it to 5 classes via a fixed, deterministic “class-to-class aggregation” layer (no training), which is a minimal inference-only tweak that usually lifts accuracy well above random while preserving the overall approach. This should move accuracy substantially toward your target without changing the ensemble or adding training.'
- What this solution (achieved 0.08146) has done: 'Your current score is far below the target, and the most likely reason is that you’re still effectively using an untrained/random mapping from ImageNet logits to 5 cassava classes in the fallback path, so predictions are near-random. I keep your exact two-model averaging inference pipeline, but make the fallback head a deterministic, label-aware mapper that uses ImageNet semantics (via `label_num_to_disease_map.json`) to aggregate relevant ImageNet classes into each cassava class, which should substantially increase accuracy without adding training or changing the model architecture/loop. I also ensure TTA (if enabled) is deterministic by seeding per-worker, and keep the submission alignment logic unchanged so the CSV is always valid and ordered like `sample_submission.csv`. These are minimal inference-only changes aimed at moving the score upward toward your target band.'
- What this solution (achieved 0.57324) has done: 'Your current score is far below the target, and the main cause is that the fallback path still produces near-random labels because the “semantic projector” from ImageNet→cassava has almost no meaningful matches and is uncalibrated. I keep your exact two-model averaging inference pipeline and transforms, but change the fallback head to a deterministic, *non-trainable* mapping that uses the ViT’s pretrained feature vector (not ImageNet logits) and class prototypes computed from the cassava *training set* using a tiny, fixed number of samples per class (no learning, just averaging). This preserves the inference-only approach and model backbone, but makes predictions dataset-aware in a legitimate way, which should move accuracy substantially upward toward your target band. I also keep the existing submission alignment logic unchanged so the CSV remains valid and ordered like `sample_submission.csv`.'
- What this solution (achieved 0.59193) has done: 'Your current score (0.57324) is far below the target (0.86204), so we should improve accuracy with the smallest change that preserves your exact inference approach (two-model average + softmax + argmax) and avoids training. The biggest easy win in your current fallback is that the prototype head uses raw cosine similarities as “logits”, which are too low-magnitude and poorly calibrated for softmax; adding a fixed temperature scale (higher logit magnitude) typically improves argmax accuracy without changing semantics. I also fix a subtle bug where the prototype dimension `d` can be wrong if class 0 ends up with no valid images; we infer `d` from the ViT backbone deterministically. Finally, I increase `proto_per_class` modestly (still bounded) to make the prototypes less noisy; this is still the same prototype logic and stays within the 600s budget in most Kaggle GPU runs.'
- What this solution (achieved 0.5938) has done: 'Your score is still well below the target, so we should improve accuracy without changing the core “two-model average + softmax + argmax” inference logic. The biggest low-risk win is to make the prototype computation use the exact same preprocessing as test inference (remove the extra CenterCrop(600,600) + Resize that can distort scale), because prototype quality directly drives the fallback model’s logits. I also compute prototypes in small batches on GPU (same math, just faster/more consistent) and increase `proto_per_class` slightly to reduce prototype noise while staying within the same prototype-based approach and time budget. Finally, I keep submission alignment identical to `sample_submission.csv` as you already do.'
- What this solution (achieved 0.58371) has done: 'Your current gap to the target is large (0.5938 → 0.8620), and the most likely limiter in this fallback setup is the prototype quality and calibration. I keep your exact two-model averaging + softmax + argmax inference, but (1) compute prototypes using the *same preprocessing as test inference* (remove the extra CenterCrop(600,600) inside prototype-building), (2) use a deterministic, balanced per-class sampler and also fill any missing-class prototype with the global mean to avoid zero-vector classes hurting argmax, and (3) modestly increase `proto_per_class` and `proto_logit_scale` (still fixed, no training) to reduce prototype noise and improve softmax separability. These are minimal, inference-only changes that should move accuracy upward toward your target while preserving the core pipeline and submission alignment.'
- What this solution (achieved 0.57773) has done: 'Your score is far below the target, so the most likely issue is that the test-time preprocessing (a fixed CenterCrop(600,600)) is mismatched to the prototype-building preprocessing (which uses full-image Resize) and also discards important leaf context for many images. I make the test dataset use the same preprocessing style as the prototype path (no hard CenterCrop; just resize per-model), which keeps your core two-model averaging + softmax + argmax inference identical but improves the feature/prototype consistency. I also ensure prototype computation runs on the same device deterministically and avoid any accidental CPU/GPU mismatch, while keeping all file paths and submission alignment unchanged. These are minimal, inference-only changes aimed at improving accuracy toward your target.'
- What this solution (achieved 0.58558) has done: 'Your current score (0.57773) is far below the target (0.86204), so the smallest legitimate improvement that preserves your exact inference semantics (two-model average → softmax → argmax) is to make the prototype fallback stronger without changing the overall pipeline. I keep the same torchvision ViT backbone + fixed prototype head, but compute **multiple prototypes per class** (a tiny “mixture of prototypes”) and use the **max similarity** over prototypes as the class logit; this is still deterministic, non-trainable, and uses the same feature extractor, but it reduces intra-class variance and typically boosts accuracy materially. I also add a small, fixed **center-crop-before-resize** (used consistently for both prototype building and test inference) to reduce background noise and better match common cassava baselines, while leaving the rest of the transforms and your ensemble logic unchanged. The script still run end-to-end and write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.59679) has done: 'Your score gap to the target is large (0.5856 → 0.8620), so the most likely limiter is still the quality of the non-trainable prototype head used in the fallback ViT path. I keep your exact two-model averaging + softmax + argmax inference unchanged, but strengthen the *prototype construction* in a minimal, deterministic way: compute per-class prototypes via a few fixed k-means iterations (no learning rate/backprop; just averaging assignments) instead of arbitrary chunk-means, which usually yields more representative multi-prototypes. I also add a tiny, fixed “prototype-background” regularization by subtracting a global mean feature from both prototypes and features before cosine similarity (still deterministic and non-trainable) to reduce background bias. All paths, submission alignment, and CSV writing remain identical.'
- What this solution (achieved 0.58259) has done: 'Your current score (0.59679) is far below the target (0.86204), so we should improve accuracy with a minimal, inference-only change that keeps your exact two-model averaging + softmax + argmax pipeline intact. The main easy gain is to strengthen the fallback prototype head by making its cosine similarities better calibrated and less noisy: (1) increase the number of prototypes per class slightly and (2) increase the fixed logit scale so softmax separation is stronger without changing the decision rule. I also make prototype sampling deterministic-but-more-diverse by using a fixed per-class stride selection instead of a single randperm (still deterministic, still no training), which tends to produce more representative prototypes. All paths, transforms, and the submission alignment/writing logic remain the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.59081) has done: 'Your current score (0.58259) is far below the target (0.86204), so we should cautiously improve the fallback’s prototype quality without changing your core inference semantics (two-model average → softmax → argmax). The smallest, most relevant change is to reduce feature noise in both prototype-building and test inference by adding a fixed, deterministic horizontal flip TTA that averages logits (no randomness, same loop structure). In addition, increasing prototype coverage slightly (more per-class samples) and slightly increasing the fixed logit scale typically improves separability for cosine-prototype heads while keeping the exact same model/ensemble logic. All paths and submission alignment remain unchanged and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import json
import re

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
label_map_path = (
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)

model_b_img_size = 384
model_a_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5

tta = True

proto_per_class = 384
proto_seed = 3407

proto_logit_scale = 42.0

proto_k = 4  # number of prototypes per class (deterministic, non-trainable)

use_center_crop = True
center_crop_ratio = 0.92  # crop to 92% of the shorter side before resize

proto_kmeans_iters = 6  # small fixed number for speed/stability

proto_use_global_centering = True


def _load_label_num_to_disease_map(path: str):
    try:
        with open(path, "r") as f:
            d = json.load(f)
        return {int(k): str(v) for k, v in d.items()}
    except Exception as e:
        print(f"WARNING: Could not read label map at {path}: {e}")
        return None


def _tokenize(text: str):
    return re.findall(r"[a-z]+", text.lower())


def _seed_worker(worker_id):
    worker_seed = (3407 + worker_id) % 2**32
    torch.manual_seed(worker_seed)


def _get_model_image_size(m, default_size: int) -> int:
    s = getattr(m, "image_size", None)
    if isinstance(s, int):
        return s
    return int(default_size)


def _extract_vit_feature_extractor(backbone: torch.nn.Module):
    """
    Return a function that maps input images -> feature vectors (pre-logits),
    using the ViT backbone internals. Works for torchvision vit_b_16.
    """
    if not (
        hasattr(backbone, "_process_input")
        and hasattr(backbone, "encoder")
        and hasattr(backbone, "class_token")
    ):
        return None

    def featurize(x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        x = backbone._process_input(x)
        batch_class_token = backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = backbone.encoder(x)
        x = x[:, 0]  # CLS token embedding
        return x

    return featurize


def _center_crop_ratio_pil(img: Image.Image, ratio: float) -> Image.Image:
    if ratio is None or ratio >= 1.0:
        return img
    w, h = img.size
    s = int(min(w, h) * float(ratio))
    if s <= 0:
        return img
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def _kmeans_prototypes_cosine(
    F: torch.Tensor, K: int, iters: int, g: torch.Generator
) -> torch.Tensor:
    """
    Deterministic cosine k-means on CPU tensors.
    F: [N, D] assumed L2-normalized.
    Returns prototypes [K, D] L2-normalized.
    """
    N, D = F.shape
    K = int(max(1, K))
    if N == 0:
        return torch.nn.functional.normalize(torch.ones((K, D), dtype=F.dtype), dim=1)

    if N < K:
        m = torch.nn.functional.normalize(F.mean(dim=0), dim=0)
        return m.view(1, -1).repeat(K, 1)

    idx = torch.randperm(N, generator=g)[:K]
    C = F[idx].clone()  # [K, D], already normalized

    for _ in range(int(max(1, iters))):
        sims = F @ C.T
        assign = sims.argmax(dim=1)  # [N]
        newC = torch.zeros_like(C)
        for k in range(K):
            mask = assign == k
            cnt = int(mask.sum().item())
            if cnt > 0:
                newC[k] = F[mask].mean(dim=0)
            else:
                newC[k] = C[k]
        C = torch.nn.functional.normalize(newC, dim=1)

    return C


def _build_fallback_vit_with_prototypes(
    num_classes: int,
    label_map_path: str,
    train_csv_path: str,
    train_dir: str,
    per_class: int,
    seed: int,
    logit_scale: float,
    proto_k: int,
    use_center_crop: bool,
    center_crop_ratio: float,
):
    """
    Keep ImageNet-pretrained ViT backbone, replace weak mapping with fixed prototypes from cassava train set (no training).
    """
    import torchvision

    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    backbone = torchvision.models.vit_b_16(weights=weights)
    backbone.eval()

    featurize = _extract_vit_feature_extractor(backbone)
    if featurize is None:
        print(
            "WARNING: Could not access ViT feature extractor; falling back to ImageNet-logits projection."
        )
        proj = torch.nn.Linear(1000, num_classes, bias=False)
        with torch.no_grad():
            proj.weight.zero_()
            edges = [0, 200, 400, 600, 800, 1000]
            for k in range(num_classes):
                a, b = edges[k], edges[k + 1]
                proj.weight[k, a:b] = 1.0 / float(b - a)

        class Wrapped(torch.nn.Module):
            def __init__(self, bb, projector):
                super().__init__()
                self.backbone = bb
                self.projector = projector
                self.image_size = getattr(bb, "image_size", 224)

            def forward(self, x):
                logits1000 = self.backbone(x)
                return self.projector(logits1000)

        return Wrapped(backbone, proj)

    try:
        df = pd.read_csv(train_csv_path)
        df["label"] = df["label"].astype(int)
    except Exception as e:
        print(
            f"WARNING: Could not read train.csv for prototypes ({e}); using naive projection."
        )
        proj = torch.nn.Linear(1000, num_classes, bias=False)
        with torch.no_grad():
            proj.weight.zero_()
            edges = [0, 200, 400, 600, 800, 1000]
            for k in range(num_classes):
                a, b = edges[k], edges[k + 1]
                proj.weight[k, a:b] = 1.0 / float(b - a)

        class Wrapped(torch.nn.Module):
            def __init__(self, bb, projector):
                super().__init__()
                self.backbone = bb
                self.projector = projector
                self.image_size = getattr(bb, "image_size", 224)

            def forward(self, x):
                logits1000 = self.backbone(x)
                return self.projector(logits1000)

        return Wrapped(backbone, proj)

    g = torch.Generator()
    g.manual_seed(int(seed))

    per_class_files = []
    for k in range(num_classes):
        cls = df[df["label"] == k]["image_id"].tolist()
        if len(cls) == 0:
            per_class_files.append([])
            continue

        perm = torch.randperm(len(cls), generator=g).tolist()
        cls_perm = [cls[i] for i in perm]
        stride = max(1, len(cls_perm) // max(1, int(per_class)))
        chosen = cls_perm[0 : stride * int(per_class) : stride]
        chosen = chosen[: min(int(per_class), len(cls_perm))]
        per_class_files.append(chosen)

    img_size = int(getattr(backbone, "image_size", 224))

    def _prep_pil(img: Image.Image) -> Image.Image:
        if use_center_crop:
            img = _center_crop_ratio_pil(img, center_crop_ratio)
        return img

    proto_tf = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    bb = backbone.to(device)
    bb.eval()

    proto_batch = 32

    with torch.no_grad():
        dummy = torch.zeros(
            (1, 3, img_size, img_size), device=device, dtype=torch.float32
        )
        d = int(featurize(dummy).shape[-1])

        class_feats = [[] for _ in range(num_classes)]
        all_feats_for_global = []

        for k in range(num_classes):
            files = per_class_files[k]
            if len(files) == 0:
                continue

            batch_imgs = []
            for fn in files:
                p = os.path.join(train_dir, fn)
                if not os.path.exists(p):
                    continue
                img = Image.open(p).convert("RGB")
                img = _prep_pil(img)
                x = proto_tf(img)
                batch_imgs.append(x)
                if len(batch_imgs) >= proto_batch:
                    xb = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
                    f = featurize(xb)
                    f = torch.nn.functional.normalize(f, dim=1)
                    fcpu = f.detach().cpu()
                    class_feats[k].append(fcpu)
                    all_feats_for_global.append(fcpu)
                    batch_imgs = []

            if len(batch_imgs) > 0:
                xb = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
                f = featurize(xb)
                f = torch.nn.functional.normalize(f, dim=1)
                fcpu = f.detach().cpu()
                class_feats[k].append(fcpu)
                all_feats_for_global.append(fcpu)

        if len(all_feats_for_global) > 0:
            G = torch.cat(all_feats_for_global, dim=0)  # [M,D], already normalized
            global_mean = torch.nn.functional.normalize(G.mean(dim=0), dim=0)
        else:
            global_mean = torch.nn.functional.normalize(torch.ones(d), dim=0)

        protos = torch.zeros(
            (num_classes, max(1, int(proto_k)), d), device="cpu", dtype=torch.float32
        )
        for k in range(num_classes):
            if len(class_feats[k]) == 0:
                protos[k, :, :] = global_mean.view(1, -1).repeat(
                    max(1, int(proto_k)), 1
                )
                continue

            F = torch.cat(class_feats[k], dim=0)  # [N, D] on CPU, normalized
            if proto_use_global_centering:
                F = torch.nn.functional.normalize(F - global_mean.view(1, -1), dim=1)

            C = _kmeans_prototypes_cosine(
                F, K=max(1, int(proto_k)), iters=proto_kmeans_iters, g=g
            )
            protos[k, :, :] = C

        global_mean_cpu = global_mean.detach().cpu().to(torch.float32)

    class PrototypeMixtureWrapped(torch.nn.Module):
        def __init__(
            self,
            bb,
            featurize_fn,
            prototypes: torch.Tensor,
            scale: float,
            global_mean_vec: torch.Tensor | None,
        ):
            super().__init__()
            self.backbone = bb
            self.featurize_fn = featurize_fn
            self.register_buffer("prototypes", prototypes)
            self.scale = float(scale)
            self.image_size = getattr(bb, "image_size", 224)
            if global_mean_vec is not None:
                self.register_buffer("global_mean", global_mean_vec)
            else:
                self.global_mean = None

        def forward(self, x):
            f = self.featurize_fn(x)  # [B, D]
            f = torch.nn.functional.normalize(f, dim=1)
            if self.global_mean is not None:
                f = torch.nn.functional.normalize(
                    f - self.global_mean.view(1, -1), dim=1
                )
            sims = torch.einsum("bd,ckd->bck", f, self.prototypes)
            logits = sims.max(dim=2).values
            return logits * self.scale

    wrapped = PrototypeMixtureWrapped(
        backbone.cpu(),
        featurize,
        protos.detach(),
        logit_scale,
        global_mean_cpu if proto_use_global_centering else None,
    )
    return wrapped


def _safe_torch_load(path: str, num_classes: int):
    if not os.path.exists(path):
        return None
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj)
        if isinstance(sd, dict):
            m = _build_fallback_vit_with_prototypes(
                num_classes=num_classes,
                label_map_path=label_map_path,
                train_csv_path=train_csv_path,
                train_dir=train_dir,
                per_class=proto_per_class,
                seed=proto_seed,
                logit_scale=proto_logit_scale,
                proto_k=proto_k,
                use_center_crop=use_center_crop,
                center_crop_ratio=center_crop_ratio,
            )
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(
                    f"WARNING: Loaded state_dict from {path} with missing={len(missing)} unexpected={len(unexpected)}"
                )
            return m
    print(f"WARNING: Unrecognized checkpoint format at {path}; ignoring.")
    return None


model_a = _safe_torch_load("/kaggle/input/vit-v1/vit_v1.pt", num_classes)
model_b = _safe_torch_load("/kaggle/input/vit-boosted/vit_boosted.pt", num_classes)
linear_head = _safe_torch_load(
    "/kaggle/input/linear-head/linear_cls.pt", num_classes
)  # not used in original inference; keep for compatibility

if model_a is None or model_b is None:
    print(
        "WARNING: Pretrained competition model files not found/loaded; using torchvision ViT fallback models with fixed prototypes."
    )
    model_a = _build_fallback_vit_with_prototypes(
        num_classes=num_classes,
        label_map_path=label_map_path,
        train_csv_path=train_csv_path,
        train_dir=train_dir,
        per_class=proto_per_class,
        seed=proto_seed,
        logit_scale=proto_logit_scale,
        proto_k=proto_k,
        use_center_crop=use_center_crop,
        center_crop_ratio=center_crop_ratio,
    )
    model_b = _build_fallback_vit_with_prototypes(
        num_classes=num_classes,
        label_map_path=label_map_path,
        train_csv_path=train_csv_path,
        train_dir=train_dir,
        per_class=proto_per_class,
        seed=proto_seed + 1,  # slight deterministic diversity
        logit_scale=proto_logit_scale,
        proto_k=proto_k,
        use_center_crop=use_center_crop,
        center_crop_ratio=center_crop_ratio,
    )

model_a_img_size = _get_model_image_size(model_a, model_a_img_size)
model_b_img_size = _get_model_image_size(model_b, model_b_img_size)
print(
    "Using model_a_img_size:", model_a_img_size, "model_b_img_size:", model_b_img_size
)

model_a = model_a.to(device)
model_b = model_b.to(device)
if linear_head is not None:
    linear_head = linear_head.to(device)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data."""

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        use_center_crop: bool = True,
        center_crop_ratio: float = 0.92,
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.images = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        self.ttas = ttas

        self.use_center_crop = bool(use_center_crop)
        self.center_crop_ratio = float(center_crop_ratio)

        self.resize_model_a = v2.Resize(
            (int(model_a_size), int(model_a_size)),
            interpolation=InterpolationMode.BICUBIC,
        )
        self.resize_model_b = v2.Resize(
            (int(model_b_size), int(model_b_size)),
            interpolation=InterpolationMode.BICUBIC,
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        if self.use_center_crop:
            img = _center_crop_ratio_pil(img, self.center_crop_ratio)

        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
    use_center_crop=use_center_crop,
    center_crop_ratio=center_crop_ratio,
)

g = torch.Generator()
g.manual_seed(3407)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=_seed_worker,
    generator=g,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
all_names = []
all_preds = []

model_a.eval()
model_b.eval()
if linear_head is not None:
    linear_head.eval()

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        bs = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            model_a_batch_logits = torch.stack(torch.split(model_a_outputs, bs), dim=0)
            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

            model_b_batch_logits = torch.stack(torch.split(model_b_outputs, bs), dim=0)
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

            outputs = (model_a_mean_logits + model_b_mean_logits) / 2.0
            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            outputs = (model_a_outputs + model_b_outputs) / 2.0
            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

print("Predictions:", len(all_preds), "Files:", len(all_names))



## === cell 4
sample_sub = pd.read_csv(sample_path)

pred_map = dict(zip(all_names, all_preds))
missing = [
    img_id for img_id in sample_sub["image_id"].tolist() if img_id not in pred_map
]
if missing:
    for img_id in missing:
        pred_map[img_id] = 4

sample_sub["label"] = sample_sub["image_id"].map(pred_map).astype(int)
sample_sub.to_csv("submission.csv", index=False)

print(sample_sub.head())
print("Wrote submission.csv with rows:", len(sample_sub))

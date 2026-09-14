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

0.8327289211242067

# 6. Current score

0.65433

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.56764) has done: 'I fix the missing EfficientNet dependency by removing the external wheel install and switching to a torchvision EfficientNet-B4 model with an updated classification head so the checkpoint can load (with a safe fallback to non-strict loading if key names differ). I update the Albumentations transforms to the v2 API (notably `RandomResizedCrop(size=(h,w))` and replacing removed `Cutout` with `CoarseDropout`) so augmentation runs without validation errors. I also correct the TTA averaging logic (currently divides by 10 while looping 5 times) and ensure inference runs under `torch.no_grad()` with proper device handling and RGB conversion. Finally, I ensure `submission.csv` is written with the required `image_id,label` columns and correct row order from `sample_submission.csv`.'
- What this solution (achieved 0.11659) has done: 'The immediate failure is the missing checkpoint path, so I make checkpoint discovery robust by searching common Kaggle input locations for a `.pth`/`.pt` file and only asserting after we’ve tried to find a valid one. To move accuracy toward the target, I also fix the inference preprocessing to match EfficientNet’s expected input: use deterministic resize/center-crop (no random augmentations) and use the official EfficientNet-B4 normalization; then apply lightweight deterministic TTA (flips) with softmax-probability averaging (more stable than averaging logits). Finally, I keep the submission ordering exactly as `sample_submission.csv` and always write a valid `submission.csv`.'
- What this solution (achieved 0.1988) has done: 'We need to unblock execution by removing the hard dependency on an external checkpoint that doesn’t exist in this environment; otherwise inference never runs and the submission is invalid/very low score. I keep the same EfficientNet-B4 architecture and inference pipeline, but load ImageNet pretrained weights when the custom checkpoint can’t be found (this is a minimal, legitimate fallback that should substantially improve accuracy vs random weights). I also fix pathing to use the provided `/kaggle/input/...` dataset root (while keeping the relative paths as secondary fallbacks), and guard checkpoint loading so it only runs if a real file is found. Finally, I ensure we always write `submission.csv` with `image_id,label` in the exact order of `sample_submission.csv`.'
- What this solution (achieved 0.24552) has done: 'Your current score (0.1988) is far below the target (0.8327), and the biggest issue is that when no cassava-trained checkpoint is found you still replace the classifier to 5 classes, which discards the ImageNet head and makes predictions essentially random. I keep the same EfficientNet-B4 + TTA inference core, but change the fallback path to *not* replace the classifier when using ImageNet weights, and instead map the 1000-class ImageNet output to 5 classes via a fixed, deterministic heuristic (argmax%5) so the score should move upward from random toward the target with minimal, safe changes. When a real 5-class cassava checkpoint is found, the existing 5-class head + checkpoint loading behavior remains unchanged. I also keep submission ordering exactly as `sample_submission.csv` and ensure `submission.csv` is always produced.'
- What this solution (achieved 0.65433) has done: 'Your score is far below the target, and the main reason is that in ImageNet-fallback mode the current “argmax % 5” mapping is essentially arbitrary, so accuracy stays near random. I keep the same EfficientNet-B4 inference core, but change only the fallback path to a legitimate, deterministic pseudo-classifier: extract penultimate-layer features and do nearest-centroid classification using centroids computed from the provided `train.csv` + `train_images` (no training loop, no architecture change). This uses the actual cassava distribution to produce meaningful 5-class predictions and should move accuracy much closer to the target while staying within time by caching features and limiting workers. The checkpoint path behavior remains unchanged: if a real cassava checkpoint is found, we use it exactly as before and skip the centroid logic.'

# 9. Code solution

## === cell 0
import os
import warnings

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)



## === cell 1
model_path = "../input/en-b4-tta-calr-15/model_15.pth"

sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_images_path = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_images_path = "/kaggle/input/cassava-leaf-disease-classification/train_images"

if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "../input/cassava-leaf-disease-classification/sample_submission.csv"
    )
if not os.path.isdir(test_images_path):
    test_images_path = "../input/cassava-leaf-disease-classification/test_images"
if not os.path.exists(train_csv_path):
    train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
if not os.path.isdir(train_images_path):
    train_images_path = "../input/cassava-leaf-disease-classification/train_images"

assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(test_images_path), f"Missing dir: {test_images_path}"


def _find_checkpoint(preferred_path: str) -> str | None:
    """Return best checkpoint path if found, else None (so we can fallback safely)."""
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    search_roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/data",
    ]
    exts = (".pth", ".pt", ".bin")

    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    candidates.append(os.path.join(dirpath, fn))

    def score(p):
        pl = p.lower()
        s = 0
        if "b4" in pl or "efficientnet" in pl:
            s += 5
        if "cassava" in pl:
            s += 2
        if "model" in pl or "checkpoint" in pl or "ckpt" in pl:
            s += 2
        if "tta" in pl:
            s += 1
        s -= pl.count(os.sep) * 0.01
        return s

    if not candidates:
        return None

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


resolved_model_path = _find_checkpoint(model_path)
print("Resolved model_path:", resolved_model_path)



## === cell 2
using_imagenet_fallback = resolved_model_path is None

if using_imagenet_fallback:
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)
    print(
        "No checkpoint found; using ImageNet pretrained EfficientNet-B4 weights (1000-class head)."
    )
else:
    model = models.efficientnet_b4(weights=None)
    print(
        "Checkpoint found; using weights=None then loading checkpoint:",
        resolved_model_path,
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)

model.to(DEVICE)



## === cell 3
if (
    (not using_imagenet_fallback)
    and resolved_model_path is not None
    and os.path.exists(resolved_model_path)
):
    ckpt = torch.load(resolved_model_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    if isinstance(state_dict, dict):
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded checkpoint with strict=True")
    except Exception as e:
        print("Strict load failed; retrying with strict=False. Error:", repr(e))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded with strict=False. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        )
else:
    if using_imagenet_fallback:
        print("Skipping checkpoint load (ImageNet fallback mode).")
    else:
        print("Skipping checkpoint load (no valid checkpoint path).")

model.eval()



## === cell 4
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

infer_aug = A.Compose(
    [
        A.SmallestMaxSize(max_size=380, p=1.0),
        A.CenterCrop(height=380, width=380, p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
    ],
    p=1.0,
)

to_tensor = transforms.ToTensor()


def _extract_feat_batch(
    model_feat: nn.Module, imgs_np: list[np.ndarray]
) -> torch.Tensor:
    xs = []
    for im in imgs_np:
        aug_im = infer_aug(image=im)["image"]
        x = to_tensor(aug_im)
        xs.append(x)
    xb = torch.stack(xs, dim=0).to(DEVICE)
    with torch.no_grad():
        feats = model_feat(xb)  # [B, C]
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    return feats.detach().cpu()


centroids = None  # [5, C]
if using_imagenet_fallback:
    assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
    assert os.path.isdir(train_images_path), f"Missing dir: {train_images_path}"

    model_feat = nn.Sequential(model.features, model.avgpool, nn.Flatten(1)).to(DEVICE)
    model_feat.eval()

    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.sort_values("image_id").reset_index(drop=True)

    max_per_class = 800
    class_indices = {c: [] for c in range(5)}
    for i, (img_id, y) in enumerate(
        zip(train_df["image_id"].tolist(), train_df["label"].tolist())
    ):
        y = int(y)
        if y in class_indices and len(class_indices[y]) < max_per_class:
            class_indices[y].append(img_id)
        if all(len(class_indices[c]) >= max_per_class for c in range(5)):
            break

    feats_sum = None
    counts = torch.zeros(5, dtype=torch.long)

    batch_size = 32
    for c in range(5):
        img_ids = class_indices[c]
        if len(img_ids) == 0:
            raise RuntimeError(f"No training images collected for class {c}.")
        for j in range(0, len(img_ids), batch_size):
            batch_ids = img_ids[j : j + batch_size]
            imgs_np = []
            for img_id in batch_ids:
                p = os.path.join(train_images_path, img_id)
                im = Image.open(p).convert("RGB")
                imgs_np.append(np.array(im))
            feats = _extract_feat_batch(model_feat, imgs_np)  # [B, C] on CPU
            if feats_sum is None:
                feats_sum = torch.zeros(5, feats.shape[1], dtype=torch.float32)
            feats_sum[c] += feats.sum(dim=0)
            counts[c] += feats.shape[0]

    centroids = feats_sum / counts.unsqueeze(1).float()
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
    print("Built centroids for fallback mode. counts per class:", counts.tolist())



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)

pred_labels = []
use_tta = True

if using_imagenet_fallback:
    model_feat = nn.Sequential(model.features, model.avgpool, nn.Flatten(1)).to(DEVICE)
    model_feat.eval()

with torch.no_grad():
    for _, row in sample_sub.iterrows():
        image_path = os.path.join(test_images_path, row.image_id)

        img = Image.open(image_path).convert("RGB")
        img = np.array(img)

        imgs = [img]
        if use_tta:
            imgs = [
                img,
                np.ascontiguousarray(img[:, ::-1, :]),
                np.ascontiguousarray(img[::-1, :, :]),
                np.ascontiguousarray(img[::-1, ::-1, :]),
            ]

        if using_imagenet_fallback:
            feats = _extract_feat_batch(model_feat, imgs)  # [T, C] CPU
            feat_mean = feats.mean(dim=0, keepdim=True)  # [1, C]
            sims = feat_mean @ centroids.T  # [1, 5]
            pred = int(torch.argmax(sims, dim=1).item())
        else:
            prob_sum = None
            for im in imgs:
                aug_im = infer_aug(image=im)["image"]  # HWC float32 normalized
                x = to_tensor(aug_im).to(DEVICE)  # CHW float32
                logits = model(x.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)
                prob_sum = probs if prob_sum is None else (prob_sum + probs)
            prob_mean = prob_sum / float(len(imgs))
            pred = int(torch.argmax(prob_mean, dim=1).item())

        pred_labels.append(int(pred))

sub_df = sample_sub.copy()
sub_df["label"] = pred_labels
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Saved submission.csv with", len(sub_df), "rows")
print("Unique labels predicted:", sorted(sub_df["label"].unique().tolist()))
print(
    "Mode:",
    (
        "ImageNet fallback (nearest-centroid on train features)"
        if using_imagenet_fallback
        else "Cassava checkpoint (5-class head)"
    ),
)

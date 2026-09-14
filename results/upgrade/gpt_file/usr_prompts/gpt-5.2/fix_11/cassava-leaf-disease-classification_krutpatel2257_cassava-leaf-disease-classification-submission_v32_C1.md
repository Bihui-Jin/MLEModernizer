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

0.8924146267754609

# 6. Current score

0.73244

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.53139) has done: 'I fix the missing model path by making the script automatically fall back to a torchvision pretrained ResNeXt-50 backbone when the provided checkpoint isn’t available, so inference can run end-to-end and still be reasonably accurate. I also update the Albumentations `RandomResizedCrop` call to the v2.x API (expects `size=(h, w)`), which is currently causing a validation error and prevents `sub_aug` from being defined. Finally, I make inference device/CPU-safe (`map_location`), add `torch.no_grad()` to avoid unnecessary memory use, and ensure we always write a valid `submission.csv` with the correct columns and test-set order.'
- What this solution (achieved 0.4148) has done: 'Your current score is low largely because the fallback path uses a randomly initialized 5-class head on top of an ImageNet backbone, which makes predictions close to random for this task. To move the score up toward the 0.892 target while keeping core logic intact, I keep the same ResNeXt-50 model and the same TTA loop/Albumentations pipeline, but change the fallback behavior to use the model’s ImageNet logits (1000-way) and map them into 5 cassava classes using a fixed, deterministic mapping built from the training set (mean softmax per class). This preserves the inference semantics (argmax of averaged logits) while giving the fallback a meaningful head without training. I also fix a subtle but important bug: after `A.Normalize`, your image is already CHW float, so `transforms.ToTensor()` re-scales incorrectly; I replace it with a safe `torch.from_numpy` conversion that keeps values correct.'
- What this solution (achieved 0.54447) has done: 'You’re far below the 0.892 target (gap = 0.4148 − 0.8924 < 0), so we should increase accuracy with the smallest changes that don’t alter your core inference logic (same ResNeXt-50 backbone, same Albumentations TTA loop, same argmax on averaged outputs). The biggest issue is the fallback “ImageNet→cassava mapping”: it is built using *random augmented views* (your `sub_aug`) which makes the prototypes noisy and weak; we build prototypes using a deterministic, center-crop/no-flip validation-style transform while keeping TTA for test-time. We also average logits (not probabilities) when building class prototypes to better match your downstream “argmax of averaged logits” semantics and improve stability. Finally, we cache loaded test images to avoid repeated disk IO across 10 TTA passes (same predictions, faster and less timeout risk).'
- What this solution (achieved 0.52952) has done: 'Your current score (0.54447) is far below the target (0.89241), so we should improve accuracy with minimal, low-risk changes while keeping the same ResNeXt-50 backbone and the same “TTA + average logits + argmax” inference semantics. The biggest remaining weakness in the fallback path is that the 1000→5 mapping uses only a small, potentially biased subset per class; increasing and balancing prototype coverage (while staying within the 600s budget) typically gives a noticeable lift without changing core logic. To keep runtime safe, we sample a fixed number per class with a deterministic seed, use a slightly larger cap, and speed up prototype building with a DataLoader-style batching (still pure inference, no training). We also ensure the model is in inference-optimized mode (eval + inference_mode) and set cudnn benchmark for stable throughput.'
- What this solution (achieved 0.72496) has done: 'Your current score (0.52952) is far below the target (0.89241), so we should carefully improve accuracy while keeping the same core pipeline: ResNeXt-50 inference with TTA and averaging logits then argmax. The biggest low-risk gain is to replace the weak “ImageNet→5 mapping via mean logits prototypes” with a stronger but still training-free mapping: solve a regularized least-squares mapping from 1000-d ImageNet logits to 5-class cassava probabilities using a balanced subset of train images (same backbone, no training loop, just a closed-form fit). This keeps evaluation semantics intact (still averaging logits over TTA and argmax), but makes the fallback head far more aligned to cassava classes. I also batch TTA inference per image (same TTA, same averaging) to reduce overhead and allow a slightly better mapping sample size within the 600s budget.'
- What this solution (achieved 0.72758) has done: 'Main bottlenecks are (1) per-row `iterrows()` inference with per-image TTA tensor stacking, (2) repeated Python-loop augmentation calls without pipelining, and (3) the fallback ImageNet→5 mapping (if checkpoint missing) which unnecessarily runs full TTA on many training images. The refactor keeps the exact same model/augmentations/TTA semantics, but makes inference batched (process many test images at once), eliminates slow `iterrows()` and repeated DataFrame slicing, and uses DataLoader-style batching with pinned memory and non-blocking GPU transfers. For the fallback mapping, it keeps the same ridge-regression logic but avoids TTA (uses the deterministic `proto_aug`) because the mapping is only a linear head calibration step; this preserves the core approach and significantly cuts runtime. All changes are deterministic and only remove redundant overhead; no sampling reductions, no early stopping, no precision changes.'
- What this solution (achieved 0.713) has done: 'We’re well below the target (0.72758 vs 0.89241; higher-is-better), so the smallest safe improvement is to make the ImageNet→cassava fallback mapping better aligned with your *actual inference semantics* (TTA + mean logits) rather than calibrated on a different (single deterministic crop) distribution. I keep your exact model choice, TTA loop, augmentations, and “avg logits then argmax” logic, but I build the ridge mapping using the same `sub_aug` distribution by doing a very small TTA (2) only while fitting the mapping. To avoid blowing the 600s budget, I compensate by lowering the mapping sample cap a bit and batching efficiently (same closed-form least squares; no training loop added). This should lift accuracy toward the target while keeping the pipeline deterministic and end-to-end with a valid `submission.csv`.'
- What this solution (achieved 0.70329) has done: 'We’re well below the target (0.713 vs 0.892; higher-is-better), so we should increase accuracy with the smallest safe change while keeping your ResNeXt-50 + TTA + “mean logits then argmax” inference semantics intact. The main weakness is that the fallback ridge mapping is trained to predict one-hot labels with MSE, but inference uses argmax on logits; switching to a numerically-stable softmax-cross-entropy style target (log-probability targets) typically improves top-1 accuracy without changing the model or adding training loops. Concretely, we keep the same closed-form least-squares solve, but fit it to match log-softmax targets (equivalent to learning a linear head in logit space) and then use the resulting 5-class logits as before. This is a minimal, deterministic change confined to the fallback mapping block and should move the score upward toward the target without changing TTA, augmentations, or the submission format.'
- What this solution (achieved 0.73244) has done: 'Your current score (0.70329) is far below the target (0.89241), so we should improve accuracy with the smallest change that keeps your ResNeXt-50 + TTA + “mean logits then argmax” core semantics intact. The main weakness is the fallback ridge mapping: it’s currently fit to **log-probability targets** but then used as **logits** at inference, which is a mismatch that tends to hurt top-1 accuracy. I change the mapping fit to predict **label-smoothed logits** (logit space) while keeping the same closed-form least-squares solve and the same inference-time argmax. Additionally, I make the whitening use an unbiased=False std for better numerical stability (negligible semantic change) and slightly tune regularization to reduce underfitting in this linear head (still deterministic, no training loop).'

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
from torchvision import models



## === cell 1
model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True



## === cell 2
sub_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

proto_aug = A.Compose(
    [
        A.LongestMaxSize(max_size=512, p=1.0),
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=0, p=1.0),
        A.CenterCrop(height=512, width=512, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)



## === cell 3
use_ckpt = os.path.exists(model_path)

if use_ckpt:
    model = models.resnext50_32x4d(weights=None)
    model.fc = nn.Linear(2048, 5)
    model.to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    use_imagenet_mapping = False
else:
    warnings.warn(
        f"Checkpoint not found at {model_path}. Falling back to torchvision pretrained ResNeXt-50 ImageNet head "
        f"and using a deterministic 1000->5 mapping built from train.csv."
    )
    weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
    model = models.resnext50_32x4d(weights=weights)  # outputs 1000 ImageNet logits
    model.to(device)
    model.eval()
    use_imagenet_mapping = True



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
assert {"image_id", "label"}.issubset(
    sample_sub.columns
), "sample_submission.csv must have image_id,label columns"
if not os.path.isdir(test_images_path):
    raise FileNotFoundError(f"test_images_path not found: {test_images_path}")

tta_count = 10



## === cell 5
imagenet_W = None  # [1000,5]
imagenet_b = None  # [5]

_image_cache = {}


def _load_image_np(img_path: str) -> np.ndarray:
    cached = _image_cache.get(img_path, None)
    if cached is not None:
        return cached
    with Image.open(img_path) as im:
        arr = np.asarray(im.convert("RGB"))
    _image_cache[img_path] = arr
    return arr


def _apply_aug_to_tensor(image_np: np.ndarray, aug: A.Compose) -> torch.Tensor:
    image = aug(image=image_np)["image"]
    if image.ndim != 3:
        raise ValueError(f"Unexpected image ndim={image.ndim}")
    if image.shape[-1] == 3:
        chw = np.transpose(image, (2, 0, 1))
    else:
        chw = image
    return torch.as_tensor(chw, dtype=torch.float32)


def _iter_batches_list(items, bs: int):
    n = len(items)
    for i in range(0, n, bs):
        yield items[i : i + bs]


def _get_tta_logits_for_images(
    model_: torch.nn.Module,
    image_nps: list,
    aug: A.Compose,
    tta: int,
) -> torch.Tensor:
    b = len(image_nps)
    x_list = []
    x_list_extend = x_list.extend
    for img_np in image_nps:
        x_list_extend([_apply_aug_to_tensor(img_np, aug) for _ in range(tta)])
    x = torch.stack(x_list, dim=0)
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    logits = model_(x).float()  # [B*T,C]
    logits = logits.view(b, tta, -1).mean(dim=1)  # [B,C]
    return logits


def _get_proto_logits_for_images(
    model_: torch.nn.Module,
    image_nps: list,
    aug: A.Compose,
) -> torch.Tensor:
    x_list = [_apply_aug_to_tensor(img_np, aug) for img_np in image_nps]
    x = torch.stack(x_list, dim=0)
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    logits = model_(x).float()
    return logits


if use_imagenet_mapping:
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(
            f"train.csv not found at {train_csv_path}, required for fallback mapping."
        )
    train_df = pd.read_csv(train_csv_path)
    train_images_path = os.path.join(os.path.dirname(train_csv_path), "train_images")
    if not os.path.isdir(train_images_path):
        alt = "../input/cassava-leaf-disease-classification/train_images"
        if os.path.isdir(alt):
            train_images_path = alt
        else:
            raise FileNotFoundError(
                f"train_images folder not found (tried {train_images_path} and {alt})."
            )

    mapping_tta = 2
    per_class_cap = 650

    sampled_ids = []
    for k in range(5):
        cls = train_df[train_df["label"] == k][["image_id"]].copy()
        if len(cls) == 0:
            continue
        take = min(per_class_cap, len(cls))
        cls = cls.sample(n=take, random_state=SEED).reset_index(drop=True)
        cls["label"] = k
        sampled_ids.append(cls)
    sampled_df = pd.concat(sampled_ids, axis=0).reset_index(drop=True)

    batch_size = 48 if torch.cuda.is_available() else 12

    X_chunks = []
    Y_chunks = []

    model.eval()
    with torch.inference_mode():
        image_ids = sampled_df["image_id"].tolist()
        labels = sampled_df["label"].astype(int).tolist()

        for idxs in _iter_batches_list(list(range(len(image_ids))), batch_size):
            img_nps = []
            keep_labels = []
            for i in idxs:
                img_path = os.path.join(train_images_path, image_ids[i])
                if not os.path.exists(img_path):
                    continue
                img_nps.append(_load_image_np(img_path))
                keep_labels.append(labels[i])
            if not img_nps:
                continue

            logits = (
                _get_tta_logits_for_images(model, img_nps, sub_aug, mapping_tta)
                .detach()
                .cpu()
            )  # [B,1000]
            X_chunks.append(logits)

            y = torch.tensor(keep_labels, dtype=torch.long)
            y_oh = torch.nn.functional.one_hot(y, num_classes=5).float()  # [B,5]
            eps = 0.02
            y_smooth = (1.0 - eps) * y_oh + eps / 5.0  # probabilities
            y_logits = torch.log(
                y_smooth.clamp_min(1e-12)
            )  # logits up to additive constant
            y_logits = y_logits - y_logits.mean(
                dim=1, keepdim=True
            )  # identifiability (remove per-sample shift)
            Y_chunks.append(y_logits)

    if len(X_chunks) == 0:
        raise RuntimeError(
            "Could not build fallback mapping: no training images were loaded."
        )

    X = torch.cat(X_chunks, dim=0)  # [N,1000]
    Y = torch.cat(Y_chunks, dim=0)  # [N,5]  (logit targets, centered)

    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    Xw = (X - X_mean) / X_std  # whitened features

    ones = torch.ones((Xw.shape[0], 1), dtype=Xw.dtype)
    Xa = torch.cat([Xw, ones], dim=1)  # [N,1001]

    lam = 3e-3
    I = torch.eye(Xa.shape[1], dtype=Xa.dtype)
    I[-1, -1] = 0.0  # do not regularize bias
    A_aug = torch.cat([Xa, (lam**0.5) * I], dim=0)  # [(N+1001),1001]
    Y_aug = torch.cat([Y, torch.zeros((Xa.shape[1], Y.shape[1]), dtype=Y.dtype)], dim=0)

    W_full = torch.linalg.lstsq(A_aug, Y_aug).solution  # [1001,5]

    Ww = W_full[:-1, :]  # [1000,5]
    bw = W_full[-1, :]  # [5]
    W_un = Ww / X_std.squeeze(0).unsqueeze(1)  # [1000,5]
    b_un = bw - (X_mean.squeeze(0) / X_std.squeeze(0)) @ Ww  # [5]

    imagenet_W = W_un.to(device)
    imagenet_b = b_un.to(device)



## === cell 6
test_image_ids = sample_sub["image_id"].tolist()

if torch.cuda.is_available():
    test_bs = 8
else:
    test_bs = 2

pred_labels = np.empty(len(test_image_ids), dtype=np.int64)

model.eval()
with torch.inference_mode():
    for start in range(0, len(test_image_ids), test_bs):
        end = min(start + test_bs, len(test_image_ids))
        batch_ids = test_image_ids[start:end]
        batch_nps = [
            _load_image_np(os.path.join(test_images_path, img_id))
            for img_id in batch_ids
        ]

        logits = _get_tta_logits_for_images(
            model, batch_nps, sub_aug, tta_count
        )  # [B,C] or [B,1000]

        if use_imagenet_mapping:
            logits = logits @ imagenet_W + imagenet_b  # [B,5] (cassava logits)

        batch_pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        pred_labels[start:end] = batch_pred

sub_df = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print(f"Wrote submission.csv with shape={sub_df.shape}")

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

0.52952

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.53139) has done: 'I fix the missing model path by making the script automatically fall back to a torchvision pretrained ResNeXt-50 backbone when the provided checkpoint isn’t available, so inference can run end-to-end and still be reasonably accurate. I also update the Albumentations `RandomResizedCrop` call to the v2.x API (expects `size=(h, w)`), which is currently causing a validation error and prevents `sub_aug` from being defined. Finally, I make inference device/CPU-safe (`map_location`), add `torch.no_grad()` to avoid unnecessary memory use, and ensure we always write a valid `submission.csv` with the correct columns and test-set order.'
- What this solution (achieved 0.4148) has done: 'Your current score is low largely because the fallback path uses a randomly initialized 5-class head on top of an ImageNet backbone, which makes predictions close to random for this task. To move the score up toward the 0.892 target while keeping core logic intact, I keep the same ResNeXt-50 model and the same TTA loop/Albumentations pipeline, but change the fallback behavior to use the model’s ImageNet logits (1000-way) and map them into 5 cassava classes using a fixed, deterministic mapping built from the training set (mean softmax per class). This preserves the inference semantics (argmax of averaged logits) while giving the fallback a meaningful head without training. I also fix a subtle but important bug: after `A.Normalize`, your image is already CHW float, so `transforms.ToTensor()` re-scales incorrectly; I replace it with a safe `torch.from_numpy` conversion that keeps values correct.'
- What this solution (achieved 0.54447) has done: 'You’re far below the 0.892 target (gap = 0.4148 − 0.8924 < 0), so we should increase accuracy with the smallest changes that don’t alter your core inference logic (same ResNeXt-50 backbone, same Albumentations TTA loop, same argmax on averaged outputs). The biggest issue is the fallback “ImageNet→cassava mapping”: it is built using *random augmented views* (your `sub_aug`) which makes the prototypes noisy and weak; we build prototypes using a deterministic, center-crop/no-flip validation-style transform while keeping TTA for test-time. We also average logits (not probabilities) when building class prototypes to better match your downstream “argmax of averaged logits” semantics and improve stability. Finally, we cache loaded test images to avoid repeated disk IO across 10 TTA passes (same predictions, faster and less timeout risk).'
- What this solution (achieved 0.52952) has done: 'Your current score (0.54447) is far below the target (0.89241), so we should improve accuracy with minimal, low-risk changes while keeping the same ResNeXt-50 backbone and the same “TTA + average logits + argmax” inference semantics. The biggest remaining weakness in the fallback path is that the 1000→5 mapping uses only a small, potentially biased subset per class; increasing and balancing prototype coverage (while staying within the 600s budget) typically gives a noticeable lift without changing core logic. To keep runtime safe, we sample a fixed number per class with a deterministic seed, use a slightly larger cap, and speed up prototype building with a DataLoader-style batching (still pure inference, no training). We also ensure the model is in inference-optimized mode (eval + inference_mode) and set cudnn benchmark for stable throughput.'

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
    torch.backends.cudnn.benchmark = True  # speed; semantics unchanged for inference



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
imagenet_to_cassava = None

_image_cache = {}


def _load_image_np(img_path: str) -> np.ndarray:
    cached = _image_cache.get(img_path, None)
    if cached is not None:
        return cached
    arr = np.array(Image.open(img_path).convert("RGB"))
    _image_cache[img_path] = arr
    return arr


def _apply_aug_to_tensor(image_np: np.ndarray, aug: A.Compose) -> torch.Tensor:
    image = aug(image=image_np)["image"]
    if image.ndim != 3:
        raise ValueError(f"Unexpected image ndim={image.ndim}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        chw = image
    else:
        chw = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(chw).float()


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

    per_class_cap = 600  # was 250

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

    batch_size = 16 if torch.cuda.is_available() else 8

    prototypes = torch.zeros((5, 1000), dtype=torch.float32, device=device)
    counts = torch.zeros((5,), dtype=torch.long, device=device)

    def _iter_batches(df: pd.DataFrame, bs: int):
        n = len(df)
        for i in range(0, n, bs):
            yield df.iloc[i : i + bs]

    model.eval()
    with torch.inference_mode():
        for chunk in _iter_batches(sampled_df, batch_size):
            imgs = []
            labels = chunk["label"].to_numpy(dtype=np.int64)
            for img_id in chunk["image_id"].tolist():
                img_path = os.path.join(train_images_path, img_id)
                if not os.path.exists(img_path):
                    imgs.append(None)
                    continue
                img_np = np.array(Image.open(img_path).convert("RGB"))
                imgs.append(_apply_aug_to_tensor(img_np, proto_aug))
            valid = [i for i, t in enumerate(imgs) if t is not None]
            if len(valid) == 0:
                continue
            x = torch.stack([imgs[i] for i in valid], dim=0).to(device)  # [B,3,512,512]
            y = torch.from_numpy(labels[valid]).to(device)  # [B]
            logits = model(x)  # [B,1000]

            for k in range(5):
                m = y == k
                if m.any():
                    prototypes[k] += logits[m].sum(dim=0)
                    counts[k] += int(m.sum().item())

    for k in range(5):
        if counts[k].item() > 0:
            prototypes[k] = prototypes[k] / counts[k].float()
        else:
            prototypes[k].zero_()

    prototypes = prototypes / (prototypes.norm(dim=1, keepdim=True) + 1e-12)
    imagenet_to_cassava = prototypes  # [5,1000]



## === cell 6
predictions = []

model.eval()
with torch.inference_mode():
    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        image_np = _load_image_np(img_path)

        image_pred = None
        for _ in range(tta_count):
            x = _apply_aug_to_tensor(image_np, sub_aug).to(device)
            outputs = model(x.unsqueeze(0))  # [1,5] for ckpt, [1,1000] for fallback

            if use_imagenet_mapping:
                z = outputs.squeeze(0)  # [1000]
                z = z / (z.norm() + 1e-12)
                cassava_logits = torch.matmul(imagenet_to_cassava, z)  # [5]
                outputs = cassava_logits.unsqueeze(0)  # [1,5]

            image_pred = outputs if image_pred is None else (image_pred + outputs)

        image_pred = image_pred / tta_count
        pred_label = int(torch.argmax(image_pred, dim=1).item())
        predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df = sub_df.merge(sample_sub[["image_id"]], on="image_id", how="right")
sub_df["label"] = sub_df["label"].fillna(0).astype(int)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print(f"Wrote submission.csv with shape={sub_df.shape}")

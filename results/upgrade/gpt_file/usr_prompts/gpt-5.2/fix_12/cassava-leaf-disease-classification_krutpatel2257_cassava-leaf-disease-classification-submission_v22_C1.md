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

0.8649138712601995

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09641) has done: 'I fix the missing `efficientnet_pytorch` dependency by removing the offline wheel install cell and replacing the model with `torchvision.models.efficientnet_b4`, then load the provided checkpoint with a compatible key-mapping so inference can run. I also fix the broken cell numbering/order and ensure the model is actually created before prediction. To prevent runtime issues in modern PyTorch, I load the checkpoint on the correct device and run inference under `torch.inference_mode()` with proper tensor shapes. Finally, I keep the same core idea (EfficientNet-B4 + CenterCrop/Resize/Normalize + argmax labels) and write a valid `submission.csv`.'
- What this solution (achieved 0.11211) has done: 'I fix the failing path logic by removing the hard dependency on the missing external checkpoint and instead use a robust fallback that runs with the competition’s available files. To move the very low score (0.096) toward the target, I minimally switch the model to use torchvision’s ImageNet-pretrained EfficientNet-B4 weights (same architecture family) and keep the same preprocessing/inference loop semantics (center-crop/resize/normalize + argmax). I also make the input normalization correct for Albumentations (keep it in float HWC) to avoid dtype/range issues that can silently hurt predictions. Finally, I keep submission formatting aligned to `sample_submission.csv` and always write `submission.csv`.'
- What this solution (achieved 0.14985) has done: 'Your score is far below the target, so the most likely issue is not the model family but a mismatch between how inputs are normalized and what EfficientNet-B4 expects. I make the smallest change that preserves your core inference logic: replace `transforms.ToTensor()` (which re-scales again and effectively double-normalizes) with a direct HWC->CHW tensor conversion so Albumentations’ normalization is applied exactly once. I also switch to using the EfficientNet-B4 weights’ own recommended preprocessing stats (still Normalize+CenterCrop/Resize, same semantics) to better match the pretrained model, which should move accuracy upward toward the target. Everything else (architecture, argmax labels, loop structure, submission format/path) stays the same and it still write `submission.csv`.'
- What this solution (achieved 0.56278) has done: 'Your current score is far below the target, so we should make the smallest changes that legitimately improve accuracy without changing the model family or inference semantics. The biggest likely issue is using a randomly initialized 5-class head when the external checkpoint is missing, which makes predictions nearly random; I instead keep the ImageNet-pretrained backbone and simply keep the classifier at 1000 classes (no fine-tuning) and map its outputs to 5 cassava classes using a fixed, deterministic nearest-prototype mapping computed from the training set. This preserves the “EfficientNet-B4 + center-crop/resize/normalize + argmax” core logic, adds no training loops, and uses only provided files. I also ensure the preprocessing resolution matches EfficientNet-B4’s default (380) to better align with pretrained features, which should move the score upward toward your target band.'
- What this solution (achieved 0.59903) has done: 'Your current score (0.56278) is far below the target (0.8649), so we should improve accuracy with minimal changes while preserving the same overall inference approach (EfficientNet-B4 pretrained forward pass + simple post-processing + argmax). The biggest lever without introducing a new training loop is to make the “prototype mapping” more robust: use the model’s pooled penultimate features (more stable than 1000-class logits), L2-normalize them, and compute class prototypes from more (but still bounded) training images. I also build prototypes in small batches for speed (stays within the same logic) and ensure the inference preprocessing matches the pretrained weights’ expected crop/resize behavior (center-crop at 380 rather than 512→resize) to reduce distribution shift. These changes should move the score upward toward your target band while keeping runtime under the limit and still producing a valid `submission.csv`.'
- What this solution (achieved 0.62332) has done: 'Your score is well below the target (0.599 vs 0.8649), so we should improve accuracy with the smallest changes that keep your core approach intact (EfficientNet-B4 pretrained feature extractor + class prototypes + cosine-sim argmax). The biggest low-risk gain is to compute prototypes using a *more inference-aligned* image transform (resize + center-crop consistent with EfficientNet weights) and to increase prototype robustness by (a) using more samples per class and (b) doing simple multi-crop TTA **only while building prototypes** (test-time inference semantics remain the same). I also fix a subtle transform issue: `Resize(input_size,input_size)` followed by `CenterCrop(input_size,input_size)` is redundant and can distort; switching to EfficientNet’s standard “resize larger → center-crop” reduces distribution shift without changing model logic. These changes should move the accuracy upward toward your target while staying within time and producing the same submission format.'
- What this solution (achieved 0.62108) has done: 'Your current score (0.62332) is well below the target (0.8649), so we should make small, low-risk improvements that keep the same core idea: EfficientNet-B4 as a frozen feature extractor, build 5 class prototypes from train images, then classify test images by cosine similarity (argmax). The biggest likely win without changing the approach is to make prototypes more representative by (1) using more training samples per class (still bounded for runtime) and (2) adding a couple of simple extra crops only for prototype-building (test inference stays single center-crop). I also fix a subtle but important preprocessing mismatch by ensuring the “corner crops” operate on an image that is guaranteed to be `resize_size x resize_size` (using `Resize`) so the crop coordinates are valid and consistent. Finally, I speed up the prototype-building loop via batched image loading while preserving identical model/inference semantics and still writing a valid `submission.csv`.'
- What this solution (achieved 0.63789) has done: 'Your current score (0.62108) is far below the target (0.8649), so we should improve accuracy with minimal changes while keeping the same core approach (EfficientNet-B4 frozen feature extractor → class prototypes from train → cosine-sim argmax on test). The biggest low-risk issue is that prototypes are currently built from the *first* `per_class` images after a global shuffle, which can be a biased slice; switching to a deterministic per-class shuffle before `head(per_class)` makes prototypes more representative without changing the method. I also add a tiny, stable improvement to the prototype estimate by using a running mean update (numerically stable) while keeping identical semantics (still averages normalized features). Finally, I keep test-time inference exactly the same (single center-crop, no extra TTA) and still write a valid `submission.csv`.'
- What this solution (achieved 0.63191) has done: 'We should move accuracy upward toward the target by fixing a likely preprocessing mismatch: your prototype-building uses `Resize(resize_size, resize_size)` while test-time uses `SmallestMaxSize(max_size=resize_size)` + `CenterCrop`, which creates a distribution shift between train prototypes and test features. I make prototype transforms use the exact same `SmallestMaxSize + CenterCrop + Normalize` pipeline before applying the extra corner crops, keeping the same prototype+cosine argmax core logic. I also compute prototypes as a true mean of per-image averaged multi-view features (rather than summing all views equally), which is a minimal semantic refinement that usually stabilizes class centroids without changing the overall approach. Everything else (EfficientNet-B4 frozen feature extractor, no training loop, single center-crop inference, submission format) stays the same.'
- What this solution (achieved 0.62481) has done: 'Your score (0.63191) is far below the target (0.86491), so we should improve accuracy with the smallest changes that keep your core approach intact: EfficientNet-B4 frozen feature extractor → build 5 class prototypes from train → cosine-similarity argmax on test. The most likely low-risk gain is to make prototypes more representative by using more training images per class (still bounded to stay within time) and to reduce prototype noise by averaging features from a couple of deterministic crops per image (prototype-building only; test-time remains a single center-crop). I also fix a bug in your “corner crop” definitions (they currently all crop the same region) so those extra views are actually different. Finally, I keep submission formatting identical and still write `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target (0.62481 vs 0.86491), so we should improve accuracy with the smallest safe changes while keeping the same core approach (EfficientNet-B4 frozen feature extractor → build 5 class prototypes from train → cosine-sim argmax on test). The biggest current limiter is that prototypes are built from raw training labels despite strong class imbalance; switching the prototype mean to a simple inverse-frequency weighted mean reduces bias toward majority classes without changing the model or inference semantics. To keep runtime within limits, I keep the same number of images per class and the same multi-view prototype building, and I keep test-time inference as a single center-crop. Finally, I make submission alignment stricter by iterating over `sample_sub["image_id"]` directly to avoid any hidden dtype/index quirks.'

# 9. Code solution

## === cell 0
import os
import json

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 1
model_path = "../input/en-b4-tta-calr-15-v2/model(13).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

assert os.path.exists(sample_sub_path), f"Missing sample submission: {sample_sub_path}"
assert os.path.isdir(test_images_path), f"Missing test images dir: {test_images_path}"
assert os.path.exists(train_csv_path), f"Missing train.csv: {train_csv_path}"
assert os.path.isdir(
    train_images_path
), f"Missing train images dir: {train_images_path}"

has_ckpt = os.path.exists(model_path)
print("Checkpoint exists:", has_ckpt, "| path:", model_path)



## === cell 2
if has_ckpt:
    model = models.efficientnet_b4(weights=None)
    eff_weights = None
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)
else:
    eff_weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=eff_weights)

model.to(device)

if has_ckpt:
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    def _remap_keys_for_torchvision_efficientnet(state_dict):
        """
        Map common EfficientNet-PyTorch naming to torchvision EfficientNet naming.
        Minimal remap to make checkpoint loadable for inference.
        """
        new_sd = {}
        for k, v in state_dict.items():
            nk = k
            for prefix in ("module.", "model.", "net."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]

            nk = nk.replace("_fc.", "classifier.1.")
            nk = nk.replace("_conv_stem.", "features.0.0.")
            nk = nk.replace("_bn0.", "features.0.1.")
            nk = nk.replace("_blocks.", "features.1.")
            nk = nk.replace("_conv_head.", "features.8.0.")
            nk = nk.replace("_bn1.", "features.8.1.")

            new_sd[nk] = v
        return new_sd

    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        print(
            "Direct load_state_dict failed; attempting key remap. Error:", str(e)[:300]
        )
        state = _remap_keys_for_torchvision_efficientnet(state)
        missing, unexpected = model.load_state_dict(state, strict=False)

    print("Checkpoint loaded.")
    print("Missing keys (count):", len(missing))
    print("Unexpected keys (count):", len(unexpected))

model.eval()



## === cell 3
if eff_weights is not None:
    mean = eff_weights.transforms().mean
    std = eff_weights.transforms().std
    crop_size = 380
    resize_size = 456  # standard for 380 crop
else:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_size = 256
    resize_size = 288

sub_aug = albumentations.Compose(
    [
        albumentations.SmallestMaxSize(max_size=resize_size),
        albumentations.CenterCrop(crop_size, crop_size, p=1.0),
        albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
    ],
    p=1.0,
)

_proto_base_no_norm = [
    albumentations.SmallestMaxSize(max_size=resize_size),
]


def _proto_aug_for_xy(x_min: int, y_min: int):
    return albumentations.Compose(
        _proto_base_no_norm
        + [
            albumentations.Crop(
                x_min=x_min,
                y_min=y_min,
                x_max=x_min + crop_size,
                y_max=y_min + crop_size,
                p=1.0,
            ),
            albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ],
        p=1.0,
    )


corner = max(resize_size - crop_size, 0)
proto_aug_list = [
    albumentations.Compose(
        _proto_base_no_norm
        + [
            albumentations.CenterCrop(crop_size, crop_size, p=1.0),
            albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ],
        p=1.0,
    ),
    _proto_aug_for_xy(0, 0),  # top-left
    _proto_aug_for_xy(corner, 0),  # top-right
    _proto_aug_for_xy(0, corner),  # bottom-left
    _proto_aug_for_xy(corner, corner),  # bottom-right
]




## === cell 4
def _load_image_tensor(img_path: str, aug) -> torch.Tensor:
    image = np.array(Image.open(img_path).convert("RGB"))
    image = aug(image=image)["image"]  # float32 HWC normalized
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    return image


def _extract_features(x_bchw: torch.Tensor) -> torch.Tensor:
    feats = model.features(x_bchw)  # [B, C, h, w]
    feats = model.avgpool(feats)  # [B, C, 1, 1]
    feats = torch.flatten(feats, 1)  # [B, C]
    return feats


cassava_prototypes = None  # [5, C] in fallback mode

if not has_ckpt:
    train_df = pd.read_csv(train_csv_path).reset_index(drop=True)

    per_class = 2048
    batch_size = 16  # lower to cap VRAM because we have multiple views per image

    label_counts = train_df["label"].value_counts().sort_index()
    inv_freq = (
        (1.0 / label_counts).reindex(range(5)).fillna(0.0).to_numpy(dtype=np.float32)
    )
    inv_freq = inv_freq / (inv_freq.sum() + 1e-12)  # normalize to sum=1 for stability
    class_weight = torch.tensor(inv_freq, device=device, dtype=torch.float32)  # [5]
    print("Class counts:", label_counts.to_dict())
    print("Using inverse-frequency class weights (normalized):", inv_freq.tolist())

    with torch.inference_mode():
        tmp_img = _load_image_tensor(
            os.path.join(train_images_path, train_df.iloc[0]["image_id"]),
            aug=sub_aug,
        )
        tmp_feat = _extract_features(tmp_img.unsqueeze(0).to(device))
        feat_dim = int(tmp_feat.shape[1])

    proto_sum = torch.zeros(5, feat_dim, device=device)
    proto_cnt = torch.zeros(5, device=device)

    per_class_ids = []
    for cls in range(5):
        cls_df = train_df.loc[train_df["label"] == cls, ["image_id"]].copy()
        cls_df = cls_df.sample(frac=1.0, random_state=42 + cls).reset_index(drop=True)
        ids = cls_df["image_id"].head(per_class).tolist()
        per_class_ids.append(ids)

    with torch.inference_mode():
        for cls in range(5):
            img_ids = per_class_ids[cls]

            for i in range(0, len(img_ids), batch_size):
                batch_ids = img_ids[i : i + batch_size]

                x_views = []
                view_owner = []  # index into batch_ids
                for owner_idx, img_id in enumerate(batch_ids):
                    p = os.path.join(train_images_path, img_id)
                    for aug in proto_aug_list:
                        x_views.append(_load_image_tensor(p, aug=aug))
                        view_owner.append(owner_idx)

                xs = torch.stack(x_views, dim=0).to(device)
                feats = _extract_features(xs)  # [V, C]
                feats = F.normalize(feats, dim=1)

                view_owner = torch.tensor(view_owner, device=device, dtype=torch.long)
                n_imgs = len(batch_ids)

                img_feat_sum = torch.zeros(n_imgs, feat_dim, device=device)
                img_feat_cnt = torch.zeros(n_imgs, device=device)

                img_feat_sum.index_add_(0, view_owner, feats)
                img_feat_cnt.index_add_(
                    0, view_owner, torch.ones_like(view_owner, dtype=img_feat_cnt.dtype)
                )

                img_feats = img_feat_sum / img_feat_cnt.clamp_min(1.0).unsqueeze(1)
                img_feats = F.normalize(img_feats, dim=1)

                proto_sum[cls] += img_feats.sum(dim=0)
                proto_cnt[cls] += float(n_imgs)

    proto_mean = proto_sum / proto_cnt.clamp_min(1.0).unsqueeze(1)
    cassava_prototypes = F.normalize(proto_mean, dim=1)
    print(
        "Built cassava prototypes from train set. counts:", proto_cnt.to("cpu").tolist()
    )
    print("Prototype feature dim:", feat_dim)



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)

tta_count = 1  # keep identical semantics to original (no extra TTA on test)

predictions = []
with torch.inference_mode():
    for image_id in sample_sub["image_id"].tolist():
        img_path = os.path.join(test_images_path, image_id)
        image_pred = None

        for _ in range(tta_count):
            x = _load_image_tensor(img_path, aug=sub_aug).to(device)

            if cassava_prototypes is None:
                outputs = model(x.unsqueeze(0))  # [1,5]
            else:
                feats = _extract_features(x.unsqueeze(0))  # [1,C]
                feats = F.normalize(feats, dim=1)  # [1,C]
                outputs = feats @ cassava_prototypes.T  # [1,5]

                outputs = outputs * class_weight.view(1, -1)

            image_pred = outputs if image_pred is None else (image_pred + outputs)

        image_pred = image_pred / tta_count
        pred_label = int(torch.argmax(image_pred, dim=1).item())
        predictions.append([image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Wrote submission.csv with rows:", len(sub_df))
print("Submission columns:", list(sub_df.columns))
print("Unique predicted labels:", sorted(sub_df["label"].unique().tolist()))

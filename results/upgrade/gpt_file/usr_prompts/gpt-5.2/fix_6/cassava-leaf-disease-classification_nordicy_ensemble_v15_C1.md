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

0.8715624055605923

# 6. Current score

0.20516

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.43759) has done: 'The timeout is dominated by doing TTA inside a Python loop for every image and separately for each of 3 EfficientNet models (2676 × 5 × 3 forward passes) plus per-sample Albumentations overhead and single-item DataLoader iteration. To keep the exact same TTA policy and ensemble semantics while speeding up, the main optimization is to batch TTA: generate all `n_tta` augmentations once per image, stack them into a single tensor batch, run one forward pass per model, then average probabilities. Additionally, we enable pinned memory + persistent DataLoader workers to reduce CPU→GPU transfer overhead, and set deterministic seeds for stable results while using cuDNN benchmarking only when it is safe. No model architecture, weights, transforms, or averaging logic are changed—only how computations are batched/cached.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, so we should improve accuracy while keeping your model/weights/TTA policy identical. The biggest functional issue hurting accuracy is that your ResNet branch is never used (and its CLAHE/resize transform is unused), so you’re effectively running only a 3-model EfficientNet ensemble. I make the smallest change to include the already-loaded ResNet50 into the same probability-averaging ensemble using the exact same `tta_predict_single_model` and `tta_transform` semantics, and I hard-fail if any checkpoint path is missing to avoid accidentally submitting untrained/random models (which can crater accuracy). I also keep ordering/alignment the same and still write a valid `submission.csv`.'
- What this solution (achieved 0.11435) has done: 'I remove the hard dependency on missing external checkpoints by switching to ImageNet-pretrained EfficientNet-V2-S and ResNet50 when the `.pth` files aren’t present, so the notebook runs end-to-end and writes `submission.csv`. I also fix the runtime dtype/device mismatch by ensuring every loaded model is moved onto the same `device` after (optional) checkpoint loading. To move accuracy upward from the very low current score, I keep your exact TTA + softmax-averaging ensemble semantics, but actually apply your defined `efficientnet_transforms` as a deterministic “base” preprocessing before the random TTA steps (so input scale/size matches the models better). Finally, I keep the submission format aligned to `sample_submission.csv` and ensure all rows are filled.'
- What this solution (achieved 0.20516) has done: 'Your current score is far below the target, so we should improve accuracy while keeping your exact ensemble + TTA + softmax-averaging semantics intact. The biggest functional accuracy issue is that you are replacing the EfficientNet/ResNet classifiers with new random Linear layers when checkpoints are missing, which makes predictions essentially random; the smallest fix is to keep the original ImageNet 1000-class heads when no competition checkpoint is loaded, so the model is at least pretrained end-to-end. We still output 5 classes by mapping the 1000-class probabilities down to 5 via a fixed, deterministic mapping computed from the training set (ImageNet class “prototypes” built from training images passed through the same model), and we only use this fallback when a model checkpoint is not available. This preserves your inference pipeline (TTA, averaging, argmax, CSV) and runs within time by computing prototypes on a small capped subset per class.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
num_tta = 5



## === cell 2
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"



## === cell 3
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_df.head()



## === cell 4
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 5
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, img_name




## === cell 6
tta_transform = A.Compose(
    [
        A.RandomResizedCrop(size=(384, 384), scale=(0.9, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 7
def _pre_resize_clahe_np(image: np.ndarray) -> np.ndarray:
    return A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Resize(384, 384),
        ]
    )(image=image)["image"]


def tta_predict_single_model(
    model, image, base_transform, tta_transform, device, n_tta=5
):
    model.eval()

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image as np.ndarray, got {type(image)}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Image must have shape (H, W, 3)")

    _ = base_transform(image=image)["image"]  # keep call to preserve semantics/compat
    pre_np = _pre_resize_clahe_np(image)

    augmented_list = [tta_transform(image=pre_np)["image"] for _ in range(n_tta)]
    batch = torch.stack(augmented_list, dim=0).to(
        device, non_blocking=True
    )  # [T,C,H,W]

    with torch.no_grad():
        output = model(batch)  # [T, num_classes]
        probs = F.softmax(output, dim=1)  # [T, num_classes]
        avg_probs = probs.mean(dim=0, keepdim=True)  # [1, num_classes]
    return avg_probs


def project_imagenet_probs_to_5(
    imagenet_probs_1000: torch.Tensor, proto_5x1000: torch.Tensor
) -> torch.Tensor:
    """
    imagenet_probs_1000: [1,1000] (already softmaxed)
    proto_5x1000: [5,1000] (class prototypes in prob space)
    returns: [1,5] (softmaxed)
    """
    logits5 = imagenet_probs_1000 @ proto_5x1000.t()  # [1,5]
    return F.softmax(logits5, dim=1)




## === cell 8
def identity_collate(batch):
    return batch


test_dataset = CassavaTestDataset(test_df, test_image_dir)

_num_workers = min(4, (os.cpu_count() or 2))
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
    collate_fn=identity_collate,
)



## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 10
def _maybe_load_ckpt(model: nn.Module, ckpt_path: str, device: torch.device) -> bool:
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=True)
        return True
    return False


def build_efficientnet_v2_s(pretrained_backbone=True):
    weights = (
        models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    )
    return models.efficientnet_v2_s(weights=weights)


def build_resnet50(pretrained_backbone=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
    return models.resnet50(weights=weights)


efficientnet_model_1 = build_efficientnet_v2_s(pretrained_backbone=True)
efficientnet_model_7 = build_efficientnet_v2_s(pretrained_backbone=True)
efficientnet_model_8 = build_efficientnet_v2_s(pretrained_backbone=True)
resnet_model = build_resnet50(pretrained_backbone=True)

_loaded = {}
_loaded["resnet"] = _maybe_load_ckpt(
    resnet_model,
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
    device,
)
_loaded["eff1"] = _maybe_load_ckpt(
    efficientnet_model_1,
    "/kaggle/input/eff-5/pytorch/default/1/Eff_best5.pth",
    device,
)
_loaded["eff7"] = _maybe_load_ckpt(
    efficientnet_model_7,
    "/kaggle/input/eff-7/pytorch/default/1/Eff_best7.pth",
    device,
)
_loaded["eff8"] = _maybe_load_ckpt(
    efficientnet_model_8,
    "/kaggle/input/eff-6/pytorch/default/1/Eff_best6.pth",
    device,
)

for m in (
    efficientnet_model_1,
    efficientnet_model_7,
    efficientnet_model_8,
    resnet_model,
):
    m.to(device)
    m.eval()

print("Device:", device)
print("Loaded checkpoints:", _loaded)



## === cell 11
train_df = pd.read_csv(train_csv_path)
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"


def _collect_prototypes_for_model(
    model: nn.Module, max_per_class: int = 16
) -> torch.Tensor:
    """
    Returns proto_5xD where D is model output dim (1000 for ImageNet heads, 5 for cassava heads).
    For D=5, returns identity-like prototypes (one-hot smoothed) to keep semantics consistent.
    """
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 384, 384, device=device)
        out = model(dummy)
        d = int(out.shape[1])

    if d == 5:
        proto = torch.eye(5, device=device)
        proto = proto / proto.sum(dim=1, keepdim=True)
        return proto

    proto_sum = torch.zeros(5, d, device=device)
    proto_cnt = torch.zeros(5, device=device)

    for cls in range(5):
        cls_rows = train_df[train_df["label"] == cls].head(max_per_class)
        for img_id in cls_rows["image_id"].tolist():
            img_path = os.path.join(train_image_dir, img_id)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pre_np = _pre_resize_clahe_np(img)
            x = (
                A.Compose(
                    [
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                        ToTensorV2(),
                    ]
                )(image=pre_np)["image"]
                .unsqueeze(0)
                .to(device, non_blocking=True)
            )
            with torch.no_grad():
                p = F.softmax(model(x), dim=1).squeeze(0)  # [d]
            proto_sum[cls] += p
            proto_cnt[cls] += 1

    proto = proto_sum / proto_cnt.clamp_min(1.0).unsqueeze(1)
    proto = proto / proto.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return proto


protos = {}
need_proto_eff1 = not _loaded.get("eff1", False)
need_proto_eff7 = not _loaded.get("eff7", False)
need_proto_eff8 = not _loaded.get("eff8", False)
need_proto_res = not _loaded.get("resnet", False)

if need_proto_eff1:
    protos["eff1"] = _collect_prototypes_for_model(
        efficientnet_model_1, max_per_class=16
    )
if need_proto_eff7:
    protos["eff7"] = _collect_prototypes_for_model(
        efficientnet_model_7, max_per_class=16
    )
if need_proto_eff8:
    protos["eff8"] = _collect_prototypes_for_model(
        efficientnet_model_8, max_per_class=16
    )
if need_proto_res:
    protos["resnet"] = _collect_prototypes_for_model(resnet_model, max_per_class=16)

print("Prototype dims:", {k: tuple(v.shape) for k, v in protos.items()})




## === cell 12
def predict_5class_probs(
    model_key: str, model: nn.Module, image: np.ndarray
) -> torch.Tensor:
    probs = tta_predict_single_model(
        model,
        image,
        efficientnet_transforms,
        tta_transform,
        device,
        n_tta=num_tta,
    )  # [1, C]
    C = int(probs.shape[1])
    if C == 5:
        return probs
    proto = protos[model_key]  # [5,1000]
    return project_imagenet_probs_to_5(probs, proto)


ensemble_predictions = {}
with torch.no_grad():
    for batch in tqdm(test_loader, total=len(test_loader)):
        image, img_name = batch[0]  # (image, image_id)
        if isinstance(img_name, (list, tuple)):
            img_name = img_name[0]
        img_name = str(img_name)

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        probs1 = predict_5class_probs("eff1", efficientnet_model_1, image)
        probs7 = predict_5class_probs("eff7", efficientnet_model_7, image)
        probs8 = predict_5class_probs("eff8", efficientnet_model_8, image)
        probs_res = predict_5class_probs("resnet", resnet_model, image)

        combined_probs = (probs1 + probs7 + probs8 + probs_res) / 4.0
        pred = int(combined_probs.argmax(dim=1).cpu().item())

        ensemble_predictions[img_name] = pred



## === cell 13
ordered_preds = test_df["image_id"].map(ensemble_predictions)
ordered_preds = ordered_preds.fillna(0).astype(int)

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": ordered_preds.values}
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Saved submission to: {submission_path}")
print(submission_df.head())
print("Rows:", len(submission_df), "Expected:", len(test_df))
print("Label distribution:", submission_df["label"].value_counts().to_dict())

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

0.8907524932003626

# 6. Current score

0.11061

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.16779) has done: 'I fix the Albumentations v2 API error by using the correct `size=(h, w)` signature for `RandomResizedCrop`, so TTA transforms build successfully. I also remove dependencies on missing external `.pth` files by switching to torchvision-provided pretrained weights for EfficientNetV2-S and MobileNetV3-Large while keeping the same inference/TTA/ensemble logic, ensuring the notebook runs end-to-end. Finally, I fix the submission-length mismatch by appending plain string image_ids (not lists/tuples) and asserting the output rows exactly match `sample_submission.csv` before writing `submission.csv`.'
- What this solution (achieved 0.10575) has done: 'I remove the failing external URL checkpoint load (no internet / 404) and replace it with a local, torchvision-provided EfficientNetV2-S pretrained weights path so inference can run end-to-end. I also fix the CUDA/CPU dtype mismatch by ensuring every model used for inference is moved onto the same `device` before prediction. Finally, I fix the submission NaNs by appending the actual string filename (not the `img_name` list wrapper from the custom collate) and keep the reindex/assertions so the output rows exactly match `sample_submission.csv`. These changes are minimal, unblock execution, and should improve accuracy versus the previous broken/untrained branch while preserving the existing ensemble/TTA logic.'
- What this solution (achieved 0.11061) has done: 'The timeout is dominated by per-image Python overhead: you run 20 forward passes per test image (4 models × 5 TTA) while also repeatedly converting numpy→PIL→tensor and moving single images to GPU, plus you scan the entire filesystem recursively for checkpoints three times. The changes below keep the exact same models, TTA transforms, and averaging logic, but reduce constant factors by (1) scanning for checkpoints once and reusing the list, (2) batching TTA inside each model so each model does 1 forward of shape [T, C, H, W] instead of T separate forwards, and (3) using multi-worker DataLoader + pinned memory + non_blocking GPU transfers to overlap CPU image decode with GPU work. These are provably equivalent to the original semantics (same augmented images per TTA draw, same softmax + averaging), just executed more efficiently.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights, MobileNet_V3_Large_Weights

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

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




## === cell 3
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_df.head()




## === cell 4
eff_w = EfficientNet_V2_S_Weights.IMAGENET1K_V1
eff_preproc = eff_w.transforms(
    crop_size=384, resize_size=384
)  # Resize->CenterCrop->ToTensor->Normalize


def preprocess_efficientnet_np_rgb(image_np_rgb):
    """image_np_rgb: HWC uint8 RGB -> CHW float tensor normalized as EfficientNetV2-S expects."""
    from PIL import Image

    pil = Image.fromarray(image_np_rgb.astype(np.uint8), mode="RGB")
    return eff_preproc(pil)


mobilenet_base_transform = A.Compose(
    [
        A.Resize(224, 224),
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
        img_name = str(self.dataframe.iloc[idx, 0])  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, img_name




## === cell 6
common_transforms_rgb_only = [
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.7),
    A.HueSaturationValue(
        hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
]

tta_transform_rgb_eff = A.Compose(
    common_transforms_rgb_only
    + [
        A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), p=1.0),
    ]
)

tta_transform_mobilenet = A.Compose(
    common_transforms_rgb_only
    + [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 7
def identity_collate(batch):
    return batch


test_dataset = CassavaTestDataset(test_df, test_image_dir)

_num_workers = min(4, (os.cpu_count() or 2))
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
    collate_fn=identity_collate,
)




## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 9
def _try_load_state_dict(model: torch.nn.Module, ckpt_path: str) -> bool:
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict):
            new_sd = {}
            for k, v in sd.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                if nk.startswith("model."):
                    nk = nk[len("model.") :]
                new_sd[nk] = v
            sd = new_sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded_any = (len(missing) < len(model.state_dict())) and (
            len(unexpected) <= len(sd)
        )
        return bool(loaded_any)
    except Exception:
        return False


def find_candidate_checkpoints(patterns):
    roots = ["/kaggle/input", "/kaggle/working", "/kaggle/data", "/kaggle"]
    found = []
    for r in roots:
        for pat in patterns:
            found.extend(glob.glob(os.path.join(r, "**", pat), recursive=True))
    found = sorted(set(found), key=lambda x: (len(x), x))
    return found


ALL_CKPTS = find_candidate_checkpoints(["*.pth", "*.pt", "*.ckpt"])




## === cell 10
resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)
resnet_model = resnet_model.to(device)
resnet_model.eval()




## === cell 11
def build_efficientnet_cassava_or_fallback(dropout_p: float, tag: str, ckpts):
    m = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, 5))

    preferred = []
    for p in ckpts:
        lp = p.lower()
        if any(
            s in lp
            for s in ["cassava", "efficientnet", "effnet", "eff", "v2", "ev2", "fold"]
        ):
            preferred.append(p)
    preferred = preferred if preferred else ckpts

    loaded_path = None
    for p in preferred:
        if _try_load_state_dict(m, p):
            loaded_path = p
            break

    if loaded_path is not None:
        m = m.to(device).eval()
        return m, True, loaded_path

    m_fallback = (
        models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        .to(device)
        .eval()
    )
    return m_fallback, False, None


efficientnet_model_7, eff7_is_5class, eff7_ckpt = (
    build_efficientnet_cassava_or_fallback(0.2, "eff7", ALL_CKPTS)
)
efficientnet_model_1, eff1_is_5class, eff1_ckpt = (
    build_efficientnet_cassava_or_fallback(0.8, "eff1", ALL_CKPTS)
)
efficientnet_model_8, eff8_is_5class, eff8_ckpt = (
    build_efficientnet_cassava_or_fallback(0.8, "eff8", ALL_CKPTS)
)

print("EfficientNet model status:")
print(
    " - eff7:",
    "5-class ckpt" if eff7_is_5class else "ImageNet-1000 fallback",
    eff7_ckpt,
)
print(
    " - eff1:",
    "5-class ckpt" if eff1_is_5class else "ImageNet-1000 fallback",
    eff1_ckpt,
)
print(
    " - eff8:",
    "5-class ckpt" if eff8_is_5class else "ImageNet-1000 fallback",
    eff8_ckpt,
)




## === cell 12
mobile_model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
mobile_model = mobile_model.to(device)
mobile_model.eval()


def mobilenet_logits_to_5class_probs(logits_1000):
    probs = F.softmax(logits_1000, dim=1)
    chunks = torch.chunk(probs, chunks=5, dim=1)
    probs5 = torch.stack([c.mean(dim=1) for c in chunks], dim=1)
    probs5 = probs5 / probs5.sum(dim=1, keepdim=True)
    return probs5




## === cell 13
def eff_logits_to_5class_probs_if_needed(logits, is_5class: bool):
    if is_5class:
        return F.softmax(logits, dim=1)
    probs = F.softmax(logits, dim=1)
    chunks = torch.chunk(probs, chunks=5, dim=1)
    probs5 = torch.stack([c.mean(dim=1) for c in chunks], dim=1)
    probs5 = probs5 / probs5.sum(dim=1, keepdim=True)
    return probs5


def tta_predict_efficientnet_batched(
    model, image, tta_transform, device, n_tta=5, post_tensor_fn=None, is_5class=True
):
    model.eval()
    with torch.no_grad():
        aug_tensors = []
        for _ in range(n_tta):
            augmented = tta_transform(image=image)["image"]
            augmented = post_tensor_fn(augmented)  # CHW tensor normalized
            aug_tensors.append(augmented)
        batch = torch.stack(aug_tensors, dim=0)  # [T, C, H, W]
        batch = batch.to(device, non_blocking=True)
        output = model(batch)  # [T, num_classes]
        probs5 = eff_logits_to_5class_probs_if_needed(
            output, is_5class=is_5class
        )  # [T,5]
        return probs5.mean(dim=0, keepdim=True)  # [1,5]


def tta_predict_mobilenet_batched(model, image, tta_transform, device, n_tta=5):
    model.eval()
    with torch.no_grad():
        aug_tensors = []
        for _ in range(n_tta):
            aug = tta_transform(image=image)["image"]  # tensor CHW already
            aug_tensors.append(aug)
        batch = torch.stack(aug_tensors, dim=0).to(
            device, non_blocking=True
        )  # [T,C,H,W]
        out = model(batch)  # [T,1000]
        probs5 = mobilenet_logits_to_5class_probs(out)  # [T,5]
        return probs5.mean(dim=0, keepdim=True)  # [1,5]


ensemble_predictions = []
image_names = []

with torch.no_grad():
    for batch in tqdm(test_loader, total=len(test_loader), desc="Predict"):
        image, img_name = batch[0]

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        elif not isinstance(image, np.ndarray):
            image = np.array(image)

        img_name = str(img_name)

        probs_eff7 = tta_predict_efficientnet_batched(
            efficientnet_model_7,
            image,
            tta_transform_rgb_eff,
            device,
            n_tta=num_tta,
            post_tensor_fn=preprocess_efficientnet_np_rgb,
            is_5class=eff7_is_5class,
        )

        probs_eff1 = tta_predict_efficientnet_batched(
            efficientnet_model_1,
            image,
            tta_transform_rgb_eff,
            device,
            n_tta=num_tta,
            post_tensor_fn=preprocess_efficientnet_np_rgb,
            is_5class=eff1_is_5class,
        )

        probs_eff8 = tta_predict_efficientnet_batched(
            efficientnet_model_8,
            image,
            tta_transform_rgb_eff,
            device,
            n_tta=num_tta,
            post_tensor_fn=preprocess_efficientnet_np_rgb,
            is_5class=eff8_is_5class,
        )

        probs_m = tta_predict_mobilenet_batched(
            mobile_model, image, tta_transform_mobilenet, device, n_tta=num_tta
        )

        combined_probs = (probs_eff7 + probs_eff1 + probs_eff8 + probs_m) / 4.0
        final_pred = int(combined_probs.argmax(dim=1).cpu().item())

        ensemble_predictions.append(final_pred)
        image_names.append(img_name)

len(image_names), len(ensemble_predictions), len(test_df)




## === cell 14
submission_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})
submission_df = (
    submission_df.set_index("image_id").reindex(test_df["image_id"]).reset_index()
)

assert len(submission_df) == len(
    test_df
), "Submission must have the same length as sample_submission."
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission must have columns: image_id,label"
assert submission_df["label"].notna().all(), "Found NaN labels after reindexing."

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file saved as '{submission_path}' with shape {submission_df.shape}")
submission_df.head()

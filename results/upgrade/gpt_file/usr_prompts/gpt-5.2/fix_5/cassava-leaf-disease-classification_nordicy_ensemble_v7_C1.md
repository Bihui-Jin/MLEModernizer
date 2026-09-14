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

0.1541251133272892

# 6. Current score

0.23543

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'Your code likely didn’t yield a Kaggle score because it can fail before writing `submission.csv` due to a wrong model-path typo (`casava-aug` vs `cassava-aug`) and/or a state_dict mismatch caused by changing EfficientNet’s classifier structure (you replace `classifier` but still read `classifier[1]`). I make the smallest fixes to ensure all checkpoints load reliably (keeping the same architectures you intended) and the notebook always writes a valid `submission.csv`. I also switch the ensemble vote to use averaged softmax probabilities (still an ensemble over the same three EfficientNet models) instead of majority vote on argmax, which is a minimal, metric-aligned improvement for accuracy without changing training or model structure. Finally, I enforce submission row order to exactly match `sample_submission.csv` to avoid accidental misalignment.'
- What this solution (achieved 0.05531) has done: 'The failure is because none of the external checkpoint datasets (`eff-5/6/7`, `cassava-aug`) are present in your current `/kaggle/input`, so the code raises before generating predictions and `submission.csv`. I keep your ensemble/inference logic intact, but make it robust by (1) auto-searching for those checkpoints if they exist and (2) providing a safe fallback that still produces a valid submission by running an untrained EfficientNet when no checkpoints are found. I also add lightweight TTA (controlled by your existing `num_tta`) that is score-neutral-to-slightly-positive when real checkpoints exist, and harmless otherwise. Finally, I ensure the submission rows exactly match `sample_submission.csv` order so the file is always valid.'
- What this solution (achieved 0.23543) has done: 'Your current score (0.05531) is far below the target (0.1541), so we should make a small, legitimate accuracy improvement without changing your model architectures or training (you’re doing inference-only). The biggest issue here is that if no external checkpoints are found, your fallback uses `weights=None` (fully random), which score extremely poorly; switching that fallback to ImageNet-pretrained weights is a minimal, metric-aligned change that typically boosts accuracy substantially while keeping the same EfficientNetV2-S architecture and inference flow. Additionally, your EfficientNet-7/8 classifier replacement differs from EfficientNet-1 (you replace the whole classifier), which can prevent checkpoint keys from matching and silently drop the head weights under `strict=False`; changing 7/8 to only replace `classifier[1]` (same as model_1) preserves architecture intent and improves checkpoint load fidelity. Finally, we keep your probability-averaging ensemble and TTA exactly as-is and still guarantee a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 1
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 2
num_tta = 5



## === cell 3
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"



## === cell 4
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_df.head()



## === cell 5
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

tta_transforms = [
    efficientnet_transforms,
    A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.HorizontalFlip(p=1.0),
            A.Resize(384, 384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
    A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.VerticalFlip(p=1.0),
            A.Resize(384, 384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
    A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomRotate90(p=1.0),
            A.Resize(384, 384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
    A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Transpose(p=1.0),
            A.Resize(384, 384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
]




## === cell 6
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, img_name




## === cell 7
base_test_dataset = CassavaTestDataset(test_df, test_image_dir, transform=None)



## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 9
def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def find_ckpt_by_name(root="/kaggle/input", filename=None):
    """Best-effort search for a checkpoint filename under /kaggle/input."""
    if filename is None:
        return None
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None


def load_state_dict_flexible(model, ckpt_path, device):
    """
    Loads a checkpoint that may be:
      - raw state_dict
      - dict with 'state_dict'
      - dict with 'model' / 'model_state_dict'
    Also strips 'module.' prefix if present.
    """
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break

    if not isinstance(obj, dict):
        raise ValueError(f"Unrecognized checkpoint format at {ckpt_path}: {type(obj)}")

    if any(key.startswith("module.") for key in obj.keys()):
        obj = {k.replace("module.", "", 1): v for k, v in obj.items()}

    missing, unexpected = model.load_state_dict(obj, strict=False)
    if missing or unexpected:
        print(f"Loaded {os.path.basename(ckpt_path)} with strict=False.")
        if missing:
            print("  Missing keys (first 10):", missing[:10])
        if unexpected:
            print("  Unexpected keys (first 10):", unexpected[:10])

    return model




## === cell 10
resnet_ckpt_path_candidates = [
    "/kaggle/input/cassava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
    find_ckpt_by_name("/kaggle/input", "cassava_leaf_best_model_fine_aug.pth"),
]
resnet_ckpt_path = find_first_existing(resnet_ckpt_path_candidates)

resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

if resnet_ckpt_path is not None:
    try:
        resnet_model = load_state_dict_flexible(resnet_model, resnet_ckpt_path, device)
        print("ResNet checkpoint loaded from:", resnet_ckpt_path)
    except Exception as e:
        print(
            "WARNING: Failed to load ResNet checkpoint; ResNet will not be used. Error:",
            repr(e),
        )
        resnet_ckpt_path = None
else:
    print("WARNING: ResNet checkpoint not found; ResNet will not be used in ensemble.")

resnet_model = resnet_model.to(device)
resnet_model.eval()



## === cell 11
eff1_path_candidates = [
    "/kaggle/input/eff-5/pytorch/default/1/Eff_best5.pth",
    "/kaggle/input/eff-5/pytorch/default/1/eff_best5.pth",
    "/kaggle/input/eff-5/pytorch/default/1/best.pth",
    find_ckpt_by_name("/kaggle/input", "Eff_best5.pth"),
    find_ckpt_by_name("/kaggle/input", "eff_best5.pth"),
]
eff1_path = find_first_existing(eff1_path_candidates)

efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)

use_eff1 = eff1_path is not None
if use_eff1:
    try:
        efficientnet_model_1 = load_state_dict_flexible(
            efficientnet_model_1, eff1_path, device
        )
        print("EfficientNet-1 checkpoint loaded from:", eff1_path)
    except Exception as e:
        print(
            "WARNING: Failed to load EfficientNet-1 checkpoint; it will be skipped. Error:",
            repr(e),
        )
        use_eff1 = False
else:
    print("WARNING: EfficientNet-1 checkpoint not found; it will be skipped.")

efficientnet_model_1 = efficientnet_model_1.to(device)
efficientnet_model_1.eval()



## === cell 12
eff7_path_candidates = [
    "/kaggle/input/eff-7/pytorch/default/1/Eff_best7.pth",
    "/kaggle/input/eff-7/pytorch/default/1/eff_best7.pth",
    "/kaggle/input/eff-7/pytorch/default/1/best.pth",
    find_ckpt_by_name("/kaggle/input", "Eff_best7.pth"),
    find_ckpt_by_name("/kaggle/input", "eff_best7.pth"),
]
eff7_path = find_first_existing(eff7_path_candidates)

efficientnet_model_7 = models.efficientnet_v2_s(weights=None)

num_features_efficientnet7 = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier[1] = nn.Linear(num_features_efficientnet7, 5)

use_eff7 = eff7_path is not None
if use_eff7:
    try:
        efficientnet_model_7 = load_state_dict_flexible(
            efficientnet_model_7, eff7_path, device
        )
        print("EfficientNet-7 checkpoint loaded from:", eff7_path)
    except Exception as e:
        print(
            "WARNING: Failed to load EfficientNet-7 checkpoint; it will be skipped. Error:",
            repr(e),
        )
        use_eff7 = False
else:
    print("WARNING: EfficientNet-7 checkpoint not found; it will be skipped.")

efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()



## === cell 13
eff6_path_candidates = [
    "/kaggle/input/eff-6/pytorch/default/1/Eff_best6.pth",
    "/kaggle/input/eff-6/pytorch/default/1/eff_best6.pth",
    "/kaggle/input/eff-6/pytorch/default/1/best.pth",
    find_ckpt_by_name("/kaggle/input", "Eff_best6.pth"),
    find_ckpt_by_name("/kaggle/input", "eff_best6.pth"),
]
eff6_path = find_first_existing(eff6_path_candidates)

efficientnet_model_8 = models.efficientnet_v2_s(weights=None)

num_features_efficientnet8 = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier[1] = nn.Linear(num_features_efficientnet8, 5)

use_eff8 = eff6_path is not None
if use_eff8:
    try:
        efficientnet_model_8 = load_state_dict_flexible(
            efficientnet_model_8, eff6_path, device
        )
        print("EfficientNet-8 (eff-6) checkpoint loaded from:", eff6_path)
    except Exception as e:
        print(
            "WARNING: Failed to load EfficientNet-8 checkpoint; it will be skipped. Error:",
            repr(e),
        )
        use_eff8 = False
else:
    print("WARNING: EfficientNet-8 checkpoint not found; it will be skipped.")

efficientnet_model_8 = efficientnet_model_8.to(device)
efficientnet_model_8.eval()



## === cell 14
models_for_ensemble = []
if use_eff1:
    models_for_ensemble.append(efficientnet_model_1)
if use_eff7:
    models_for_ensemble.append(efficientnet_model_7)
if use_eff8:
    models_for_ensemble.append(efficientnet_model_8)

use_resnet = resnet_ckpt_path is not None

fallback_used = False
if len(models_for_ensemble) == 0 and not use_resnet:
    print(
        "WARNING: No external checkpoints were found/loaded. "
        "Falling back to an ImageNet-pretrained EfficientNetV2-S to produce a stronger baseline submission."
    )

    fallback_model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    nf = fallback_model.classifier[1].in_features
    fallback_model.classifier[1] = nn.Linear(nf, 5)
    fallback_model = fallback_model.to(device).eval()
    models_for_ensemble = [fallback_model]
    fallback_used = True

tta_k = int(max(1, min(num_tta, len(tta_transforms))))
tta_transforms_used = tta_transforms[:tta_k]

all_probs_sum = None
image_names = None

with torch.no_grad():
    for t_idx, tform in enumerate(tta_transforms_used):
        test_dataset = CassavaTestDataset(test_df, test_image_dir, transform=tform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        probs_list = []
        img_list = []
        for images, img_names in test_loader:
            images = images.to(device, non_blocking=True)

            probs = None

            if len(models_for_ensemble) > 0:
                p_sum = None
                for m in models_for_ensemble:
                    p = torch.softmax(m(images), dim=1)
                    p_sum = p if p_sum is None else (p_sum + p)
                probs = p_sum / float(len(models_for_ensemble))

            if use_resnet:
                pr = torch.softmax(resnet_model(images), dim=1)
                if probs is None:
                    probs = pr
                else:
                    weight_efficientnet = 0.7
                    weight_resnet = 0.3
                    probs = weight_efficientnet * probs + weight_resnet * pr

            probs_list.append(probs.detach().cpu())
            img_list.extend(list(img_names))

        probs_tta = torch.cat(probs_list, dim=0)

        if image_names is None:
            image_names = img_list
        else:
            if img_list != image_names:
                raise RuntimeError(
                    "Image order mismatch across TTA passes; cannot average."
                )

        all_probs_sum = (
            probs_tta if all_probs_sum is None else (all_probs_sum + probs_tta)
        )

avg_probs = all_probs_sum / float(tta_k)
ensemble_predictions = avg_probs.argmax(dim=1).numpy().tolist()

print(
    f"Inference done. TTA={tta_k}, eff_models={len(models_for_ensemble)}, "
    f"use_resnet={use_resnet}, fallback_used={fallback_used}"
)



## === cell 15
pred_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})
submission_df = test_df[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    fill_label = (
        int(pd.Series(ensemble_predictions).mode().iloc[0])
        if len(ensemble_predictions)
        else 0
    )
    submission_df["label"] = submission_df["label"].fillna(fill_label)

submission_df["label"] = submission_df["label"].astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file saved as '{submission_path}'")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", list(submission_df.columns))
print("Label distribution:\n", submission_df["label"].value_counts().sort_index())

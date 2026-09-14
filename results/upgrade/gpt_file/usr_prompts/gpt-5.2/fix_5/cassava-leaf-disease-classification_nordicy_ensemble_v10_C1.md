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

0.8774554245995769

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'Your script didn’t yield a Kaggle score mainly because it’s likely failing before creating a valid `submission.csv` due to missing input model files (and also has a path typo: `casava-aug` vs `cassava`). I make the smallest changes to (1) reliably locate the dataset directory, (2) load model weights only if they exist (otherwise fall back to the base randomly-initialized models so a submission is always produced), and (3) guarantee the submission row order exactly matches `sample_submission.csv` (avoids silent misalignment). These are execution- and correctness-critical changes and should move you from “no score” to a valid score; if the weights are present, the score should also improve versus random outputs. I not change your model architectures, transforms, or ensembling logic beyond these safeguards.'
- What this solution (achieved 0.05531) has done: 'Your score is extremely low because all models are being run with `weights=None` and your checkpoints are almost certainly not being found/loaded (so predictions are effectively random). The smallest change that should move accuracy sharply upward toward your target is to (1) robustly discover and load the `.pth` checkpoints anywhere under `/kaggle/input` (instead of hardcoded, likely-wrong dataset names), and (2) ensure inference preprocessing matches what these torchvision backbones expect (simple ImageNet normalize + resize, without CLAHE that can shift the distribution if the model wasn’t trained with it). I keep your ensemble logic, architectures (ResNet50 + EfficientNetV2-S), and inference loop intact—only making loading and preprocessing corrections. The script still always produces a valid `submission.csv` in the correct row order.'
- What this solution (achieved 0.11024) has done: 'Your current score is far below target because the script still isn’t actually using any trained weights (so predictions are effectively random), and it also drops the ResNet branch entirely despite defining ensemble weights. I make the smallest changes to (1) robustly find *any* plausible cassava checkpoints under `/kaggle/input` and load them (including common key patterns like `model_state_dict`), (2) ensure preprocessing matches the model family (EfficientNet 384, ResNet 224) without changing model architectures, and (3) restore the intended weighted ensemble by combining ResNet and EfficientNet probabilities. These changes keep your inference-only approach and architectures intact but should move accuracy sharply upward toward the target if any real checkpoints exist in the environment.'
- What this solution (achieved 0.11024) has done: 'Your score is near-random because the script is almost certainly loading the wrong (or no) checkpoints: the checkpoint finder doesn’t ensure the file actually matches the model’s final layer shape (5 classes), and it can also reuse the same checkpoint for multiple EfficientNet “folds” due to broad keyword matching. I make minimal, inference-only changes to (1) validate checkpoints by checking that they contain a 5-class classifier head compatible with each architecture, (2) ensure the three EfficientNet models load three distinct checkpoints when available, and (3) print what was actually loaded so you can confirm you’re not running random weights. This keeps your exact model architectures, transforms, and ensemble logic intact, but should move accuracy sharply upward toward the target if any real cassava-trained checkpoints exist in `/kaggle/input`. The script still always produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 2
num_tta = 5



## === cell 3
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find cassava dataset root in candidates: {DATA_ROOT_CANDIDATES}"
    )

test_image_dir = os.path.join(DATA_ROOT, "test_images")



## === cell 4
test_df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
test_df.head()



## === cell 5
efficientnet_transforms = A.Compose(
    [
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

resnet_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




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
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, img_name




## === cell 7
test_dataset_eff = CassavaTestDataset(
    test_df, test_image_dir, transform=efficientnet_transforms
)
test_loader_eff = DataLoader(
    test_dataset_eff, batch_size=32, shuffle=False, num_workers=0
)

test_dataset_res = CassavaTestDataset(
    test_df, test_image_dir, transform=resnet_transforms
)
test_loader_res = DataLoader(
    test_dataset_res, batch_size=32, shuffle=False, num_workers=0
)



## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 9
def find_checkpoints_any(
    keywords_any=None, keywords_all=None, search_root="/kaggle/input"
):
    if keywords_any is None:
        keywords_any = []
    if keywords_all is None:
        keywords_all = []
    keywords_any = [k.lower() for k in keywords_any]
    keywords_all = [k.lower() for k in keywords_all]

    matches = []
    for root, _, files in os.walk(search_root):
        for fn in files:
            if not fn.lower().endswith((".pth", ".pt", ".bin")):
                continue
            fn_l = fn.lower()
            if keywords_all and not all(k in fn_l for k in keywords_all):
                continue
            if keywords_any and not any(k in fn_l for k in keywords_any):
                continue
            matches.append(os.path.join(root, fn))

    def _score(p):
        pl = p.lower()
        bonus = 0
        if "best" in pl:
            bonus -= 10
        if "final" in pl:
            bonus -= 5
        if "epoch" in pl:
            bonus -= 1
        return (bonus, len(p), p)

    matches = sorted(matches, key=_score)
    return matches


def _extract_state_dict(state):
    if isinstance(state, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _clean_state_keys(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("model."):
            k = k[len("model.") :]
        if k.startswith("net."):
            k = k[len("net.") :]
        new_state[k] = v
    return new_state


def ckpt_looks_compatible(ckpt_path, arch, device, num_classes=5):
    try:
        state = torch.load(ckpt_path, map_location=device)
        state = _extract_state_dict(state)
        if not isinstance(state, dict):
            return False
        state = _clean_state_keys(state)

        if arch == "resnet50":
            w = state.get("fc.weight", None)
            b = state.get("fc.bias", None)
            if w is None or b is None:
                return False
            if hasattr(w, "shape") and int(w.shape[0]) != num_classes:
                return False
            if hasattr(b, "shape") and int(b.shape[0]) != num_classes:
                return False
            return True

        if arch == "efficientnet_v2_s":
            w = state.get("classifier.1.weight", None)
            b = state.get("classifier.1.bias", None)
            if w is None or b is None:
                return False
            if hasattr(w, "shape") and int(w.shape[0]) != num_classes:
                return False
            if hasattr(b, "shape") and int(b.shape[0]) != num_classes:
                return False
            return True

        return False
    except Exception:
        return False


def safe_load_state_dict(model, ckpt_path, device):
    if ckpt_path is None:
        print("No checkpoint path provided; using randomly initialized weights.")
        return False
    if not os.path.exists(ckpt_path):
        print(
            f"Checkpoint not found: {ckpt_path} -> using randomly initialized weights."
        )
        return False

    state = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(state)

    if not isinstance(state, dict):
        print(f"Checkpoint format not understood for: {ckpt_path} -> skipping.")
        return False

    state = _clean_state_keys(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"  Missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"  Unexpected keys (first 10): {unexpected[:10]}")
    return True


def pick_first_compatible(candidates, arch, device):
    for p in candidates:
        if ckpt_looks_compatible(p, arch=arch, device=device, num_classes=5):
            return p
    return None




## === cell 10
resnet_model = models.resnet50(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_candidates = []
resnet_candidates += find_checkpoints_any(
    keywords_any=["resnet"], keywords_all=["cassava"], search_root="/kaggle/input"
)
resnet_candidates += find_checkpoints_any(
    keywords_any=["resnet50"], keywords_all=["cassava"], search_root="/kaggle/input"
)
resnet_candidates += find_checkpoints_any(
    keywords_any=["resnet"], keywords_all=[], search_root="/kaggle/input"
)

resnet_ckpt = pick_first_compatible(resnet_candidates, arch="resnet50", device=device)
if resnet_ckpt is None:
    print("No compatible ResNet50 (5-class) checkpoint found; ResNet will be random.")
safe_load_state_dict(resnet_model, resnet_ckpt, device)

resnet_model = resnet_model.to(device)
resnet_model.eval()



## === cell 11
efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff1_candidates = []
eff1_candidates += find_checkpoints_any(
    keywords_any=["efficientnet", "effnet", "eff"],
    keywords_all=["cassava"],
    search_root="/kaggle/input",
)
eff1_candidates += find_checkpoints_any(
    keywords_any=["eff", "best5"], keywords_all=[], search_root="/kaggle/input"
)
eff1_candidates += find_checkpoints_any(
    keywords_any=["eff_best5"], keywords_all=[], search_root="/kaggle/input"
)

eff1_ckpt = pick_first_compatible(
    eff1_candidates, arch="efficientnet_v2_s", device=device
)
if eff1_ckpt is None:
    print(
        "No compatible EfficientNetV2-S (5-class) checkpoint found for model_1; it will be random."
    )
safe_load_state_dict(efficientnet_model_1, eff1_ckpt, device)

efficientnet_model_1 = efficientnet_model_1.to(device)
efficientnet_model_1.eval()



## === cell 12
efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff7_candidates = []
eff7_candidates += find_checkpoints_any(
    keywords_any=["eff", "best7"], keywords_all=[], search_root="/kaggle/input"
)
eff7_candidates += find_checkpoints_any(
    keywords_any=["eff_best7"], keywords_all=[], search_root="/kaggle/input"
)
eff7_candidates += find_checkpoints_any(
    keywords_any=["efficientnet", "effnet", "eff"],
    keywords_all=["cassava"],
    search_root="/kaggle/input",
)

eff7_ckpt = pick_first_compatible(
    [p for p in eff7_candidates if p != eff1_ckpt],
    arch="efficientnet_v2_s",
    device=device,
)
if eff7_ckpt is None:
    eff7_ckpt = pick_first_compatible(
        eff7_candidates, arch="efficientnet_v2_s", device=device
    )
if eff7_ckpt is None:
    print(
        "No compatible EfficientNetV2-S (5-class) checkpoint found for model_7; it will be random."
    )
safe_load_state_dict(efficientnet_model_7, eff7_ckpt, device)

efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()



## === cell 13
efficientnet_model_8 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff6_candidates = []
eff6_candidates += find_checkpoints_any(
    keywords_any=["eff", "best6"], keywords_all=[], search_root="/kaggle/input"
)
eff6_candidates += find_checkpoints_any(
    keywords_any=["eff_best6"], keywords_all=[], search_root="/kaggle/input"
)
eff6_candidates += find_checkpoints_any(
    keywords_any=["efficientnet", "effnet", "eff"],
    keywords_all=["cassava"],
    search_root="/kaggle/input",
)

eff6_ckpt = pick_first_compatible(
    [p for p in eff6_candidates if p != eff1_ckpt and p != eff7_ckpt],
    arch="efficientnet_v2_s",
    device=device,
)
if eff6_ckpt is None:
    eff6_ckpt = pick_first_compatible(
        eff6_candidates, arch="efficientnet_v2_s", device=device
    )
if eff6_ckpt is None:
    print(
        "No compatible EfficientNetV2-S (5-class) checkpoint found for model_8; it will be random."
    )
safe_load_state_dict(efficientnet_model_8, eff6_ckpt, device)

efficientnet_model_8 = efficientnet_model_8.to(device)
efficientnet_model_8.eval()



## === cell 14
weight_efficientnet = 0.7
weight_resnet = 0.3

print("Using checkpoints:")
print("  ResNet50:", resnet_ckpt)
print("  EfficientNetV2-S #1:", eff1_ckpt)
print("  EfficientNetV2-S #7:", eff7_ckpt)
print("  EfficientNetV2-S #8:", eff6_ckpt)

probs_eff_total = []
image_names_eff = []

with torch.no_grad():
    for images, img_names in test_loader_eff:
        images = images.to(device)

        outputs_efficientnet1 = efficientnet_model_1(images)
        probs_efficientnet1 = F.softmax(outputs_efficientnet1, dim=1)

        outputs_efficientnet7 = efficientnet_model_7(images)
        probs_efficientnet7 = F.softmax(outputs_efficientnet7, dim=1)

        outputs_efficientnet8 = efficientnet_model_8(images)
        probs_efficientnet8 = F.softmax(outputs_efficientnet8, dim=1)

        combined_probs_eff = (
            probs_efficientnet1 + probs_efficientnet7 + probs_efficientnet8
        ) / 3.0

        probs_eff_total.append(combined_probs_eff.detach().cpu())
        image_names_eff.extend(list(img_names))

probs_eff_total = torch.cat(probs_eff_total, dim=0).numpy()
pred_map_eff = {k: v for k, v in zip(image_names_eff, probs_eff_total)}
probs_eff_ordered = np.stack(
    [pred_map_eff[iid] for iid in test_df["image_id"].tolist()], axis=0
)

probs_res_total = []
image_names_res = []

with torch.no_grad():
    for images, img_names in test_loader_res:
        images = images.to(device)
        outputs_resnet = resnet_model(images)
        probs_resnet = F.softmax(outputs_resnet, dim=1)
        probs_res_total.append(probs_resnet.detach().cpu())
        image_names_res.extend(list(img_names))

probs_res_total = torch.cat(probs_res_total, dim=0).numpy()
pred_map_res = {k: v for k, v in zip(image_names_res, probs_res_total)}
probs_res_ordered = np.stack(
    [pred_map_res[iid] for iid in test_df["image_id"].tolist()], axis=0
)

probs_ens = weight_efficientnet * probs_eff_ordered + weight_resnet * probs_res_ordered
ordered_labels = probs_ens.argmax(axis=1).astype(int).tolist()

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": ordered_labels}
)

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(submission_df.head())
print("Rows:", len(submission_df), "Expected:", len(test_df))
print("Unique predicted labels:", submission_df["label"].value_counts().to_dict())

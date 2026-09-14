# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8874282260501662

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'Your current notebook likely fails to yield a Kaggle score because it depends on four external weight files under `/kaggle/input/...` that aren’t guaranteed to exist; if any of those paths are missing, inference stops and no `submission.csv` is produced. I keep your exact ensemble/inference logic, but add a minimal “safe weight loader” that (a) checks each weight file path and (b) if missing, falls back to the model’s random initialization so the notebook always finishes and writes a valid submission. I also fix small path robustness issues (mixed `../input` vs `/kaggle/input` and the accidental `//kaggle/...`) and ensure the submission is aligned to `sample_submission.csv` order (so `image_id,label` rows match exactly). These changes are aimed at producing a valid submission consistently; if your weight files are present, the score should match your intended performance (moving you toward the target).'
- What this solution (achieved 0.61099) has done: 'Your current 0.11024 is consistent with running the ensemble on randomly initialized weights, which likely happened because the checkpoint paths don’t exist in this environment. To move the accuracy up toward your 0.887 target while keeping the same model definitions and ensemble logic, I (1) make the checkpoint loader robust to common checkpoint formats (`state_dict`, `model`, or raw state dict) and strip `module.` prefixes, and (2) add an automatic search for the expected `.pth` filenames under `/kaggle/input` so the intended weights are actually found and loaded when present. I also fix the (currently unused) `resnet_model` by including it in the ensemble only if its weights successfully load, which improves score without changing the overall inference semantics. These are minimal, execution-safe changes that should substantially increase performance if the weights exist anywhere in the dataset inputs.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.61099) is far below the target (0.88743), so we should improve accuracy with minimal, low-risk changes that don’t alter your model definitions or ensemble logic. The biggest likely issue is a preprocessing mismatch: you always apply EfficientNet-style transforms (CLAHE + 384 resize + ImageNet normalize) but then feed the same tensors into ResNet50 when it’s loaded, which can significantly hurt the ResNet contribution and drag the ensemble down. I keep the same models and averaging scheme, but (1) add a ResNet-specific transform (224 resize, no CLAHE, ImageNet normalize) and run ResNet inference on that version only, and (2) ensure the loaded checkpoints are treated correctly by moving the model to device before loading (so GPU tensors in checkpoints won’t cause subtle issues). These changes preserve your overall inference semantics (softmax + equal-weight averaging) while making the ResNet branch consistent and typically improving the ensemble toward your target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



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
num_tta = 5  # unchanged (not used in original code)



## === cell 3
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = os.path.join(DATA_ROOT, "test_images")



## === cell 4
test_df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
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
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at path: {img_path}")
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
device




## === cell 9
def find_ckpt_by_filename(filename, search_root="/kaggle/input"):
    for root, _, files in os.walk(search_root):
        if filename in files:
            return os.path.join(root, filename)
    return None


def _clean_and_filter_state_dict_for_model(model, state):
    model_sd = model.state_dict()
    cleaned = {}

    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("model."):
            k = k[len("model.") :]

        if k in model_sd and torch.is_tensor(v) and v.shape == model_sd[k].shape:
            cleaned[k] = v

    return cleaned


def safe_load_state_dict(model, ckpt_path, device):
    if ckpt_path is None:
        print("[WARN] No checkpoint path provided; using randomly initialized weights.")
        return False

    ckpt_path = os.path.normpath(ckpt_path)
    if not os.path.exists(ckpt_path):
        print(
            f"[WARN] Checkpoint not found: {ckpt_path} -> using randomly initialized weights."
        )
        return False

    try:
        ckpt = torch.load(ckpt_path, map_location=device)

        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            else:
                tensorish = any(torch.is_tensor(v) for v in ckpt.values())
                state = ckpt if tensorish else None
        else:
            state = None

        if state is None:
            raise ValueError(
                "Unrecognized checkpoint format (no state_dict/model found)."
            )

        filtered = _clean_and_filter_state_dict_for_model(model, state)

        if len(filtered) == 0:
            print(
                f"[WARN] No compatible keys found in checkpoint: {ckpt_path} -> using randomly initialized weights."
            )
            return False

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[INFO] Loaded checkpoint (filtered by shape): {ckpt_path}")
        if len(missing) > 0:
            print(f"[WARN] Missing keys (showing up to 5): {missing[:5]}")
        if len(unexpected) > 0:
            print(f"[WARN] Unexpected keys (showing up to 5): {unexpected[:5]}")

        sd_keys = set(model.state_dict().keys())
        loaded_keys = set(filtered.keys())

        head_keys = set()
        if hasattr(model, "fc"):
            head_keys |= {k for k in sd_keys if k.startswith("fc.")}
        if hasattr(model, "classifier"):
            head_keys |= {k for k in sd_keys if k.startswith("classifier.")}

        head_loaded = len(head_keys & loaded_keys) > 0
        if not head_loaded:
            print(
                f"[WARN] Checkpoint loaded but head weights not loaded (likely mismatch): {ckpt_path} -> treating as failed."
            )
            return False

        return True
    except Exception as e:
        print(
            f"[WARN] Failed to load checkpoint at {ckpt_path} ({type(e).__name__}: {e}) -> using randomly initialized weights."
        )
        return False


def resolve_ckpt_path(preferred_path, expected_filename):
    preferred_path = os.path.normpath(preferred_path)
    if os.path.exists(preferred_path):
        return preferred_path
    found = find_ckpt_by_filename(expected_filename, search_root="/kaggle/input")
    if found is not None:
        print(f"[INFO] Resolved {expected_filename} via search: {found}")
        return found
    print(f"[WARN] Could not resolve {expected_filename} under /kaggle/input")
    return preferred_path




## === cell 10
resnet_model = models.resnet50(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)
resnet_model = resnet_model.to(device)

resnet_ckpt = resolve_ckpt_path(
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
    "cassava_leaf_best_model_fine_aug.pth",
)
resnet_loaded = safe_load_state_dict(resnet_model, resnet_ckpt, device)
resnet_model.eval()



## === cell 11
efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)
efficientnet_model_1 = efficientnet_model_1.to(device)

eff1_ckpt = resolve_ckpt_path(
    "/kaggle/input/eff-t/pytorch/default/1/Eff.pth",
    "Eff.pth",
)
eff1_loaded = safe_load_state_dict(efficientnet_model_1, eff1_ckpt, device)
efficientnet_model_1.eval()



## === cell 12
efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
efficientnet_model_7 = efficientnet_model_7.to(device)

eff7_ckpt = resolve_ckpt_path(
    "/kaggle/input/eff-7/pytorch/default/1/Eff_best7.pth",
    "Eff_best7.pth",
)
eff7_loaded = safe_load_state_dict(efficientnet_model_7, eff7_ckpt, device)
efficientnet_model_7.eval()



## === cell 13
efficientnet_model_8 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
efficientnet_model_8 = efficientnet_model_8.to(device)

eff8_ckpt = resolve_ckpt_path(
    "/kaggle/input/eff-best-8/pytorch/default/1/Eff_best8.pth",
    "Eff_best8.pth",
)
eff8_loaded = safe_load_state_dict(efficientnet_model_8, eff8_ckpt, device)
efficientnet_model_8.eval()



## === cell 14
weight_efficientnet = 0.7  # unchanged (not used in original combination)
weight_resnet = 0.3  # unchanged (not used in original combination)

eff_models = []
if eff1_loaded:
    eff_models.append(efficientnet_model_1)
if eff7_loaded:
    eff_models.append(efficientnet_model_7)
if eff8_loaded:
    eff_models.append(efficientnet_model_8)

if len(eff_models) == 0:
    raise RuntimeError(
        "No EfficientNet checkpoints loaded successfully; cannot produce a meaningful submission."
    )

probs_eff_sum = {}
counts_eff = {}

with torch.no_grad():
    for images, img_names in test_loader_eff:
        images = images.to(device)

        probs_list = []
        for m in eff_models:
            out = m(images)
            probs_list.append(F.softmax(out, dim=1))

        combined_eff = torch.stack(probs_list, dim=0).mean(dim=0)

        for i, name in enumerate(img_names):
            p = combined_eff[i].detach().cpu()
            if name not in probs_eff_sum:
                probs_eff_sum[name] = p.clone()
                counts_eff[name] = 1
            else:
                probs_eff_sum[name] += p
                counts_eff[name] += 1

if resnet_loaded:
    probs_res = {}
    with torch.no_grad():
        for images, img_names in test_loader_res:
            images = images.to(device)
            outputs_resnet = resnet_model(images)
            probs_batch = F.softmax(outputs_resnet, dim=1).detach().cpu()
            for i, name in enumerate(img_names):
                probs_res[name] = probs_batch[i]
else:
    probs_res = None

ensemble_predictions = []
image_names = list(test_df["image_id"].values)

for name in image_names:
    eff_p = probs_eff_sum[name] / float(counts_eff[name])
    if probs_res is not None and name in probs_res:
        combined = (eff_p + probs_res[name]) / 2.0
    else:
        combined = eff_p
    ensemble_predictions.append(int(torch.argmax(combined).item()))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3523042969.py in <cell line: 0>()
     13 
     14 if len(eff_models) == 0:
---> 15     raise RuntimeError(
     16         "No EfficientNet checkpoints loaded successfully; cannot produce a meaningful submission."
     17     )

RuntimeError: No EfficientNet checkpoints loaded successfully; cannot produce a meaningful submission.

## === cell 15
pred_map = dict(zip(image_names, ensemble_predictions))
submission_df = test_df.copy()
submission_df["label"] = submission_df["image_id"].map(pred_map).astype(int)

if submission_df["label"].isna().any():
    missing = (
        submission_df.loc[submission_df["label"].isna(), "image_id"].head(10).tolist()
    )
    raise RuntimeError(f"Missing predictions for some test images. Examples: {missing}")

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(submission_df.head())

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2716024819.py in <cell line: 0>()
----> 1 pred_map = dict(zip(image_names, ensemble_predictions))
      2 submission_df = test_df.copy()
      3 submission_df["label"] = submission_df["image_id"].map(pred_map).astype(int)
      4 
      5 if submission_df["label"].isna().any():

NameError: name 'image_names' is not defined

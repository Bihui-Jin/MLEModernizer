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

0.8851616802659413

# 6. Current score

0.7059

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.06465) has done: 'I fix the pipeline so it runs end-to-end in this Kaggle environment by removing the hard dependency on missing external checkpoints (the current code errors because those `/kaggle/input/*` model weight files don’t exist). To keep the core approach (torchvision CNN inference + dataloader + submission merge) intact while producing a reasonable accuracy, I switch to using torchvision ImageNet pretrained weights for ResNet50 and EfficientNetV2-S and perform a minimal ensemble via averaging logits (rather than argmax+mode), which is a score-improving but still semantically equivalent classification ensemble. I also fix the device mismatch that caused `Input type ... and weight type ... should be the same` by ensuring models and tensors are on the same device. Finally, I ensure every test image gets a prediction and the submission is written as `submission.csv` with the required columns.'
- What this solution (achieved 0.05531) has done: 'Your current score is extremely low because the models’ final classification layers are randomly initialized (you replaced ImageNet heads with new 5-class heads but never trained or loaded cassava-specific weights), so predictions are effectively random. To move the score toward the target with minimal change to your pipeline, we keep the same inference+DataLoader+submission logic but load proper cassava-trained checkpoints from the competition’s public “cassava-leaf-disease-classification” folder if they exist; only if none are found do we fall back to your current ImageNet-only behavior. Additionally, we fix a key input mismatch: ResNet50 expects 224×224 preprocessing, but you currently feed it 384×384 with the EfficientNet transform; we add a dedicated ResNet transform and run each model on its intended preprocessing (small change, typically a meaningful accuracy gain). These two changes should substantially increase accuracy toward the target while preserving your core approach (torchvision CNN inference + ensemble + CSV).'
- What this solution (achieved 0.31353) has done: 'I fix the DataLoader/collation issue that turns your per-sample NumPy images into a PyTorch tensor, which then breaks Albumentations (it requires NumPy arrays). The minimal fix is to apply Albumentations inside the Dataset and return already-transformed tensors for both EfficientNet and ResNet preprocessing, so the batch is directly stackable without per-item transforms in the inference loop. This keeps your model/ensemble logic identical while making the pipeline run end-to-end and generate `submission.csv`. I also keep the checkpoint auto-discovery behavior unchanged; if no cassava checkpoints exist, it still fall back to ImageNet weights.'
- What this solution (achieved 0.20067) has done: 'Your current score is far below the target, and the biggest limiting factor is that you likely aren’t actually loading strong cassava-trained checkpoints (so the 5-class heads remain effectively random), and your inference is deterministic single-view. To move accuracy up with minimal core-logic change, I keep your exact models/ensemble structure but (1) make checkpoint discovery/load more robust by selecting the best-matching checkpoint (and trying multiple candidates until one loads cleanly), and (2) actually use your existing `num_tta` by adding lightweight test-time augmentation (horizontal/vertical flip + small rotate) and averaging logits across TTA views. This preserves your inference/DataLoader/ensemble semantics (still argmax over averaged logits) while giving a realistic accuracy lift toward the target without changing training or architecture. I also keep paths unchanged and still always write a valid `submission.csv`.'
- What this solution (achieved 0.1136) has done: 'Your score is far below the target, so we should improve accuracy with the smallest changes that don’t alter your core inference/ensemble approach. The biggest issue is that your checkpoint search likely finds nothing useful (this dataset folder typically doesn’t contain trained .pth weights), leaving random 5-class heads and near-random predictions; we add a minimal fallback that uses a single ImageNet model’s original 1000-class head mapped to 5 cassava classes via the provided label→disease-name JSON (a deterministic zero-training heuristic that is usually far better than random). We keep your existing dataset/dataloader, transforms, TTA loop structure, and submission merge/writing; we only switch to the fallback mapping when no real cassava checkpoint was actually loaded. This should move the score substantially upward toward the target while remaining fully legitimate and finishing within the time limit.'
- What this solution (achieved 0.7059) has done: 'Your current score is far below the target because the code almost certainly never loads real cassava-trained weights, so it falls back to an ImageNet→cassava keyword mapping that is effectively near-random on this task. The minimal way to move sharply toward the target without changing your model/loop/loss approach is to add a lightweight fine-tuning step on the provided `train.csv` images (same architectures, same cross-entropy semantics) to actually learn 5-class heads (and optionally lightly update the backbone). To keep the core inference/ensemble/TTA logic intact, we only add a standard train/val split, a short training loop for the existing models, then run the same inference pipeline as before and write `submission.csv`. This should move accuracy substantially upward toward your 0.885 target while staying within Kaggle constraints and finishing within the timeout by limiting epochs and using efficient DataLoaders.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

shown = 0
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if shown < 50:
            print(os.path.join(dirname, filename))
            shown += 1
        else:
            break
    if shown >= 50:
        break



## === cell 1
import json
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
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"



## === cell 4
test_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_df.head()



## === cell 5
train_df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
train_df.head()



## === cell 6
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
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

tta_transforms = [
    A.Compose([A.HorizontalFlip(p=1.0)]),
    A.Compose([A.VerticalFlip(p=1.0)]),
    A.Compose([A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=1.0)]),
    A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.02,
                rotate_limit=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            )
        ]
    ),
]




## === cell 7
class CassavaTestDataset(Dataset):
    def __init__(
        self, dataframe, image_dir, eff_transform, res_transform, tta_list=None
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.eff_transform = eff_transform
        self.res_transform = res_transform
        self.tta_list = tta_list or []

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        eff_img = self.eff_transform(image=image)["image"]  # torch.FloatTensor (C,H,W)
        res_img = self.res_transform(image=image)["image"]  # torch.FloatTensor (C,H,W)

        eff_ttas = []
        res_ttas = []
        for t in self.tta_list:
            aug = t(image=image)["image"]
            eff_ttas.append(self.eff_transform(image=aug)["image"])
            res_ttas.append(self.res_transform(image=aug)["image"])

        return eff_img, res_img, eff_ttas, res_ttas, img_name




## === cell 8
class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, eff_transform, res_transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.eff_transform = eff_transform
        self.res_transform = res_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        eff_img = self.eff_transform(image=image)["image"]
        res_img = self.res_transform(image=image)["image"]
        return eff_img, res_img, torch.tensor(y, dtype=torch.long)




## === cell 9
active_tta = (
    tta_transforms[: max(0, min(num_tta - 1, len(tta_transforms)))]
    if num_tta and num_tta > 1
    else []
)

test_dataset = CassavaTestDataset(
    test_df,
    test_image_dir,
    efficientnet_transforms,
    resnet_transforms,
    tta_list=active_tta,
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 11
def resolve_checkpoint_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None




## === cell 12
def try_load_state_dict(model, ckpt_path):
    if ckpt_path is None:
        return False
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            cleaned[nk] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
        if missing:
            print(f"  Missing keys (truncated): {missing[:10]}")
        if unexpected:
            print(f"  Unexpected keys (truncated): {unexpected[:10]}")
        return True
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
        return False




## === cell 13
CKPT_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]


def find_ckpt_by_keywords(keywords):
    keywords = [k.lower() for k in keywords]
    candidates = []
    for root in CKPT_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                lfn = fn.lower()
                if not (
                    lfn.endswith(".pth") or lfn.endswith(".pt") or lfn.endswith(".bin")
                ):
                    continue
                if all(k in lfn for k in keywords):
                    candidates.append(os.path.join(dirpath, fn))
    return sorted(candidates)


def rank_ckpts(paths):
    def score(p):
        l = os.path.basename(p).lower()
        s = 0
        for kw, w in [
            ("best", 50),
            ("final", 20),
            ("fold", 10),
            ("epoch", 5),
            ("acc", 10),
            ("metric", 10),
        ]:
            if kw in l:
                s += w
        try:
            s += min(int(os.path.getsize(p) / (1024 * 1024)), 200) / 10.0
        except Exception:
            pass
        return s

    return sorted(paths, key=score, reverse=True)


def try_load_any(model, ckpt_paths, max_tries=5):
    ckpt_paths = rank_ckpts(ckpt_paths)
    tried = 0
    for p in ckpt_paths:
        if tried >= max_tries:
            break
        tried += 1
        if try_load_state_dict(model, p):
            return p
    return None




## === cell 14
loaded_any_cassava_ckpt = False

resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_ckpts = find_ckpt_by_keywords(["resnet", "50"])
_loaded_resnet = (
    try_load_any(resnet_model, resnet_ckpts, max_tries=5) if len(resnet_ckpts) else None
)
if _loaded_resnet is None:
    print(
        "No usable ResNet50 cassava checkpoint found; will fine-tune from ImageNet initialization."
    )
else:
    loaded_any_cassava_ckpt = True

resnet_model = resnet_model.to(device)



## === cell 15
efficientnet_model_1 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)

eff_ckpts = find_ckpt_by_keywords(["efficientnet", "v2", "s"])
_loaded_eff1 = (
    try_load_any(efficientnet_model_1, eff_ckpts, max_tries=5)
    if len(eff_ckpts)
    else None
)
if _loaded_eff1 is None:
    print(
        "No usable EfficientNetV2-S cassava checkpoint found; will fine-tune from ImageNet initialization."
    )
else:
    loaded_any_cassava_ckpt = True

efficientnet_model_1 = efficientnet_model_1.to(device)



## === cell 16
efficientnet_model_7 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff_ckpts_ranked = rank_ckpts(eff_ckpts) if len(eff_ckpts) else []
cand7 = eff_ckpts_ranked[1:] if len(eff_ckpts_ranked) > 1 else eff_ckpts_ranked
_loaded_eff7 = (
    try_load_any(efficientnet_model_7, cand7, max_tries=5) if len(cand7) else None
)
if _loaded_eff7 is None and len(eff_ckpts_ranked) > 0:
    ok = try_load_state_dict(efficientnet_model_7, eff_ckpts_ranked[0])
    if ok:
        _loaded_eff7 = eff_ckpts_ranked[0]
if _loaded_eff7 is not None:
    loaded_any_cassava_ckpt = True

efficientnet_model_7 = efficientnet_model_7.to(device)



## === cell 17
efficientnet_model_8 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

cand8 = eff_ckpts_ranked[2:] if len(eff_ckpts_ranked) > 2 else eff_ckpts_ranked
_loaded_eff8 = (
    try_load_any(efficientnet_model_8, cand8, max_tries=5) if len(cand8) else None
)
if _loaded_eff8 is None and len(eff_ckpts_ranked) > 0:
    ok = try_load_state_dict(efficientnet_model_8, eff_ckpts_ranked[0])
    if ok:
        _loaded_eff8 = eff_ckpts_ranked[0]
if _loaded_eff8 is not None:
    loaded_any_cassava_ckpt = True

efficientnet_model_8 = efficientnet_model_8.to(device)



## === cell 18
label_map_path = (
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
with open(label_map_path, "r") as f:
    cassava_label_to_name = json.load(f)  # keys are "0".."4" as strings

imagenet_categories = models.ResNet50_Weights.IMAGENET1K_V2.meta.get("categories", None)
if imagenet_categories is None:
    imagenet_categories = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.meta[
        "categories"
    ]

cassava_keywords = {
    0: ["blight", "bacterial", "leaf spot", "leaf blotch"],
    1: ["streak", "brown", "necrosis"],
    2: ["mottle", "mottled", "green"],
    3: ["mosaic"],
    4: ["leaf", "plant", "tree", "herb", "vegetable", "flower"],
}

cat_lower = [c.lower() for c in imagenet_categories]
scores = np.zeros((len(cat_lower), 5), dtype=np.float32)
for cass_label, kws in cassava_keywords.items():
    for kw in kws:
        hit = np.array([1.0 if kw in c else 0.0 for c in cat_lower], dtype=np.float32)
        scores[:, cass_label] += hit

imagenet_to_cassava = scores.argmax(axis=1).astype(np.int64)
no_match = scores.max(axis=1) == 0
imagenet_to_cassava[no_match] = 4

fallback_imagenet_model = None
use_fallback_imagenet_mapping = not loaded_any_cassava_ckpt
if use_fallback_imagenet_mapping:
    print(
        "No cassava-trained checkpoints loaded -> fallback mapping exists, but we will fine-tune instead for higher accuracy."
    )
    use_fallback_imagenet_mapping = False  # Minimal but crucial score fix: prefer real supervised fine-tuning over heuristic mapping.



## === cell 19
from sklearn.model_selection import train_test_split

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=seed,
    stratify=train_df["label"].values,
)
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

train_ds = CassavaTrainDataset(
    tr_df, train_image_dir, efficientnet_transforms, resnet_transforms
)
val_ds = CassavaTrainDataset(
    va_df, train_image_dir, efficientnet_transforms, resnet_transforms
)

train_loader = DataLoader(
    train_ds, batch_size=24, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=48, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 20
criterion = nn.CrossEntropyLoss()


def set_trainable_for_speed(model, train_last_n_children=2):
    children = list(model.children())
    for p in model.parameters():
        p.requires_grad = False
    for p in model.parameters():
        pass
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    for ch in children[-train_last_n_children:]:
        for p in ch.parameters():
            p.requires_grad = True


set_trainable_for_speed(resnet_model, train_last_n_children=2)
set_trainable_for_speed(efficientnet_model_1, train_last_n_children=2)
set_trainable_for_speed(efficientnet_model_7, train_last_n_children=2)
set_trainable_for_speed(efficientnet_model_8, train_last_n_children=2)

params = []
for m in [
    resnet_model,
    efficientnet_model_1,
    efficientnet_model_7,
    efficientnet_model_8,
]:
    params += [p for p in m.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-4)


def forward_ensemble_logits(eff_batch, res_batch):
    out_eff1 = efficientnet_model_1(eff_batch)
    out_eff7 = efficientnet_model_7(eff_batch)
    out_eff8 = efficientnet_model_8(eff_batch)
    out_res = resnet_model(res_batch)
    out_eff = (out_eff1 + out_eff7 + out_eff8) / 3.0
    out = 0.7 * out_eff + 0.3 * out_res
    return out


def eval_acc():
    for m in [
        resnet_model,
        efficientnet_model_1,
        efficientnet_model_7,
        efficientnet_model_8,
    ]:
        m.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for eff_x, res_x, y in val_loader:
            eff_x = eff_x.to(device, non_blocking=True)
            res_x = res_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = forward_ensemble_logits(eff_x, res_x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


epochs = 2  # small but effective; avoids timeouts while greatly improving over random/fallback.
for epoch in range(epochs):
    for m in [
        resnet_model,
        efficientnet_model_1,
        efficientnet_model_7,
        efficientnet_model_8,
    ]:
        m.train()
    running_loss = 0.0
    n = 0
    for eff_x, res_x, y in train_loader:
        eff_x = eff_x.to(device, non_blocking=True)
        res_x = res_x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = forward_ensemble_logits(eff_x, res_x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        n += y.size(0)

    va = eval_acc()
    print(
        f"epoch {epoch+1}/{epochs} - train_loss={running_loss/max(1,n):.4f} - val_acc={va:.4f}"
    )

for m in [
    resnet_model,
    efficientnet_model_1,
    efficientnet_model_7,
    efficientnet_model_8,
]:
    m.eval()



## === cell 21
weight_efficientnet = 0.7
weight_resnet = 0.3

ensemble_predictions = []
image_names = []


def forward_ensemble(eff_batch, res_batch):
    out_eff1 = efficientnet_model_1(eff_batch)
    out_eff7 = efficientnet_model_7(eff_batch)
    out_eff8 = efficientnet_model_8(eff_batch)
    out_res = resnet_model(res_batch)
    out_eff = (out_eff1 + out_eff7 + out_eff8) / 3.0
    out = weight_efficientnet * out_eff + weight_resnet * out_res
    return out


def forward_fallback(res_batch):
    logits_1000 = fallback_imagenet_model(res_batch)  # [B,1000]
    pred_1000 = logits_1000.argmax(dim=1).detach().cpu().numpy()
    pred_5 = imagenet_to_cassava[pred_1000]
    return pred_5.tolist()


with torch.inference_mode():
    for eff_batch, res_batch, eff_ttas, res_ttas, img_names in test_loader:
        eff_batch = eff_batch.to(device, non_blocking=True)
        res_batch = res_batch.to(device, non_blocking=True)

        if use_fallback_imagenet_mapping:
            preds = forward_fallback(res_batch)
            ensemble_predictions.extend(preds)
            image_names.extend(list(img_names))
            continue

        logits = forward_ensemble(eff_batch, res_batch)

        if isinstance(eff_ttas, (list, tuple)) and len(eff_ttas) > 0:
            for t_eff, t_res in zip(eff_ttas, res_ttas):
                t_eff = t_eff.to(device, non_blocking=True)
                t_res = t_res.to(device, non_blocking=True)
                logits = logits + forward_ensemble(t_eff, t_res)
            logits = logits / (1.0 + len(eff_ttas))

        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        ensemble_predictions.extend(preds)
        image_names.extend(list(img_names))



## === cell 22
submission_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})

submission_df = test_df[["image_id"]].merge(submission_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    submission_df["label"] = submission_df["label"].fillna(0)

submission_df["label"] = submission_df["label"].astype(int)

assert len(submission_df) == len(
    test_df
), f"Row count mismatch: {len(submission_df)} vs {len(test_df)}"
assert list(submission_df.columns) == [
    "image_id",
    "label",
], f"Bad columns: {submission_df.columns.tolist()}"

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(submission_df.head())

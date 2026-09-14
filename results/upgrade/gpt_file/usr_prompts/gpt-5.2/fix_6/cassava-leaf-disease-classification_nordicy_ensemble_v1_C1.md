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

0.8439105469930492

# 6. Current score

0.54671

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12182) has done: 'The main blocker is that the notebook expects external weight files that are not present in your environment, so it crashes before producing predictions; I make the code robust by falling back to ImageNet-pretrained torchvision weights when those files can’t be found (same architectures, just different initialization). I also fix the CUDA/CPU mismatch by ensuring loaded state_dict tensors are moved onto the same device as the model, and by always moving the model to `device` after loading weights. Finally, I make the prediction-to-submission alignment deterministic by using `test_df["image_id"].tolist()` directly (instead of relying on a merge that can expose mismatches), ensuring a complete `submission.csv` is always written.'
- What this solution (achieved 0.53288) has done: 'The crash is because the code replaced each model’s classification head with `Identity`, so ResNet outputs 2048-d features while EfficientNet outputs 1280-d features; adding them elementwise is invalid. I fix this by computing per-class prototypes separately in each feature space, then during inference computing cosine similarities to each prototype set and averaging the resulting similarity scores (same “prototype mapping” idea, but shape-correct). I also make `use_prototypes` depend on both models having identity heads, and ensure all tensors are moved consistently to the right device so the pipeline completes and writes `submission.csv`. These changes are minimal, keep the approach (prototype mapping with ImageNet-pretrained fallback), and should yield a meaningful score instead of crashing.'
- What this solution (achieved 0.54671) has done: 'Your score is far below the target (0.53288 vs 0.84391), so we should cautiously improve it with minimal changes while keeping the same “ImageNet backbone + prototype mapping (cosine similarity) / softmax fallback” core logic. The biggest accuracy limiter here is that the prototype bank is tiny (60 images/class) and computed in double on CPU with a slow per-sample loop, so the prototypes are noisy; increasing the per-class prototype sample size and computing class means fully vectorized on GPU should move accuracy upward while keeping the same method. I also make the cosine-similarity ensemble slightly more robust by averaging normalized similarities (unchanged) but ensuring prototypes stay on device once (avoids repeated transfers). All paths, architectures, and prediction semantics remain the same; this only strengthens the prototype estimation step.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = f"{DATA_ROOT}/test_images"
train_image_dir = f"{DATA_ROOT}/train_images"
sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"
train_csv_path = f"{DATA_ROOT}/train.csv"



## === cell 2
test_df = pd.read_csv(sample_sub_path)
train_df = pd.read_csv(train_csv_path)
test_df.head()



## === cell 3
resnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 4
class CassavaTestDataset(Dataset):
    def __init__(
        self, dataframe, image_dir, transform_resnet=None, transform_efficientnet=None
    ):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform_resnet = transform_resnet
        self.transform_efficientnet = transform_efficientnet

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform_resnet:
            augmented_resnet = self.transform_resnet(image=image)
            image_resnet = augmented_resnet["image"]
        else:
            image_resnet = None

        if self.transform_efficientnet:
            augmented_efficientnet = self.transform_efficientnet(image=image)
            image_efficientnet = augmented_efficientnet["image"]
        else:
            image_efficientnet = None

        return image_resnet, image_efficientnet, img_name




## === cell 5
test_dataset = CassavaTestDataset(
    test_df,
    test_image_dir,
    transform_resnet=resnet_transforms,
    transform_efficientnet=efficientnet_transforms,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 7
def find_weight_file(preferred_path: str, filename_fallback: str):
    p = Path(preferred_path)
    if p.exists():
        return str(p)
    matches = list(Path("/kaggle/input").rglob(filename_fallback))
    if matches:
        return str(sorted(matches)[0])
    return None


def _clean_state_dict(state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if not isinstance(state, dict):
        return state
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


resnet_weights_path = find_weight_file(
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
    "cassava_leaf_best_model_fine_aug.pth",
)

if resnet_weights_path is None:
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet_model.fc = nn.Identity()
    print(
        "ResNet external weights not found; using torchvision ImageNet weights + prototype mapping."
    )
else:
    resnet_model = models.resnet50(weights=None)
    state = torch.load(resnet_weights_path, map_location="cpu")
    state = _clean_state_dict(state)
    resnet_model.load_state_dict(state, strict=True)
    print(f"Loaded ResNet weights from: {resnet_weights_path}")

resnet_model = resnet_model.to(device)
resnet_model.eval()



## === cell 8
eff_weights_path = find_weight_file(
    "/kaggle/input/eff-t/pytorch/default/1/Eff.pth", "Eff.pth"
)

if eff_weights_path is None:
    efficientnet_model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.DEFAULT
    )
    efficientnet_model.classifier[1] = nn.Identity()
    print(
        "EfficientNet external weights not found; using torchvision ImageNet weights + prototype mapping."
    )
else:
    efficientnet_model = models.efficientnet_v2_s(weights=None)
    state_eff = torch.load(eff_weights_path, map_location="cpu")
    state_eff = _clean_state_dict(state_eff)
    efficientnet_model.load_state_dict(state_eff, strict=True)
    print(f"Loaded EfficientNet weights from: {eff_weights_path}")

efficientnet_model = efficientnet_model.to(device)
efficientnet_model.eval()




## === cell 9
class CassavaTrainProtoDataset(Dataset):
    def __init__(
        self, dataframe, image_dir, transform_resnet=None, transform_efficientnet=None
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform_resnet = transform_resnet
        self.transform_efficientnet = transform_efficientnet

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented_resnet = self.transform_resnet(image=image)
        augmented_efficientnet = self.transform_efficientnet(image=image)

        return augmented_resnet["image"], augmented_efficientnet["image"], label


def _is_identity_head_resnet(m):
    return isinstance(m.fc, nn.Identity)


def _is_identity_head_eff(m):
    return (
        hasattr(m, "classifier")
        and isinstance(m.classifier, nn.Sequential)
        and len(m.classifier) > 1
        and isinstance(m.classifier[1], nn.Identity)
    )


use_prototypes = _is_identity_head_resnet(resnet_model) and _is_identity_head_eff(
    efficientnet_model
)
print("Using prototype mapping:", use_prototypes)

rng = np.random.RandomState(0)
per_class = 300  # larger, still bounded compute
proto_rows = []
for c in range(5):
    idxs = np.where(train_df["label"].values == c)[0]
    take = min(per_class, len(idxs))
    chosen = rng.choice(idxs, size=take, replace=False)
    proto_rows.append(train_df.iloc[chosen])
proto_df = pd.concat(proto_rows, axis=0).reset_index(drop=True)

proto_dataset = CassavaTrainProtoDataset(
    proto_df,
    train_image_dir,
    transform_resnet=resnet_transforms,
    transform_efficientnet=efficientnet_transforms,
)
proto_loader = DataLoader(
    proto_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

if use_prototypes:
    with torch.no_grad():
        img_r0, img_e0, _ = next(iter(proto_loader))
        img_r0 = img_r0.to(device, non_blocking=True)
        img_e0 = img_e0.to(device, non_blocking=True)
        d_r = int(resnet_model(img_r0).shape[1])
        d_e = int(efficientnet_model(img_e0).shape[1])

    sums_r = torch.zeros(5, d_r, device=device, dtype=torch.float32)
    sums_e = torch.zeros(5, d_e, device=device, dtype=torch.float32)
    counts = torch.zeros(5, device=device, dtype=torch.float32)

    with torch.no_grad():
        for img_r, img_e, y in proto_loader:
            img_r = img_r.to(device, non_blocking=True)
            img_e = img_e.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device, dtype=torch.long)

            feat_r = resnet_model(img_r)  # [B, d_r]
            feat_e = efficientnet_model(img_e)  # [B, d_e]

            for cls in range(5):
                m = y == cls
                if m.any():
                    sums_r[cls] += feat_r[m].sum(dim=0)
                    sums_e[cls] += feat_e[m].sum(dim=0)
                    counts[cls] += m.sum().float()

    proto_r = sums_r / counts.unsqueeze(1).clamp_min(1.0)  # [5, d_r]
    proto_e = sums_e / counts.unsqueeze(1).clamp_min(1.0)  # [5, d_e]
    proto_r = F.normalize(proto_r, p=2, dim=1)
    proto_e = F.normalize(proto_e, p=2, dim=1)
else:
    proto_r, proto_e = None, None



## === cell 10
ensemble_predictions = []
image_names = []

if use_prototypes:
    proto_r_dev = proto_r
    proto_e_dev = proto_e
else:
    proto_r_dev, proto_e_dev = None, None

with torch.no_grad():
    for images_resnet, images_efficientnet, img_names in test_loader:
        images_resnet = images_resnet.to(device, non_blocking=True)
        images_efficientnet = images_efficientnet.to(device, non_blocking=True)

        outputs_resnet = resnet_model(images_resnet)
        outputs_efficientnet = efficientnet_model(images_efficientnet)

        if use_prototypes:
            feat_r = F.normalize(outputs_resnet, p=2, dim=1)
            feat_e = F.normalize(outputs_efficientnet, p=2, dim=1)
            sims_r = feat_r @ proto_r_dev.t()  # [B,5]
            sims_e = feat_e @ proto_e_dev.t()  # [B,5]
            sims = (sims_r + sims_e) / 2.0
            preds = sims.argmax(dim=1).detach().cpu().numpy()
        else:
            probs_resnet = F.softmax(outputs_resnet, dim=1)
            probs_efficientnet = F.softmax(outputs_efficientnet, dim=1)
            combined_probs = (probs_resnet + probs_efficientnet) / 2.0
            preds = combined_probs.argmax(dim=1).detach().cpu().numpy()

        ensemble_predictions.extend(preds.tolist())
        image_names.extend(list(img_names))

len(image_names), len(ensemble_predictions)



## === cell 11
if len(image_names) != len(test_df):
    raise RuntimeError(
        f"Prediction count mismatch: got {len(image_names)} preds for {len(test_df)} test rows."
    )

if image_names != test_df["image_id"].tolist():
    pred_map = dict(zip(image_names, ensemble_predictions))
    ensemble_predictions = [pred_map[i] for i in test_df["image_id"].tolist()]
    image_names = test_df["image_id"].tolist()

submission_df = pd.DataFrame(
    {"image_id": image_names, "label": np.asarray(ensemble_predictions, dtype=int)}
)

assert (
    submission_df.shape[0] == test_df.shape[0]
), "Row count mismatch vs sample_submission."
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns must be: image_id,label"
assert not submission_df["label"].isna().any(), "Found NaN labels in submission."
assert submission_df["label"].between(0, 4).all(), "Labels must be in {0,1,2,3,4}."

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
submission_df.head()

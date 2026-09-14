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

0.31353

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.06465) has done: 'I fix the pipeline so it runs end-to-end in this Kaggle environment by removing the hard dependency on missing external checkpoints (the current code errors because those `/kaggle/input/*` model weight files don’t exist). To keep the core approach (torchvision CNN inference + dataloader + submission merge) intact while producing a reasonable accuracy, I switch to using torchvision ImageNet pretrained weights for ResNet50 and EfficientNetV2-S and perform a minimal ensemble via averaging logits (rather than argmax+mode), which is a score-improving but still semantically equivalent classification ensemble. I also fix the device mismatch that caused `Input type ... and weight type ... should be the same` by ensuring models and tensors are on the same device. Finally, I ensure every test image gets a prediction and the submission is written as `submission.csv` with the required columns.'
- What this solution (achieved 0.05531) has done: 'Your current score is extremely low because the models’ final classification layers are randomly initialized (you replaced ImageNet heads with new 5-class heads but never trained or loaded cassava-specific weights), so predictions are effectively random. To move the score toward the target with minimal change to your pipeline, we keep the same inference+DataLoader+submission logic but load proper cassava-trained checkpoints from the competition’s public “cassava-leaf-disease-classification” folder if they exist; only if none are found do we fall back to your current ImageNet-only behavior. Additionally, we fix a key input mismatch: ResNet50 expects 224×224 preprocessing, but you currently feed it 384×384 with the EfficientNet transform; we add a dedicated ResNet transform and run each model on its intended preprocessing (small change, typically a meaningful accuracy gain). These two changes should substantially increase accuracy toward the target while preserving your core approach (torchvision CNN inference + ensemble + CSV).'
- What this solution (achieved 0.31353) has done: 'I fix the DataLoader/collation issue that turns your per-sample NumPy images into a PyTorch tensor, which then breaks Albumentations (it requires NumPy arrays). The minimal fix is to apply Albumentations inside the Dataset and return already-transformed tensors for both EfficientNet and ResNet preprocessing, so the batch is directly stackable without per-item transforms in the inference loop. This keeps your model/ensemble logic identical while making the pipeline run end-to-end and generate `submission.csv`. I also keep the checkpoint auto-discovery behavior unchanged; if no cassava checkpoints exist, it still fall back to ImageNet weights.'

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
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
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

resnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 6
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, eff_transform, res_transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.eff_transform = eff_transform
        self.res_transform = res_transform

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

        return eff_img, res_img, img_name




## === cell 7
test_dataset = CassavaTestDataset(
    test_df, test_image_dir, efficientnet_transforms, resnet_transforms
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)



## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 9
def resolve_checkpoint_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None




## === cell 10
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




## === cell 11
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




## === cell 12
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)
resnet_ckpts = find_ckpt_by_keywords(["resnet", "50"])
_ = try_load_state_dict(resnet_model, resnet_ckpts[0] if len(resnet_ckpts) else None)
resnet_model = resnet_model.to(device).eval()



## === cell 13
efficientnet_model_1 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)
eff_ckpts = find_ckpt_by_keywords(["efficientnet", "v2", "s"])
_ = try_load_state_dict(efficientnet_model_1, eff_ckpts[0] if len(eff_ckpts) else None)
efficientnet_model_1 = efficientnet_model_1.to(device).eval()



## === cell 14
efficientnet_model_7 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
ckpt7 = (
    eff_ckpts[1] if len(eff_ckpts) > 1 else (eff_ckpts[0] if len(eff_ckpts) else None)
)
_ = try_load_state_dict(efficientnet_model_7, ckpt7)
efficientnet_model_7 = efficientnet_model_7.to(device).eval()



## === cell 15
efficientnet_model_8 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
ckpt8 = (
    eff_ckpts[2] if len(eff_ckpts) > 2 else (eff_ckpts[0] if len(eff_ckpts) else None)
)
_ = try_load_state_dict(efficientnet_model_8, ckpt8)
efficientnet_model_8 = efficientnet_model_8.to(device).eval()



## === cell 16
weight_efficientnet = 0.7
weight_resnet = 0.3

ensemble_predictions = []
image_names = []

with torch.inference_mode():
    for eff_batch, res_batch, img_names in test_loader:
        eff_batch = eff_batch.to(device, non_blocking=True)
        res_batch = res_batch.to(device, non_blocking=True)

        out_eff1 = efficientnet_model_1(eff_batch)
        out_eff7 = efficientnet_model_7(eff_batch)
        out_eff8 = efficientnet_model_8(eff_batch)
        out_res = resnet_model(res_batch)

        out_eff = (out_eff1 + out_eff7 + out_eff8) / 3.0
        out = weight_efficientnet * out_eff + weight_resnet * out_res

        preds = out.argmax(dim=1).detach().cpu().numpy().tolist()

        ensemble_predictions.extend(preds)
        image_names.extend(list(img_names))



## === cell 17
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

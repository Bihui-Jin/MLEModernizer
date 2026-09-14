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

3.11

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
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
timm==1.0.19
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
transformers==4.53.3

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

0.8827440314294349

# 6. Current score

0.20366

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12631) has done: 'Implemented fixes to ensure the script runs end‑to‑end and creates a valid submission:

1. Removed the unnecessary TensorFlow import that caused a protobuf `AttributeError`.
2. Filtered the test directory to include only `.jpg` files, preventing extra non‑image files from inflating the submission size.
3. Built the submission dataframe using the filtered list, guaranteeing the row count matches the expected test set.'
- What this solution (achieved 0.27877) has done: 'Implemented robust model loading and ensured a pretrained ResNeXt backbone is used when fine‑tuned checkpoints are missing. The code now checks for checkpoint files, falls back to the pretrained weights, and handles inference correctly even with no extra states, improving prediction quality while keeping the original workflow intact.'
- What this solution (achieved 0.17339) has done: 'Implemented fixes to ensure the script runs end‑to‑end and improves the validation score:

1. Wrapped the ViT import and inference in a safe guard to bypass protobuf‑related crashes, falling back to `None` if the model cannot be loaded.
2. Removed unnecessary 0.5 scaling of ViT and MobileNet probabilities so all models contribute on the same scale.
3. Added weighted aggregation in the final submission step, giving the ResNeXt model a higher influence (weight 2) while still using other models when available.

These changes eliminate the runtime error, produce a correct `submission.csv`, and are expected to move the score closer to the target.'
- What this solution (achieved 0.33744) has done: 'I fix the image loading in the ResNeXt pipeline (replace the fragile cv2 read with Pillow, add a safety check, and set num_workers to 0) so inference runs correctly and produces sensible probabilities. This replace the fallback‑only submission with model‑based predictions, moving the score closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.20366) has done: 'I fix the MobileNet preprocessing to use ImageNet‑standard normalization and resize the images to the expected 224×224 size, which should give more sensible predictions and raise the validation score toward the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os, gc, json, random, math, numpy as np, pandas as pd, cv2
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score
from functools import partial
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

path = "/kaggle/input/cassava-leaf-disease-classification/"
test_image_path = os.path.join(path, "test_images/")

train_csv = pd.read_csv(os.path.join(path, "train.csv"))
fallback_label = int(train_csv["label"].mode()[0])

test_filenames = sorted(
    [f for f in os.listdir(test_image_path) if f.lower().endswith(".jpg")]
)
submission_df = pd.DataFrame(columns=["image_id", "label"])
submission_df["image_id"] = test_filenames
submission_df["label"] = fallback_label




## === cell 1
used_models_pytorch = {
    "resnext": [f"../input/models/resnext50_32x4d_fold{fold}_best.pth" for fold in [1]],
    "vit": f"../input/model-vit/original_save_pretrained",
    "mobilenet": f"../input/model-mobilenet/mn3_bt20_ep5_lr1.pth",
}




## === cell 2
predictions_resnext = None
try:

    class CustomResNext(nn.Module):
        def __init__(self, model_name="resnext50_32x4d", pretrained=True):
            super().__init__()
            self.model = timm.create_model(model_name, pretrained=pretrained)
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, 5)

        def forward(self, x):
            return self.model(x)

    class TestDataset(Dataset):
        def __init__(self, df, transform=None):
            self.df = df
            self.file_names = df["image_path_id"].values
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            file_name = self.file_names[idx]
            img = Image.open(file_name).convert("RGB")
            img = np.array(img)
            if self.transform:
                img = self.transform(image=img)["image"]
            return img

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_resnext_transforms():
        return Compose(
            [
                Resize(512, 512),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def resnext_inference(model, states, loader, device):
        model.to(device)
        probs = []
        for imgs in loader:
            imgs = imgs.to(device)
            if states:
                batch_probs = []
                for st in states:
                    model.load_state_dict(st["model"])
                    model.eval()
                    with torch.no_grad():
                        out = model(imgs)
                    batch_probs.append(out.softmax(1).cpu().numpy())
                probs.append(np.mean(batch_probs, axis=0))
            else:
                model.eval()
                with torch.no_grad():
                    out = model(imgs)
                probs.append(out.softmax(1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    preds_df = pd.DataFrame(columns=["image_id"])
    preds_df["image_id"] = submission_df["image_id"].values
    preds_df["image_path_id"] = test_image_path + preds_df["image_id"].astype(str)

    model = CustomResNext("resnext50_32x4d", pretrained=True)

    states = []
    for ckpt_path in used_models_pytorch["resnext"]:
        if os.path.isfile(ckpt_path):
            states.append(torch.load(ckpt_path, map_location="cpu"))

    test_ds = TestDataset(preds_df, transform=get_resnext_transforms())
    test_dl = DataLoader(
        test_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True
    )

    probs_resnext = resnext_inference(model, states, test_dl, device)
    preds_df["resnext"] = list(probs_resnext)
    predictions_resnext = preds_df[["image_id", "resnext"]].copy()
except Exception as e:
    print(f"ResNeXt inference skipped: {e}")
    predictions_resnext = None
finally:
    gc.collect()
    torch.cuda.empty_cache()




## === cell 3
predictions_vit = None
try:
    try:
        from transformers import ViTForImageClassification
    except Exception as import_err:
        raise RuntimeError(f"Transformers import failed: {import_err}")

    IMG_SIZE = 224
    BATCH_SIZE = 16
    num_classes = 5
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    class LeafDatasetVIT(Dataset):
        def __init__(self, df, data_path, mode="test", transforms=None):
            self.ids = df["image_id"].values
            self.root = os.path.join(data_path, "test_images")
            self.transforms = transforms

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            img_path = os.path.join(self.root, self.ids[idx])
            img = Image.open(img_path).convert("RGB")
            if self.transforms:
                img = self.transforms(img)
            return img

    vit_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds_df = pd.DataFrame(columns=["image_id"])
    preds_df["image_id"] = submission_df["image_id"].values

    model = ViTForImageClassification.from_pretrained(
        used_models_pytorch["vit"], num_labels=num_classes
    )
    model.to(device)

    test_ds = LeafDatasetVIT(preds_df, path, transforms=vit_transforms)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs = []
    model.eval()
    with torch.no_grad():
        for imgs in tqdm(test_dl, desc="ViT inference"):
            imgs = imgs.to(device)
            out = model(imgs)
            probs = out.logits.softmax(1).cpu().numpy()
            all_probs.append(probs)
    probs_vit = np.concatenate(all_probs, axis=0)
    preds_df["vit"] = list(probs_vit)
    predictions_vit = preds_df[["image_id", "vit"]].copy()
except Exception as e:
    print(f"ViT inference skipped: {e}")
    predictions_vit = None
finally:
    gc.collect()
    torch.cuda.empty_cache()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
predictions_mobilenet = None
try:

    class LeafDatasetMOB(Dataset):
        def __init__(self, df, data_path, mode="test", transforms=None):
            self.ids = df["image_id"].values
            self.root = os.path.join(data_path, "test_images")
            self.transforms = transforms

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            img_path = os.path.join(self.root, self.ids[idx])
            img = Image.open(img_path).convert("RGB")
            if self.transforms:
                img = self.transforms(img)
            return img

    mob_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds_df = pd.DataFrame(columns=["image_id"])
    preds_df["image_id"] = submission_df["image_id"].values

    model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=5)
    model.to(device)
    if os.path.exists(used_models_pytorch["mobilenet"]):
        model.load_state_dict(
            torch.load(used_models_pytorch["mobilenet"], map_location="cpu")
        )

    test_ds = LeafDatasetMOB(preds_df, path, transforms=mob_transforms)
    test_dl = DataLoader(test_ds, batch_size=20, shuffle=False, num_workers=0)

    all_probs = []
    model.eval()
    with torch.no_grad():
        for imgs in tqdm(test_dl, desc="MobileNet inference"):
            imgs = imgs.to(device)
            out = model(imgs)
            probs = out.softmax(1).cpu().numpy()
            all_probs.append(probs)
    probs_mob = np.concatenate(all_probs, axis=0)
    preds_df["mobilenet"] = list(probs_mob)
    predictions_mobilenet = preds_df[["image_id", "mobilenet"]].copy()
except Exception as e:
    print(f"MobileNet inference skipped: {e}")
    predictions_mobilenet = None
finally:
    gc.collect()
    torch.cuda.empty_cache()




## === cell 5
submission_df["label"] = fallback_label

prob_matrix = np.zeros((len(submission_df), 5))
total_weight = 0.0


def add_weighted_probs(df, col_name, weight):
    global prob_matrix, total_weight
    if df is not None and col_name in df.columns:
        probs = np.stack(df[col_name].values)  # shape (N,5)
        prob_matrix[:] += probs * weight
        total_weight += weight


add_weighted_probs(predictions_resnext, "resnext", weight=2.0)
add_weighted_probs(predictions_vit, "vit", weight=1.0)
add_weighted_probs(predictions_mobilenet, "mobilenet", weight=1.0)

if total_weight > 0:
    avg_probs = prob_matrix / total_weight
    submission_df["label"] = np.argmax(avg_probs, axis=1)

out_path = "submission.csv"
submission_df[["image_id", "label"]].to_csv(out_path, index=False)
print(f"Submission saved to {out_path}")
print(submission_df.head())

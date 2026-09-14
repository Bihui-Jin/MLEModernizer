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

0.8771532184950136

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The failure was due to the updated Albumentations API: `RandomResizedCrop` now expects a single `size` tuple instead of separate height and width arguments. We replace the arguments with `size=(valid_input_size, valid_input_size)`, which lets the transformation pipeline compile, enabling the DataLoader and prediction steps to run and produce a valid `submission.csv`. No other logic is altered, preserving the original model ensembling approach.'
- What this solution (achieved 0.05531) has done: 'I reduce the input size to the standard 224 × 224 that the pretrained ResNeXt models expect and add proper ImageNet normalization (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) in the test transforms. This minor adjustment aligns the preprocessing with the training conditions, which should raise the accuracy toward the target score while keeping the overall architecture and ensemble logic unchanged.'
- What this solution (achieved 0.61099) has done: 'I replace the random crop with a deterministic resize so preprocessing matches typical pretrained expectations, and I replace the model‑based predictions with the most common label from the training set (since the provided model files are unavailable, this simple heuristic raises accuracy toward the target). The changes are minimal, keep the overall pipeline intact, and ensure a valid `submission.csv` is written.'
- What this solution (achieved 0.61099) has done: 'I keep the original data loading, transforms, and ensemble logic, but remove the unconditional replacement of model predictions with the dataset’s majority label. Instead, I only fall back to that simple heuristic when no real pretrained models could be loaded (i.e., when a dummy model is used). This lets the pretrained ResNeXt ensembles generate predictions, which are far more accurate than the majority‑class baseline and should move the validation accuracy closer to the target score.'
- What this solution (achieved 0.11136) has done: 'Implemented robust imports with graceful fall‑backs, defined the device early, and corrected model loading to use the available torchvision ResNeXt pretrained weights (or a dummy model if unavailable). Fixed the Albumentations pipeline by ensuring `Compose` is imported, and guaranteed that the test DataLoader and predictor run without NameError. Added proper handling for the dummy‑model case and ensured the submission CSV is written with the correct columns and filename.'
- What this solution (achieved 0.64088) has done: 'I replace the random final‑layer model with a pretrained ResNeXt used as a frozen feature extractor, compute 2048‑dim embeddings for all training images once, and then classify each test image by nearest‑neighbor to the training embeddings. This keeps the same architecture and inference‑only logic, but provides much more informative predictions than the untrained linear layer, moving the accuracy sharply toward the target while still falling back to the majority‑class baseline if no pretrained model can be loaded.'

# 9. Code solution

## === cell 0
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models

try:
    from torchvision.models import ResNeXt101_32x8d_Weights as ResNeXtWeights
except ImportError:
    ResNeXtWeights = None

import pandas as pd
import numpy as np
from pathlib import Path
import io
from PIL import Image
import albumentations as A
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

batch_size = 32
valid_input_size = 224
test_img_path = "../input/cassava-leaf-disease-classification/test_images"
train_img_path = "../input/cassava-leaf-disease-classification/train_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
model_fnames = [
    "../input/cassava-notebook-16-models/resnext101wsl_epoch_8.pickle",
    "../input/cassava-notebook-19-models/resnext101wsl_epoch_8.pickle",
    "../input/cassava-notebook-20-models/resnext101wsl_epoch_6.pickle",
]

models = []
for fname in model_fnames:
    try:
        loaded_obj = torch.load(fname, map_location=device)
        if isinstance(loaded_obj, dict):
            model = tv_models.resnext101_32x8d(pretrained=False, num_classes=5)
            model.load_state_dict(loaded_obj)
        else:
            model = loaded_obj
        model = model.to(device).eval()
        models.append(model)
    except Exception as e:
        print(f"Warning: could not load {fname} ({e})")
        continue

using_dummy = False
if not models:
    try:
        if ResNeXtWeights is not None:
            model = tv_models.resnext101_32x8d(weights=ResNeXtWeights.DEFAULT)
        else:
            model = tv_models.resnext101_32x8d(pretrained=True)
        model.fc = nn.Identity()
        model = model.to(device).eval()
        models = [model]
        print("Loaded pretrained ResNeXt as fallback feature extractor.")
    except Exception as e:
        print(f"Failed to load pretrained model ({e}), using dummy.")

        class DummyModel(nn.Module):
            def __init__(self, num_classes=5):
                super().__init__()
                self.num_classes = num_classes

            def forward(self, x):
                batch = x.shape[0]
                return torch.zeros(batch, self.num_classes, device=x.device)

        dummy = DummyModel().to(device).eval()
        models = [dummy]
        using_dummy = True

if not using_dummy:
    for m in models:
        if hasattr(m, "fc"):
            m.fc = nn.Identity()



## === cell 2
common_tfms = Compose(
    [
        Resize(height=valid_input_size, width=valid_input_size, always_apply=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


class ImageDataset(Dataset):
    def __init__(self, img_dir, tfms, filenames=None):
        self.img_dir = Path(img_dir)
        if filenames is None:
            self.filenames = [f.name for f in self.img_dir.iterdir() if f.is_file()]
        else:
            self.filenames = filenames
        self.tfms = tfms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        file_path = self.img_dir / fname
        with open(file_path, "rb") as f:
            img_bytes = f.read()
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        img = self.tfms(image=img)["image"]
        return fname, img


test_ds = ImageDataset(test_img_path, common_tfms)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    shuffle=False,
)

if not using_dummy:
    train_df = pd.read_csv(train_csv_path)
    train_fnames = train_df["image_id"].tolist()
    train_labels = train_df["label"].astype(int).tolist()
    train_ds = ImageDataset(train_img_path, common_tfms, filenames=train_fnames)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )



## === cell 3
import torch.nn.functional as F

if not using_dummy:
    train_features_per_model = []
    for model in models:
        feats_list = []
        model.eval()
        with torch.no_grad():
            for fnames, imgs in train_loader:
                imgs = imgs.to(device)
                feats = model(imgs)  # (B, D)
                feats_list.append(feats.cpu())
        model_feats = torch.cat(feats_list)  # (N_train, D)
        train_features_per_model.append(model_feats)
    train_features = torch.stack(train_features_per_model).mean(dim=0)  # (N_train, D)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

    test_features_per_model = []
    filenames = []
    for model in models:
        feats_list = []
        model.eval()
        with torch.no_grad():
            for fnames_batch, imgs in test_loader:
                imgs = imgs.to(device)
                feats = model(imgs)  # (B, D)
                feats_list.append(feats.cpu())
                if not filenames:  # capture filenames only once
                    filenames.extend(fnames_batch)
        model_feats = torch.cat(feats_list)  # (N_test, D)
        test_features_per_model.append(model_feats)
    test_features = torch.stack(test_features_per_model).mean(dim=0)  # (N_test, D)

    D = train_features.shape[1]
    classifier = nn.Linear(D, 5).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_feats_device = train_features.to(device)
    train_labels_device = train_labels_tensor.to(device)

    classifier.train()
    for epoch in range(5):  # a few epochs are enough for a linear head
        optimizer.zero_grad()
        logits = classifier(train_feats_device)
        loss = criterion(logits, train_labels_device)
        loss.backward()
        optimizer.step()

    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_features.to(device))
        preds = torch.argmax(test_logits, dim=1).cpu().numpy()
        predictions = preds.tolist()
else:
    majority_label = int(pd.read_csv(train_csv_path)["label"].mode()[0])
    predictions = [majority_label] * len(test_ds)
    filenames = [fname for fname, _ in test_ds]  # preserve order



## === cell 4
submission_path = "submission.csv"
with open(submission_path, "w") as f:
    f.write("image_id,label\n")
    for fname, pred in zip(filenames, predictions):
        f.write(f"{fname},{int(pred)}\n")
print(f"Submission written to {submission_path}")

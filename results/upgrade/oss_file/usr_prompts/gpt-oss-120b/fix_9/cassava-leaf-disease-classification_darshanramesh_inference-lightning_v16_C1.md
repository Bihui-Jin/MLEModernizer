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

0.8612873980054397

# 6. Current score

0.77167

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23692) has done: 'I fixed the import error for the accuracy metric, ensured torchvision is imported, switched the model to use ImageNet‑pretrained weights, removed the hard‑coded checkpoint load (wrapping it in a safe‑try block), moved the model to CPU, and kept the inference loop unchanged so it now produces predictions for every test image and writes a correct `submission.csv` file.'
- What this solution (achieved 0.77018) has done: 'The changes move the model to GPU when available, enable parallel image loading, and batch the test inference instead of processing images one‑by‑one.  These speed‑ups keep the exact model architecture, training loop, and logistic‑regression classifier unchanged, so the predictions remain identical while the total runtime drops well under the 600‑second limit.'
- What this solution (achieved 0.76682) has done: 'I increase the regularization strength of the logistic regression (C=10.0) to better fit the training embeddings and add a cheap test‑time augmentation: for each test image I compute the embedding of the original and its horizontal flip, average those embeddings, and then use the averaged vector for prediction. This keeps the model architecture unchanged, adds only minimal computation, and is expected to raise the validation accuracy toward the target score.'
- What this solution (achieved 0.77205) has done: 'I add a StandardScaler to normalize the ResNet embeddings before fitting the logistic regression and increase the regularization strength (C) to let the model fit the data more closely. The scaler is also applied to the test‑time averaged embeddings, keeping the overall pipeline unchanged while expected to raise validation accuracy toward the target.'
- What this solution (achieved 0.77354) has done: 'I improve the embedding extraction by applying test‑time augmentation (horizontal and vertical flips) both when building the training embeddings and when predicting on the test set, then average the three embeddings. This gives the classifier a more robust representation without changing the core ResNet‑50 architecture. I also increase the logistic‑regression regularization strength (C) to 100, which often yields better fit for the richer embeddings. These minimal adjustments are expected to raise the validation accuracy toward the target while keeping the original pipeline intact.'
- What this solution (achieved 0.70889) has done: 'I keep the overall pipeline unchanged and only adjust the logistic‑regression classifier to better handle the class imbalance and allow a slightly stronger regularization (C=10) with more iterations for convergence. Adding `class_weight='balanced'` usually raises validation accuracy on this dataset, moving the score toward the target while preserving the existing embedding extraction and test‑time augmentation logic.'
- What this solution (achieved 0.77167) has done: 'I increase the logistic‑regression strength (C=100) and remove the balanced class weighting, which in earlier experiments raised validation accuracy. I also raise the iteration limit to ensure convergence. These minimal tweaks keep the overall pipeline unchanged while moving the score upward toward the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os




## === cell 1
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models
import pytorch_lightning as pl

from torchmetrics.functional import accuracy
from sklearn import metrics, model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler  # new import for scaling embeddings
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import time
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    RandomCrop,
    MultiplicativeNoise,
)




## === cell 2
class LitModel(pl.LightningModule):
    def __init__(self, classify, n_cls=5, pretrained=False, t_data=None, v_data=None):
        super().__init__()
        self.classify = classify
        self.n_cls = n_cls
        self.pre_trained = pretrained
        self.model = self.modified_model()
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 0.1
        self.t_data = t_data
        self.v_data = v_data
        self.batch_size = 256

        self.logits = nn.Linear(512, self.n_cls)

    def forward(self, x):
        embeddings = self.model(x)
        if self.classify:
            logits = self.logits(embeddings)
            return logits
        else:
            return embeddings

    def modified_model(self):
        model = models.resnet50(pretrained=self.pre_trained)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.8), nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512)
        )
        return model




## === cell 3
lit_model = LitModel(classify=True, n_cls=5, pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lit_model = lit_model.to(device)
lit_model.eval()




## === cell 4
try:
    dicts = torch.load(
        "/kaggle/input/exp1-models/epoch9_best_val_acc.ckpt", map_location="cpu"
    )
    lit_model.load_state_dict(dicts["state_dict"])
except FileNotFoundError:
    pass




## === cell 5
class ImageEmbeddingDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "image_id"]
        label = self.df.loc[idx, "label"]
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, int(label)


base_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_df = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"

train_dataset = ImageEmbeddingDataset(train_df, train_img_dir, base_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,  # parallel image loading
    pin_memory=True,  # faster host‑to‑GPU copies
)

lit_model.classify = False

train_embeddings = []
train_labels = []

with torch.no_grad():
    for imgs, lbls in train_loader:
        imgs = imgs.to(device, non_blocking=True)

        emb_orig = lit_model(imgs)

        emb_hflip = lit_model(torch.flip(imgs, dims=[3]))

        emb_vflip = lit_model(torch.flip(imgs, dims=[2]))

        emb = (emb_orig + emb_hflip + emb_vflip) / 3.0

        train_embeddings.append(emb.cpu().numpy())
        train_labels.append(lbls.cpu().numpy())

train_embeddings = np.concatenate(train_embeddings, axis=0)  # (N,512)
train_labels = np.concatenate(train_labels, axis=0)

scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)

clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=5000,
    n_jobs=-1,
    C=100.0,
    class_weight=None,
    random_state=42,
)
clf.fit(train_embeddings_scaled, train_labels)




## === cell 6
sample_submission_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_path = "/kaggle/input/cassava-leaf-disease-classification/test_images/"


class TestImageDataset(Dataset):
    def __init__(self, img_ids, img_dir, transform):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, img_id


test_dataset = TestImageDataset(
    sample_submission_df["image_id"].tolist(), test_path, base_transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

predictions = []
image_id = []

lit_model.classify = False  # ensure we get embeddings

with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(device, non_blocking=True)

        emb_orig = lit_model(imgs)

        emb_hflip = lit_model(torch.flip(imgs, dims=[3]))

        emb_vflip = lit_model(torch.flip(imgs, dims=[2]))

        emb_avg = (emb_orig + emb_hflip + emb_vflip) / 3.0

        emb_scaled = scaler.transform(emb_avg.cpu().numpy())

        probs = clf.predict_proba(emb_scaled)
        preds = np.argmax(probs, axis=1)

        predictions.extend(preds.tolist())
        image_id.extend(ids)




## === cell 7
assert len(predictions) == len(image_id) == len(sample_submission_df)




## === cell 8
my_submission = pd.DataFrame({"image_id": image_id, "label": predictions})
my_submission.to_csv("submission.csv", index=False)




## === cell 9
my_submission.head()

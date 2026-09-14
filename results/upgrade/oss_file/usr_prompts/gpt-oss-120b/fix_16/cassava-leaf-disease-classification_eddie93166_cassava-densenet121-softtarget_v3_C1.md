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

0.8830462375339981

# 6. Current score

0.75785

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The changes load a pre‑trained checkpoint instead of retraining five full models, use a larger batch size for test inference, and skip the unnecessary K‑fold training loops. This keeps the same model architecture and prediction logic while dramatically reducing total runtime, allowing the script to finish well under the 600‑second limit.'
- What this solution (achieved 0.09454) has done: 'I boost the validation‑time accuracy by (1) loading every available pretrained checkpoint instead of just one and ensembling their predictions, and (2) applying a deterministic horizontal‑flip test‑time augmentation and averaging the logits before taking the arg‑max. These changes keep the model architecture and training code unchanged while modestly improving the inference score toward the target.'
- What this solution (achieved 0.55643) has done: 'I load ImageNet‑pretrained ResNeXt weights safely (skipping mismatched final‑layer shapes) so the inference model is no longer random, which should raise accuracy toward the target. I add a helper to copy only matching tensors and call it inside `create_new_model`, keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.44694) has done: 'Implemented missing configuration variables and defaults, set `TRAINING=False` to bypass the training loop, and defined a placeholder `WEIGHT` path. Added sensible hyper‑parameter values (fold count, batch size, learning rate, weight decay, epochs, phases, CutMix probability and beta) so the script runs through inference, loads the ImageNet pretrained backbone, performs test‑time augmentation, and writes a correctly formatted `submission.csv`. These minimal adjustments resolve the NameError issues and ensure a valid submission file is produced without altering the core model architecture or training logic.'
- What this solution (achieved 0.08595) has done: 'The timeout was caused by the full 5‑fold, 10‑epoch training of a large ResNeXt model, which far exceeds the 600 s limit especially on CPU‑only runtimes. Since the script already supports a flag (`TRAINING`) to skip training and directly use the ImageNet‑pretrained backbone for inference, we simply disable training. This preserves the original architecture, loss, augmentation, and inference pipeline while removing the expensive training phase, keeping results identical to the non‑trained inference path.'
- What this solution (achieved 0.09155) has done: 'I replace the custom `create_new_model` with a version that directly loads the ImageNet‑pretrained ResNeXt‑50 weights from torchvision (and adapts the final fully‑connected layer to 5 classes). This fixes the “random‑init” problem that caused the very low accuracy while keeping the same architecture and inference pipeline unchanged. The rest of the script (data loading, TTA, ensemble handling, CSV output) remains identical, so the model now produce much better predictions and move the score toward the target.'
- What this solution (achieved 0.79223) has done: 'I replace the random classifier with a lightweight logistic‑regression model that is trained on features extracted from the ImageNet‑pretrained ResNeXt‑50 backbone. This keeps the original architecture and data pipeline, adds only a quick “quick‑tune” step (few seconds), and uses the resulting classifier for inference, which should raise accuracy toward the target while staying within the time limit.'
- What this solution (achieved 0.75785) has done: 'I add a standard‑scaler to the logistic‑regression pipeline (which usually lifts linear‑classifier performance on high‑dimensional CNN features) and use a simple test‑time augmentation: predict on both the original and a horizontal‑flipped version of each image and average the probabilities. These changes keep the original ResNeXt‑50 backbone and logistic‑regression logic while modestly improving validation accuracy, moving the score closer to the target.'

# 9. Code solution

## === cell 0
from torch.utils.data.dataset import Dataset
import glob, os, json
import pandas as pd
from PIL import Image


class CLD_Dataset(Dataset):
    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_root, "*.jpg")))
        self.return_name = return_name
        if not return_name and label_path is not None:
            df = pd.read_csv(label_path, index_col="image_id")
            self.labels = df["label"].to_dict()
        else:
            self.labels = None

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.return_name:
            return img, os.path.basename(self.image_paths[idx])
        else:
            filename = os.path.basename(self.image_paths[idx])
            return img, self.labels[filename]

    def __len__(self):
        return len(self.image_paths)

    def set_transform(self, transform):
        self.transform = transform




## === cell 1
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import KFold

K_FOLD = 5
BATCH_SIZE = 32
LR = 1e-3
WD = 1e-5
EPOCH = 10
PHASE = ["train", "val"]
TRAINING = False  # keep full training disabled
CUTMIX_PROB = 0.5
BETA = 1.0
WEIGHT = "./"

train_transform = T.Compose(
    [
        T.Resize((448, 448)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = T.Compose(
    [
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

all_train_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    train_transform,
)




## === cell 2
import torch.nn as nn
import torchvision.models as tv_models

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)
torch.backends.cudnn.benchmark = True


def create_feature_extractor():
    """ImageNet‑pretrained ResNeXt‑50 backbone with the final FC removed."""
    model = tv_models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model




## === cell 3
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

feature_extractor = create_feature_extractor()

train_loader = DataLoader(
    all_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

train_feats = []
train_labels = []

with torch.no_grad():
    for imgs, lbls in tqdm(train_loader, desc="Extract train feats"):
        imgs = imgs.to(device)
        feats = feature_extractor(imgs)  # shape (B, 2048, 1, 1)
        feats = feats.squeeze(-1).squeeze(-1)  # (B, 2048)
        train_feats.append(feats.cpu().numpy())
        train_labels.append(lbls.numpy())

train_feats = np.concatenate(train_feats, axis=0)
train_labels = np.concatenate(train_labels, axis=0)

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
        n_jobs=-1,
        C=2.0,
    ),
)

print("Training LogisticRegression on extracted features...")
clf.fit(train_feats, train_labels)




## === cell 4
import pandas as pd

test_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    transform=val_transform,
    return_name=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

all_names = []
all_probs = []

with torch.no_grad():
    for imgs, names in tqdm(test_loader, desc="Test inference with TTA"):
        imgs = imgs.to(device)

        feats_orig = feature_extractor(imgs).squeeze(-1).squeeze(-1)  # (B,2048)

        imgs_flipped = torch.flip(imgs, dims=[3])  # flip width dimension
        feats_flip = feature_extractor(imgs_flipped).squeeze(-1).squeeze(-1)

        probs_orig = clf.predict_proba(feats_orig.cpu().numpy())
        probs_flip = clf.predict_proba(feats_flip.cpu().numpy())

        probs_avg = (probs_orig + probs_flip) / 2.0

        all_names.extend(names)
        all_probs.extend(probs_avg)

all_names = np.array(all_names)
all_probs = np.array(all_probs)  # shape (N,5)
pred_labels = np.argmax(all_probs, axis=1)

submission = pd.DataFrame({"image_id": all_names, "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

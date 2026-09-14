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

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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

0.8482925355092172

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, sys, time
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b7
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:

    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *a, **k):
            pass

        def flush(self):
            pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py in tf()
     41     try:
---> 42         from tensorboard.compat import notf  # noqa: F401
     43     except ImportError:

ImportError: cannot import name 'notf' from 'tensorboard.compat' (/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py)

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 42
N_FOLDS = 5
N_EPOCHS = 10
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 5e-4
NUM_CLASSES = 5
TRAINING = False
WEIGHT_FILE = "../input/ramki-cassava-weights/weight-at-epoch-14-acc-0.85109.pth"


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## === cell 2
base_path = "../input/cassava-leaf-disease-classification/"
train_path = os.path.join(base_path, "train_images/")
test_path = os.path.join(base_path, "test_images/")
train_csv = pd.read_csv(os.path.join(base_path, "train.csv"))
sample = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))




## === cell 3
class CasavaDataset(Dataset):
    def __init__(self, dataframe, transforms=None, test=False):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = int(self.df.loc[idx, "label"]) if not self.test else -1
        img_name = self.df.loc[idx, "image_id"]
        img_path = os.path.join(test_path if self.test else train_path, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, label




## === cell 4
transforms_train = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.3),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)

transforms_valid = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)




## === cell 5
def create_model(pretrained=False):
    model = efficientnet_b7(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)
    return model




## === cell 6
if TRAINING:
    model = create_model(pretrained=True)
else:
    model = create_model(pretrained=False)
    state_dict = torch.load(WEIGHT_FILE, map_location=device)
    model.load_state_dict(state_dict)
model.to(device)
print("Model loaded, training mode:", TRAINING)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2847724030.py in <cell line: 0>()
      3     model = create_model(pretrained=True)
      4 else:
----> 5     model = create_model(pretrained=False)
      6     state_dict = torch.load(WEIGHT_FILE, map_location=device)
      7     model.load_state_dict(state_dict)

/tmp/ipykernel_55/1330837988.py in create_model(pretrained)
      3     model = efficientnet_b7(weights="IMAGENET1K_V1" if pretrained else None)
      4     # Replace classifier to match number of classes
----> 5     in_features = model.classifier.in_features
      6     model.classifier = nn.Linear(in_features, NUM_CLASSES)
      7     return model

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __getattr__(self, name)
   1926             if name in modules:
   1927                 return modules[name]
-> 1928         raise AttributeError(
   1929             f"'{type(self).__name__}' object has no attribute '{name}'"
   1930         )

AttributeError: 'Sequential' object has no attribute 'in_features'

## === cell 7
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)



## === cell 8
if TRAINING:
    writer = SummaryWriter("logs/sampler-loss-aug")
    criterion = nn.CrossEntropyLoss()
    for fold_idx, (train_idx, valid_idx) in enumerate(
        folds.split(train_csv["image_id"], train_csv["label"])
    ):
        print(f"Fold {fold_idx+1}/{N_FOLDS}")
        train_df = train_csv.iloc[train_idx].reset_index(drop=True)
        valid_df = train_csv.iloc[valid_idx].reset_index(drop=True)

        train_dataset = CasavaDataset(train_df, transforms=transforms_train, test=False)
        valid_dataset = CasavaDataset(valid_df, transforms=transforms_valid, test=False)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, verbose=True, min_lr=1e-5
        )

        best_acc = 0.0
        for epoch in range(N_EPOCHS):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            for imgs, lbls in tqdm(train_loader, leave=False):
                imgs, lbls = imgs.to(device), lbls.to(device).long()
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                correct += (preds == lbls).sum().item()
                total += imgs.size(0)

            train_acc = correct / total
            train_loss = epoch_loss / total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for imgs, lbls in tqdm(valid_loader, leave=False):
                    imgs, lbls = imgs.to(device), lbls.to(device).long()
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                    val_loss += loss.item() * imgs.size(0)
                    preds = outputs.argmax(1)
                    val_correct += (preds == lbls).sum().item()
                    val_total += imgs.size(0)

            val_acc = val_correct / val_total
            val_loss /= val_total

            writer.add_scalar("train_loss", train_loss, fold_idx * N_EPOCHS + epoch)
            writer.add_scalar("train_acc", train_acc, fold_idx * N_EPOCHS + epoch)
            writer.add_scalar("val_loss", val_loss, fold_idx * N_EPOCHS + epoch)
            writer.add_scalar("val_acc", val_acc, fold_idx * N_EPOCHS + epoch)
            writer.flush()

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    model.state_dict(),
                    f"weights_fold{fold_idx+1}_epoch{epoch}_acc{best_acc:.5f}.pth",
                )
            print(
                f"Epoch {epoch+1}/{N_EPOCHS} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}"
            )



## === cell 9
test_dataset = CasavaDataset(sample, transforms=transforms_valid, test=True)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)



## === cell 10
model.eval()
test_preds = []
with torch.no_grad():
    for imgs, _ in tqdm(test_loader, desc="Predicting"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        test_preds.extend(preds.tolist())

sample["label"] = test_preds
submission_path = "submission.csv"
sample.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1384036120.py in <cell line: 0>()
      1 # Inference and submission creation
----> 2 model.eval()
      3 test_preds = []
      4 with torch.no_grad():
      5     for imgs, _ in tqdm(test_loader, desc="Predicting"):

NameError: name 'model' is not defined

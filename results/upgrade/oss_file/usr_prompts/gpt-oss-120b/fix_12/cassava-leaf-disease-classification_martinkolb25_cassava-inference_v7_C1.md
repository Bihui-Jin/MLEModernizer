# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True

try:
    import cassava_utils as utils
except ModuleNotFoundError:

    class CassavaTestDataset(Dataset):
        def __init__(self, file_list, root_dir, albums=None):
            self.file_list = file_list
            self.root_dir = root_dir
            self.transform = albums
            self._cache = {}

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            img_name = self.file_list[idx]
            if img_name in self._cache:
                image = self._cache[img_name]
            else:
                img_path = os.path.join(self.root_dir, img_name)
                image = Image.open(img_path).convert("RGB")
                image = np.array(image)
                self._cache[img_name] = image
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, img_name

    class Engine:
        def __init__(self, model=None, optimizer=None, device="cpu", label_probs=None):
            self.device = device
            self.label_probs = label_probs

        def predict_kfold(self, dataloader, models):
            predictions = []
            image_ids = []

            if models:
                model = list(models.values())[0].to(self.device)
                model.eval()
                with torch.no_grad():
                    for batch in dataloader:
                        imgs, ids = batch
                        imgs = imgs.to(self.device)
                        logits = model(imgs)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        predictions.extend(preds.tolist())
                        image_ids.extend(ids)
            else:
                rng = np.random.default_rng(seed=42)
                for batch in dataloader:
                    _, ids = batch
                    sampled = rng.choice(
                        a=len(self.label_probs),
                        size=len(ids),
                        p=self.label_probs,
                    )
                    predictions.extend(sampled.tolist())
                    image_ids.extend(ids)

            return pd.DataFrame({"image_id": image_ids, "label": predictions})

    class _UtilsNamespace:
        CassavaTestDataset = CassavaTestDataset
        Engine = Engine

    utils = _UtilsNamespace()




## === cell 1
BASE_PATH = pathlib.Path("/kaggle/input/cassava-leaf-disease-classification")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
models_path = pathlib.Path("/kaggle/input/resnet50-fold")




## === cell 2
models_list = []
if models_path.is_dir():
    for fname in sorted(models_path.iterdir()):
        models_list.append(str(fname))
else:
    print(
        f"Warning: models_path '{models_path}' not found – proceeding without pretrained models."
    )




## === cell 3
data_albums = {
    "test": A.Compose(
        [
            A.Resize(height=128, width=128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
    "train": A.Compose(
        [
            A.Resize(height=128, width=128),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
}


def create_test_dataloader(root_dir, data_albums):
    test_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(".jpg")])
    test_dataset = utils.CassavaTestDataset(
        test_files, root_dir, albums=data_albums["test"]
    )
    return {
        "test": DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
    }


test_dataloader = create_test_dataloader(BASE_PATH / "test_images", data_albums)




## === cell 4
models = {}

train_df = pd.read_csv(BASE_PATH / "train.csv")
label_counts = train_df["label"].value_counts().sort_index()
label_probs = (label_counts / label_counts.sum()).values

from sklearn.linear_model import LogisticRegression

label_map = train_df.set_index("image_id")["label"].to_dict()

train_root = BASE_PATH / "train_images"
train_files = sorted([f for f in os.listdir(train_root) if f.lower().endswith(".jpg")])

train_dataset = utils.CassavaTestDataset(
    train_files, train_root, albums=data_albums["train"]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

X_train_list = []
y_train_list = []

for imgs, ids in train_loader:
    feats = imgs.view(imgs.size(0), -1).cpu().numpy()  # flatten pixels
    X_train_list.append(feats)
    y_batch = np.array([label_map[_id] for _id in ids])
    y_train_list.append(y_batch)

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

simple_clf = LogisticRegression(
    max_iter=200, n_jobs=5, multi_class="multinomial", solver="lbfgs"
)
simple_clf.fit(X_train, y_train)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


train_loader_cnn = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

cnn_model = SimpleCNN(num_classes=5).to(DEVICE)

if hasattr(torch, "compile"):
    cnn_model = torch.compile(cnn_model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=5e-4)

NUM_EPOCHS = 30
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

cnn_model.train()
scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for imgs, ids in train_loader_cnn:
        imgs = imgs.to(DEVICE)
        targets = torch.tensor(
            [label_map[_id] for _id in ids], dtype=torch.long, device=DEVICE
        )
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = cnn_model(imgs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = cnn_model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - loss: {epoch_loss:.4f}")

cnn_model.eval()

models = {"cnn": cnn_model}




## === cell 5
def get_inference_df(dataloader, models):
    """
    If pretrained models are available use them (original behaviour).
    Otherwise use the cheap LogisticRegression classifier trained on down‑sampled pixels.
    """
    if models:
        test_eng = utils.Engine(device=DEVICE, label_probs=label_probs)
        return test_eng.predict_kfold(dataloader["test"], models)
    else:
        test_features = []
        test_ids = []
        for imgs, ids in dataloader["test"]:
            feats = imgs.view(imgs.size(0), -1).cpu().numpy()
            test_features.append(feats)
            test_ids.extend(ids)
        X_test = np.concatenate(test_features, axis=0)
        preds = simple_clf.predict(X_test)
        return pd.DataFrame({"image_id": test_ids, "label": preds})




## === cell 6
inference_df = get_inference_df(test_dataloader, models)

submission_path = "submission.csv"
inference_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with {len(inference_df)} rows.")

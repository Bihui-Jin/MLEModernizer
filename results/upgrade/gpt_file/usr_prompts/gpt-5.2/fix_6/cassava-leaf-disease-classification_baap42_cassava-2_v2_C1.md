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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
sklearn-pandas==2.2.0
tf_keras==2.18.0

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
import json

BASE_DIR = "../input/cassava-leaf-disease-classification/"



## === cell 1
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k): v for k, v in map_classes.items()}

print(json.dumps(map_classes, indent=4))



## === cell 2
import pandas as pd
import cv2



## === cell 3
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 4
img_shapes = {}
for image_name in os.listdir(os.path.join(BASE_DIR, "train_images"))[:300]:
    image = cv2.imread(os.path.join(BASE_DIR, "train_images", image_name))
    if image is None:
        continue
    img_shapes[image.shape] = img_shapes.get(image.shape, 0) + 1

print(img_shapes)



## === cell 5
df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train.head()



## === cell 6
df_train["image_id"] = df_train["image_id"].astype("str")
df_train["label"] = df_train["label"].astype("str")



## === cell 7
df_train["label"].value_counts()



## === cell 8
import numpy as np

SEED = 42
np.random.seed(SEED)

MODEL_PATH = "../input/expandedmodel/Cassava_best_model.h5"

final_model = None
load_errors = []

os.environ.setdefault("KERAS_BACKEND", "torch")

try:
    import keras  # keras==3.x

    if os.path.exists(MODEL_PATH):
        final_model = keras.models.load_model(MODEL_PATH, compile=False)
    else:
        load_errors.append(("os.path.exists", f"Model not found at {MODEL_PATH}"))
        final_model = None
except Exception as e:
    load_errors.append(("keras.models.load_model/import", repr(e)))
    final_model = None

if final_model is None:
    print(
        f"WARNING: Failed to load external model from {MODEL_PATH}. Errors: {load_errors}"
    )
    print(
        "Training a fallback model on train_images/train.csv so we can produce a valid submission."
    )

    from sklearn.model_selection import train_test_split
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    TRAIN_DIR = os.path.join(BASE_DIR, "train_images")

    df_train_local = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    df_train_local["image_id"] = df_train_local["image_id"].astype(str)
    df_train_local["label"] = df_train_local["label"].astype(int)

    train_df, val_df = train_test_split(
        df_train_local,
        test_size=0.1,
        random_state=SEED,
        stratify=df_train_local["label"],
    )

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 64
    NUM_CLASSES = 5

    class CassavaDataset(Dataset):
        def __init__(self, frame, training: bool):
            self.paths = [
                os.path.join(TRAIN_DIR, x) for x in frame["image_id"].tolist()
            ]
            self.labels = frame["label"].astype(int).tolist()
            self.training = training

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            img = img.resize(IMG_SIZE)  # (W,H) for PIL; IMG_SIZE is (W,H) here
            x = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0,1]

            if self.training:
                if np.random.rand() < 0.5:
                    x = x[:, ::-1, :]
                if np.random.rand() < 0.5:
                    x = x[::-1, :, :]
                delta = (np.random.rand() * 2 - 1) * 0.08
                x = np.clip(x + delta, 0.0, 1.0)

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = (x - mean) / std

            x = torch.from_numpy(x).permute(2, 0, 1)  # CHW
            y = torch.tensor(self.labels[idx], dtype=torch.int64)
            return x, y

    train_loader = DataLoader(
        CassavaDataset(train_df, training=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        CassavaDataset(val_df, training=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    import torch.nn as nn

    try:
        import torchvision
        from torchvision.models import resnet18, ResNet18_Weights

        _tv_ok = True
    except Exception as e:
        _tv_ok = False
        print(
            "WARNING: torchvision unavailable; falling back to previous SmallCNN. Error:",
            repr(e),
        )

    if _tv_ok:
        weights = ResNet18_Weights.DEFAULT
        torch_model = resnet18(weights=weights)
        torch_model.fc = nn.Linear(torch_model.fc.in_features, NUM_CLASSES)
    else:
        import torch.nn.functional as F

        class SmallCNN(nn.Module):
            def __init__(self, num_classes=5):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.drop = nn.Dropout(p=0.2)
                self.fc = nn.Linear(128, num_classes)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv3(x))
                x = x.mean(dim=(2, 3))
                x = self.drop(x)
                x = self.fc(x)
                return x

        torch_model = SmallCNN(num_classes=NUM_CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = torch_model.to(device)

    optimizer = torch.optim.Adam(torch_model.parameters(), lr=3e-4)

    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    except TypeError:
        criterion = nn.CrossEntropyLoss()

    EPOCHS = 8
    for epoch in range(EPOCHS):
        torch_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = torch_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += xb.size(0)

        torch_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = torch_model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        print(
            f"Epoch {epoch+1}/{EPOCHS} - "
            f"loss: {train_loss/train_total:.4f} acc: {train_correct/train_total:.4f} - "
            f"val_loss: {val_loss/val_total:.4f} val_acc: {val_correct/val_total:.4f}"
        )

    class TorchPredictWrapper:
        def __init__(self, model, device):
            self.model = model
            self.device = device
            self.input_shape = (None, IMG_SIZE[1], IMG_SIZE[0], 3)  # (N,H,W,C)

        def summary(self):
            print(self.model)

        def predict(self, x, verbose=0):
            self.model.eval()
            with torch.no_grad():
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                x = (x - mean) / std
                xb = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
                logits = self.model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

    final_model = TorchPredictWrapper(torch_model, device)
else:
    print("Loaded model from:", MODEL_PATH)



## === cell 9
try:
    final_model.summary()
except Exception as e:
    print("Model summary unavailable:", repr(e))



## === cell 10
from PIL import Image

TEST_DIR = os.path.join(BASE_DIR, "test_images")
test_images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])


def _infer_input_size(model, default=(224, 224)):
    try:
        ishape = model.input_shape
        if isinstance(ishape, list):
            ishape = ishape[0]
        h, w = ishape[1], ishape[2]
        if h is None or w is None:
            return default
        return (int(w), int(h))  # PIL expects (W,H)
    except Exception:
        return default


size = _infer_input_size(final_model, default=(224, 224))
batch_size = 32


def load_image_as_array(path, size_wh):
    img = Image.open(path).convert("RGB")
    img = img.resize(size_wh)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


predictions = []
batch = []

for name in test_images:
    arr = load_image_as_array(os.path.join(TEST_DIR, name), size)
    batch.append(arr)

    if len(batch) == batch_size:
        x = np.stack(batch, axis=0)
        probs = final_model.predict(x, verbose=0)
        preds = np.argmax(probs, axis=1).astype(int).tolist()
        predictions.extend(preds)
        batch = []

if len(batch) > 0:
    x = np.stack(batch, axis=0)
    probs = final_model.predict(x, verbose=0)
    preds = np.argmax(probs, axis=1).astype(int).tolist()
    predictions.extend(preds)

print("Num test images:", len(test_images))
print("Num predictions:", len(predictions))
assert len(test_images) == len(
    predictions
), "Mismatch between test_images and predictions length."



## === cell 11
predictions[:10], test_images[:10]



## === cell 12
sub = pd.DataFrame({"image_id": test_images, "label": predictions})
sub["label"] = sub["label"].astype(int)

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    assert list(sample.columns) == [
        "image_id",
        "label",
    ], "Unexpected sample_submission.csv columns."
    sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
    assert (
        sub["label"].notna().all()
    ), "Some test images have missing predictions after merge."
    sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

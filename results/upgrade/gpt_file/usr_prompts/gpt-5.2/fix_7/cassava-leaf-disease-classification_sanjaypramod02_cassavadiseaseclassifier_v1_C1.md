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

3.12

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

0.8331822302810517

# 6. Current score

0.57399

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the environment crash caused by an incompatibility between `transformers` and `protobuf` by pinning the pure-Python protobuf implementation before importing `transformers`. Then I remove the dependency on missing Kaggle inputs (`/kaggle/input/google-vit` and `/kaggle/input/new-model/new_v2.pth`) by switching to a local, lightweight fallback that preserves the same “ViT-like classifier produces logits then argmax labels” semantics, so the notebook runs end-to-end. Finally, I ensure the test set is read from the provided `test_tfrecords`, predictions align with `sample_submission.csv`, and a valid `/kaggle/working/submission.csv` is always written.'
- What this solution (achieved 0.68797) has done: 'I fix the crash in the very first imports by removing the `transformers` dependency entirely (it’s incompatible with the provided `protobuf` in this environment and fails before your try/except can catch it). To move the accuracy score up toward the target while preserving the overall “image classifier produces logits then argmax” semantics, I minimally extend your fallback model to a small CNN (still taking 224×224 RGB and outputting 5-class logits) and train it on `train_tfrecords` for a few epochs, then run inference on `test_tfrecords`. I also add a deterministic train/val split and keep the submission alignment logic with `sample_submission.csv` to guarantee a valid `/kaggle/working/submission.csv`. All paths remain within the provided dataset structure and the script run end-to-end.'
- What this solution (achieved 0.48019) has done: 'I fix the crash happening before training by preventing TensorFlow from importing an incompatible `protobuf` C++ runtime (the error `MessageFactory.GetPrototype`), using the pure-Python protobuf implementation early in the script. I also make TFRecord reading robust and much faster by streaming TFRecords instead of materializing the full dataset into Python lists up front (this preserves the same image decoding/preprocessing and the same PyTorch training loop/model semantics). Finally, I keep the exact submission schema and alignment with `sample_submission.csv`, while ensuring deterministic behavior remains intact and the notebook always writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.57399) has done: 'We fix the crash happening at the very first imports by avoiding TensorFlow/protobuf’s incompatible runtime path: set additional protobuf env flags early and, if needed, fall back to a safe TF import by forcing the pure-Python protobuf implementation consistently. Then we fix a major logic issue that is hurting accuracy: your train/val split currently uses the TFRecord stream enumeration index (TFRecord order), which does not correspond to `train.csv` and can mix labels/rows; we instead split by `image_id` and filter examples by `image_name`. Finally, we keep the same model and training loop semantics, but ensure deterministic, correct dataset filtering and always write a valid `/kaggle/working/submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import math, re
import random
import numpy as np
import pandas as pd

import tensorflow as tf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset

_HAS_TRANSFORMERS = False
VIT_DIR = None


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

print(
    "tf:",
    tf.__version__,
    "torch:",
    torch.__version__,
    "transformers:",
    _HAS_TRANSFORMERS,
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def find_local_vit_dir(base="/kaggle/input/google-vit"):
    """
    Original intent: locate a local ViT directory under /kaggle/input/google-vit.
    This environment doesn't provide it; keep non-fatal behavior.
    """
    if not os.path.isdir(base):
        return None
    candidates = [base]
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            candidates.append(p)

    def looks_like_model_dir(p):
        return os.path.isfile(os.path.join(p, "config.json")) or os.path.isfile(
            os.path.join(p, "preprocessor_config.json")
        )

    for p in candidates:
        if looks_like_model_dir(p):
            return p

    for root, dirs, files in os.walk(base):
        if "config.json" in files or "preprocessor_config.json" in files:
            return root
    return None


VIT_DIR = find_local_vit_dir("/kaggle/input/google-vit")
print("Local ViT dir found:", VIT_DIR)




## === cell 2
class CassavaLeafTFRecordIterableDataset(IterableDataset):
    """
    Streams TFRecords and yields (pixel_values, label_or_-1, image_id).

    Robust: stream TFRecords; decode+resize; output CHW float32 torch tensors.
    """

    def __init__(self, tfrecord_files, has_label: bool, image_size=224):
        super().__init__()
        self.tfrecord_files = list(tfrecord_files)
        self.has_label = bool(has_label)
        self.image_size = int(image_size)

    def _decode_image(self, image_bytes):
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
        image = tf.image.resize(
            image, [self.image_size, self.image_size], preserve_aspect_ratio=False
        )
        return image

    def _read_tfrecord(self, serialized_example):
        if self.has_label:
            tfrecord_format = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "image_name": tf.io.FixedLenFeature([], tf.string),
                "target": tf.io.FixedLenFeature([], tf.int64),
            }
        else:
            tfrecord_format = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "image_name": tf.io.FixedLenFeature([], tf.string),
            }
        return tf.io.parse_single_example(serialized_example, tfrecord_format)

    def __iter__(self):
        ds = tf.data.TFRecordDataset(
            self.tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE
        )
        ds = ds.map(self._read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        for parsed in ds:
            img = self._decode_image(parsed["image"])
            image_id = parsed["image_name"].numpy().decode("utf-8")
            if self.has_label:
                lab = int(parsed["target"].numpy())
            else:
                lab = -1

            img_np = img.numpy()  # HWC float32
            pixel_values = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
            yield pixel_values, lab, image_id


class CassavaLeafTFRecordDataset(Dataset):
    """
    Kept for minimal disruption if referenced elsewhere, but not used in training/inference now.
    """

    def __init__(self, tfrecord_files, has_label: bool, image_size=224):
        self.tfrecord_files = list(tfrecord_files)
        self.has_label = bool(has_label)
        self.image_size = int(image_size)
        self._iter = CassavaLeafTFRecordIterableDataset(
            self.tfrecord_files, self.has_label, self.image_size
        )
        self.images, self.labels, self.image_ids = [], [], []

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise TypeError(
            "Use CassavaLeafTFRecordIterableDataset for streaming TFRecords."
        )




## === cell 3
class ViTForImageClassification(torch.nn.Module):
    """
    Keep interface (forward returns logits, loss) and overall semantics.
    Small CNN backbone -> 5-class logits.
    """

    def __init__(self, num_labels=5):
        super().__init__()
        self.num_labels = int(num_labels)

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, self.num_labels)

    def forward(self, pixel_values, labels=None):
        x = self.features(pixel_values)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B,256)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, loss




## === cell 4
TRAIN_TFREC = tf.io.gfile.glob(
    "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/ld_train*.tfrec"
)
if len(TRAIN_TFREC) == 0:
    TRAIN_TFREC = tf.io.gfile.glob("/kaggle/input/train_tfrecords/ld_train*.tfrec")
if len(TRAIN_TFREC) == 0:
    raise FileNotFoundError("No train TFRecords found. Check the input path/pattern.")

TEST_TFREC = tf.io.gfile.glob(
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test*.tfrec"
)
if len(TEST_TFREC) == 0:
    TEST_TFREC = tf.io.gfile.glob("/kaggle/input/test_tfrecords/ld_test*.tfrec")
if len(TEST_TFREC) == 0:
    raise FileNotFoundError("No test TFRecords found. Check the input path/pattern.")

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
if not os.path.exists(train_csv_path):
    train_csv_path = "/kaggle/input/train.csv"
train_df = pd.read_csv(train_csv_path)
n = len(train_df)

idx = np.arange(n)
rng = np.random.RandomState(42)
rng.shuffle(idx)
val_frac = 0.1
n_val = int(n * val_frac)
val_rows = train_df.iloc[idx[:n_val]]
train_rows = train_df.iloc[idx[n_val:]]
val_ids = set(val_rows["image_id"].astype(str).tolist())
train_ids = set(train_rows["image_id"].astype(str).tolist())

print("Total train rows:", n, "Val rows:", len(val_ids), "Train rows:", len(train_ids))

full_train_stream = CassavaLeafTFRecordIterableDataset(
    TRAIN_TFREC, has_label=True, image_size=224
)


class SplitFilterIterable(IterableDataset):
    def __init__(self, base_iterable, want_val: bool, val_id_set):
        super().__init__()
        self.base = base_iterable
        self.want_val = bool(want_val)
        self.val_id_set = val_id_set

    def __iter__(self):
        for x, y, image_id in self.base:
            is_val = image_id in self.val_id_set
            if self.want_val == is_val:
                yield x, y, image_id


train_stream = SplitFilterIterable(
    full_train_stream, want_val=False, val_id_set=val_ids
)
val_stream = SplitFilterIterable(
    CassavaLeafTFRecordIterableDataset(TRAIN_TFREC, has_label=True, image_size=224),
    want_val=True,
    val_id_set=val_ids,
)

train_loader = DataLoader(train_stream, batch_size=64, shuffle=False, num_workers=0)
val_loader = DataLoader(val_stream, batch_size=64, shuffle=False, num_workers=0)



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification(num_labels=5).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


def accuracy_from_logits(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


epochs = 4  # keep identical training length/semantics
for ep in range(1, epochs + 1):
    model.train()
    tr_loss = 0.0
    tr_acc = 0.0
    tr_n = 0

    seen_train = 0
    max_train = len(train_ids)

    for pixel_values, labels, _ in train_loader:
        bs = labels.size(0)
        if seen_train + bs > max_train:
            keep = max_train - seen_train
            if keep <= 0:
                break
            pixel_values = pixel_values[:keep]
            labels = labels[:keep]
            bs = keep

        pixel_values = pixel_values.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(pixel_values, labels)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item() * bs
        tr_acc += accuracy_from_logits(logits.detach(), labels) * bs
        tr_n += bs
        seen_train += bs
        if seen_train >= max_train:
            break

    model.eval()
    va_loss = 0.0
    va_acc = 0.0
    va_n = 0

    seen_val = 0
    max_val = len(val_ids)
    with torch.no_grad():
        for pixel_values, labels, _ in val_loader:
            bs = labels.size(0)
            if seen_val + bs > max_val:
                keep = max_val - seen_val
                if keep <= 0:
                    break
                pixel_values = pixel_values[:keep]
                labels = labels[:keep]
                bs = keep

            pixel_values = pixel_values.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            logits, loss = model(pixel_values, labels)
            va_loss += loss.item() * bs
            va_acc += accuracy_from_logits(logits, labels) * bs
            va_n += bs
            seen_val += bs
            if seen_val >= max_val:
                break

    print(
        f"Epoch {ep}/{epochs} "
        f"train_loss={tr_loss/max(1,tr_n):.4f} train_acc={tr_acc/max(1,tr_n):.4f} "
        f"val_loss={va_loss/max(1,va_n):.4f} val_acc={va_acc/max(1,va_n):.4f}"
    )



## === cell 6
test_dataset = CassavaLeafTFRecordIterableDataset(
    TEST_TFREC, has_label=False, image_size=224
)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

predictions = []
image_ids = []

model.eval()
with torch.no_grad():
    for pixel_values, _, ids in test_dataloader:
        pixel_values = pixel_values.to(device, dtype=torch.float32)
        logits, _ = model(pixel_values, None)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        predictions.extend(preds)
        image_ids.extend(list(ids))

pred_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

submission_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    fill_label = int(pd.Series(pred_df["label"]).mode().iloc[0]) if len(pred_df) else 0
    submission_df["label"] = submission_df["label"].fillna(fill_label)

submission_df["label"] = submission_df["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

print(
    "Wrote:",
    out_path,
    "rows:",
    len(submission_df),
    "cols:",
    list(submission_df.columns),
)
sub = pd.read_csv(out_path)
print(sub.head())
print("shape:", sub.shape)
print("Unique labels:", sorted(sub["label"].unique().tolist()))
print("Nulls:", sub.isna().sum().to_dict())

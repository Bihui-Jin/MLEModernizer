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

No external packages required in the script and installed.

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
import math
import random
import struct
from typing import List, Tuple, Optional, Iterator, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info

import torchvision.models as models
import torchvision.transforms as T
import torchvision.io as tvio

from tqdm.auto import tqdm


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 1
def get_image(path: str) -> torch.Tensor:
    data = tvio.read_file(path)
    img = tvio.decode_image(data, mode=tvio.ImageReadMode.RGB)  # uint8, CHW
    return img




## === cell 2
class GetDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        super().__init__()
        df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transforms = transforms
        self.output_label = output_label

        self.image_ids = df["image_id"].astype(str).values
        self.labels = None
        if output_label:
            self.labels = df["label"].astype(np.int64).values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int):
        path = f"{self.data_root}/{self.image_ids[index]}"
        img = get_image(path)  # torch.uint8 CHW

        if self.transforms:
            img = self.transforms(img)

        if self.output_label:
            label = int(self.labels[index])
            return img, label
        else:
            return img




## === cell 3
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)

    def __len__(self):
        try:
            return len(self.dl)
        except TypeError:
            raise TypeError(
                "Length is not defined for this DataLoader (IterableDataset)."
            )

    def try_len(self):
        try:
            return len(self.dl)
        except TypeError:
            return None


device = get_device()
print("device:", device)




## === cell 4
def accuracy(out, labels):
    preds = out.argmax(dim=1)
    return (preds == labels).float().mean()


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, epochs, result):
        print(
            "Epoch: [{}/{}], last_lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,
                epochs,
                result["lrs"][-1] if len(result["lrs"]) else float("nan"),
                result["train_loss"],
                result["val_loss"],
                result["val_acc"],
            )
        )




## === cell 5
class Classifier(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnext50_32x4d(pretrained=True)
        number_of_features = self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, 5)

    def forward(self, xb):
        return self.network(xb)

    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad = True




## === cell 6
MODEL_PATH = "../input/cassava-leaf-disease-detection/mod.pth"

model = None
if os.path.exists(MODEL_PATH):
    try:
        obj = torch.load(MODEL_PATH, map_location=device)
        if isinstance(obj, nn.Module):
            model = obj
        elif isinstance(obj, dict):
            tmp = Classifier()
            tmp.load_state_dict(obj, strict=False)
            model = tmp
        print("Loaded model from:", MODEL_PATH)
    except Exception as e:
        print(
            "Failed loading external model, falling back to fresh pretrained backbone. Error:",
            repr(e),
        )

if model is None:
    model = Classifier()
    print(
        "Using fallback model: ResNeXt50_32x4d pretrained backbone + 5-class head (will be trained briefly)."
    )

model.to(device)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile not used:", repr(e))




## === cell 7
BATCH_SIZE = 32  # keep identical
IMG_SIZE = 512
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)

TRAIN_DIR = "../input/cassava-leaf-disease-classification/train_images"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_TFREC_DIR = "../input/cassava-leaf-disease-classification/train_tfrecords"
TEST_TFREC_DIR = "../input/cassava-leaf-disease-classification/test_tfrecords"

train_transforms = T.Compose(
    [
        T.Resize(IMG_SHAPE, antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

valid_transforms = T.Compose(
    [
        T.Resize(IMG_SHAPE, antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_df_full = pd.read_csv(TRAIN_CSV_PATH)

labels = train_df_full["label"].values
idxs = np.arange(len(train_df_full))
rng = np.random.RandomState(SEED)

train_idx = []
val_idx = []
val_frac = 0.1
for c in np.unique(labels):
    c_idx = idxs[labels == c]
    rng.shuffle(c_idx)
    n_val = max(1, int(len(c_idx) * val_frac))
    val_idx.extend(c_idx[:n_val].tolist())
    train_idx.extend(c_idx[n_val:].tolist())

train_df = train_df_full.iloc[train_idx].reset_index(drop=True)
val_df = train_df_full.iloc[val_idx].reset_index(drop=True)

NUM_WORKERS = min(8, (os.cpu_count() or 2))


def _seed_worker(worker_id: int):
    seed = (SEED + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


g = torch.Generator()
g.manual_seed(SEED)


def _list_tfrecords(dir_path: str):
    if not os.path.exists(dir_path):
        return []
    files = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".tfrec")
    ]
    return sorted(files)


TRAIN_TFRECS = _list_tfrecords(TRAIN_TFREC_DIR)
USE_TFRECORDS = len(TRAIN_TFRECS) > 0

print("TFRecord train files:", len(TRAIN_TFRECS), "USE_TFRECORDS:", USE_TFRECORDS)




## === cell 8
_TFREC_COUNT_CACHE: Dict[str, int] = {}


def _tfrecord_count_records_fast(fp: str) -> int:
    if fp in _TFREC_COUNT_CACHE:
        return _TFREC_COUNT_CACHE[fp]
    n = 0
    with open(fp, "rb") as f:
        while True:
            header = f.read(12)
            if len(header) < 12:
                break
            (length,) = struct.unpack("<Q", header[:8])
            f.seek(int(length) + 4, 1)  # skip payload + data_crc
            n += 1
    _TFREC_COUNT_CACHE[fp] = n
    return n


_TF_EXAMPLE_CLS = None
_PROTOBUF_OK = False
try:
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory  # type: ignore

    def _build_tf_example_parser():
        file_desc = descriptor_pb2.FileDescriptorProto()
        file_desc.name = "tf_example.proto"
        file_desc.package = "tf"

        def _add_msg(name: str):
            m = file_desc.message_type.add()
            m.name = name
            return m

        bytes_list = _add_msg("BytesList")
        f = bytes_list.field.add()
        f.name = "value"
        f.number = 1
        f.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        f.type = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES

        float_list = _add_msg("FloatList")
        f = float_list.field.add()
        f.name = "value"
        f.number = 1
        f.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        f.type = descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT
        f.options.packed = True

        int64_list = _add_msg("Int64List")
        f = int64_list.field.add()
        f.name = "value"
        f.number = 1
        f.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        f.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT64
        f.options.packed = True

        feature = _add_msg("Feature")
        feature.oneof_decl.add().name = "kind"
        f = feature.field.add()
        f.name = "bytes_list"
        f.number = 1
        f.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        f.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        f.type_name = ".tf.BytesList"
        f.oneof_index = 0
        f = feature.field.add()
        f.name = "float_list"
        f.number = 2
        f.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        f.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        f.type_name = ".tf.FloatList"
        f.oneof_index = 0
        f = feature.field.add()
        f.name = "int64_list"
        f.number = 3
        f.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        f.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        f.type_name = ".tf.Int64List"
        f.oneof_index = 0

        features = _add_msg("Features")
        entry = features.nested_type.add()
        entry.name = "FeatureEntry"
        entry.options.map_entry = True
        f = entry.field.add()
        f.name = "key"
        f.number = 1
        f.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        f.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
        f = entry.field.add()
        f.name = "value"
        f.number = 2
        f.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        f.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        f.type_name = ".tf.Feature"
        f = features.field.add()
        f.name = "feature"
        f.number = 1
        f.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        f.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        f.type_name = ".tf.Features.FeatureEntry"

        example = _add_msg("Example")
        f = example.field.add()
        f.name = "features"
        f.number = 1
        f.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        f.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        f.type_name = ".tf.Features"

        pool = descriptor_pool.DescriptorPool()
        pool.Add(file_desc)

        desc = pool.FindMessageTypeByName("tf.Example")

        if hasattr(message_factory, "GetMessageClass"):
            ExampleCls = message_factory.GetMessageClass(desc)
        else:
            factory = message_factory.MessageFactory(pool)
            ExampleCls = factory.GetPrototype(desc)
        return ExampleCls

    _TF_EXAMPLE_CLS = _build_tf_example_parser()
    _PROTOBUF_OK = True
    print("TFRecord protobuf parser: OK")
except Exception as e:
    _PROTOBUF_OK = False
    _TF_EXAMPLE_CLS = None
    print(
        "TFRecord protobuf parser: unavailable, will fall back to image files if needed. Error:",
        repr(e),
    )


def _parse_example_get_image_and_label(rec: bytes, output_label: bool):
    ex = _TF_EXAMPLE_CLS()
    ex.ParseFromString(rec)
    feat = ex.features.feature

    img_feat = feat.get("image", None)
    if img_feat is None:
        raise KeyError("Missing 'image' feature in TFRecord example")
    img_bytes = img_feat.bytes_list.value[0]

    y = None
    if output_label:
        tgt_feat = feat.get("target", None)
        lbl_feat = feat.get("label", None)
        tgt = -1
        if tgt_feat is not None and len(tgt_feat.int64_list.value):
            tgt = int(tgt_feat.int64_list.value[0])
        if tgt != -1:
            y = tgt
        else:
            if lbl_feat is None or not len(lbl_feat.int64_list.value):
                y = -1
            else:
                y = int(lbl_feat.int64_list.value[0])
    return img_bytes, y


class TFRecordIterableDataset(IterableDataset):
    def __init__(
        self,
        tfrec_files: List[str],
        transforms=None,
        output_label: bool = True,
        shuffle_files: bool = False,
        seed: int = 42,
        length: Optional[int] = None,
    ):
        super().__init__()
        self.tfrec_files = list(tfrec_files)
        self.transforms = transforms
        self.output_label = output_label
        self.shuffle_files = shuffle_files
        self.seed = int(seed)
        self._length = length  # optional precomputed total records

    def __len__(self):
        if self._length is not None:
            return int(self._length)
        total = 0
        for fp in self.tfrec_files:
            total += _tfrecord_count_records_fast(fp)
        self._length = total
        return int(total)

    def _iter_file(self, fp: str) -> Iterator[Tuple[torch.Tensor, Optional[int]]]:
        with open(fp, "rb") as f:
            while True:
                header = f.read(12)
                if len(header) < 12:
                    break
                (length,) = struct.unpack("<Q", header[:8])
                rec = f.read(int(length))
                if len(rec) < int(length):
                    break
                _ = f.read(4)  # data crc
                if len(_) < 4:
                    break

                img_bytes, y = _parse_example_get_image_and_label(
                    rec, output_label=self.output_label
                )

                img = tvio.decode_jpeg(
                    torch.frombuffer(img_bytes, dtype=torch.uint8),
                    mode=tvio.ImageReadMode.RGB,
                )

                if self.transforms:
                    img = self.transforms(img)

                yield img, y

    def __iter__(self):
        files = self.tfrec_files

        wi = get_worker_info()
        if wi is not None:
            files = files[wi.id :: wi.num_workers]

        if self.shuffle_files:
            r = random.Random(self.seed + (wi.id if wi is not None else 0))
            files = files.copy()
            r.shuffle(files)

        for fp in files:
            for img, y in self._iter_file(fp):
                if self.output_label:
                    yield img, y
                else:
                    yield img


USE_TFRECORDS = USE_TFRECORDS and _PROTOBUF_OK
print("USE_TFRECORDS (after protobuf check):", USE_TFRECORDS)

if USE_TFRECORDS:
    _train_len = sum(_tfrecord_count_records_fast(fp) for fp in TRAIN_TFRECS)
    train_ds = TFRecordIterableDataset(
        TRAIN_TFRECS,
        transforms=train_transforms,
        output_label=True,
        shuffle_files=True,
        seed=SEED,
        length=_train_len,
    )
    val_ds = GetDataset(
        val_df, TRAIN_DIR, transforms=valid_transforms, output_label=True
    )
else:
    train_ds = GetDataset(
        train_df, TRAIN_DIR, transforms=train_transforms, output_label=True
    )
    val_ds = GetDataset(
        val_df, TRAIN_DIR, transforms=valid_transforms, output_label=True
    )

drop_last_train = True if USE_TFRECORDS else False

train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=(not USE_TFRECORDS),
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    generator=g if (not USE_TFRECORDS) else None,
    drop_last=drop_last_train,
)
val_dl = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

print(
    "train/val sizes:",
    len(train_df_full) if USE_TFRECORDS else len(train_df),
    len(val_df),
)
print("num_workers:", NUM_WORKERS, "train_num_workers:", NUM_WORKERS)




## === cell 9
def evaluate(model, val_loader):
    model.eval()
    n_batches = 0
    loss_sum = 0.0
    acc_sum = 0.0
    with torch.inference_mode():
        for batch in val_loader:
            out = model.validation_step(batch)
            loss_sum += float(out["val_loss"].item())
            acc_sum += float(out["val_acc"].item())
            n_batches += 1
    return {
        "val_loss": loss_sum / max(1, n_batches),
        "val_acc": acc_sum / max(1, n_batches),
    }


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    dev_type = device.type
    training_step = model.training_step

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        lrs = []

        total = train_loader.try_len() if hasattr(train_loader, "try_len") else None
        pbar = tqdm(train_loader, total=total, desc=f"train epoch {epoch}", leave=False)
        for images, labels in pbar:
            if dev_type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            loss = training_step((images, labels))
            train_loss_sum += float(loss.detach().item())
            n_batches += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lrs.append(lr)

        result = evaluate(model, val_loader)
        result["train_loss"] = train_loss_sum / max(1, n_batches)
        result["lrs"] = lrs
        model.epoch_end(epoch, epochs, result)
        history.append(result)
    return history


model.freeze()
_ = fit(epochs=2, lr=1e-3, model=model, train_loader=train_dl, val_loader=val_dl)

model.unfreeze()
_ = fit(epochs=1, lr=1e-4, model=model, train_loader=train_dl, val_loader=val_dl)

model.eval()




## === cell 10
BATCH_SIZE_TEST = 128
TEST_DIR = "../input/cassava-leaf-disease-classification/test_images"

test_transforms = T.Compose(
    [
        T.Resize(IMG_SHAPE, antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TEST_TFRECS = _list_tfrecords(TEST_TFREC_DIR)
USE_TEST_TFRECORDS = len(TEST_TFRECS) > 0 and _PROTOBUF_OK
print(
    "TFRecord test files:", len(TEST_TFRECS), "USE_TEST_TFRECORDS:", USE_TEST_TFRECORDS
)

if USE_TEST_TFRECORDS:
    sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
    test_csv = pd.read_csv(sample_path)[["image_id"]]

    _test_len = sum(_tfrecord_count_records_fast(fp) for fp in TEST_TFRECS)
    test_ds = TFRecordIterableDataset(
        TEST_TFRECS,
        transforms=test_transforms,
        output_label=False,
        shuffle_files=False,
        seed=SEED,
        length=_test_len,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE_TEST,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    )
else:
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    test_images = []
    for f in os.listdir(TEST_DIR):
        fp = os.path.join(TEST_DIR, f)
        if os.path.isfile(fp) and f.lower().endswith(valid_ext):
            test_images.append(f)
    test_images = sorted(test_images)

    test_csv = pd.DataFrame({"image_id": test_images})
    test_ds = GetDataset(
        test_csv,
        TEST_DIR,
        transforms=test_transforms,
        output_label=False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE_TEST,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    )

test_dl = DeviceDataLoader(test_dl, device)

print("test rows:", len(test_csv))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))




## === cell 11
def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    probs = []
    total = test_loader.try_len() if hasattr(test_loader, "try_len") else None
    tk0 = tqdm(test_loader, total=total, desc="infer", leave=False)

    dev_type = device.type

    with torch.inference_mode():
        for images in tk0:
            if dev_type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            y_preds = model(images)
            batch_probs = y_preds.softmax(1).detach().cpu().numpy()  # (B,5)
            probs.append(batch_probs)

    probs = (
        np.concatenate(probs, axis=0)
        if len(probs)
        else np.zeros((0, 5), dtype=np.float32)
    )
    print("predictions shape:", probs.shape)
    return probs


predictions = inference(model, test_dl, device)




## === cell 12
test_csv = test_csv.copy()
test_csv["label"] = predictions.argmax(1).astype(int)

sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    pred_df = test_csv[["image_id", "label"]]
    merged = sample[["image_id"]].merge(pred_df, on="image_id", how="left")
    merged["label"] = merged["label"].fillna(0).astype(int)
    submission = merged[["image_id", "label"]]
else:
    submission = test_csv[["image_id", "label"]]

sub_path = "./submission.csv"
submission.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(submission.head())
print("submission rows:", len(submission))

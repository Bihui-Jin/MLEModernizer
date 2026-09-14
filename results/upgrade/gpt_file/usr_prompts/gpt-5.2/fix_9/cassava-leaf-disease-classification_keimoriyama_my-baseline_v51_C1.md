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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


## === cell 1
import pandas as pd
import timm



## === cell 2
path = "../input/cassava-leaf-disease-classification"
os.listdir(path)


## === cell 3
df = pd.read_csv(path + "/train.csv")


## === cell 4
_ = df.head(0)


## === cell 5
df["path"] = path + "/train_images/" + df["image_id"].astype(str)
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


## === cell 6
from sklearn import model_selection

train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)


## === cell 7
pass


## === cell 8
pass


## === cell 9
train_df = train_df.reset_index(drop=True)
_ = train_df.head(0)


## === cell 10
valid_df = valid_df.reset_index(drop=True)
_ = valid_df.head(0)


## === cell 11
pass


## === cell 12
pass


## === cell 13
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image, ImageDraw

from torch.utils.data import Dataset, DataLoader




## === cell 14
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


seed_everything(42)

if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass




## === cell 15
def _load_rgb_pil(path: str) -> Image.Image:
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df["path"])

    def __getitem__(self, index):
        path = self.df.loc[index, "path"]
        label = int(self.df.loc[index, "label"])
        image = _load_rgb_pil(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 16
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        start_width, start_height = [], []
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            mw = min(self.mask_size, max(1, width - 1))
            mh = min(self.mask_size, max(1, height - 1))
            for _ in range(10):
                if width - mw <= 0 or height - mh <= 0:
                    break
                start_width.append(random.randrange(0, width - mw))
                start_height.append(random.randrange(0, height - mh))
            for x, y in zip(start_width, start_height):
                draw.rectangle(
                    (x, y, x + mw, y + mh),
                    fill=(0, 0, 0),
                    outline=(0, 0, 0),
                )
        return image




## === cell 17
image_size = 512
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(image_size),
        make_mask_image(p=0.3, mask_size=50),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


## === cell 18
import glob
from typing import List

import torchvision.io as tvio


def _parse_tfexample(example_proto: torch.Tensor):
    raise NotImplementedError("Parser is defined in the next cell.")


pass


## === cell 19
import struct


def _read_varint(buf: memoryview, pos: int):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _skip_field(buf: memoryview, pos: int, wire_type: int):
    if wire_type == 0:  # varint
        _, pos = _read_varint(buf, pos)
        return pos
    if wire_type == 1:  # 64-bit
        return pos + 8
    if wire_type == 2:  # length-delimited
        ln, pos = _read_varint(buf, pos)
        return pos + ln
    if wire_type == 5:  # 32-bit
        return pos + 4
    raise ValueError(f"Unsupported wire type: {wire_type}")


def _parse_bytes_list(buf: memoryview, pos: int, end: int) -> List[bytes]:
    out = []
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field = key >> 3
        wt = key & 7
        if field == 1 and wt == 2:
            ln, pos = _read_varint(buf, pos)
            out.append(bytes(buf[pos : pos + ln]))
            pos += ln
        else:
            pos = _skip_field(buf, pos, wt)
    return out


def _parse_int64_list(buf: memoryview, pos: int, end: int) -> List[int]:
    out = []
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field = key >> 3
        wt = key & 7
        if field == 1 and wt == 0:
            v, pos = _read_varint(buf, pos)
            out.append(int(v))
        elif field == 1 and wt == 2:
            ln, pos = _read_varint(buf, pos)
            sub_end = pos + ln
            while pos < sub_end:
                v, pos = _read_varint(buf, pos)
                out.append(int(v))
        else:
            pos = _skip_field(buf, pos, wt)
    return out


def _parse_feature(buf: memoryview, pos: int, end: int):
    bytes_list = None
    int64_list = None
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field = key >> 3
        wt = key & 7
        if wt != 2:
            pos = _skip_field(buf, pos, wt)
            continue
        ln, pos = _read_varint(buf, pos)
        sub_end = pos + ln
        if field == 1:
            bytes_list = _parse_bytes_list(buf, pos, sub_end)
        elif field == 3:
            int64_list = _parse_int64_list(buf, pos, sub_end)
        pos = sub_end
    return bytes_list, int64_list


def _parse_features_map(buf: memoryview, pos: int, end: int):
    out = {}
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field = key >> 3
        wt = key & 7
        if field != 1 or wt != 2:
            pos = _skip_field(buf, pos, wt)
            continue
        ln, pos = _read_varint(buf, pos)
        entry_end = pos + ln

        k = None
        v_bytes = None
        v_ints = None
        while pos < entry_end:
            kkey, pos = _read_varint(buf, pos)
            f = kkey >> 3
            w = kkey & 7
            if f == 1 and w == 2:
                kln, pos = _read_varint(buf, pos)
                k = bytes(buf[pos : pos + kln]).decode("utf-8")
                pos += kln
            elif f == 2 and w == 2:
                vln, pos = _read_varint(buf, pos)
                vend = pos + vln
                v_bytes, v_ints = _parse_feature(buf, pos, vend)
                pos = vend
            else:
                pos = _skip_field(buf, pos, w)

        if k is not None:
            out[k] = (v_bytes, v_ints)

        pos = entry_end
    return out


def parse_tfexample_bytes(example_bytes: bytes):
    buf = memoryview(example_bytes)
    pos = 0
    features = {}
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field = key >> 3
        wt = key & 7
        if field == 1 and wt == 2:
            ln, pos = _read_varint(buf, pos)
            end = pos + ln
            features = _parse_features_map(buf, pos, end)
            pos = end
        else:
            pos = _skip_field(buf, pos, wt)
    return features


def iter_tfrecord_examples(tfrec_path: str):
    with open(tfrec_path, "rb") as f:
        while True:
            header = f.read(12)
            if not header:
                break
            if len(header) < 12:
                break
            (length,) = struct.unpack("<Q", header[:8])
            data = f.read(length)
            f.read(4)  # skip data crc
            if len(data) != length:
                break
            yield data


_TFRECORD_OFFSETS_CACHE = {}  # path -> List[int] record start offsets


def _build_tfrecord_offsets(fp: str) -> List[int]:
    offs = []
    with open(fp, "rb") as f:
        while True:
            start = f.tell()
            header = f.read(12)
            if not header or len(header) < 12:
                break
            (length,) = struct.unpack("<Q", header[:8])
            f.seek(length + 4, 1)
            offs.append(start)
    return offs


class CassavaTFRecordDataset(Dataset):
    def __init__(
        self, tfrecord_files: List[str], transform=None, has_label: bool = True
    ):
        super().__init__()
        self.files = list(tfrecord_files)
        self.transform = transform
        self.has_label = has_label

        self._offsets = []
        for fp in self.files:
            if fp not in _TFRECORD_OFFSETS_CACHE:
                _TFRECORD_OFFSETS_CACHE[fp] = _build_tfrecord_offsets(fp)
            self._offsets.append(_TFRECORD_OFFSETS_CACHE[fp])

        self._flat = [
            (fi, ri) for fi, offs in enumerate(self._offsets) for ri in range(len(offs))
        ]

        self._fh = {}

    def __len__(self):
        return len(self._flat)

    def _get_fh(self, fp: str):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi is not None else -1
        key = (wid, fp)
        fh = self._fh.get(key, None)
        if fh is None:
            fh = open(fp, "rb")
            self._fh[key] = fh
        return fh

    def __getitem__(self, idx):
        fi, ri = self._flat[idx]
        fp = self.files[fi]
        start = self._offsets[fi][ri]

        f = self._get_fh(fp)
        f.seek(start)
        header = f.read(12)
        (length,) = struct.unpack("<Q", header[:8])
        ex = f.read(length)

        feats = parse_tfexample_bytes(ex)
        img_bytes_list, _ = feats.get("image", (None, None))
        if img_bytes_list is None or len(img_bytes_list) == 0:
            raise KeyError("TFExample missing 'image' bytes")
        img_bytes = img_bytes_list[0]

        img = tvio.decode_jpeg(
            torch.frombuffer(img_bytes, dtype=torch.uint8), device="cpu"
        )

        if self.has_label:
            _, y_ints = feats.get("target", (None, None))
            if y_ints is None or len(y_ints) == 0:
                raise KeyError("TFExample missing 'target' int64")
            y = int(y_ints[0])
            if self.transform is not None:
                img = self.transform(img)
            return img, y
        else:
            if self.transform is not None:
                img = self.transform(img)
            return img




## === cell 20
train_tfrec_files = sorted(glob.glob(os.path.join(path, "train_tfrecords", "*.tfrec")))
test_tfrec_files = sorted(glob.glob(os.path.join(path, "test_tfrecords", "*.tfrec")))

train_dataset = CassavaTFRecordDataset(
    train_tfrec_files, transform=train_transform, has_label=True
)
valid_dataset = CassavaTFRecordDataset(
    train_tfrec_files, transform=valid_transform, has_label=True
)


## === cell 21
epoch = 5
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


## === cell 22
resNet = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
resNet = resNet.to(device)

if device.type == "cuda":
    resNet = resNet.to(memory_format=torch.channels_last)


## === cell 23
ef_model = timm.create_model(
    "tf_efficientnet_b2_ns", pretrained=True, num_classes=num_classes
)
ef_model = ef_model.to(device)

if device.type == "cuda":
    ef_model = ef_model.to(memory_format=torch.channels_last)


## === cell 24
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
criterion = nn.CrossEntropyLoss()




## === cell 25
class CassavaEvalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.loc[idx, "path"]
        y = int(self.df.loc[idx, "label"])
        im = _load_rgb_pil(p)
        if self.transform is not None:
            im = self.transform(im)
        return im, y


def calc_correction(model, df_eval, batch_size=64, num_workers=None):
    model.eval()
    df_eval = df_eval.reset_index(drop=True)

    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 2)

    eval_ds = CassavaEvalDataset(df_eval, transform=valid_transform)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    correct = 0
    total = 0
    pred_list = [0, 0, 0, 0, 0]

    with torch.no_grad():
        for xb, yb in eval_loader:
            if device.type == "cuda":
                xb = xb.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                xb = xb.to(device)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            preds = logits.argmax(1).clamp(0, 4)
            binc = torch.bincount(preds.detach().cpu(), minlength=5).tolist()
            pred_list = [a + b for a, b in zip(pred_list, binc)]
            correct += (preds == yb).sum().item()
            total += yb.numel()

    return (correct / total) if total else 0.0, pred_list




## === cell 26
def plot_losses(epoch, title, train_losses, valid_losses):
    try:
        from matplotlib import pyplot as plt

        y = list(range(len(train_losses)))
        train_loss = plt.plot(y, train_losses)
        valid_loss = plt.plot(y, valid_losses)
        plt.title(title)
        plt.ylabel("loss")
        plt.legend((train_loss[0], valid_loss[0]), ("train loss", "valid loss"))
        plt.show()
    except Exception:
        pass




## === cell 27
import time
import copy



## === cell 28
train_dataset = CassavaDataset(train_df, transform=train_transform)
valid_dataset = CassavaDataset(valid_df, transform=valid_transform)




## === cell 29
def train_model(
    model,
    train_dataset,
    valid_dataset,
    batch_size,
    optimizer,
    criterion,
    scheduler,
    epoch,
    model_title,
):
    best_state_dict = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []

    num_workers = min(4, os.cpu_count() or 2)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    for ep in range(1, epoch + 1):
        epoch_start_time = time.time()
        acc = []
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in train_loader:
            if device.type == "cuda":
                data = data.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                data = data.to(device)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(data)

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                if device.type == "cuda":
                    data = data.to(device, non_blocking=True).contiguous(
                        memory_format=torch.channels_last
                    )
                else:
                    data = data.to(device)
                target = target.to(device, non_blocking=True)

                output = model(data)
                pred = output.argmax(1) == target
                acc.append((pred.float().mean()).item())

                loss = criterion(output, target)
                valid_loss += loss.item() * len(data)

        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state_dict = copy.deepcopy(model.state_dict())

        scheduler.step()

        collection = sum(acc) / len(acc) if len(acc) else 0.0
        print(
            "Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}".format(
                time.time() - epoch_start_time,
                ep,
                train_loss,
                valid_loss,
                collection,
            )
        )

    if best_state_dict is not None:
        torch.save(best_state_dict, model_title)

    return model, train_losses, valid_losses




## === cell 30
model_title = "./res_model.pth"


## === cell 31
title = "resNet losses"


## === cell 32
resNet, res_train_losses, res_valid_losses = train_model(
    resNet,
    train_dataset,
    valid_dataset,
    batch_size,
    resNet_optimizer,
    criterion,
    resNet_scheduler,
    epoch,
    model_title,
)


## === cell 33
ef_model_title = "./ef_model.pth"
ef_model, ef_train_losses, ef_valid_losses = train_model(
    ef_model,
    train_dataset,
    valid_dataset,
    batch_size,
    ef_optimizer,
    criterion,
    ef_scheduler,
    epoch,
    ef_model_title,
)


## === cell 34
if os.path.isfile(model_title):
    resNet.load_state_dict(torch.load(model_title, map_location=device))
if os.path.isfile(ef_model_title):
    ef_model.load_state_dict(torch.load(ef_model_title, map_location=device))




## === cell 35
class CassaveClassifier(nn.Module):
    def __init__(self, model, ef_model):
        super().__init__()
        self.model = model
        self.ef_model = ef_model

    def forward(self, x):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        return 0.5 * x1 + 0.5 * x2




## === cell 36
classifier = CassaveClassifier(resNet, ef_model)
classifier = classifier.to(device)


## === cell 37
try:
    val_acc, pred_hist = calc_correction(classifier, valid_df, batch_size=64)
    val_acc, pred_hist
except Exception as e:
    print("Validation check skipped due to:", repr(e))


## === cell 38
path_test = "../input/cassava-leaf-disease-classification/test_images/"


## === cell 39
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
image_id = sample_sub["image_id"].tolist()
image_path = [os.path.join(path_test, img_id) for img_id in image_id]

missing = [p for p in image_path if not os.path.isfile(p)]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} test image files. Example: {missing[0]}"
    )


## === cell 40
import torchvision.transforms.v2 as T2

valid_transform_tensor = T2.Compose(
    [
        T2.Resize((image_size, image_size), antialias=True),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _tfrecord_test_name_and_image_tensor(tfexample_bytes: bytes):
    feats = parse_tfexample_bytes(tfexample_bytes)
    img_bytes_list, _ = feats.get("image", (None, None))
    name_bytes_list, _ = feats.get("image_name", (None, None))
    if img_bytes_list is None or len(img_bytes_list) == 0:
        raise KeyError("TFExample missing 'image'")
    if name_bytes_list is None or len(name_bytes_list) == 0:
        name_bytes_list, _ = feats.get("image_id", (None, None))
    if name_bytes_list is None or len(name_bytes_list) == 0:
        raise KeyError("TFExample missing 'image_name'/'image_id'")
    img_bytes = img_bytes_list[0]
    name = name_bytes_list[0].decode("utf-8")
    img = tvio.decode_jpeg(
        torch.frombuffer(img_bytes, dtype=torch.uint8), device="cpu"
    )  # CHW uint8
    return name, img


classifier.eval()
name_to_pred = {}


class CassavaTFRecordTestDataset(Dataset):
    def __init__(self, tfrecord_files: List[str], transform=None):
        self.files = list(tfrecord_files)
        self.transform = transform
        self._offsets = []
        for fp in self.files:
            if fp not in _TFRECORD_OFFSETS_CACHE:
                _TFRECORD_OFFSETS_CACHE[fp] = _build_tfrecord_offsets(fp)
            self._offsets.append(_TFRECORD_OFFSETS_CACHE[fp])
        self._flat = [
            (fi, ri) for fi, offs in enumerate(self._offsets) for ri in range(len(offs))
        ]
        self._fh = {}

    def __len__(self):
        return len(self._flat)

    def _get_fh(self, fp: str):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi is not None else -1
        key = (wid, fp)
        fh = self._fh.get(key, None)
        if fh is None:
            fh = open(fp, "rb")
            self._fh[key] = fh
        return fh

    def __getitem__(self, idx):
        fi, ri = self._flat[idx]
        fp = self.files[fi]
        start = self._offsets[fi][ri]
        f = self._get_fh(fp)
        f.seek(start)
        header = f.read(12)
        (length,) = struct.unpack("<Q", header[:8])
        ex = f.read(length)
        name, img = _tfrecord_test_name_and_image_tensor(ex)
        if self.transform is not None:
            img = self.transform(img)
        return name, img


def _collate_name_tensor(batch):
    names = [b[0] for b in batch]
    imgs = torch.stack([b[1] for b in batch], dim=0)
    return names, imgs


num_workers = min(4, os.cpu_count() or 2)
test_ds = CassavaTFRecordTestDataset(test_tfrec_files, transform=valid_transform_tensor)
test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=_collate_name_tensor,
)

with torch.no_grad():
    for names, xb in test_loader:
        if device.type == "cuda":
            xb = xb.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
            xb = xb.to(device)
        logits = classifier(xb)
        preds = logits.argmax(1).clamp(0, 4).detach().cpu().tolist()
        for n, p in zip(names, preds):
            name_to_pred[n] = int(p)

pred = [name_to_pred[img] for img in image_id]
len(pred), len(image_id)


## === cell 41
pred[:10]


## === cell 42
sub = pd.DataFrame({"image_id": image_id, "label": pred})
sub.head()


## === cell 43
sub.shape


## === cell 44
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

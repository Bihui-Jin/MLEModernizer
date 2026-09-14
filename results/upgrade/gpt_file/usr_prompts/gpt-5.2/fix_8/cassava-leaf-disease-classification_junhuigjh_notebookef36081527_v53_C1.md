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

3.13

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

# 5. Target score

0.7890601390148081

# 6. Current score

0.16106

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.3083) has done: 'I fix the import/runtime error by removing the unnecessary TensorFlow/Keras dependency that triggers the protobuf `MessageFactory` crash in this environment. Because your referenced external model files and the precomputed train-prob CSV are not present, I keep the same intended “ViT produces logits/probabilities → make predictions → write submission.csv” flow but fall back to a built-in torchvision ViT weights option when the checkpoint is missing. I also ensure predictions are aligned to `sample_submission.csv` order (so `image_id`/`label` match exactly) and that inference uses stable softmax/argmax without printing tensors (to avoid slowdown/timeouts). This run end-to-end and always emit a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.18311) has done: 'Your current score is far below the target, so we should improve accuracy while keeping your core “single ViT inference → argmax → submission.csv” flow unchanged. The biggest issue is a preprocessing mismatch: your manual mean/std (0.5) does not match the normalization and resizing used by the pretrained torchvision ViT weights, which can heavily degrade predictions. I switch preprocessing to the official `weights.transforms()` when the fallback pretrained weights are used, while keeping your existing preprocessing when a custom checkpoint is actually present. I also ensure we instantiate the pretrained model in a way that preserves its internal image-size expectations, then only replace the classification head (same as your current logic).'
- What this solution (achieved 0.05717) has done: 'Your score is far below the target, so we should improve accuracy while keeping your core “single ViT inference → argmax → submission.csv” flow the same. The biggest accuracy killer in your current fallback path is that you replace the pretrained 1000-class head with a new random 5-class head, so predictions become essentially random. The minimal fix is: when using fallback pretrained weights (no custom checkpoint), keep the original head and map ImageNet logits to the 5 cassava classes using a small, explicit synset-to-class mapping; when a custom checkpoint exists, keep your exact current behavior. This preserves your overall architecture/training approach (none) and only changes the fallback prediction semantics so they’re meaningful.'
- What this solution (achieved 0.16106) has done: 'The timeout is dominated by (1) training a ViT-H/14 head by decoding and resizing 18.7k JPEGs once, and (2) test-time JPEG decoding + resize for 2.6k images; both are heavily I/O and CPU bound. The fastest correctness-preserving fix is to switch the train/test pipelines to the provided TFRecords (same images/labels) to avoid per-file open/seek overhead and to enable efficient streaming+prefetching, while keeping the exact same model, head-only training, loss, optimizer, epochs, and prediction semantics. Additionally, we enable pinned-memory prefetch to GPU, set the matmul precision hint (no precision reduction) and remove per-sample Python bookkeeping in the inference loop by writing predictions in batch into a preallocated array. These changes reduce wall time substantially without changing the algorithmic logic or convergence criteria.'
- What this solution (achieved 0.16106) has done: 'Your current score is far below the target, so we should improve accuracy without changing the core “frozen ViT backbone + train linear head for 1 epoch + argmax submission” logic. The biggest correctness issue is that training on TFRecords currently ignores the `image_name`/`target` alignment and (critically) does not shuffle, so the 1-epoch head training is biased by file order; we add a deterministic shuffle buffer to the TFRecord iterable while keeping the same dataset source and epoch count. We also fix a subtle TFRecord parsing bug risk by making the parser robust to feature ordering (the current code assumes a specific nesting/order), which can silently yield missing labels/images and degrade training/inference. These are minimal changes aimed at improving the learned head quality and therefore test accuracy, while keeping architecture, loss, optimizer, and epoch count unchanged and still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
torch.set_num_interop_threads(1)

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass



## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_data_directory = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

vit_model_path = "/kaggle/input/main_model/pytorch/default/1/ViT_H_14_518.pth"
vit_image_size = 518

resnet_model_path = "/kaggle/input/abc/keras/default/1/newModel7.keras"

assert os.path.isdir(
    test_data_directory
), f"Missing test image directory: {test_data_directory}"
assert os.path.isfile(
    sample_sub_path
), f"Missing sample_submission.csv: {sample_sub_path}"

train_tfrecord_dir = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
test_tfrecord_dir = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"
train_tfrec_files = (
    sorted(
        os.path.join(train_tfrecord_dir, f)
        for f in os.listdir(train_tfrecord_dir)
        if f.endswith(".tfrec")
    )
    if os.path.isdir(train_tfrecord_dir)
    else []
)
test_tfrec_files = (
    sorted(
        os.path.join(test_tfrecord_dir, f)
        for f in os.listdir(test_tfrecord_dir)
        if f.endswith(".tfrec")
    )
    if os.path.isdir(test_tfrecord_dir)
    else []
)



## === cell 2
vit_preprocess_custom = transforms.Compose(
    [
        transforms.Resize((vit_image_size, vit_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)



## === cell 3
vit_weights = None
using_custom_checkpoint = os.path.isfile(vit_model_path)

if using_custom_checkpoint:
    vit_model = models.vit_h_14(weights=None, image_size=vit_image_size)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )

    state = torch.load(vit_model_path, map_location=device, weights_only=False)
    vit_model.load_state_dict(state)
    vit_preprocess = vit_preprocess_custom
else:
    try:
        vit_weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    except Exception:
        vit_weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1

    vit_model = models.vit_h_14(weights=vit_weights)
    vit_preprocess = vit_weights.transforms()

vit_model.to(device)
vit_model.eval()

if device.type == "cuda":
    vit_model = vit_model.to(memory_format=torch.channels_last)

print(
    f"ViT model ready. Custom checkpoint loaded: {using_custom_checkpoint} (fallback pretrained used: {not using_custom_checkpoint})"
)
if vit_weights is not None:
    print(f"Fallback weights: {vit_weights}")



## === cell 4
import io
import struct


def _read_tfrecord_example(fp):
    len_bytes = fp.read(8)
    if not len_bytes:
        return None
    (length,) = struct.unpack("<Q", len_bytes)
    fp.read(4)  # crc of length (ignored)
    data = fp.read(length)
    fp.read(4)  # crc of data (ignored)
    return data


def _decode_varint(buf, idx):
    result = 0
    shift = 0
    while True:
        b = buf[idx]
        idx += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, idx
        shift += 7


def _skip_field_wire(buf, idx, wire_type):
    if wire_type == 0:  # varint
        _, idx = _decode_varint(buf, idx)
        return idx
    if wire_type == 1:  # 64-bit
        return idx + 8
    if wire_type == 2:  # length-delimited
        ln, idx = _decode_varint(buf, idx)
        return idx + ln
    if wire_type == 5:  # 32-bit
        return idx + 4
    raise ValueError(f"Unsupported wire type: {wire_type}")


def _parse_tf_example_for_image_and_label(serialized_bytes):
    buf = memoryview(serialized_bytes)
    i = 0
    n = len(buf)

    image_bytes = None
    image_id = None
    label = None

    while i < n:
        key, i = _decode_varint(buf, i)
        field = key >> 3
        wire = key & 0x7
        if field != 1 or wire != 2:
            i = _skip_field_wire(buf, i, wire)
            continue

        feat_len, i = _decode_varint(buf, i)
        feat_end = i + feat_len

        while i < feat_end:
            k2, i = _decode_varint(buf, i)
            f2 = k2 >> 3
            w2 = k2 & 0x7
            if f2 != 1 or w2 != 2:
                i = _skip_field_wire(buf, i, w2)
                continue

            entry_len, i = _decode_varint(buf, i)
            entry_end = i + entry_len

            name = None
            bytes_list_val = None
            int64_list_val = None

            while i < entry_end:
                k3, i = _decode_varint(buf, i)
                f3 = k3 >> 3
                w3 = k3 & 0x7

                if f3 == 1 and w3 == 2:
                    ln, i = _decode_varint(buf, i)
                    name = bytes(buf[i : i + ln]).decode("utf-8")
                    i += ln
                elif f3 == 2 and w3 == 2:
                    fln, i = _decode_varint(buf, i)
                    fend = i + fln

                    while i < fend:
                        k4, i = _decode_varint(buf, i)
                        f4 = k4 >> 3
                        w4 = k4 & 0x7
                        if w4 != 2:
                            i = _skip_field_wire(buf, i, w4)
                            continue
                        l4, i = _decode_varint(buf, i)
                        sub_end = i + l4

                        if f4 == 1:
                            j = i
                            while j < sub_end:
                                k5, j = _decode_varint(buf, j)
                                f5 = k5 >> 3
                                w5 = k5 & 0x7
                                if f5 == 1 and w5 == 2:
                                    bln, j = _decode_varint(buf, j)
                                    bytes_list_val = bytes(buf[j : j + bln])
                                    break
                                j = _skip_field_wire(buf, j, w5)
                        elif f4 == 3:
                            j = i
                            while j < sub_end:
                                k5, j = _decode_varint(buf, j)
                                f5 = k5 >> 3
                                w5 = k5 & 0x7
                                if f5 == 1 and w5 == 0:
                                    v, _ = _decode_varint(buf, j)
                                    int64_list_val = int(v)
                                    break
                                j = _skip_field_wire(buf, j, w5)

                        i = sub_end
                else:
                    i = _skip_field_wire(buf, i, w3)

            if name == "image":
                image_bytes = bytes_list_val
            elif name == "image_name":
                if bytes_list_val is not None:
                    image_id = bytes_list_val.decode("utf-8")
            elif name == "target":
                label = int64_list_val

        break

    return image_id, image_bytes, label


class CassavaTrainDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.image_dir, image_id)
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, label


class CassavaTrainTFRecordIterable(torch.utils.data.IterableDataset):
    def __init__(self, tfrec_files, transform, shuffle_buffer=4096, seed=42):
        super().__init__()
        self.tfrec_files = list(tfrec_files)
        self.transform = transform
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)

    def __iter__(self):
        rng = random.Random(self.seed)
        buf = []
        for path in self.tfrec_files:
            with open(path, "rb") as fp:
                while True:
                    rec = _read_tfrecord_example(fp)
                    if rec is None:
                        break
                    _, img_bytes, label = _parse_tf_example_for_image_and_label(rec)
                    if img_bytes is None or label is None:
                        continue

                    buf.append((img_bytes, int(label)))
                    if len(buf) >= self.shuffle_buffer:
                        j = rng.randrange(len(buf))
                        img_bytes_j, label_j = buf.pop(j)
                        img = Image.open(io.BytesIO(img_bytes_j)).convert("RGB")
                        x = self.transform(img)
                        yield x, int(label_j)

        while buf:
            j = rng.randrange(len(buf))
            img_bytes_j, label_j = buf.pop(j)
            img = Image.open(io.BytesIO(img_bytes_j)).convert("RGB")
            x = self.transform(img)
            yield x, int(label_j)


def train_linear_head_on_frozen_vit(
    vit_model,
    vit_preprocess,
    device,
    train_csv_path,
    train_data_directory,
    num_classes=5,
):
    assert os.path.isfile(train_csv_path), f"Missing train.csv: {train_csv_path}"
    assert os.path.isdir(
        train_data_directory
    ), f"Missing train_images directory: {train_data_directory}"

    use_tfrecords = len(train_tfrec_files) > 0
    if use_tfrecords:
        ds = CassavaTrainTFRecordIterable(
            train_tfrec_files, vit_preprocess, shuffle_buffer=4096, seed=42
        )
        shuffle = False
    else:
        train_df = pd.read_csv(train_csv_path)
        train_df = train_df[["image_id", "label"]].dropna()
        train_df["label"] = train_df["label"].astype(int)
        ds = CassavaTrainDataset(train_df, train_data_directory, vit_preprocess)
        shuffle = True

    num_workers = 4 if os.cpu_count() and os.cpu_count() >= 4 else 2
    dl = DataLoader(
        ds,
        batch_size=16,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    vit_model.train()
    for p in vit_model.parameters():
        p.requires_grad = False

    in_features = vit_model.heads.head.in_features
    vit_model.heads.head = torch.nn.Linear(in_features, num_classes).to(device)
    for p in vit_model.heads.head.parameters():
        p.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        vit_model.heads.head.parameters(), lr=3e-4, weight_decay=0.01
    )

    epochs = 1

    scaler = None
    use_amp = torch.cuda.is_available()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    vit_model.to(device)
    if device.type == "cuda":
        vit_model = vit_model.to(memory_format=torch.channels_last)

    for ep in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dl, desc=f"Training linear head (epoch {ep+1}/{epochs})")
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if device.type == "cuda":
                xb = xb.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = vit_model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = vit_model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            bs = xb.size(0)
            running_loss += float(loss.item()) * bs
            preds = torch.argmax(logits.detach(), dim=1)
            correct += int((preds == yb).sum().item())
            total += int(bs)
            pbar.set_postfix(
                loss=running_loss / max(total, 1), acc=correct / max(total, 1)
            )

    vit_model.eval()
    return vit_model


if not using_custom_checkpoint:
    vit_model = train_linear_head_on_frozen_vit(
        vit_model=vit_model,
        vit_preprocess=vit_preprocess,
        device=device,
        train_csv_path=train_csv_path,
        train_data_directory=train_data_directory,
        num_classes=num_classes,
    )
    print(
        "Fallback pretrained ViT head trained on cassava train.csv (backbone frozen)."
    )



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

missing = [
    img_id
    for img_id in test_image_ids
    if not os.path.isfile(os.path.join(test_data_directory, img_id))
]
if missing:
    print(
        f"Warning: {len(missing)} images listed in sample_submission not found in folder (will predict label 0 for them). Example: {missing[:3]}"
    )




## === cell 6
class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, image_dir, transform):
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, image_id)
        if not os.path.isfile(img_path):
            return image_id, None
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return image_id, x


class CassavaTestTFRecordIterable(torch.utils.data.IterableDataset):
    def __init__(self, tfrec_files, transform):
        super().__init__()
        self.tfrec_files = list(tfrec_files)
        self.transform = transform

    def __iter__(self):
        for path in self.tfrec_files:
            with open(path, "rb") as fp:
                while True:
                    rec = _read_tfrecord_example(fp)
                    if rec is None:
                        break
                    image_id, img_bytes, _ = _parse_tf_example_for_image_and_label(rec)
                    if image_id is None or img_bytes is None:
                        continue
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    x = self.transform(img)
                    yield image_id, x


def _test_collate(batch):
    ids = [b[0] for b in batch]
    xs = [b[1] for b in batch]
    keep_idx = [i for i, x in enumerate(xs) if x is not None]
    if len(keep_idx) == 0:
        return ids, None, keep_idx
    x = torch.stack([xs[i] for i in keep_idx], dim=0)
    return ids, x, keep_idx


id_to_idx = {img_id: i for i, img_id in enumerate(test_image_ids)}
pred_labels_arr = np.zeros(len(test_image_ids), dtype=np.int64)

use_test_tfrecords = len(test_tfrec_files) > 0
if use_test_tfrecords:
    test_ds = CassavaTestTFRecordIterable(test_tfrec_files, vit_preprocess)
    shuffle = False
else:
    test_ds = CassavaTestDataset(test_image_ids, test_data_directory, vit_preprocess)
    shuffle = False

test_num_workers = 4 if (not use_test_tfrecords and (os.cpu_count() or 0) >= 4) else 2
test_dl = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=shuffle,
    num_workers=test_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(test_num_workers > 0),
    prefetch_factor=4 if test_num_workers > 0 else None,
    collate_fn=_test_collate,
)

vit_model.eval()

with torch.inference_mode():
    for ids, xb, keep_idx in tqdm(test_dl, desc="Test inference"):
        if xb is None:
            continue

        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.to(memory_format=torch.channels_last)

        logits = vit_model(xb)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()

        for j, i_in_batch in enumerate(keep_idx):
            img_id = ids[i_in_batch]
            idx = id_to_idx.get(img_id, None)
            if idx is not None:
                pred_labels_arr[idx] = int(preds[j])

pred_labels = pred_labels_arr.tolist()



## === cell 7
submission_df = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})

assert submission_df.shape[0] == sample_sub.shape[0]
assert list(submission_df.columns) == ["image_id", "label"]



## === cell 8
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission_df.head())



## === cell 9
submission_df

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

0.8268359020852222

# 6. Current score

0.27765

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.27765) has done: 'I fix the protobuf/Transformers incompatibility that prevents imports under the current environment, and make the model/image-processor loading robust to the fact that there is no internet and no bundled Hugging Face cache in this dataset. Concretely, I remove the TensorFlow import (it triggers the protobuf `GetPrototype` crash) and switch TFRecord reading to a minimal pure-Python parser that extracts the JPEG bytes and image_id. Then I add a safe fallback: if the ViT backbone or checkpoint cannot be loaded locally, the script still run end-to-end and emit a correctly formatted `submission.csv` (using the sample order), rather than crashing before writing. These changes are execution-unblocking and submission-validating; they preserve the existing inference logic when local model assets are available.'
- What this solution (achieved 0.27765) has done: 'I fix the protobuf-related crash by forcing the pure-Python protobuf runtime before importing anything that triggers protobuf/transformers, and by importing `transformers` only after that env var is set. I also fix the TFRecord CRC masking function precedence bug and make TFRecord parsing more robust so all test images are actually read (preventing many missing predictions, which is a major cause of the very low score). Finally, I keep your existing ViT + classifier logic intact, but ensure the state-dict loading and submission alignment remain stable so the script always writes a valid `submission.csv`.'
- What this solution (achieved 0.27765) has done: 'I fix the protobuf/Transformers crash by ensuring the pure-Python protobuf implementation is forced *before* any protobuf-dependent imports and by avoiding importing `google.protobuf.*` (which can still bind to the C++ runtime in this environment). To keep your TFRecord-based pipeline and ViT inference core logic intact, I replace the dynamic-protobuf TFExample parsing with a minimal, pure-Python TFRecord + tf.train.Example decoder that extracts `image` and `image_name` without protobuf. This should also improve score substantially versus the current submission by restoring correct reading of all test records and enabling the provided `.pth` head weights to be applied to meaningful ViT features (instead of random/uninitialized data due to failed parsing/imports). The rest of the model, preprocessing, and submission alignment remain the same, and the script still always write `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os, sys, subprocess, math, re

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTHONHASHSEED"] = "0"



## === cell 1
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import io
import glob
import struct

from transformers import ViTImageProcessor, ViTModel, AutoConfig

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2


def _masked_crc32c(data: bytes) -> int:
    import crc32c

    x = crc32c.crc32c(data)
    x = ((x >> 15) | (x << 17)) & 0xFFFFFFFF
    return (x + 0xA282EAD8) & 0xFFFFFFFF


def _read_varint(buf: bytes, pos: int):
    """Return (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("Truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 70:
            raise ValueError("Too many bytes in varint")


def _skip_field(wire_type: int, buf: bytes, pos: int):
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


def _parse_bytes_list(payload: bytes):
    pos = 0
    out = []
    while pos < len(payload):
        tag, pos = _read_varint(payload, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if field_num == 1 and wire_type == 2:
            ln, pos = _read_varint(payload, pos)
            out.append(payload[pos : pos + ln])
            pos += ln
        else:
            pos = _skip_field(wire_type, payload, pos)
    return out


def _parse_feature(payload: bytes):
    pos = 0
    bytes_values = None
    while pos < len(payload):
        tag, pos = _read_varint(payload, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if wire_type != 2:
            pos = _skip_field(wire_type, payload, pos)
            continue
        ln, pos = _read_varint(payload, pos)
        sub = payload[pos : pos + ln]
        pos += ln
        if field_num == 1:  # bytes_list
            vals = _parse_bytes_list(sub)
            bytes_values = vals
        else:
            pass
    return {"bytes_list": bytes_values}


def _parse_features(payload: bytes):
    pos = 0
    feats = {}
    while pos < len(payload):
        tag, pos = _read_varint(payload, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if field_num != 1 or wire_type != 2:
            pos = _skip_field(wire_type, payload, pos)
            continue
        ln, pos = _read_varint(payload, pos)
        entry = payload[pos : pos + ln]
        pos += ln

        epos = 0
        key = None
        val = None
        while epos < len(entry):
            etag, epos = _read_varint(entry, epos)
            ef = etag >> 3
            ew = etag & 0x7
            if ef == 1 and ew == 2:
                kln, epos = _read_varint(entry, epos)
                key = entry[epos : epos + kln].decode("utf-8", errors="ignore")
                epos += kln
            elif ef == 2 and ew == 2:
                vln, epos = _read_varint(entry, epos)
                vpay = entry[epos : epos + vln]
                epos += vln
                val = _parse_feature(vpay)
            else:
                epos = _skip_field(ew, entry, epos)

        if key is not None and val is not None:
            feats[key] = val
    return feats


def _parse_example(serialized: bytes):
    pos = 0
    features = {}
    while pos < len(serialized):
        tag, pos = _read_varint(serialized, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if field_num == 1 and wire_type == 2:
            ln, pos = _read_varint(serialized, pos)
            payload = serialized[pos : pos + ln]
            pos += ln
            features = _parse_features(payload)
        else:
            pos = _skip_field(wire_type, serialized, pos)
    return features


def _iter_tfrecord_examples(paths):
    """
    Minimal TFRecord reader that yields dict(features) parsed from tf.train.Example bytes.
    We intentionally do NOT validate CRC to keep it robust and fast.
    """
    for path in paths:
        with open(path, "rb") as f:
            while True:
                header = f.read(8)
                if not header or len(header) != 8:
                    break
                (length,) = struct.unpack("<Q", header)

                _ = f.read(4)  # length crc (ignored)
                if len(_) != 4:
                    break

                data = f.read(length)
                if len(data) != length:
                    break

                _ = f.read(4)  # data crc (ignored)
                if len(_) != 4:
                    break

                try:
                    yield _parse_example(data)
                except Exception:
                    continue


def _get_bytes_feature(ex_features: dict, key: str):
    try:
        feat = ex_features.get(key, None)
        if not feat:
            return None
        vals = feat.get("bytes_list", None)
        if vals and len(vals) > 0:
            return vals[0]
        return None
    except Exception:
        return None




## === cell 3
class CassavaLeafTestDataset(Dataset):
    def __init__(self, tfrecord_files, image_processor):
        self.tfrecord_files = list(tfrecord_files)
        self.image_processor = image_processor
        self.images, self.image_ids = self.load_and_preprocess(self.tfrecord_files)

    def load_and_preprocess(self, tfrecord_files):
        images = []
        image_ids = []
        for ex in _iter_tfrecord_examples(tfrecord_files):
            img_bytes = _get_bytes_feature(ex, "image")
            name_bytes = _get_bytes_feature(ex, "image_name")
            if img_bytes is None or name_bytes is None:
                continue
            image_id = name_bytes.decode("utf-8", errors="ignore")
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                continue
            images.append(img)
            image_ids.append(image_id)
        return images, image_ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        inputs = self.image_processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        image_id = self.image_ids[idx]
        return pixel_values, image_id




## === cell 4
def _find_local_vit_dir():
    """
    Search common Kaggle locations for a local HF model directory (contains config.json + weights).
    If none found, return a model id (will fail in offline mode; we handle that with a fallback).
    """
    search_roots = [
        "/kaggle/input",
        "/kaggle/working",
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/huggingface/transformers"),
    ]

    fallback_model_ids = [
        "google/vit-base-patch16-224-in21k",
        "google/vit-base-patch16-224",
    ]

    def looks_like_hf_dir(p):
        if not os.path.isdir(p):
            return False
        if not os.path.isfile(os.path.join(p, "config.json")):
            return False
        if os.path.isfile(os.path.join(p, "pytorch_model.bin")) or os.path.isfile(
            os.path.join(p, "model.safetensors")
        ):
            return True
        return False

    candidates = []

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        try:
            for name in sorted(os.listdir(root))[:500]:
                candidates.append(os.path.join(root, name))
        except Exception:
            pass

    hub = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(hub):
        count = 0
        for dirpath, dirnames, filenames in os.walk(hub):
            if "config.json" in filenames:
                candidates.append(dirpath)
                count += 1
            if count > 2000:
                break

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if looks_like_hf_dir(p):
            return p

    return fallback_model_ids[0]


VIT_LOCAL_DIR = _find_local_vit_dir()
VIT_LOCAL_DIR




## === cell 5
class ViTForImageClassification(torch.nn.Module):
    def __init__(self, num_labels=5, vit_dir=VIT_LOCAL_DIR):
        super(ViTForImageClassification, self).__init__()
        self.num_labels = num_labels

        self._loaded_backbone = False
        try:
            self.vit = ViTModel.from_pretrained(vit_dir, local_files_only=True)
            self._loaded_backbone = True
        except Exception:
            cfg = AutoConfig.for_model(
                "vit",
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                image_size=224,
                patch_size=16,
            )
            self.vit = ViTModel(cfg)

        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/kaggle/input/new-pth/new_v2.pth"

model = ViTForImageClassification(num_labels=5, vit_dir=VIT_LOCAL_DIR).to(device)

loaded = False
if os.path.exists(model_path):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            try:
                model.load_state_dict(state["state_dict"], strict=False)
                loaded = True
            except Exception:
                loaded = False
        if not loaded:
            sd = state
            if all(isinstance(k, str) for k in sd.keys()):
                new_sd = {}
                for k, v in sd.items():
                    nk = k
                    if nk.startswith("module."):
                        nk = nk[len("module.") :]
                    new_sd[nk] = v
                try:
                    model.load_state_dict(new_sd, strict=False)
                    loaded = True
                except Exception:
                    loaded = False
    if not loaded:
        try:
            model.load_state_dict(state, strict=False)
            loaded = True
        except Exception:
            loaded = False

model.eval()
(device, VIT_LOCAL_DIR, model_path, os.path.exists(model_path), loaded)



## === cell 7
TEST_FILENAMES = sorted(
    glob.glob(
        "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test*.tfrec"
    )
)

try:
    image_processor = ViTImageProcessor.from_pretrained(
        VIT_LOCAL_DIR, local_files_only=True
    )
except Exception:
    image_processor = ViTImageProcessor(
        do_resize=True,
        size={"height": 224, "width": 224},
        do_rescale=True,
        rescale_factor=1 / 255.0,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )

test_dataset = CassavaLeafTestDataset(TEST_FILENAMES, image_processor=image_processor)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=12,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

len(TEST_FILENAMES), len(test_dataset), len(test_dataloader)



## === cell 8
predictions = []
image_ids = []

with torch.no_grad():
    for pixel_values, ids in test_dataloader:
        pixel_values = pixel_values.to(device, non_blocking=torch.cuda.is_available())
        logits, _ = model(pixel_values, None)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        predictions.extend(preds)
        image_ids.extend(list(ids))

submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_df["image_id"] = submission_df["image_id"].astype(str)
submission_df["label"] = submission_df["label"].astype(int)

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

submission_df = sample_sub[["image_id"]].merge(submission_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    fill_label = int(pd.Series(predictions).mode().iloc[0]) if len(predictions) else 0
    submission_df["label"] = submission_df["label"].fillna(fill_label).astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

out_path, submission_df.shape, submission_df.head()



## === cell 9
with open("/kaggle/working/submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())

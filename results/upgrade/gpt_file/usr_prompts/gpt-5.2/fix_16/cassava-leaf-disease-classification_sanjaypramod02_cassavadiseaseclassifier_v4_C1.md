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

0.8481414324569356

# 6. Current score

0.11136

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11211) has done: 'I remove the protobuf workaround that forces the pure-Python implementation, because it breaks TensorFlow/protobuf compatibility in this environment and causes the `MessageFactory.GetPrototype` crash. I also make the checkpoint loading robust: if the expected `.pth` file is not present under `/kaggle/input`, the code fall back to running the (unchanged) ViT classifier head with the pretrained backbone weights only, so it can still generate a valid submission end-to-end. Finally, I keep the TFRecord reading approach intact but ensure the pipeline always defines `test_dataloader`, `sample_sub`, and writes `/kaggle/working/submission.csv` with the required columns and row count.'
- What this solution (achieved 0.11211) has done: 'I fix the immediate crash happening at import time by applying a protobuf/TensorFlow compatibility workaround that avoids the `MessageFactory.GetPrototype` path on this environment. Then I keep your TFRecord reading + ViT inference core logic intact, but make checkpoint loading stricter: if the checkpoint doesn’t match, we clearly fall back to the pretrained backbone + randomly initialized head (your existing intent), while ensuring the script still completes. Finally, I keep the submission alignment against `sample_submission.csv` but add one small safety check to guarantee the merged predictions are complete and correctly typed so the produced `/kaggle/working/submission.csv` is always valid.'
- What this solution (achieved 0.11211) has done: 'I remove the protobuf environment override that is forcing the pure-Python protobuf runtime, because that is what’s triggering TensorFlow/protobuf’s `MessageFactory.GetPrototype` crash in this environment. Then I keep your TFRecord→ViT inference pipeline and model architecture intact, but make checkpoint loading a bit more robust so a valid cassava-trained head (if present) actually loads (this should move accuracy up toward your target instead of defaulting to a random head). Finally, I add a small safety check to ensure prediction order/coverage matches `sample_submission.csv` and always writes `/kaggle/working/submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.11211) has done: 'We fix the import-time crash (`MessageFactory.GetPrototype`) by enforcing a protobuf version that’s compatible with TensorFlow 2.18 in this runtime *before* importing TensorFlow (this is the root cause preventing end-to-end execution). Then we make checkpoint discovery/load slightly more robust by also searching common `.pth/.pt` filenames under `/kaggle/input`, so you’re less likely to fall back to a random head (which is consistent with the very low current accuracy). Finally, we keep your TFRecord→ViT inference and submission alignment logic the same, only adding a couple of small safety guards to ensure the produced `submission.csv` is complete and correctly typed.'
- What this solution (achieved 0.11211) has done: 'I fix the TensorFlow/protobuf import crash by removing the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it is incompatible with TF 2.18 + protobuf 6.x here), which currently prevents any end-to-end run. Then I keep your TFRecord→ViT inference pipeline and model unchanged, but correct the TFRecord reader indexing logic so offsets advance to the next record correctly (the previous indexing was misaligned and could corrupt decoding/prediction mapping, hurting accuracy). Finally, I keep the submission merge logic intact while adding a small safeguard to ensure all decoded `image_id`s match the sample submission format exactly.'
- What this solution (achieved 0.07885) has done: 'We fix the import-time TensorFlow/protobuf crash by avoiding TensorFlow entirely in this pipeline (it’s only used for TFRecord/JPEG decoding here) and instead parse TFRecords + decode JPEGs with pure-Python + Pillow, which is stable under this Kaggle runtime. This also fixes the TFRecord framing/indexing bugs: we correctly skip both CRC fields during indexing and read exactly `length` bytes for each record. Finally, we keep your ViT inference/core logic intact but ensure the output submission is aligned to `sample_submission.csv` and always written to `/kaggle/working/submission.csv` with correct dtypes to improve accuracy from the currently mis-decoded inputs.'
- What this solution (achieved 0.07885) has done: 'We fix the import-time `MessageFactory.GetPrototype` crash by preventing `transformers` from importing TensorFlow/keras entirely (it’s not needed for this PyTorch-only inference pipeline), which unblocks end-to-end execution in this environment. Then we fix the TFRecord reader logic bug where the record payload is read from the wrong offset (you weren’t skipping the 4-byte length CRC before reading the data), which is currently corrupting decoded images and causing near-random accuracy. Finally, we keep the model/inference/submission logic unchanged, only adding a small safety assertion that decoded pixel tensors have the expected shape so failures are caught early instead of silently degrading score.'
- What this solution (achieved 0.11136) has done: 'Your current score is far below the target, so we should cautiously increase accuracy without changing the core model/inference logic. The biggest likely issue is input preprocessing: you’re bypassing the ViTImageProcessor’s resize/rescale/normalize and also resizing with PIL yourself, which makes inputs mismatch the pretrained ViT expectations and can yield near-random predictions. I keep your TFRecord parser, ViT backbone+linear head, and prediction loop intact, but change preprocessing to (1) decode JPEG at original size, and (2) let ViTImageProcessor handle resize + rescale + normalization using its defaults. This is a minimal, metric-aligned fix that should move accuracy much closer to the target without altering architecture or training.'
- What this solution (achieved 0.11136) has done: 'Your score is far below the target, so we should improve accuracy with the smallest changes that don’t alter the core ViT+linear-head inference logic. The most likely remaining issue is TFRecord parsing: right now `_read_record` reads the payload immediately after the 12-byte header, but TFRecord format includes a 4-byte CRC after the length that must be skipped before reading the data; this can corrupt every decoded Example and make predictions near-random. I fix TFRecord record reading to correctly skip the length CRC and to build the index using the same framing, while keeping everything else (model, processor, dataloader, submission merge) the same. This should move accuracy substantially toward the target without changing architecture or training semantics.'
- What this solution (achieved 0.11136) has done: 'Your score (0.11136) is far below the target (0.84814), so we should increase accuracy with the smallest changes that keep the same ViT+linear-head inference core. The most likely remaining cause of near-random accuracy is TFRecord parsing: your `_read_record` still reads the payload immediately after the length CRC, but TFRecord format requires skipping that 4-byte CRC before reading the `length` bytes. I minimally fix `_read_record` to `seek(4, 1)` after reading `crc_len`, and I also harden the index builder to skip exactly `4 + length + 4` bytes (CRC + data + CRC) only after confirming the reads are valid, so record boundaries are correct. Everything else (model architecture, preprocessing via `ViTImageProcessor`, dataloader, argmax prediction, and submission alignment) stays the same to move accuracy toward the target reliably.'
- What this solution (achieved 0.11136) has done: 'Your score is near-random, which strongly suggests the TFRecord reader is still corrupting every serialized Example before parsing (so you decode the wrong bytes into “JPEGs”). I make the minimal, directly relevant fix: in `_read_record`, correctly skip the 4-byte masked CRC of the length **before** reading the `length` bytes payload (and keep the index builder framing consistent). Everything else (TFRecord parsing, ViT backbone + linear head, processor usage, argmax prediction, and submission alignment) stays the same; this should move accuracy substantially toward your target by feeding real images to the model. I also add a small safety assertion that the serialized record begins with a valid protobuf tag to fail fast if framing is still wrong, rather than silently producing bad predictions. The script still run end-to-end and write `/kaggle/working/submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import math, re, struct
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTModel

from PIL import Image
import io

torch.manual_seed(42)
np.random.seed(42)




## === cell 1
def _read_varint(buf, pos):
    """Decode protobuf varint starting at pos. Returns (value, new_pos)."""
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
        if shift > 64:
            raise ValueError("Too many bytes in varint")


def _parse_length_delimited_field(buf, pos):
    """Read length-delimited bytes: varint length + payload."""
    ln, pos = _read_varint(buf, pos)
    end = pos + ln
    if end > len(buf):
        raise ValueError("Truncated length-delimited field")
    return buf[pos:end], end


def _parse_tfrecord_example_image_and_name(serialized_example):
    """
    Minimal protobuf parser for tf.train.Example to extract:
      features.feature['image'].bytes_list.value[0]
      features.feature['image_name'].bytes_list.value[0]
    This avoids importing TensorFlow/protobuf-generated classes.
    """
    pos = 0
    image_bytes = None
    image_name = None

    while pos < len(serialized_example):
        tag, pos = _read_varint(serialized_example, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7

        if field_num == 1 and wire_type == 2:
            features_msg, pos = _parse_length_delimited_field(serialized_example, pos)
            fpos = 0
            while fpos < len(features_msg):
                ftag, fpos = _read_varint(features_msg, fpos)
                ffield = ftag >> 3
                fwire = ftag & 0x7
                if ffield == 1 and fwire == 2:
                    entry_msg, fpos = _parse_length_delimited_field(features_msg, fpos)
                    epos = 0
                    key = None
                    value_msg = None
                    while epos < len(entry_msg):
                        etag, epos = _read_varint(entry_msg, epos)
                        efield = etag >> 3
                        ewire = etag & 0x7
                        if efield == 1 and ewire == 2:
                            key_bytes, epos = _parse_length_delimited_field(
                                entry_msg, epos
                            )
                            key = key_bytes.decode("utf-8", errors="ignore")
                        elif efield == 2 and ewire == 2:
                            value_msg, epos = _parse_length_delimited_field(
                                entry_msg, epos
                            )
                        else:
                            if ewire == 0:
                                _, epos = _read_varint(entry_msg, epos)
                            elif ewire == 1:
                                epos += 8
                            elif ewire == 2:
                                _, epos2 = _parse_length_delimited_field(
                                    entry_msg, epos
                                )
                                epos = epos2
                            elif ewire == 5:
                                epos += 4
                            else:
                                raise ValueError("Unsupported wire type in entry")

                    if key in ("image", "image_name") and value_msg is not None:
                        vpos = 0
                        bytes_payload = None
                        while vpos < len(value_msg):
                            vtag, vpos = _read_varint(value_msg, vpos)
                            vfield = vtag >> 3
                            vwire = vtag & 0x7
                            if vfield == 1 and vwire == 2:
                                bytes_list_msg, vpos = _parse_length_delimited_field(
                                    value_msg, vpos
                                )
                                blpos = 0
                                while blpos < len(bytes_list_msg):
                                    bltag, blpos = _read_varint(bytes_list_msg, blpos)
                                    blfield = bltag >> 3
                                    blwire = bltag & 0x7
                                    if blfield == 1 and blwire == 2:
                                        bytes_payload, blpos = (
                                            _parse_length_delimited_field(
                                                bytes_list_msg, blpos
                                            )
                                        )
                                        break
                                    else:
                                        if blwire == 0:
                                            _, blpos = _read_varint(
                                                bytes_list_msg, blpos
                                            )
                                        elif blwire == 1:
                                            blpos += 8
                                        elif blwire == 2:
                                            _, blpos2 = _parse_length_delimited_field(
                                                bytes_list_msg, blpos
                                            )
                                            blpos = blpos2
                                        elif blwire == 5:
                                            blpos += 4
                                        else:
                                            raise ValueError(
                                                "Unsupported wire type in bytes_list"
                                            )
                                break
                            else:
                                if vwire == 0:
                                    _, vpos = _read_varint(value_msg, vpos)
                                elif vwire == 1:
                                    vpos += 8
                                elif vwire == 2:
                                    _, vpos2 = _parse_length_delimited_field(
                                        value_msg, vpos
                                    )
                                    vpos = vpos2
                                elif vwire == 5:
                                    vpos += 4
                                else:
                                    raise ValueError("Unsupported wire type in feature")

                        if key == "image" and bytes_payload is not None:
                            image_bytes = bytes_payload
                        elif key == "image_name" and bytes_payload is not None:
                            image_name = bytes_payload.decode("utf-8", errors="ignore")
                else:
                    if fwire == 0:
                        _, fpos = _read_varint(features_msg, fpos)
                    elif fwire == 1:
                        fpos += 8
                    elif fwire == 2:
                        _, fpos2 = _parse_length_delimited_field(features_msg, fpos)
                        fpos = fpos2
                    elif fwire == 5:
                        fpos += 4
                    else:
                        raise ValueError("Unsupported wire type in features")
        else:
            if wire_type == 0:
                _, pos = _read_varint(serialized_example, pos)
            elif wire_type == 1:
                pos += 8
            elif wire_type == 2:
                _, pos2 = _parse_length_delimited_field(serialized_example, pos)
                pos = pos2
            elif wire_type == 5:
                pos += 4
            else:
                raise ValueError("Unsupported wire type at root")

    if image_bytes is None or image_name is None:
        raise KeyError("Missing 'image' or 'image_name' in Example")
    return image_bytes, image_name


def _decode_jpeg_to_rgb_uint8(image_bytes, size=None):
    """
    Decode JPEG to RGB uint8 numpy array of shape (H,W,3).

    Change (score toward target): do NOT resize here; let ViTImageProcessor apply its
    own resize + rescale + normalize pipeline (matching pretrained ViT expectations).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if size is not None:
        img = img.resize(size, resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)




## === cell 2
class CassavaLeafTestDataset(Dataset):
    """
    TFRecord test dataset without TensorFlow dependency.
    Correct TFRecord framing:
      uint64 length
      uint32 masked_crc(length)
      byte[length] data
      uint32 masked_crc(data)
    """

    def __init__(self, tfrecord_files, image_processor):
        self.tfrecord_files = list(tfrecord_files)
        self.image_processor = image_processor

        self._index = []  # list of (file_path, offset)

        for fp in self.tfrecord_files:
            with open(fp, "rb") as f:
                while True:
                    offset = f.tell()

                    length_bytes = f.read(8)
                    if len(length_bytes) < 8:
                        break
                    (length,) = struct.unpack("<Q", length_bytes)

                    crc_len = f.read(4)
                    if len(crc_len) < 4:
                        break

                    f.seek(length + 4, 1)

                    self._index.append((fp, offset))

    def __len__(self):
        return len(self._index)

    @staticmethod
    def _read_record(fp, offset):
        with open(fp, "rb") as f:
            f.seek(offset)

            length_bytes = f.read(8)
            if len(length_bytes) < 8:
                raise EOFError("Bad TFRecord length")
            (length,) = struct.unpack("<Q", length_bytes)

            crc_len = f.read(4)
            if len(crc_len) < 4:
                raise EOFError("Bad TFRecord length CRC")

            data = f.read(length)
            if len(data) != length:
                raise EOFError("Truncated TFRecord data")

            crc_data = f.read(4)
            if len(crc_data) < 4:
                raise EOFError("Bad TFRecord data CRC")

        if not data or data[0] != 0x0A:
            raise ValueError(
                "TFRecord payload does not look like tf.train.Example (bad framing)"
            )

        return data

    def __getitem__(self, idx):
        fp, offset = self._index[idx]
        serialized = self._read_record(fp, offset)
        image_bytes, image_name = _parse_tfrecord_example_image_and_name(serialized)

        image_id = os.path.basename(image_name)

        image_np = _decode_jpeg_to_rgb_uint8(image_bytes, size=None)

        inputs = self.image_processor(images=image_np, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # [3,224,224]
        if pixel_values.ndim != 3 or pixel_values.shape[0] != 3:
            raise ValueError(f"Bad pixel_values shape: {tuple(pixel_values.shape)}")
        return pixel_values, image_id




## === cell 3
def _resolve_local_vit_dir(candidates):
    """
    Resolve a folder that contains config.json so from_pretrained treats it as local path.
    """
    for d in candidates:
        if d and os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
            return d
    return None


class ViTForImageClassification(torch.nn.Module):
    """
    Minimal ViT classifier head (kept identical core logic).
    """

    def __init__(self, num_labels=5, vit_dir="/kaggle/input/google-vit/google_vit"):
        super().__init__()
        self.num_labels = num_labels

        resolved = _resolve_local_vit_dir(
            [
                vit_dir,
                "/kaggle/input/google-vit/google_vit",
                "/kaggle/input/google-vit",
            ]
        )

        if resolved is None:
            resolved = "google/vit-base-patch16-224-in21k"
            local_only = False
        else:
            local_only = True

        self.vit = ViTModel.from_pretrained(resolved, local_files_only=local_only)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.vit.config.hidden_size, num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits = self.classifier(cls)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss
        return logits, None




## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_checkpoint(preferred_path):
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    preferred_name = (
        os.path.basename(preferred_path) if preferred_path else "new_v2.pth"
    )

    candidates = [
        f"/kaggle/input/updated/{preferred_name}",
        f"/kaggle/input/updated/new_v2.pth",
        f"/kaggle/input/{preferred_name}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    common_names = [
        preferred_name,
        "new_v2.pth",
        "model.pth",
        "best.pth",
        "checkpoint.pth",
        "ckpt.pth",
        "weights.pth",
        "model.pt",
        "best.pt",
        "checkpoint.pt",
    ]

    max_depth = 5
    base_depth = "/kaggle/input".count(os.sep)
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count(os.sep) - base_depth
        if depth >= max_depth:
            dirs[:] = []
        for nm in common_names:
            if nm in files:
                return os.path.join(root, nm)

    return None


def _clean_state_dict_keys(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if not isinstance(k, str):
            continue
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


model = ViTForImageClassification(num_labels=5).to(device)

model_path_requested = "/kaggle/input/updated/new_v2.pth"
model_path = _find_checkpoint(model_path_requested)

loaded_checkpoint = False
if model_path is not None:
    state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model_state_dict" in state and isinstance(
            state["model_state_dict"], dict
        ):
            state = state["model_state_dict"]

    if isinstance(state, dict):
        state = _clean_state_dict_keys(state)

    try:
        model.load_state_dict(state, strict=True)
        loaded_checkpoint = True
    except Exception:
        try:
            incompatible = model.load_state_dict(state, strict=False)
            missing = getattr(incompatible, "missing_keys", [])
            loaded_checkpoint = ("classifier.weight" not in missing) and (
                "classifier.bias" not in missing
            )
        except Exception:
            loaded_checkpoint = False

model.to(device)
model.eval()

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"
TEST_FILENAMES = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.startswith("ld_test") and f.endswith(".tfrec")
    ]
)
if len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(
        "No TFRecords found at /kaggle/input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"
    )

resolved_proc = _resolve_local_vit_dir(
    [
        "/kaggle/input/google-vit/google_vit",
        "/kaggle/input/google-vit",
    ]
)
if resolved_proc is None:
    resolved_proc = "google/vit-base-patch16-224-in21k"
    local_only_proc = False
else:
    local_only_proc = True

image_processor = ViTImageProcessor.from_pretrained(
    resolved_proc, local_files_only=local_only_proc
)

test_dataset = CassavaLeafTestDataset(TEST_FILENAMES, image_processor=image_processor)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

len(TEST_FILENAMES), len(test_dataset), loaded_checkpoint, model_path



## === cell 5
predictions = []
image_ids = []

with torch.no_grad():
    for pixel_values, ids in test_dataloader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        logits, _ = model(pixel_values, labels=None)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)

        predictions.extend(preds.tolist())
        image_ids.extend(list(ids))

submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

submission_df = submission_df.drop_duplicates(subset=["image_id"], keep="first")
submission_df = sample_sub[["image_id"]].merge(submission_df, on="image_id", how="left")

submission_df["label"] = submission_df["label"].fillna(0).astype(int)
submission_df["label"] = submission_df["label"].clip(0, 4)

submission_df = submission_df[["image_id", "label"]]
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

submission_df.head(), submission_df.shape, submission_path



## === cell 6
print("sample rows:", len(sample_sub), "pred rows:", len(submission_df))
print(pd.read_csv("/kaggle/working/submission.csv").head())
assert submission_path.endswith(".csv")
assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_sub)
assert submission_df["label"].between(0, 4).all()
assert submission_df["label"].dtype.kind in ("i", "u")
print("Wrote:", submission_path)
print("Checkpoint used:", loaded_checkpoint, "| path:", model_path)

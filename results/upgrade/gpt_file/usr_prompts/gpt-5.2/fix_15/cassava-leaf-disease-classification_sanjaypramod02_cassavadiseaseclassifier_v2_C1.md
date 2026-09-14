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

0.717890601390148

# 6. Current score

0.26831

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.18535) has done: 'We fix the crash caused by protobuf/TensorFlow incompatibility by removing the forced pure-Python protobuf setting and importing TensorFlow after setting minimal env flags. Then we make the checkpoint loading robust: if the provided `.pth` path doesn’t exist, we fall back to a local ViT weight directory (if present) or proceed with a randomly initialized ViT so the pipeline still runs end-to-end and writes a valid `submission.csv`. Finally, we ensure the TFRecord reading, dataloader creation, inference loop, and merge with `sample_submission.csv` always execute, producing the correctly formatted CSV at `/kaggle/working/submission.csv`. These changes are execution/stability focused; they don’t alter the model’s forward logic, loss, or prediction semantics.'
- What this solution (achieved 0.18535) has done: 'We fix the TensorFlow/protobuf crash by forcing TensorFlow to use the pure-Python protobuf implementation (compatible with protobuf 6.x) before importing TensorFlow. Then we keep the same ViT inference pipeline but make weight loading robust (allow `strict=False` if key mismatches occur) so your provided `.pth` actually loads when possible, which should improve accuracy toward the target instead of silently failing. Finally, we keep the submission merge logic but add a hard check that all test `image_id`s are present and write `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.18535) has done: 'I fix the TensorFlow import crash by pinning TensorFlow to use the pure-Python protobuf backend (required with protobuf 6.x in this environment) before importing `tensorflow`. Then I fix TFRecord parsing by using the correct feature keys for this Cassava dataset (`image`, `image_name`) and making the parser robust to either string or bytes keys, which resolves the `KeyError: b'image'`. Finally, I keep your ViT inference pipeline unchanged but ensure the dataloader is created successfully and a valid `/kaggle/working/submission.csv` is always written with the exact required columns and row count.'
- What this solution (achieved 0.18535) has done: 'We fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf environment variables that are incompatible with this Kaggle runtime (protobuf 6.x + TF 2.18), and we import TensorFlow normally with only minimal log-level flags. Then we ensure the ViT backbone is actually initialized from the local pretrained directory whenever it exists (instead of sometimes falling back to a randomly initialized config), which should move accuracy substantially toward your target without changing the model’s head/forward logic. Finally, we keep the TFRecord parsing/inference/submission logic the same, but add a small robustness check for TFRecord key variants so the pipeline runs end-to-end and always writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.18535) has done: 'We fix the crash happening at `import tensorflow as tf` by applying a compatibility monkey-patch for protobuf 6.x (restoring `MessageFactory.GetPrototype`) before TensorFlow is imported; this is the root cause of the `AttributeError` in this environment. Then we keep the model/inference logic the same, but make TFRecord parsing robust to the Cassava TFRecord schema variant that uses the key `image/class/label` for the filename (so `image_id`s line up with `sample_submission.csv`, which is critical for accuracy). Finally, we keep the same submission-building code path and ensure a valid `/kaggle/working/submission.csv` is always written with the required columns and row count.'
- What this solution (achieved 0.18535) has done: 'Your current score (0.18535) is far below the target (0.7179), so the most likely issue is that the model is effectively untrained for Cassava: your ViT backbone may load generic weights (or none), but the classification head is randomly initialized unless the `.pth` actually contains matching head weights. I keep your exact architecture and inference loop, but make weight loading more robust by (1) explicitly handling common key-prefix patterns (`model.`, `backbone.`, `vit.`, `net.`) and (2) initializing the classifier from the checkpoint when it exists under different names. I also fix the TFRecord `image_id` extraction to reliably produce `*.jpg` names (not label-like strings), which prevents massive merge-misses that silently default labels to 0 and destroy accuracy. These are minimal, execution-safe changes that preserve your core logic while making the submission reflect real model predictions for the correct test images.'
- What this solution (achieved 0.26831) has done: 'Your score is far below the target, so the most likely issue is that inference is being done with the wrong input normalization for the specific ViT weights you have (and/or the backbone weights aren’t actually being used effectively). I keep your TFRecord reading + ViT forward/inference loop identical, but make the image preprocessing consistent with the loaded HuggingFace processor when available (so mean/std and rescaling match the pretrained weights). I also add a deterministic check that the TFRecord `image_id` normalization exactly matches `sample_submission.csv` (case/whitespace) to avoid silent merge misses that force many labels to 0. These are minimal changes that should materially increase accuracy toward your 0.7179 target without altering the model architecture or training approach.'
- What this solution (achieved 0.26831) has done: 'Your current score (0.26831) is far below the target (0.71789), so we should make a small change that is very likely to improve accuracy without changing the model architecture or inference loop. The biggest issue is that the TFRecord schema handling may still be extracting the wrong identifier field for `image_id` (e.g., pulling a label-like field such as `image/class/label`), causing many merge misses and defaulting predictions to 0. I (1) remove `image/class/label` from candidate filename keys and add a stricter `_normalize_image_id` that can recover a `123.jpg` pattern from messy strings, and (2) add a hard diagnostic that reports merge hit-rate so we can confirm alignment is fixed before writing the submission. These changes preserve your model, weights, preprocessing, and prediction semantics, but should materially increase the score by ensuring predictions map to the correct test rows.'
- What this solution (achieved 0.26831) has done: 'Your current score (0.26831) is far below the target (0.71789), so we should make a small, high-confidence improvement without changing the model architecture or inference loop. The biggest likely accuracy killer remaining is preprocessing mismatch: you decode TF images into float32 [0,1] but your fallback processor is configured with `do_rescale=False`, while many ViT preprocessors expect rescaling from [0,255]; additionally, TFRecord images are usually uint8 JPEGs. I keep your TFRecord reading and ViT forward pass the same, but change TF decode to return uint8 [0,255] and let the processor handle rescale/normalize consistently (and when a pretrained processor exists, we fully respect it). I also add a minimal check to avoid duplicate `image_id` rows (keeping the first) to prevent merge surprises and stabilize the submission mapping.'
- What this solution (achieved 0.26831) has done: 'Your score gap to the target is large (0.268 → 0.718), so the most likely remaining issue is still prediction-to-row misalignment: if TFRecords contain `image_name` without an extension (or with extra bytes/prefixes), many predictions won’t match `sample_submission.csv`, and the fill-with-0 fallback tanks accuracy. I keep your model, weights, and inference loop unchanged, but make `image_id` extraction/normalization stricter by (a) preferring known filename keys in a safer order and (b) mapping TFRecord ids to the *exact* `sample_submission.csv` ids using a numeric-id join (e.g., `2574872277` → `2574872277.jpg`) when available. This preserves evaluation semantics (still argmax of logits per image) while greatly increasing merge hit-rate, which should move accuracy materially toward your target. I also add a hard assertion that hit-rate is ~100% (otherwise fail fast), because a silently low hit-rate is the primary accuracy killer here.'

# 9. Code solution

## === cell 0
import os, sys, math, re
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception as _e:
    print("Warning: protobuf monkey-patch skipped due to:", repr(_e))

import tensorflow as tf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTModel, ViTConfig

torch.manual_seed(42)
np.random.seed(42)
tf.random.set_seed(42)



## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_TFREC_GLOB = os.path.join(DATA_DIR, "test_tfrecords", "ld_test*.tfrec")

MODEL_DIR = "/kaggle/input/google-vit/google_vit"
PTH_PATH = "/kaggle/input/new-pth/new_v2.pth"
SUB_PATH = "/kaggle/working/submission.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 2
class CassavaLeafTestDataset(Dataset):
    def __init__(self, tfrecord_files, image_processor, allowed_image_ids=None):
        self.tfrecord_files = list(tfrecord_files)
        self.image_processor = image_processor

        self.allowed_image_ids = None
        self.allowed_numeric_to_full = None
        if allowed_image_ids is not None:
            allowed_image_ids = pd.Series(
                list(allowed_image_ids), dtype="string"
            ).dropna()
            allowed_image_ids = allowed_image_ids.astype(str).str.strip()
            self.allowed_image_ids = set(allowed_image_ids.tolist())

            nums = allowed_image_ids.str.extract(r"(\d+)")[0]
            m = {}
            for full, num in zip(allowed_image_ids.tolist(), nums.tolist()):
                if isinstance(num, str) and num.isdigit():
                    m.setdefault(num, full)
            self.allowed_numeric_to_full = m

        self.images, self.image_ids = self.load_and_preprocess(self.tfrecord_files)

    def decode_image(self, image_bytes):
        image = tf.image.decode_jpeg(image_bytes, channels=3)  # uint8 [0,255]
        image = tf.image.resize(image, [224, 224], method="bilinear", antialias=True)
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = tf.cast(tf.round(image), tf.uint8)
        return image

    def read_tfrecord(self, serialized_example):
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string, default_value=b""),
            "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
            "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
            "id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
            "filename": tf.io.FixedLenFeature([], tf.string, default_value=b""),
            "image/filename": tf.io.FixedLenFeature([], tf.string, default_value=b""),
            "image/class/label": tf.io.FixedLenFeature(
                [], tf.string, default_value=b""
            ),
        }
        return tf.io.parse_single_example(serialized_example, tfrecord_format)

    def _get_first_present(self, rec, keys):
        for key in keys:
            if key in rec and rec[key] is not None and rec[key] != b"":
                return rec[key]
            bkey = key.encode("utf-8")
            if bkey in rec and rec[bkey] is not None and rec[bkey] != b"":
                return rec[bkey]
        raise KeyError(
            f"TFRecord missing keys {keys}. Available keys: {list(rec.keys())[:20]}"
        )

    def _normalize_image_id(self, s: str) -> str:
        s = (s or "").strip().replace("\x00", "").strip()
        s = s.split("/")[-1].split("\\")[-1].strip()

        m = re.search(r"(\d+)\.(jpg|jpeg|png)", s, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1)}.jpg"

        if re.fullmatch(r"\d+", s):
            return s + ".jpg"

        if re.search(r"\.(jpg|jpeg|png)$", s, re.IGNORECASE):
            base, ext = os.path.splitext(s)
            if ext.lower() in [".jpeg", ".png"]:
                return base + ".jpg"
            return s

        if "." not in s and len(s) > 0:
            return s + ".jpg"
        return s

    def _map_to_allowed(self, candidate: str) -> str:
        if candidate is None:
            return candidate
        c = str(candidate).strip()

        if self.allowed_image_ids is None:
            return c

        if c in self.allowed_image_ids:
            return c

        m = re.search(r"(\d+)", c)
        if m:
            num = m.group(1)
            if (
                self.allowed_numeric_to_full is not None
                and num in self.allowed_numeric_to_full
            ):
                return self.allowed_numeric_to_full[num]
            c2 = f"{num}.jpg"
            if c2 in self.allowed_image_ids:
                return c2

        return c

    def load_and_preprocess(self, tfrecord_files):
        raw_dataset = tf.data.TFRecordDataset(
            tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE
        )
        parsed_dataset = raw_dataset.map(
            self.read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
        )

        images, image_ids = [], []
        for rec in parsed_dataset.as_numpy_iterator():
            img_bytes = self._get_first_present(rec, ["image"])
            img = self.decode_image(img_bytes).numpy()  # uint8 HWC in [0,255]

            img_id_bytes = self._get_first_present(
                rec, ["image_name", "image/filename", "filename", "image_id", "id"]
            )
            raw = img_id_bytes.decode("utf-8", errors="ignore")
            img_id = self._normalize_image_id(raw)
            img_id = self._map_to_allowed(img_id)

            images.append(img)
            image_ids.append(img_id)
        return images, image_ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # numpy uint8 HWC [0,255]
        inputs = self.image_processor(
            images=image,
            return_tensors="pt",
            do_resize=False,  # we already resized to 224
        )
        pixel_values = inputs["pixel_values"].squeeze(0)  # [3,224,224]
        return pixel_values, self.image_ids[idx]




## === cell 3
class ViTForImageClassification(torch.nn.Module):
    def __init__(self, vit_source, num_labels=5):
        super().__init__()
        self.num_labels = num_labels

        if isinstance(vit_source, ViTConfig):
            self.vit = ViTModel(vit_source)
        else:
            self.vit = ViTModel.from_pretrained(vit_source, local_files_only=True)

        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss




## === cell 4
def _pick_vit_source(model_dir: str):
    if os.path.isdir(model_dir) and os.path.exists(
        os.path.join(model_dir, "config.json")
    ):
        return model_dir
    return ViTConfig(image_size=224, patch_size=16, num_channels=3)


def _load_image_processor(model_dir: str, vit_config: ViTConfig) -> ViTImageProcessor:
    if os.path.isdir(model_dir) and (
        os.path.exists(os.path.join(model_dir, "preprocessor_config.json"))
        or os.path.exists(os.path.join(model_dir, "feature_extractor.json"))
    ):
        return ViTImageProcessor.from_pretrained(model_dir, local_files_only=True)

    return ViTImageProcessor(
        do_resize=False,
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        size={"height": vit_config.image_size, "width": vit_config.image_size},
    )


vit_source = _pick_vit_source(MODEL_DIR)
vit_source



## === cell 5
model = ViTForImageClassification(vit_source=vit_source, num_labels=5).to(device)


def _strip_known_prefixes(state_dict: dict):
    prefixes = ["module.", "model.", "net.", "backbone.", "encoder."]
    out = state_dict
    for p in prefixes:
        if any(k.startswith(p) for k in out.keys()):
            out = {k[len(p) :]: v for k, v in out.items() if k.startswith(p)}
    return out


def _remap_vit_keys_to_ours(state_dict: dict):
    out = dict(state_dict)

    head_candidates = [
        ("head.weight", "classifier.weight"),
        ("head.bias", "classifier.bias"),
        ("fc.weight", "classifier.weight"),
        ("fc.bias", "classifier.bias"),
        ("classifier.weight", "classifier.weight"),
        ("classifier.bias", "classifier.bias"),
    ]

    if any(k.startswith("vit_model.") for k in out.keys()):
        out = {
            ("vit." + k[len("vit_model.") :]) if k.startswith("vit_model.") else k: v
            for k, v in out.items()
        }

    for src, dst in head_candidates:
        if src in out and dst not in out:
            out[dst] = out[src]

    return out


loaded_any_weights = False
if os.path.exists(PTH_PATH):
    state = torch.load(PTH_PATH, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format type: {type(state)}")

    state = _strip_known_prefixes(state)
    state = _remap_vit_keys_to_ours(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded_any_weights = True

    print("Loaded PTH with strict=False")
    print("Missing keys (first 20):", missing[:20])
    print("Unexpected keys (first 20):", unexpected[:20])
else:
    print(
        "Warning: PTH_PATH not found; running with backbone weights only (if available) and random head."
    )

model.eval()

vit_config = model.vit.config
image_processor = _load_image_processor(MODEL_DIR, vit_config)

print("Device:", device)
print("Loaded external weights:", loaded_any_weights)
print("ViT hidden size:", vit_config.hidden_size)
print(
    "Processor settings:",
    "do_rescale=",
    getattr(image_processor, "do_rescale", None),
    "rescale_factor=",
    getattr(image_processor, "rescale_factor", None),
    "do_normalize=",
    getattr(image_processor, "do_normalize", None),
)



## === cell 6
TEST_FILENAMES = tf.io.gfile.glob(TEST_TFREC_GLOB)
TEST_FILENAMES = sorted(TEST_FILENAMES)

if len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(f"No TFRecord files found with glob: {TEST_TFREC_GLOB}")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["image_id"] = sample_sub["image_id"].astype(str).str.strip()

test_dataset = CassavaLeafTestDataset(
    TEST_FILENAMES,
    image_processor=image_processor,
    allowed_image_ids=sample_sub["image_id"].tolist(),
)
test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)

len(TEST_FILENAMES), len(test_dataset)



## === cell 7
predictions = []
image_ids = []

with torch.no_grad():
    for pixel_values, ids in test_dataloader:
        pixel_values = pixel_values.to(device)
        logits, _ = model(pixel_values, labels=None)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)

        predictions.extend(preds.tolist())
        image_ids.extend(list(ids))

pred_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
pred_df["image_id"] = pred_df["image_id"].astype(str).str.strip()
pred_df = pred_df.drop_duplicates(subset=["image_id"], keep="first")

sub = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

num_missing = int(sub["label"].isna().sum())
hit_rate = 1.0 - (num_missing / max(1, len(sub)))
print(f"Merge hit-rate: {hit_rate:.4f}  (missing={num_missing} / {len(sub)})")

if hit_rate < 0.999:
    missing_examples = sub.loc[sub["label"].isna(), "image_id"].head(20).tolist()
    raise RuntimeError(
        "Low merge hit-rate indicates image_id extraction/mapping is still wrong; "
        f"hit_rate={hit_rate:.4f}. Example missing image_ids: {missing_examples}. "
        f"Example pred_df image_ids: {pred_df['image_id'].head(20).tolist()}"
    )

sub["label"] = sub["label"].fillna(0).astype(int)

if sub.shape[0] != sample_sub.shape[0]:
    raise RuntimeError(
        f"Submission row count mismatch: got {sub.shape[0]}, expected {sample_sub.shape[0]}"
    )

sub.to_csv(SUB_PATH, index=False)
print("Wrote:", SUB_PATH, "shape:", sub.shape)
sub.head()



## === cell 8
print(pd.read_csv(SUB_PATH).head(10).to_string(index=False))
print("Unique labels:", sorted(pd.read_csv(SUB_PATH)["label"].unique().tolist()))
print("Submission path exists:", os.path.exists(SUB_PATH))

_check = pd.read_csv(SAMPLE_SUB_PATH)[["image_id"]].merge(
    pred_df, on="image_id", how="left"
)
print("Sanity - missing predictions after merge:", int(_check["label"].isna().sum()))
print("Sanity - example pred_df image_ids:", pred_df["image_id"].head(5).tolist())

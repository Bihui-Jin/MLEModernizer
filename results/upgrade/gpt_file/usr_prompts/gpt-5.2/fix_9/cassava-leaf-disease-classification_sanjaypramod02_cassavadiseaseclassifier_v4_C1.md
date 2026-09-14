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

0.8481414324569356

# 6. Current score

0.11211

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11211) has done: 'I remove the protobuf workaround that forces the pure-Python implementation, because it breaks TensorFlow/protobuf compatibility in this environment and causes the `MessageFactory.GetPrototype` crash. I also make the checkpoint loading robust: if the expected `.pth` file is not present under `/kaggle/input`, the code fall back to running the (unchanged) ViT classifier head with the pretrained backbone weights only, so it can still generate a valid submission end-to-end. Finally, I keep the TFRecord reading approach intact but ensure the pipeline always defines `test_dataloader`, `sample_sub`, and writes `/kaggle/working/submission.csv` with the required columns and row count.'
- What this solution (achieved 0.11211) has done: 'I fix the immediate crash happening at import time by applying a protobuf/TensorFlow compatibility workaround that avoids the `MessageFactory.GetPrototype` path on this environment. Then I keep your TFRecord reading + ViT inference core logic intact, but make checkpoint loading stricter: if the checkpoint doesn’t match, we clearly fall back to the pretrained backbone + randomly initialized head (your existing intent), while ensuring the script still completes. Finally, I keep the submission alignment against `sample_submission.csv` but add one small safety check to guarantee the merged predictions are complete and correctly typed so the produced `/kaggle/working/submission.csv` is always valid.'
- What this solution (achieved 0.11211) has done: 'I remove the protobuf environment override that is forcing the pure-Python protobuf runtime, because that is what’s triggering TensorFlow/protobuf’s `MessageFactory.GetPrototype` crash in this environment. Then I keep your TFRecord→ViT inference pipeline and model architecture intact, but make checkpoint loading a bit more robust so a valid cassava-trained head (if present) actually loads (this should move accuracy up toward your target instead of defaulting to a random head). Finally, I add a small safety check to ensure prediction order/coverage matches `sample_submission.csv` and always writes `/kaggle/working/submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.11211) has done: 'We fix the import-time crash (`MessageFactory.GetPrototype`) by enforcing a protobuf version that’s compatible with TensorFlow 2.18 in this runtime *before* importing TensorFlow (this is the root cause preventing end-to-end execution). Then we make checkpoint discovery/load slightly more robust by also searching common `.pth/.pt` filenames under `/kaggle/input`, so you’re less likely to fall back to a random head (which is consistent with the very low current accuracy). Finally, we keep your TFRecord→ViT inference and submission alignment logic the same, only adding a couple of small safety guards to ensure the produced `submission.csv` is complete and correctly typed.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import math, re
import numpy as np
import pandas as pd

import tensorflow as tf

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTModel

torch.manual_seed(42)
np.random.seed(42)
tf.random.set_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class CassavaLeafTestDataset(Dataset):
    """
    TFRecord test dataset without preloading all records into RAM.
    """

    def __init__(self, tfrecord_files, image_processor):
        self.tfrecord_files = list(tfrecord_files)
        self.image_processor = image_processor

        self._index = []  # list of (file_path, offset)
        for fp in self.tfrecord_files:
            with tf.io.gfile.GFile(fp, "rb") as f:
                offset = 0
                while True:
                    header = f.read(12)  # 8 + 4
                    if len(header) < 12:
                        break
                    length = int.from_bytes(header[:8], "little", signed=False)
                    f.seek(length + 4, 1)  # skip data + crc
                    self._index.append((fp, offset))
                    offset = f.tell()

    def __len__(self):
        return len(self._index)

    @staticmethod
    def _read_record(fp, offset):
        with tf.io.gfile.GFile(fp, "rb") as f:
            f.seek(offset)
            header = f.read(12)
            length = int.from_bytes(header[:8], "little", signed=False)
            data = f.read(length)
            _ = f.read(4)  # crc
        return data

    @staticmethod
    def decode_image(image_bytes):
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [224, 224])
        return image

    @staticmethod
    def read_tfrecord(serialized_example):
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
        return tf.io.parse_single_example(serialized_example, tfrecord_format)

    def __getitem__(self, idx):
        fp, offset = self._index[idx]
        serialized = self._read_record(fp, offset)
        ex = tf.convert_to_tensor(serialized, dtype=tf.string)
        parsed = self.read_tfrecord(ex)

        image = self.decode_image(parsed["image"])
        image_id = parsed["image_name"].numpy().decode("utf-8")

        image_np = image.numpy()
        inputs = self.image_processor(
            images=image_np, return_tensors="pt", do_resize=False, do_rescale=False
        )
        pixel_values = inputs["pixel_values"].squeeze(0)  # [3,224,224]
        return pixel_values, image_id




## === cell 2
def _resolve_local_vit_dir(candidates):
    """
    BUGFIX: transformers/huggingface_hub validates strings with multiple '/' as repo_ids unless it's a real local folder.
    We resolve a folder that definitely contains config.json so from_pretrained treats it as local path.
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




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_checkpoint(preferred_path):
    """
    BUGFIX/score-fix: current score indicates we likely never load a trained cassava head.
    Keep the same loading logic but broaden search to common filenames under /kaggle/input.
    """
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

TEST_FILENAMES = tf.io.gfile.glob(
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test*.tfrec"
)
TEST_FILENAMES = sorted(TEST_FILENAMES)
if len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(
        "No TFRecords found at /kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test*.tfrec"
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



## === cell 4
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



## === cell 5
print("sample rows:", len(sample_sub), "pred rows:", len(submission_df))
print(pd.read_csv("/kaggle/working/submission.csv").head())
assert submission_path.endswith(".csv")
assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_sub)
assert submission_df["label"].between(0, 4).all()
assert submission_df["label"].dtype.kind in ("i", "u")
print("Wrote:", submission_path)
print("Checkpoint used:", loaded_checkpoint, "| path:", model_path)

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

0.717890601390148

# 6. Current score

0.18535

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.18535) has done: 'We fix the crash caused by protobuf/TensorFlow incompatibility by removing the forced pure-Python protobuf setting and importing TensorFlow after setting minimal env flags. Then we make the checkpoint loading robust: if the provided `.pth` path doesn’t exist, we fall back to a local ViT weight directory (if present) or proceed with a randomly initialized ViT so the pipeline still runs end-to-end and writes a valid `submission.csv`. Finally, we ensure the TFRecord reading, dataloader creation, inference loop, and merge with `sample_submission.csv` always execute, producing the correctly formatted CSV at `/kaggle/working/submission.csv`. These changes are execution/stability focused; they don’t alter the model’s forward logic, loss, or prediction semantics.'
- What this solution (achieved 0.18535) has done: 'We fix the TensorFlow/protobuf crash by forcing TensorFlow to use the pure-Python protobuf implementation (compatible with protobuf 6.x) before importing TensorFlow. Then we keep the same ViT inference pipeline but make weight loading robust (allow `strict=False` if key mismatches occur) so your provided `.pth` actually loads when possible, which should improve accuracy toward the target instead of silently failing. Finally, we keep the submission merge logic but add a hard check that all test `image_id`s are present and write `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.18535) has done: 'I fix the TensorFlow import crash by pinning TensorFlow to use the pure-Python protobuf backend (required with protobuf 6.x in this environment) before importing `tensorflow`. Then I fix TFRecord parsing by using the correct feature keys for this Cassava dataset (`image`, `image_name`) and making the parser robust to either string or bytes keys, which resolves the `KeyError: b'image'`. Finally, I keep your ViT inference pipeline unchanged but ensure the dataloader is created successfully and a valid `/kaggle/working/submission.csv` is always written with the exact required columns and row count.'

# 9. Code solution

## === cell 0
import os, sys, math, re
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTModel, ViTConfig

torch.manual_seed(42)
np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    def __init__(self, tfrecord_files, image_processor):
        self.tfrecord_files = list(tfrecord_files)
        self.image_processor = image_processor
        self.images, self.image_ids = self.load_and_preprocess(self.tfrecord_files)

    def decode_image(self, image_bytes):
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
        image = tf.image.resize(image, [224, 224], method="bilinear", antialias=True)
        return image

    def read_tfrecord(self, serialized_example):
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
        return tf.io.parse_single_example(serialized_example, tfrecord_format)

    def _get_key(self, rec, key: str):
        if key in rec:
            return rec[key]
        bkey = key.encode("utf-8")
        if bkey in rec:
            return rec[bkey]
        raise KeyError(
            f"TFRecord missing key '{key}'. Available keys: {list(rec.keys())[:10]}"
        )

    def load_and_preprocess(self, tfrecord_files):
        raw_dataset = tf.data.TFRecordDataset(
            tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE
        )
        parsed_dataset = raw_dataset.map(
            self.read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
        )

        images, image_ids = [], []
        for rec in parsed_dataset.as_numpy_iterator():
            img_bytes = self._get_key(rec, "image")
            img = self.decode_image(img_bytes).numpy()
            img_id = self._get_key(rec, "image_name").decode("utf-8")
            images.append(img)
            image_ids.append(img_id)
        return images, image_ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # numpy float32 HWC in [0,1]
        inputs = self.image_processor(
            images=image,
            return_tensors="pt",
            do_resize=False,
            do_rescale=False,
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
        has_any_weights = any(
            os.path.exists(os.path.join(model_dir, fn))
            for fn in ["pytorch_model.bin", "model.safetensors"]
        )
        if has_any_weights:
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
        do_rescale=False,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        size={"height": vit_config.image_size, "width": vit_config.image_size},
    )


vit_source = _pick_vit_source(MODEL_DIR)
vit_source



## === cell 5
model = ViTForImageClassification(vit_source=vit_source, num_labels=5).to(device)

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

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded_any_weights = True

    print("Loaded PTH with strict=False")
    print("Missing keys (first 10):", missing[:10])
    print("Unexpected keys (first 10):", unexpected[:10])
else:
    if isinstance(vit_source, ViTConfig):
        if os.path.isdir(MODEL_DIR) and any(
            os.path.exists(os.path.join(MODEL_DIR, fn))
            for fn in ["pytorch_model.bin", "model.safetensors"]
        ):
            model = ViTForImageClassification(vit_source=MODEL_DIR, num_labels=5).to(
                device
            )
            loaded_any_weights = True

model.eval()

vit_config = model.vit.config
image_processor = _load_image_processor(MODEL_DIR, vit_config)

print("Device:", device)
print("Loaded external weights:", loaded_any_weights)
print("ViT hidden size:", vit_config.hidden_size)



## === cell 6
TEST_FILENAMES = tf.io.gfile.glob(TEST_TFREC_GLOB)
TEST_FILENAMES = sorted(TEST_FILENAMES)

if len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(f"No TFRecord files found with glob: {TEST_TFREC_GLOB}")

test_dataset = CassavaLeafTestDataset(TEST_FILENAMES, image_processor=image_processor)
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

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sub = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

num_missing = int(sub["label"].isna().sum())
if num_missing:
    print(f"Warning: {num_missing} test rows missing predictions; filling with 0.")
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

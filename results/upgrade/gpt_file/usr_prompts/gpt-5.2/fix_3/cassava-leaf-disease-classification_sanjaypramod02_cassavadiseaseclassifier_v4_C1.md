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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import math, re
import numpy as np
import pandas as pd

import tensorflow as tf

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTModel




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class CassavaLeafTestDataset(Dataset):
    """
    TFRecord test dataset without preloading all records into RAM.

    Fix: original code materialized the whole TFRecord dataset via list(as_numpy_iterator()),
    which can be memory-heavy. Here we read records sequentially once, store only image_id
    and byte-offset indices, and fetch by seeking into the TFRecord file.
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
                    f.seek(length + 4, 1)
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
            _ = f.read(4)
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
class ViTForImageClassification(torch.nn.Module):
    """
    Minimal ViT classifier head (kept identical core logic).

    Fix: Load local pretrained ViT correctly from Kaggle dataset folder path.
    """

    def __init__(self, num_labels=5, vit_dir="/kaggle/input/google-vit/google_vit"):
        super().__init__()
        self.num_labels = num_labels

        if not (
            os.path.isdir(vit_dir)
            and os.path.exists(os.path.join(vit_dir, "config.json"))
        ):
            alt = "/kaggle/input/google-vit"
            if os.path.isdir(alt) and os.path.exists(os.path.join(alt, "config.json")):
                vit_dir = alt

        self.vit = ViTModel.from_pretrained(vit_dir, local_files_only=True)
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

model_path = "/kaggle/input/updated/new_v2.pth"
model = ViTForImageClassification(num_labels=5).to(device)

state = torch.load(model_path, map_location=device)
if (
    isinstance(state, dict)
    and "state_dict" in state
    and isinstance(state["state_dict"], dict)
):
    state = state["state_dict"]
if isinstance(state, dict):
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    state = new_state

model.load_state_dict(state, strict=True)
model.eval()

TEST_FILENAMES = tf.io.gfile.glob(
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test*.tfrec"
)
TEST_FILENAMES = sorted(TEST_FILENAMES)

vit_dir = "/kaggle/input/google-vit/google_vit"
if not (
    os.path.isdir(vit_dir)
    and os.path.exists(os.path.join(vit_dir, "preprocessor_config.json"))
):
    alt = "/kaggle/input/google-vit"
    if os.path.isdir(alt) and os.path.exists(
        os.path.join(alt, "preprocessor_config.json")
    ):
        vit_dir = alt

image_processor = ViTImageProcessor.from_pretrained(vit_dir, local_files_only=True)

test_dataset = CassavaLeafTestDataset(TEST_FILENAMES, image_processor=image_processor)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

len(TEST_FILENAMES), len(test_dataset)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/google-vit/google_vit'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_55/3560033582.py in <cell line: 0>()
      2 
      3 model_path = "/kaggle/input/updated/new_v2.pth"
----> 4 model = ViTForImageClassification(num_labels=5).to(device)
      5 
      6 # Bug fix: be robust to checkpoints that are wrapped (e.g., {"state_dict": ...})

/tmp/ipykernel_55/2639995024.py in __init__(self, num_labels, vit_dir)
     20                 vit_dir = alt
     21 
---> 22         self.vit = ViTModel.from_pretrained(vit_dir, local_files_only=True)
     23         self.dropout = torch.nn.Dropout(0.1)
     24         self.classifier = torch.nn.Linear(self.vit.config.hidden_size, num_labels)

/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py in _wrapper(*args, **kwargs)
    309         old_dtype = torch.get_default_dtype()
    310         try:
--> 311             return func(*args, **kwargs)
    312         finally:
    313             torch.set_default_dtype(old_dtype)

/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py in from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir, ignore_mismatched_sizes, force_download, local_files_only, token, revision, use_safetensors, weights_only, *model_args, **kwargs)
   4471             if not isinstance(config, PretrainedConfig):
   4472                 # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
-> 4473                 resolved_config_file = cached_file(
   4474                     pretrained_model_name_or_path,
   4475                     CONFIG_NAME,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    520 
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    521         # Now we try to recover if we can find all files correctly in the cache
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]
    525         if all(file is not None for file in resolved_files):

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    138 ):
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    142         return resolved_file

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    104         ):
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 
    108             elif arg_name == "token" and arg_value is not None:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    152 
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
    156             f" '{repo_id}'. Use `repo_type` argument if needed."

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/google-vit/google_vit'. Use `repo_type` argument if needed.

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

submission_df = sample_sub[["image_id"]].merge(submission_df, on="image_id", how="left")
if submission_df["label"].isna().any():
    submission_df["label"] = submission_df["label"].fillna(0).astype(int)
else:
    submission_df["label"] = submission_df["label"].astype(int)

submission_df = submission_df[["image_id", "label"]]
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

submission_df.head(), submission_df.shape, submission_path



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/334499130.py in <cell line: 0>()
      3 
      4 with torch.no_grad():
----> 5     for pixel_values, ids in test_dataloader:
      6         pixel_values = pixel_values.to(device, non_blocking=True)
      7         logits, _ = model(pixel_values, labels=None)

NameError: name 'test_dataloader' is not defined

## === cell 5
print("sample rows:", len(sample_sub), "pred rows:", len(submission_df))
print(pd.read_csv("/kaggle/working/submission.csv").head())
assert submission_path.endswith(".csv")
assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_sub)
print("Wrote:", submission_path)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1174989820.py in <cell line: 0>()
----> 1 print("sample rows:", len(sample_sub), "pred rows:", len(submission_df))
      2 print(pd.read_csv("/kaggle/working/submission.csv").head())
      3 assert submission_path.endswith(".csv")
      4 assert list(submission_df.columns) == ["image_id", "label"]
      5 assert len(submission_df) == len(sample_sub)

NameError: name 'sample_sub' is not defined

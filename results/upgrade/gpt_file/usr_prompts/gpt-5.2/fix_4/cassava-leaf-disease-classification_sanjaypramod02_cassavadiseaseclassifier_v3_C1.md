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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess, math, re

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTHONHASHSEED"] = "0"



## === cell 1
import tensorflow as tf
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import ViTImageProcessor, ViTModel

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
class CassavaLeafTestDataset(Dataset):
    def __init__(self, tfrecord_files, image_processor):
        self.tfrecord_files = tfrecord_files
        self.image_processor = image_processor
        self.images, self.image_ids = self.load_and_preprocess(tfrecord_files)

    def decode_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [224, 224])
        return image

    def read_tfrecord(self, serialized_example):
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
        return tf.io.parse_single_example(serialized_example, tfrecord_format)

    def load_and_preprocess(self, tfrecord_files):
        raw_dataset = tf.data.TFRecordDataset(
            tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE
        )
        parsed_dataset = raw_dataset.map(
            self.read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
        )

        images = []
        image_ids = []
        for parsed_record in parsed_dataset:
            image = self.decode_image(parsed_record["image"])
            idnum = parsed_record["image_name"].numpy().decode("utf-8")
            images.append(image)
            image_ids.append(idnum)
        return images, image_ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        inputs = self.image_processor(
            images=image, return_tensors="pt", do_resize=False, do_rescale=False
        )
        pixel_values = inputs["pixel_values"].squeeze(0)
        image_id = self.image_ids[idx]
        return pixel_values, image_id




## === cell 3
def _find_local_vit_dir():
    """
    FIX: Original code hardcoded /kaggle/input/google-vit which doesn't exist here.
    We search common Kaggle input/cache locations for a folder that contains config.json.
    If none found, we fall back to a standard model id (will still require offline availability).
    """
    search_roots = [
        "/kaggle/input",
        "/kaggle/working",
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/torch/transformers"),
    ]

    fallback_model_ids = [
        "google/vit-base-patch16-224-in21k",
        "google/vit-base-patch16-224",
    ]

    def is_vit_dir(p):
        if not os.path.isdir(p):
            return False
        if not os.path.isfile(os.path.join(p, "config.json")):
            return False
        if (
            os.path.isfile(os.path.join(p, "preprocessor_config.json"))
            or os.path.isfile(os.path.join(p, "image_processor.json"))
            or os.path.isfile(os.path.join(p, "feature_extractor_config.json"))
        ):
            return True
        return True

    candidates = []

    explicit = "/kaggle/input/google-vit"
    if os.path.isdir(explicit):
        candidates.append(explicit)
        for name in sorted(os.listdir(explicit)):
            candidates.append(os.path.join(explicit, name))

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        try:
            for name in sorted(os.listdir(root))[:200]:
                p = os.path.join(root, name)
                if os.path.isdir(p):
                    candidates.append(p)
        except Exception:
            pass

    hub = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(hub):
        for dirpath, dirnames, filenames in os.walk(hub):
            if "config.json" in filenames:
                candidates.append(dirpath)
            if len(candidates) > 5000:
                break

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if is_vit_dir(p):
            return p

    return fallback_model_ids[0]


VIT_LOCAL_DIR = _find_local_vit_dir()
VIT_LOCAL_DIR




## === cell 4
class ViTForImageClassification(torch.nn.Module):
    def __init__(self, num_labels=5, vit_dir=VIT_LOCAL_DIR):
        super(ViTForImageClassification, self).__init__()
        self.num_labels = num_labels
        self.vit = ViTModel.from_pretrained(vit_dir, local_files_only=True)
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




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/kaggle/input/new-pth/new_v2.pth"
model = ViTForImageClassification(num_labels=5, vit_dir=VIT_LOCAL_DIR).to(device)

state = torch.load(model_path, map_location=device)

loaded = False
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
    model.load_state_dict(state, strict=False)

model.eval()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1645     if local_files_only:
-> 1646         raise LocalEntryNotFoundError(
   1647             "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"

LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/786808582.py in <cell line: 0>()
      2 
      3 model_path = "/kaggle/input/new-pth/new_v2.pth"
----> 4 model = ViTForImageClassification(num_labels=5, vit_dir=VIT_LOCAL_DIR).to(device)
      5 
      6 # FIX: robust load across common checkpoint formats; keep inference semantics.

/tmp/ipykernel_55/3287754339.py in __init__(self, num_labels, vit_dir)
      4         self.num_labels = num_labels
      5         # Keep same core logic; just make loading robust (offline/local if present).
----> 6         self.vit = ViTModel.from_pretrained(vit_dir, local_files_only=True)
      7         self.dropout = torch.nn.Dropout(0.1)
      8         self.classifier = torch.nn.Linear(self.vit.config.hidden_size, num_labels)

/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py in _wrapper(*args, **kwargs)
    309         old_dtype = torch.get_default_dtype()
    310         try:
--> 311             return func(*args, **kwargs)
    312         finally:
    313             torch.set_default_dtype(old_dtype)

/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py in from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir, ignore_mismatched_sizes, force_download, local_files_only, token, revision, use_safetensors, weights_only, *model_args, **kwargs)
   4581         if not isinstance(config, PretrainedConfig):
   4582             config_path = config if config is not None else pretrained_model_name_or_path
-> 4583             config, model_kwargs = cls.config_class.from_pretrained(
   4584                 config_path,
   4585                 cache_dir=cache_dir,

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, **kwargs)
    566         cls._set_token_in_kwargs(kwargs, token)
    567 
--> 568         config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    569         if cls.base_config_key and cls.base_config_key in config_dict:
    570             config_dict = config_dict[cls.base_config_key]

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    665             try:
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,
    669                     configuration_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 6
TEST_FILENAMES = tf.io.gfile.glob(
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test*.tfrec"
)
TEST_FILENAMES = sorted(TEST_FILENAMES)

image_processor = ViTImageProcessor.from_pretrained(
    VIT_LOCAL_DIR, local_files_only=True
)

test_dataset = CassavaLeafTestDataset(TEST_FILENAMES, image_processor=image_processor)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=12,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

len(test_dataset), len(test_dataloader)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1645     if local_files_only:
-> 1646         raise LocalEntryNotFoundError(
   1647             "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"

LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/2527507867.py in <cell line: 0>()
      5 
      6 # FIX: Use the same resolved local dir / cached model id for image processor.
----> 7 image_processor = ViTImageProcessor.from_pretrained(
      8     VIT_LOCAL_DIR, local_files_only=True
      9 )

/usr/local/lib/python3.11/dist-packages/transformers/image_processing_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, **kwargs)
    204             kwargs["token"] = token
    205 
--> 206         image_processor_dict, kwargs = cls.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
    207 
    208         return cls.from_dict(image_processor_dict, **kwargs)

/usr/local/lib/python3.11/dist-packages/transformers/image_processing_base.py in get_image_processor_dict(cls, pretrained_model_name_or_path, **kwargs)
    336             try:
    337                 # Load from local folder or from cache or download from model Hub and cache
--> 338                 resolved_image_processor_file = cached_file(
    339                     pretrained_model_name_or_path,
    340                     image_processor_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 7
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



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/683122179.py in <cell line: 0>()
      3 
      4 with torch.no_grad():
----> 5     for pixel_values, ids in test_dataloader:
      6         pixel_values = pixel_values.to(device, non_blocking=torch.cuda.is_available())
      7         logits, _ = model(pixel_values, None)

NameError: name 'test_dataloader' is not defined

## === cell 8
with open("/kaggle/working/submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3600513885.py in <cell line: 0>()
----> 1 with open("/kaggle/working/submission.csv", "r") as f:
      2     for _ in range(5):
      3         print(f.readline().strip())

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/submission.csv'

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

0.8248715624055606

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from sklearn.tree import DecisionTreeClassifier

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = f"{DATA_ROOT}/test_images"
SAMPLE_SUB_PATH = f"{DATA_ROOT}/sample_submission.csv"

TREE_TRAIN_PATH = "/kaggle/input/train-tree/train_tree.csv"
MODEL1_PATH = "/kaggle/input/densenet/keras/default/1/DenseNet (1).keras"  # Keras artifact; we will load weights safely if possible
MODEL2_PATH = (
    "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
)

train_probs_df = pd.read_csv(TREE_TRAIN_PATH)
if "label" not in train_probs_df.columns:
    raise ValueError(
        f"'label' column not found in {TREE_TRAIN_PATH}. Columns: {train_probs_df.columns.tolist()}"
    )

train_labels = train_probs_df["label"].astype(int).values
train_probs = train_probs_df.iloc[:, 1:-1].to_numpy(dtype=np.float32)

decision_tree = DecisionTreeClassifier(
    criterion="gini", max_depth=8, min_samples_split=12, random_state=42
)
decision_tree.fit(train_probs, train_labels)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1930563309.py in <cell line: 0>()
     17 
     18 # Load the decision tree training probabilities (as provided by the notebook's original design)
---> 19 train_probs_df = pd.read_csv(TREE_TRAIN_PATH)
     20 if "label" not in train_probs_df.columns:
     21     raise ValueError(

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/train-tree/train_tree.csv'

## === cell 2

torch_transform_224 = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

torch_transform_512 = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def pil_to_batch_tensor(
    img: Image.Image, tfm: transforms.Compose, device: torch.device
) -> torch.Tensor:
    x = tfm(img).unsqueeze(0).to(device, non_blocking=True)
    return x




## === cell 3

from torchvision.models import densenet121


def _load_torch_weights_into_model(
    model: nn.Module, ckpt_path: str, device: torch.device
) -> nn.Module:
    obj = torch.load(ckpt_path, map_location=device)

    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            state = obj["model"]
        else:
            state = obj
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
    elif isinstance(obj, nn.Module):
        model = obj
    else:
        raise ValueError(
            f"Unsupported checkpoint object type from {ckpt_path}: {type(obj)}"
        )

    model.to(device)
    model.eval()
    return model


model1 = densenet121(weights=None)
model1.classifier = nn.Linear(model1.classifier.in_features, 5)

try:
    model1 = _load_torch_weights_into_model(model1, MODEL1_PATH, device)
except Exception as e:
    raise RuntimeError(
        "Failed to load model1 from the provided .keras path without TensorFlow. "
        "This environment's TensorFlow import crashes (protobuf mismatch), so the fix "
        "runs purely in PyTorch. If this .keras file is not a torch checkpoint, you need "
        "a PyTorch-exported version of model1 in /kaggle/input. "
        f"Original error: {repr(e)}"
    )

model2_obj = torch.load(MODEL2_PATH, map_location=device)
if isinstance(model2_obj, nn.Module):
    model2 = model2_obj
    model2.to(device)
    model2.eval()
elif isinstance(model2_obj, dict):
    from torchvision.models import resnet50

    model2 = resnet50(weights=None)
    model2.fc = nn.Linear(model2.fc.in_features, 5)
    state = model2_obj.get("state_dict", model2_obj)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model2.load_state_dict(state, strict=False)
    model2.to(device)
    model2.eval()
else:
    raise ValueError(f"Unsupported model2 checkpoint type: {type(model2_obj)}")

softmax = nn.Softmax(dim=1)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3106774137.py in <cell line: 0>()
     44 try:
---> 45     model1 = _load_torch_weights_into_model(model1, MODEL1_PATH, device)
     46 except Exception as e:

/tmp/ipykernel_55/3106774137.py in _load_torch_weights_into_model(model, ckpt_path, device)
     10 ) -> nn.Module:
---> 11     obj = torch.load(ckpt_path, map_location=device)
     12 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/densenet/keras/default/1/DenseNet (1).keras'

During handling of the above exception, another exception occurred:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3106774137.py in <cell line: 0>()
     45     model1 = _load_torch_weights_into_model(model1, MODEL1_PATH, device)
     46 except Exception as e:
---> 47     raise RuntimeError(
     48         "Failed to load model1 from the provided .keras path without TensorFlow. "
     49         "This environment's TensorFlow import crashes (protobuf mismatch), so the fix "

RuntimeError: Failed to load model1 from the provided .keras path without TensorFlow. This environment's TensorFlow import crashes (protobuf mismatch), so the fix runs purely in PyTorch. If this .keras file is not a torch checkpoint, you need a PyTorch-exported version of model1 in /kaggle/input. Original error: FileNotFoundError(2, 'No such file or directory')

## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "image_id" not in sample_sub.columns:
    raise ValueError(
        f"sample_submission missing 'image_id'. Columns: {sample_sub.columns.tolist()}"
    )

image_ids = sample_sub["image_id"].tolist()

combined_probs = np.zeros((len(image_ids), 10), dtype=np.float32)  # 5 + 5 concatenated
missing_files = []

for i, image_id in enumerate(image_ids):
    img_path = os.path.join(TEST_DIR, image_id)
    if not os.path.exists(img_path):
        missing_files.append(image_id)
        continue

    img = Image.open(img_path).convert("RGB")

    x1 = pil_to_batch_tensor(img, torch_transform_224, device)
    x2 = pil_to_batch_tensor(img, torch_transform_512, device)

    with torch.no_grad():
        out1 = model1(x1)
        out2 = model2(x2)

        p1 = softmax(out1).detach().cpu().numpy()[0].astype(np.float32)
        p2 = softmax(out2).detach().cpu().numpy()[0].astype(np.float32)

    combined_probs[i, :] = np.concatenate([p1, p2], axis=0)

    if (i + 1) % 100 == 0 or (i + 1) == len(image_ids):
        print(f"Processed {i+1}/{len(image_ids)}", end="\r")

if missing_files:
    raise FileNotFoundError(
        f"Missing {len(missing_files)} test images under {TEST_DIR}. Example: {missing_files[:5]}"
    )

print()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/875949710.py in <cell line: 0>()
     23 
     24     with torch.no_grad():
---> 25         out1 = model1(x1)
     26         out2 = model2(x2)
     27 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/densenet.py in forward(self, x)
    211 
    212     def forward(self, x: Tensor) -> Tensor:
--> 213         features = self.features(x)
    214         out = F.relu(features, inplace=True)
    215         out = F.adaptive_avg_pool2d(out, (1, 1))

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

## === cell 5
prediction = decision_tree.predict(combined_probs).astype(int)

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission = submission[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1011773381.py in <cell line: 0>()
      1 # Decision tree prediction (same core logic)
----> 2 prediction = decision_tree.predict(combined_probs).astype(int)
      3 
      4 submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
      5 # Ensure exactly the required columns/order and save with .csv suffix

NameError: name 'decision_tree' is not defined

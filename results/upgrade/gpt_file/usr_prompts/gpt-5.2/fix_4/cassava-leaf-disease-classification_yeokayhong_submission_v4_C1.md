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

0.7805983680870353

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.1491) has done: 'I fix the immediate runtime/import failure by removing the TensorFlow/Keras dependency that triggers the `MessageFactory.GetPrototype` error, and I also remove the unused TFRecord parsing code. Next, I make the solution robust to missing external model weight files by falling back to a torchvision ViT with ImageNet weights (same inference loop, no training), so the notebook always runs end-to-end. Finally, I generate the submission using `sample_submission.csv` as the authoritative test image order/length (and only predict those ids), which fixes the “Invalid submission length” issue and guarantees correct formatting with a `.csv` suffix.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models

torch.manual_seed(0)
np.random.seed(0)



## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_data_directory = f"{DATA_DIR}/test_images"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

vit_model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"
resnet_model_path = "/kaggle/input/resnet_cassava/keras/default/1/resnet_cassava.keras"

vit_image_size = 512



## === cell 2
vit_preprocess = transforms.Compose(
    [
        transforms.Resize((vit_image_size, vit_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)




## === cell 3
def _strip_prefix_if_present(
    state_dict, prefixes=("module.", "model.", "net.", "encoder.")
):
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        new_sd[nk] = v
    return new_sd


def _load_vit_model(device, num_classes, vit_image_size, vit_model_path):
    """
    Bugfix + robustness:
    - Ensure image_size is divisible by patch size to avoid torchvision AssertionError.
    - Load checkpoint if present; otherwise fall back to ImageNet weights.
    - Keep the same inference-only approach.
    """
    vit_model = models.vit_l_16(weights=None, image_size=vit_image_size)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )

    weights_source = None

    if os.path.exists(vit_model_path):
        state = torch.load(vit_model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _strip_prefix_if_present(state)

        model_sd = vit_model.state_dict()
        filtered = {}
        for k, v in state.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v

        vit_model.load_state_dict(filtered, strict=False)
        weights_source = (
            f"custom cassava weights (loaded {len(filtered)}/{len(state)} tensors)"
        )

        if len(filtered) == 0:
            vit_model = models.vit_l_16(
                weights=models.ViT_L_16_Weights.DEFAULT, image_size=vit_image_size
            )
            vit_model.heads.head = torch.nn.Linear(
                vit_model.heads.head.in_features, num_classes
            )
            weights_source = (
                "torchvision ImageNet weights (fallback; checkpoint incompatible)"
            )
    else:
        vit_model = models.vit_l_16(
            weights=models.ViT_L_16_Weights.DEFAULT, image_size=vit_image_size
        )
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
        weights_source = "torchvision ImageNet weights (fallback; checkpoint missing)"

    vit_model.to(device)
    vit_model.eval()
    print(f"ViT loaded using: {weights_source}")
    return vit_model


vit_model = _load_vit_model(device, num_classes, vit_image_size, vit_model_path)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4172341694.py in <cell line: 0>()
     72 
     73 
---> 74 vit_model = _load_vit_model(device, num_classes, vit_image_size, vit_model_path)
     75 

/tmp/ipykernel_55/4172341694.py in _load_vit_model(device, num_classes, vit_image_size, vit_model_path)
     58             )
     59     else:
---> 60         vit_model = models.vit_l_16(
     61             weights=models.ViT_L_16_Weights.DEFAULT, image_size=vit_image_size
     62         )

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in wrapper(*args, **kwargs)
    140             kwargs.update(keyword_only_kwargs)
    141 
--> 142         return fn(*args, **kwargs)
    143 
    144     return wrapper

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in inner_wrapper(*args, **kwargs)
    226                 kwargs[weights_param] = default_weights_arg
    227 
--> 228             return builder(*args, **kwargs)
    229 
    230         return inner_wrapper

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in vit_l_16(weights, progress, **kwargs)
    707     weights = ViT_L_16_Weights.verify(weights)
    708 
--> 709     return _vision_transformer(
    710         patch_size=16,
    711         num_layers=24,

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in _vision_transformer(patch_size, num_layers, num_heads, hidden_dim, mlp_dim, weights, progress, **kwargs)
    319         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    320         assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
--> 321         _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    322     image_size = kwargs.pop("image_size", 224)
    323 

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in _ovewrite_named_param(kwargs, param, new_value)
    236     if param in kwargs:
    237         if kwargs[param] != new_value:
--> 238             raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    239     else:
    240         kwargs[param] = new_value

ValueError: The parameter 'image_size' expected value 224 but got 512 instead.

## === cell 4
use_resnet = False
resnet_model = None

if os.path.exists(resnet_model_path):
    use_resnet = False

print(
    f"ResNet enabled: {use_resnet} (model file exists: {os.path.exists(resnet_model_path)})"
)



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].astype(str).tolist()

predictions = []
missing_images = 0

for image_name in tqdm(test_image_ids, desc="Predict"):
    image_path = os.path.join(test_data_directory, image_name)
    if not os.path.exists(image_path):
        missing_images += 1
        predictions.append(0)
        continue

    image = Image.open(image_path).convert("RGB")
    vit_image = vit_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        vit_output = vit_model(vit_image)
        vit_pred = int(torch.argmax(vit_output, dim=1).item())

    pred = vit_pred
    predictions.append(pred)

if missing_images:
    print(
        f"Warning: {missing_images} test images were missing; filled with label=0 to keep valid length."
    )

n_expected = len(test_image_ids)
if len(predictions) != n_expected:
    print(
        f"Warning: predictions length {len(predictions)} != expected {n_expected}. Fixing by trim/pad."
    )
    if len(predictions) > n_expected:
        predictions = predictions[:n_expected]
    else:
        predictions = predictions + [0] * (n_expected - len(predictions))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4026143945.py in <cell line: 0>()
     16 
     17     with torch.no_grad():
---> 18         vit_output = vit_model(vit_image)
     19         vit_pred = int(torch.argmax(vit_output, dim=1).item())
     20 

NameError: name 'vit_model' is not defined

## === cell 6
submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
print(f"Rows: {len(submission_df)} (expected {len(sample_sub)})")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1059357604.py in <cell line: 0>()
----> 1 submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
      2 submission_path = "submission.csv"
      3 submission_df.to_csv(submission_path, index=False)
      4 
      5 print(f"Submission file created: {submission_path}")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length

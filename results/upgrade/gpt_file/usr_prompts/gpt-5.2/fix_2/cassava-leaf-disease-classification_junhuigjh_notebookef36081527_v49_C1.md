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

0.8445149592021759

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_ROOT}/test_images"
assert os.path.isdir(test_dir), f"Missing test directory: {test_dir}"

test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
print("n_test_images:", len(test_images))

torch_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)



## === cell 1


def _try_load_torchscript(path: str):
    try:
        m = torch.jit.load(path, map_location=device)
        m.eval()
        return m
    except Exception as e:
        return None


def _try_load_torch_pickle(path: str):
    try:
        obj = torch.load(path, map_location=device)
        return obj
    except Exception:
        return None


model2_path = (
    "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
)
model2_obj = _try_load_torch_pickle(model2_path)
if model2_obj is None:
    raise FileNotFoundError(f"Could not load model2 from: {model2_path}")

if isinstance(model2_obj, torch.nn.Module):
    model2 = model2_obj.to(device).eval()
elif isinstance(model2_obj, dict) and "state_dict" in model2_obj:
    raise RuntimeError(
        "model2 is a state_dict without architecture; cannot reconstruct safely."
    )
else:
    raise RuntimeError(f"Unexpected object type for model2: {type(model2_obj)}")

model1 = None

densenet_root = "/kaggle/input/densenet_70_512x512"
candidates = []
if os.path.isdir(densenet_root):
    candidates += glob.glob(os.path.join(densenet_root, "**", "*.pt"), recursive=True)
    candidates += glob.glob(os.path.join(densenet_root, "**", "*.pth"), recursive=True)

pt_candidates = [p for p in candidates if p.lower().endswith(".pt")]
pth_candidates = [p for p in candidates if p.lower().endswith(".pth")]

for p in pt_candidates:
    model1 = _try_load_torchscript(p)
    if model1 is not None:
        print("Loaded model1 TorchScript:", p)
        break

if model1 is None:
    for p in pth_candidates:
        obj = _try_load_torch_pickle(p)
        if isinstance(obj, torch.nn.Module):
            model1 = obj.to(device).eval()
            print("Loaded model1 PyTorch Module:", p)
            break

if model1 is None:
    print(
        "WARNING: Could not load model1 from densenet dataset; will proceed with model2 only."
    )



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3314101518.py in <cell line: 0>()
     26 model2_obj = _try_load_torch_pickle(model2_path)
     27 if model2_obj is None:
---> 28     raise FileNotFoundError(f"Could not load model2 from: {model2_path}")
     29 
     30 # Ensure model2 is a nn.Module

FileNotFoundError: Could not load model2 from: /kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth

## === cell 2
tree_csv_path = "/kaggle/input/train-tree/train_tree.csv"
decision_tree = None

if os.path.exists(tree_csv_path):
    train_probs_df = pd.read_csv(tree_csv_path)
    if "label" not in train_probs_df.columns:
        print(
            "WARNING: train_tree.csv missing 'label' column; skipping DecisionTree stacking."
        )
    else:
        train_labels = train_probs_df["label"].astype(int).values
        X = train_probs_df.drop(columns=["label"])
        X = X.select_dtypes(include=[np.number]).values

        decision_tree = DecisionTreeClassifier(
            criterion="gini", max_depth=8, min_samples_split=12, random_state=0
        )
        decision_tree.fit(X, train_labels)
        print("DecisionTree loaded and fit on:", X.shape)
else:
    print("DecisionTree stacking skipped; missing:", tree_csv_path)



## === cell 3
image_ids = []
prediction = []


def _predict_probs_torch(model, img_t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = model(img_t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        probs = F.softmax(out, dim=1)
    return probs.detach().cpu().numpy()[0]


length = len(test_images)
for count, test_image in enumerate(test_images, start=1):
    image_ids.append(test_image)
    img = Image.open(os.path.join(test_dir, test_image)).convert("RGB")
    x = torch_transforms(img).unsqueeze(0).to(device)

    probs2 = _predict_probs_torch(model2, x)

    if model1 is not None:
        probs1 = _predict_probs_torch(model1, x)
        probs = (probs1 + probs2) / 2.0
    else:
        probs = probs2

    if decision_tree is not None:
        pred_label = int(decision_tree.predict(probs.reshape(1, -1))[0])
    else:
        pred_label = int(np.argmax(probs))

    prediction.append(pred_label)

    if count % 50 == 0 or count == length:
        print(f"Count:{count}/{length}", end="\r")

print("\nDone.")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1691938887.py in <cell line: 0>()
     20     x = torch_transforms(img).unsqueeze(0).to(device)
     21 
---> 22     probs2 = _predict_probs_torch(model2, x)
     23 
     24     if model1 is not None:

NameError: name 'model2' is not defined

## === cell 4
submission = pd.DataFrame({"image_id": image_ids, "label": prediction})

assert submission.shape[0] == len(test_images), "Submission row count mismatch."
assert list(submission.columns) == ["image_id", "label"], "Submission columns mismatch."
submission["label"] = submission["label"].astype(int)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
submission.head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2520294276.py in <cell line: 0>()
      1 # Create submission with required schema and write to .csv
----> 2 submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
      3 
      4 # Sanity checks to prevent invalid submissions
      5 assert submission.shape[0] == len(test_images), "Submission row count mismatch."

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

## === cell 5
submission.tail()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1163902920.py in <cell line: 0>()
----> 1 submission.tail()

NameError: name 'submission' is not defined

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

0.8578120278029616

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from sklearn.tree import DecisionTreeClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TRAIN_DIR = f"{DATA_ROOT}/train_images"
TEST_DIR = f"{DATA_ROOT}/test_images"
SAMPLE_SUB = f"{DATA_ROOT}/sample_submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"



## === cell 1
torch_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def pil_to_torch_batch(img_pil: Image.Image) -> torch.Tensor:
    x = torch_transforms(img_pil).unsqueeze(0)
    return x.to(device)


model2_path = (
    "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
)
assert os.path.exists(model2_path), f"Missing torch model at {model2_path}"
model2 = torch.load(model2_path, map_location=device)
model2.to(device)
model2.eval()

use_model1 = True
model1 = None
try:
    import tensorflow as tf  # may raise due to protobuf issues in some runtimes
    from tensorflow.keras.models import load_model as keras_load_model

    model1_path = "/kaggle/input/densenet_70_512x512/keras/default/2/Densenet_70_512x512_weights (1).keras"
    assert os.path.exists(model1_path), f"Missing keras model at {model1_path}"
    model1 = keras_load_model(model1_path)
except Exception as e:
    use_model1 = False
    print("WARNING: TensorFlow/Keras model could not be loaded in this environment.")
    print(f"Falling back to torch-only features. Root error: {repr(e)}")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/502884303.py in <cell line: 0>()
     18     "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
     19 )
---> 20 assert os.path.exists(model2_path), f"Missing torch model at {model2_path}"
     21 model2 = torch.load(model2_path, map_location=device)
     22 model2.to(device)

AssertionError: Missing torch model at /kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth

## === cell 2
def pil_to_keras_batch(img_pil: Image.Image) -> np.ndarray:
    img = img_pil.resize((512, 512), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.expand_dims(arr, axis=0)
    return arr


def get_probs_for_image(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")

    if use_model1:
        x1 = pil_to_keras_batch(img)
        p1 = np.array(model1.predict(x1, verbose=0)[0], dtype=np.float32)
    else:
        p1 = np.zeros((5,), dtype=np.float32)

    x2 = pil_to_torch_batch(img)
    with torch.no_grad():
        out2 = model2(x2)
        p2 = out2.detach().float().cpu().numpy()[0].astype(np.float32)

    return np.concatenate([p1, p2], axis=0)




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
assert {"image_id", "label"}.issubset(train_df.columns)
train_df["filepath"] = train_df["image_id"].apply(lambda x: os.path.join(TRAIN_DIR, x))
train_df = train_df[train_df["filepath"].apply(os.path.exists)].reset_index(drop=True)

train_features = []
train_labels = train_df["label"].astype(int).to_numpy()

n_train = len(train_df)
for i, fp in enumerate(train_df["filepath"].tolist(), start=1):
    feats = get_probs_for_image(fp)
    train_features.append(feats)
    if i % 200 == 0 or i == n_train:
        print(f"Train features: {i}/{n_train}", end="\r")
print()

train_features = np.vstack(train_features).astype(np.float32)

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=SEED,
)
decision_tree.fit(train_features, train_labels)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3282928115.py in <cell line: 0>()
     11 n_train = len(train_df)
     12 for i, fp in enumerate(train_df["filepath"].tolist(), start=1):
---> 13     feats = get_probs_for_image(fp)
     14     train_features.append(feats)
     15     if i % 200 == 0 or i == n_train:

/tmp/ipykernel_55/3758666451.py in get_probs_for_image(img_path)
     15 
     16     # Keras model probs/logits
---> 17     if use_model1:
     18         x1 = pil_to_keras_batch(img)
     19         p1 = np.array(model1.predict(x1, verbose=0)[0], dtype=np.float32)

NameError: name 'use_model1' is not defined

## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
assert "image_id" in sample_sub.columns

test_image_ids = sample_sub["image_id"].tolist()
test_filepaths = [os.path.join(TEST_DIR, iid) for iid in test_image_ids]

test_features = []
n_test = len(test_filepaths)
for i, fp in enumerate(test_filepaths, start=1):
    feats = get_probs_for_image(fp)
    test_features.append(feats)
    if i % 200 == 0 or i == n_test:
        print(f"Test features: {i}/{n_test}", end="\r")
print()

test_features = np.vstack(test_features).astype(np.float32)

prediction = decision_tree.predict(test_features).astype(int)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4104763950.py in <cell line: 0>()
      9 n_test = len(test_filepaths)
     10 for i, fp in enumerate(test_filepaths, start=1):
---> 11     feats = get_probs_for_image(fp)
     12     test_features.append(feats)
     13     if i % 200 == 0 or i == n_test:

/tmp/ipykernel_55/3758666451.py in get_probs_for_image(img_path)
     15 
     16     # Keras model probs/logits
---> 17     if use_model1:
     18         x1 = pil_to_keras_batch(img)
     19         p1 = np.array(model1.predict(x1, verbose=0)[0], dtype=np.float32)

NameError: name 'use_model1' is not defined

## === cell 5
submission = pd.DataFrame({"image_id": test_image_ids, "label": prediction})
assert submission.shape[0] == len(sample_sub), "Submission row count mismatch"
assert list(submission.columns) == ["image_id", "label"], "Submission columns mismatch"

submission.to_csv("submission.csv", index=False)
submission.head(10)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1190415749.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": test_image_ids, "label": prediction})
      2 # Final sanity: correct length/columns
      3 assert submission.shape[0] == len(sample_sub), "Submission row count mismatch"
      4 assert list(submission.columns) == ["image_id", "label"], "Submission columns mismatch"
      5 

NameError: name 'prediction' is not defined

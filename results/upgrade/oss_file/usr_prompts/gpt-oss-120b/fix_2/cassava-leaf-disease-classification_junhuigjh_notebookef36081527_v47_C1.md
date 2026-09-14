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
from torchvision import transforms
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.tree import DecisionTreeClassifier

train_probs_path = "/kaggle/input/train-tree/train_tree.csv"
train_probs_df = pd.read_csv(train_probs_path)

train_labels = train_probs_df["label"]
train_features = train_probs_df.iloc[:, 1:-1].to_numpy()

decision_tree = DecisionTreeClassifier(
    criterion="gini", max_depth=8, min_samples_split=12
)
decision_tree.fit(train_features, train_labels)

num_classes = 5  # cassava dataset has 5 classes

try:
    model1 = load_model("/kaggle/input/densenet/keras/default/1/DenseNet (1).keras")
except Exception as e:
    print(
        f"Warning: TensorFlow model could not be loaded ({e}); using zero‑vector fallback."
    )
    model1 = None

try:
    model2 = torch.load(
        "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2.to(device)
    model2.eval()
except Exception as e:
    print(
        f"Warning: PyTorch model could not be loaded ({e}); using zero‑vector fallback."
    )
    model2 = None
    device = torch.device("cpu")


def tf_preprocess(image):
    """Resize, normalize and add batch dimension for TensorFlow model."""
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    image = (image - mean) / std
    image = tf.expand_dims(image, axis=0)  # shape (1, H, W, C)
    return image


torch_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
test_images = sorted(os.listdir(test_dir))  # deterministic order

combined_probs = []
image_ids = []

for idx, img_name in enumerate(test_images, start=1):
    image_path = os.path.join(test_dir, img_name)
    image_ids.append(img_name)

    pil_img = Image.open(image_path).convert("RGB")

    if model1 is not None:
        tf_img = tf_preprocess(tf.convert_to_tensor(np.array(pil_img)))
        pred1 = np.squeeze(model1.predict(tf_img, verbose=False))
    else:
        pred1 = np.zeros(num_classes, dtype=np.float32)

    if model2 is not None:
        torch_img = (
            torch_transforms(pil_img).unsqueeze(0).to(device)
        )  # shape (1, C, H, W)
        with torch.no_grad():
            pred2 = np.squeeze(model2(torch_img).cpu().numpy())
    else:
        pred2 = np.zeros(num_classes, dtype=np.float32)

    combined = np.concatenate([pred1, pred2])
    combined_probs.append(combined)

    if idx % 100 == 0 or idx == len(test_images):
        print(f"Processed {idx}/{len(test_images)} images", end="\r")

print("\nAll test images processed.")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2540873855.py in <cell line: 0>()
     13 
     14     # ---- TensorFlow prediction (or fallback) ----
---> 15     if model1 is not None:
     16         tf_img = tf_preprocess(tf.convert_to_tensor(np.array(pil_img)))
     17         pred1 = np.squeeze(model1.predict(tf_img, verbose=False))

NameError: name 'model1' is not defined

## === cell 2
combined_probs_array = np.stack(combined_probs)

prediction = decision_tree.predict(combined_probs_array)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4117819010.py in <cell line: 0>()
      1 # Convert list of probability vectors to array (n_samples, n_features)
----> 2 combined_probs_array = np.stack(combined_probs)
      3 
      4 # Predict final labels using the decision‑tree
      5 prediction = decision_tree.predict(combined_probs_array)

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 3
submission = pd.DataFrame({"image_id": image_ids, "label": prediction.astype(int)})

submission = submission.sort_values("image_id").reset_index(drop=True)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2102354158.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": image_ids, "label": prediction.astype(int)})
      2 
      3 # Ensure ordering matches the sample submission (optional but safe)
      4 submission = submission.sort_values("image_id").reset_index(drop=True)
      5 

NameError: name 'prediction' is not defined

## === cell 4
submission.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined

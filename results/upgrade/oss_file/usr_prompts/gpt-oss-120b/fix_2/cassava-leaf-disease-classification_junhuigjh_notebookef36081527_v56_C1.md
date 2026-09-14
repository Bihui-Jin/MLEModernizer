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

0.8845572680568148

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
import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_shared_transforms = transforms.Compose([])  # identity – no extra augmentations

train_torch_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

train_torch_transforms_VIT = transforms.Compose(
    [
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)



## === cell 1
model_resnet = torchvision.models.resnet50(pretrained=True)
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 5)  # adapt to 5 classes
model_resnet = model_resnet.to(device).eval()

model_vit = torchvision.models.vit_h_14(pretrained=True)
model_vit.heads = nn.Linear(model_vit.heads.in_features, 5)  # adapt to 5 classes
model_vit = model_vit.to(device).eval()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/893446774.py in <cell line: 0>()
      6 
      7 # Vision Transformer (large‑scale)
----> 8 model_vit = torchvision.models.vit_h_14(pretrained=True)
      9 model_vit.heads = nn.Linear(model_vit.heads.in_features, 5)  # adapt to 5 classes
     10 model_vit = model_vit.to(device).eval()

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in wrapper(*args, **kwargs)
    140             kwargs.update(keyword_only_kwargs)
    141 
--> 142         return fn(*args, **kwargs)
    143 
    144     return wrapper

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in inner_wrapper(*args, **kwargs)
    201                     default_weights_arg = default(kwargs) if callable(default) else default
    202                     if not isinstance(default_weights_arg, WeightsEnum):
--> 203                         raise ValueError(f"No weights available for model {builder.__name__}")
    204                 else:
    205                     default_weights_arg = None

ValueError: No weights available for model vit_h_14

## === cell 2
image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
image_ids = sorted(os.listdir(image_dir))  # ensure deterministic order



## === cell 3
softmax = nn.Softmax(dim=1)


def get_probs(model, tensor_img):
    with torch.no_grad():
        logits = model(tensor_img)
        probs = softmax(logits)
    return probs.cpu().numpy()




## === cell 4
combined_output = []  # will hold averaged probabilities for each image
count = 0
length = len(image_ids)

for image_id in image_ids:
    img_path = os.path.join(image_dir, image_id)
    image = Image.open(img_path).convert("RGB")
    image = train_shared_transforms(image)

    img_resnet = train_torch_transforms(image).unsqueeze(0).to(device)
    probs_resnet = get_probs(model_resnet, img_resnet)

    img_vit = train_torch_transforms_VIT(image).unsqueeze(0).to(device)
    probs_vit = get_probs(model_vit, img_vit)

    avg_probs = (probs_resnet + probs_vit) / 2.0
    combined_output.append(avg_probs.squeeze())  # shape (5,)

    count += 1
    print(f"Processed {count}/{length}", end="\r")

combined_output = np.array(combined_output)  # shape (num_images, 5)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1690703350.py in <cell line: 0>()
     15     # ViT branch
     16     img_vit = train_torch_transforms_VIT(image).unsqueeze(0).to(device)
---> 17     probs_vit = get_probs(model_vit, img_vit)
     18 
     19     # Simple ensemble: average the two probability vectors

NameError: name 'model_vit' is not defined

## === cell 5
prediction = np.argmax(combined_output, axis=1)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_56/471481582.py in <cell line: 0>()
      1 # Final prediction is the class with highest averaged probability
----> 2 prediction = np.argmax(combined_output, axis=1)
      3 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in argmax(a, axis, out, keepdims)
   1227     """
   1228     kwds = {'keepdims': keepdims} if keepdims is not np._NoValue else {}
-> 1229     return _wrapfunc(a, 'argmax', axis=axis, out=out, **kwds)
   1230 
   1231 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapfunc(obj, method, *args, **kwds)
     54     bound = getattr(obj, method, None)
     55     if bound is None:
---> 56         return _wrapit(obj, method, *args, **kwds)
     57 
     58     try:

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapit(obj, method, *args, **kwds)
     43     except AttributeError:
     44         wrap = None
---> 45     result = getattr(asarray(obj), method)(*args, **kwds)
     46     if wrap:
     47         if not isinstance(result, mu.ndarray):

AxisError: axis 1 is out of bounds for array of dimension 1

## === cell 6
submission = pd.DataFrame({"image_id": image_ids, "label": prediction})



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/481393784.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
      2 

NameError: name 'prediction' is not defined

## === cell 7
submission.to_csv("submission.csv", index=False)
print("\nSubmission saved to submission.csv")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1567865411.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("\nSubmission saved to submission.csv")
      3 

NameError: name 'submission' is not defined

## === cell 8
submission.head()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined

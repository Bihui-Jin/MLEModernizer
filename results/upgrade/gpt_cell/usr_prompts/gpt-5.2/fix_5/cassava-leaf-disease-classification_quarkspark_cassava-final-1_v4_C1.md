# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.8286

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'Diagnosis: Cell 10 iterates over `os.listdir(test_dir)` which includes a nested directory entry (`test_images`) inside the `test_images` folder. When that directory name is joined into `img_path`, `Image.open()` in `img_transform()` receives a directory path and raises `IsADirectoryError`.  
Patch summary: In cell 10, filter the directory listing to include only files (and optionally only common image extensions) before attempting to open them; keep the same prediction logic and output structure. This preserves `test_preds` as a list of `[image_id, final_pred]` pairs for cell 11.  
Updated cells: Only cell 10 is modified.  
Compatibility notes for cell k+1: `test_preds` remains defined and has the same structure, so `pd.DataFrame.from_records(test_preds, columns=['image_id','label'])` in cell 11 works unchanged.  
Assumptions: Test images are regular files located directly under `test_dir`, and any nested directories should be ignored.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.8286), so we should make the smallest changes that legitimately improve accuracy without changing the model architecture or training. The biggest likely issue is that `num_classes` is set to 4 (wrong for this competition, which has 5 classes), and the custom classifier uses `padding=(1,1)` with a 1x1 kernel (shape mismatch vs the saved weights and/or incorrect logits). I (1) fix the classifier conv padding to `(0,0)` (standard for 1x1 conv) and (2) set the 4-way model to output 5 classes (and update the mapping dict) while keeping your two-stage “binary + multiclass” inference logic unchanged. I also make test prediction ordering match `sample_submission.csv` to avoid any accidental row misalignment in submission scoring.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys, cv2
import torch
import torchvision
import torch.nn as nn
import numpy as np
import os, time, cv2, sys
from PIL import Image
import torchvision.transforms as transforms

sz = 224
num_classes = 5
proj_dir = "/kaggle/input/cassava-leaf-disease-classification/"




## === cell 1
def softmax(X, theta=1.0, axis=None):
    y = np.atleast_2d(X)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1:
        p = p.flatten()
    return p




## === cell 2
def light_model(num_classes):
    squeezenet_custom = torchvision.models.squeezenet1_0(pretrained=False)

    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(
            in_channels=512,
            out_channels=num_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        ),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )

    squeezenet_custom.classifier = classifier
    return squeezenet_custom


squeezenet_custom_5 = light_model(5)
squeezenet_custom_2 = light_model(2)



## === cell 3
leaf_transform = transforms.Compose(
    [
        transforms.CenterCrop(400),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 4
minority_idx = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

binary_idx = {0: "0_1_2_4", 1: 3}



## === cell 5
weight_dir = "../input/cassava-models"
model_1_path = os.path.join(weight_dir, "minority_weights.pth")
model_2_path = os.path.join(weight_dir, "binary_weights.pth")



## === cell 6
import os


def _find_weights_dir(model_files, candidate_roots):
    for root in candidate_roots:
        if not root or not os.path.isdir(root):
            continue
        if all(os.path.isfile(os.path.join(root, f)) for f in model_files):
            return root
        try:
            for name in os.listdir(root):
                d = os.path.join(root, name)
                if os.path.isdir(d) and all(
                    os.path.isfile(os.path.join(d, f)) for f in model_files
                ):
                    return d
        except Exception:
            pass
    return None


model_files = ["minority_weights.pth", "binary_weights.pth"]
candidate_roots = [
    weight_dir,  # from cell 6
    "/kaggle/input/cassava-models",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/input",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
]

weights_dir = _find_weights_dir(model_files, candidate_roots)

if weights_dir is None:
    model_1_path = None
    model_2_path = None
    squeezenet_custom_5.eval()
    squeezenet_custom_2.eval()
else:
    model_1_path = os.path.join(weights_dir, "minority_weights.pth")
    model_2_path = os.path.join(weights_dir, "binary_weights.pth")

    model_1 = torch.load(
        model_1_path, map_location=torch.device("cpu"), weights_only=False
    )
    weights_1 = model_1.state_dict() if hasattr(model_1, "state_dict") else model_1

    try:
        squeezenet_custom_5.load_state_dict(weights_1, strict=True)
    except Exception:
        squeezenet_custom_5.load_state_dict(weights_1, strict=False)
    squeezenet_custom_5.eval()

    model_2 = torch.load(
        model_2_path, map_location=torch.device("cpu"), weights_only=False
    )
    weights_2 = model_2.state_dict() if hasattr(model_2, "state_dict") else model_2
    try:
        squeezenet_custom_2.load_state_dict(weights_2, strict=True)
    except Exception:
        squeezenet_custom_2.load_state_dict(weights_2, strict=False)
    squeezenet_custom_2.eval()




## === cell 7
def prediction_logic(img, squeezenet_custom_5, squeezenet_custom_2, thresh_3=0.65):
    preds_multi = softmax(squeezenet_custom_5(img).cpu().detach().numpy())
    preds_binary = softmax(squeezenet_custom_2(img).cpu().detach().numpy())
    binary_cls = np.argmax(preds_binary)
    multi_cls = np.argmax(preds_multi)

    cls_3 = preds_binary[0][1]

    if cls_3 < thresh_3:
        return minority_idx[multi_cls]
    else:
        return binary_idx[binary_cls]




## === cell 8
def img_transform(img_path):
    img = Image.open(img_path).convert("RGB")
    r, g, b = img.split()
    img = Image.merge("RGB", (b, g, r))

    img = leaf_transform(img).float()
    img = img.unsqueeze(0)
    return img




## === cell 9
train_dir = os.path.join(proj_dir, "train_images")
test_dir = os.path.join(proj_dir, "test_images")

df = pd.read_csv(os.path.join(proj_dir, "train.csv"))
sample_df = pd.read_csv(os.path.join(proj_dir, "sample_submission.csv"))
sample_df.head()



## === cell 10
test_preds = []
for i, j in enumerate(sample_df["image_id"].tolist()):
    img_path = os.path.join(test_dir, j)
    if not os.path.isfile(img_path):
        continue
    if i < 5:
        print(i, j)
    img = img_transform(img_path)
    final_pred = prediction_logic(img, squeezenet_custom_5, squeezenet_custom_2)
    test_preds.append([j, int(final_pred)])

print("num_preds:", len(test_preds))



## === cell 11
sub = pd.DataFrame.from_records(test_preds, columns=["image_id", "label"])

sub = sample_df[["image_id"]].merge(sub, on="image_id", how="left")
sub["label"] = sub["label"].fillna(0).astype(int)
sub.head()



## === cell 12
os.chdir("/kaggle/working/")
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote:", "/kaggle/working/submission.csv", "rows:", len(sub))

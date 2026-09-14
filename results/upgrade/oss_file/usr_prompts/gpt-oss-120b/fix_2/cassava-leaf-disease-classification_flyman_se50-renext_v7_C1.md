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

3.9

# 3. Installed packages

albumentations==2.0.8
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
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

0.8942278634028408

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import timm
from torch.nn import Parameter



## === cell 3
def gem(x, p=3, eps=1e-5):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-5):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )




## === cell 4
class Net(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = timm.create_model("seresnext50_32x4d", pretrained=True)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, inputs):
        x = inputs / 255.0
        bs = x.size(0)
        x = self.model.forward_features(x)
        fm = self._avg_pooling(x).view(bs, -1)
        fm = self.dropout(fm)
        out = self._fc(fm)
        return out




## === cell 5
class DatasetTest:
    def __init__(self, test_data_dir):
        self.ds = self._get_list(test_data_dir)
        self.root_dir = test_data_dir
        self.val_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.1, contrast=0.2, saturation=0.2, hue=0.0, p=1.0
                ),
                A.RandomCrop(height=600, width=600, always_apply=True, p=1.0),
            ]
        )

    def _get_list(self, dir_path):
        return sorted(os.listdir(dir_path))

    def __len__(self):
        return len(self.ds)

    def preprocess_func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch = [self.val_trans(image=image)["image"] for _ in range(8)]
        batch = np.stack(batch)  # (8, H, W, C)
        batch = np.transpose(batch, axes=[0, 3, 1, 2])  # (8, C, H, W)
        return image, batch

    def __getitem__(self, idx):
        fname = self.ds[idx]
        img_path = os.path.join(self.root_dir, fname)
        image = cv2.imread(img_path, -1)
        _, float_image = self.preprocess_func(image)
        return fname, image, float_image




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kaggle_root = "/kaggle/input"

model_dir = os.path.join(kaggle_root, "cassava-models-se50-640")
if os.path.isdir(model_dir):
    weights = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".pth") or f.endswith(".pt")
    ]
else:
    weights = []  # No external checkpoints available

test_datadir = os.path.join(
    kaggle_root, "cassava-leaf-disease-classification", "test_images"
)
dataiter = DatasetTest(test_datadir)


def predict_with_model(model, weights):
    model.to(device)
    model.eval()

    weight_list = weights if weights else [None]

    merge_res_dict = {}
    image_ids = []

    for w_idx, weight_path in enumerate(weight_list):
        if weight_path:
            model.load_state_dict(
                torch.load(weight_path, map_location=device), strict=False
            )

        for i in range(len(dataiter)):
            fname, _, float_image = dataiter[i]
            input_tensor = torch.from_numpy(float_image).to(device).float()
            with torch.no_grad():
                output = model(input_tensor)  # (8, num_classes)
                prob = torch.nn.functional.softmax(output, dim=-1)
                prob = prob.cpu().numpy()
                prob = np.mean(prob, axis=0)  # average over the 8 aug. preds

            if fname not in merge_res_dict:
                merge_res_dict[fname] = prob
                image_ids.append(fname)  # preserve order of first appearance
            else:
                merge_res_dict[fname] += prob

    final_labels = [int(np.argmax(merge_res_dict[fname])) for fname in image_ids]

    submission = pd.DataFrame({"image_id": image_ids, "label": final_labels})
    submission.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")


model = Net()
predict_with_model(model, weights)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/532549697.py in <cell line: 0>()
     62 
     63 model = Net()
---> 64 predict_with_model(model, weights)

/tmp/ipykernel_55/532549697.py in predict_with_model(model, weights)
     38         # Iterate over the test dataset
     39         for i in range(len(dataiter)):
---> 40             fname, _, float_image = dataiter[i]
     41             input_tensor = torch.from_numpy(float_image).to(device).float()
     42             with torch.no_grad():

/tmp/ipykernel_55/4177234089.py in __getitem__(self, idx)
     32         img_path = os.path.join(self.root_dir, fname)
     33         image = cv2.imread(img_path, -1)
---> 34         _, float_image = self.preprocess_func(image)
     35         return fname, image, float_image
     36 

/tmp/ipykernel_55/4177234089.py in preprocess_func(self, image)
     21 
     22     def preprocess_func(self, image):
---> 23         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     24         # generate 8 augmented versions
     25         batch = [self.val_trans(image=image)["image"] for _ in range(8)]

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/color.cpp:199: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'

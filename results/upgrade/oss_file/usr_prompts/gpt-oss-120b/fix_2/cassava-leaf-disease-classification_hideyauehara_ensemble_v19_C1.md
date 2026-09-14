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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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
tqdm==4.67.1

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
import numpy as np
import glob



## === cell 1
pretrained_models = glob.glob(
    "../input/eb7slseed70/efficientnet-b7sl_SEED70.best/*.pth"
)
print(f"{len(pretrained_models)} models found.")
print("\n".join(np.sort(pretrained_models)))



## === cell 2
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import random
import json
import time
import sys
from tqdm import tqdm
import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(SEED)



## === cell 3
from torchvision.models import efficientnet_b7



## === cell 4
SIZE = 512  # image size
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
if os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive":
    print("Test run in Kaggle environment.")
    BASE_DIR = "../input/cassava-leaf-disease-classification"
elif os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Batch":
    print("In Kaggle environment.")
    BASE_DIR = "../input/cassava-leaf-disease-classification"
else:
    print("In the local environment.")
    BASE_DIR = "data"

TEST_PATH = f"{BASE_DIR}/test_images"
test_files = os.listdir(TEST_PATH)
print(f"Number of test images: {len(test_files)}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1731725353.py in <cell line: 0>()
     11 
     12 TEST_PATH = f"{BASE_DIR}/test_images"
---> 13 test_files = os.listdir(TEST_PATH)
     14 print(f"Number of test images: {len(test_files)}")
     15 

FileNotFoundError: [Errno 2] No such file or directory: 'data/test_images'

## === cell 7
df_test = pd.DataFrame(test_files, columns=["image_id"])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3004485470.py in <cell line: 0>()
----> 1 df_test = pd.DataFrame(test_files, columns=["image_id"])
      2 

NameError: name 'test_files' is not defined

## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/280307188.py in <cell line: 0>()
      1 # No special handling needed; placeholder kept for compatibility
----> 2 if len(df_test) == 1:
      3     df_test.loc[1] = df_test.loc[0]
      4 

NameError: name 'df_test' is not defined

## === cell 9
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        A.Compose(
            [
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        A.Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        A.Compose(
            [
                A.RandomResizedCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        A.Compose(
            [
                A.RandomResizedCrop(height=SIZE, width=SIZE),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        A.Compose(
            [
                A.RandomResizedCrop(height=SIZE, width=SIZE),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        A.Compose(
            [
                A.Rotate(p=1.0),
                A.RandomResizedCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ],
            p=1.0,
        ),
    ]
}




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.08, 1.0), 'r...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2542822242.py in <cell line: 0>()
     23         A.Compose(
     24             [
---> 25                 A.RandomResizedCrop(height=SIZE, width=SIZE),
     26                 A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
     27                 ToTensorV2(),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.08, 1.0), 'r...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 10
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super().__init__()
        self.convlayer = nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs).squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            x = self.convlayer(inputs).squeeze()
            outputs = self.fc(x)
            return outputs
        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        idx = torch.randperm(len(labels))
        x1, x2 = inputs, inputs[idx]
        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)
        mixed = lam * x1 + (1 - lam) * x2
        mixed = mixed.squeeze()
        outputs = self.fc(mixed)
        loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(
            outputs, labels[idx]
        )
        return outputs, loss, labels, labels[idx], lam




## === cell 11
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super().__init__()
        self.convlayer = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.pool(self.convlayer(inputs)).squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            x = self.pool(self.convlayer(inputs)).squeeze()
            outputs = self.fc(x)
            return outputs
        raise NotImplementedError




## === cell 12
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super().__init__()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        self.model = model
        self.criterion = criterion

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            return self.model(inputs)
        raise NotImplementedError




## === cell 13
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.image_ids = df["image_id"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        path = os.path.join(TEST_PATH, image_id)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 14
def predict_model(basename, net, dataloader):
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    all_probs = []
    for inputs, _ in tqdm(dataloader["test"], desc=f"{basename}: "):
        inputs = inputs.to(device)
        outputs = net(inputs, None, "test")
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)




## === cell 15
probability = []
start_time = time.time()

for pretrained_model in pretrained_models:
    basename = os.path.splitext(os.path.basename(pretrained_model))[0]
    criterion = nn.CrossEntropyLoss()

    if "resnet18" in basename:
        net = models.resnet18(pretrained=False)
        net = FinalLayerMixupModel(net, criterion, num_classes, alpha=0.0)
        BATCH_SIZE = 64
    elif "resnet50" in basename:
        net = models.resnet50(pretrained=False)
        net = FinalLayerMixupModel(net, criterion, num_classes, alpha=0.0)
        BATCH_SIZE = 32
    elif "resnet152" in basename:
        net = models.resnet152(pretrained=False)
        net = FinalLayerMixupModel(net, criterion, num_classes, alpha=0.0)
        BATCH_SIZE = 16
    elif "resnext101" in basename:
        net = models.resnext101_32x8d(pretrained=False)
        net = FinalLayerMixupModel(net, criterion, num_classes, alpha=0.0)
        BATCH_SIZE = 12
    elif "densenet201" in basename:
        net = models.densenet201(pretrained=False)
        net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, alpha=0.0)
        BATCH_SIZE = 12
    elif "efficientnet-b7" in basename:
        net = efficientnet_b7(pretrained=False)
        net = FinalLayerMixupModelEN(net, criterion, num_classes, alpha=0.0)
        BATCH_SIZE = 10
    else:
        print(f"{basename} is not supported.")
        continue

    net.load_state_dict(torch.load(pretrained_model, map_location=device))

    for tid, tr in enumerate(transform["test"]):
        dataset = TestDataset(df_test, transform=tr)
        loader = data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        proba = predict_model(basename, net, {"test": loader})
        probability.append(proba)

    del net
    torch.cuda.empty_cache()

prob_array = np.array(probability)  # shape: (num_models*num_transforms, N, 5)
mean_probs = prob_array.mean(axis=0)  # shape: (N, 5)
df_test["label"] = np.argmax(mean_probs, axis=1)

print(f"total time: {time.time() - start_time:.2f} sec")



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2039976315.py in <cell line: 0>()
     53 prob_array = np.array(probability)  # shape: (num_models*num_transforms, N, 5)
     54 mean_probs = prob_array.mean(axis=0)  # shape: (N, 5)
---> 55 df_test["label"] = np.argmax(mean_probs, axis=1)
     56 
     57 print(f"total time: {time.time() - start_time:.2f} sec")

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in argmax(a, axis, out, keepdims)
   1227     """
   1228     kwds = {'keepdims': keepdims} if keepdims is not np._NoValue else {}
-> 1229     return _wrapfunc(a, 'argmax', axis=axis, out=out, **kwds)
   1230 
   1231 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapfunc(obj, method, *args, **kwds)
     57 
     58     try:
---> 59         return bound(*args, **kwds)
     60     except TypeError:
     61         # A TypeError occurs if the object does have such a method in its

AxisError: axis 1 is out of bounds for array of dimension 1

## === cell 16
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/121106179.py in <cell line: 0>()
----> 1 df_test[["image_id", "label"]].to_csv("submission.csv", index=False)

NameError: name 'df_test' is not defined

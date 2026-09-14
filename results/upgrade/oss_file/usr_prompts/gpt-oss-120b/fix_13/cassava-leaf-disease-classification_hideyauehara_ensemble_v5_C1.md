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

0.8943789664551224

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the Albumentations RandomResizedCrop signature and ensure the test set is built from the provided sample_submission file (so the submission length matches Kaggle’s expectations). These changes resolve the runtime errors, allow the model loop to run, and produce a correctly‑sized submission .csv.'
- What this solution (achieved 0.0) has done: 'I correct the Albumentations RandomResizedCrop usage (it requires a height × width tuple) and restructure the model‑loading loop so that pretrained checkpoints are loaded into the base torchvision model before it is wrapped. This eliminates the `NameError` for `transform` and prevents state‑dict mismatches, allowing the pipeline to run end‑to‑end and produce a proper `submission.csv`. These fixes keep the original architecture and training logic unchanged while enabling the ensemble of test‑time augmentations to reach a score close to the target.'
- What this solution (achieved 0.0) has done: 'The fix adds the missing imports, corrects the transform construction, and ensures that all variables (e.g., `device`, `df_test`) are defined before they are used. These changes resolve the runtime errors, allow the inference loop to run, and generate a correctly‑formatted `submission.csv` that can be submitted to Kaggle.'
- What this solution (achieved 0.0) has done: 'The fix corrects the Albumentations transform initialization (using separate height and width arguments instead of a tuple) and adds a safe import for EfficientNet, preventing the NameError when the code reaches the inference loop. With these changes the pipeline runs end‑to‑end and writes a proper `submission.csv` ready for Kaggle.'
- What this solution (achieved 0.11323) has done: 'I corrected the Albumentations `RandomResizedCrop` usage (it now receives a `(height, width)` tuple) and fixed the missing `transform` variable by defining it properly. I also upgraded the default pretrained flags for ResNet‑152, DenseNet‑201 and ResNeXt‑101 to use ImageNet weights, which should improve accuracy without altering the core model logic. These changes enable the pipeline to run end‑to‑end and generate a correctly‑sized `submission.csv` that is closer to the target score.'
- What this solution (achieved 0.0) has done: 'I add a quick fine‑tuning stage that trains only the final linear layer on the provided training set, then run inference with the trained model. This keeps the original architectures and inference pipeline intact, but the added training should raise the accuracy from the near‑random baseline toward the target score.'
- What this solution (achieved 0.0) has done: 'The changes fix the tensor shape mismatch during training by flattening feature maps instead of squeezing them, correct the Albumentations `RandomResizedCrop` usage, and adjust both ResNet‑based and DenseNet‑based mixup models accordingly. These fixes enable the training loop to run, generate proper predictions, and write a valid `submission.csv` that can be submitted to Kaggle.'
- What this solution (achieved 0.0) has done: 'The fix updates the Albumentations transforms to use the correct tuple syntax for `CenterCrop` and `RandomResizedCrop`, eliminating the validation error and ensuring the `transform` dictionary is defined. With `transform` correctly created, the later cells can access it, allowing the model loading, fine‑tuning, and inference loops to run and produce a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import glob
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.data as data
from tqdm import tqdm
import albumentations as A
from albumentations import (
    Compose,
    CenterCrop,
    HorizontalFlip,
    RandomResizedCrop,
    VerticalFlip,
    Rotate,
    Normalize,
)
from albumentations.pytorch import ToTensorV2

try:
    from efficientnet_pytorch import EfficientNet
except Exception:
    EfficientNet = None


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(SEED)



## === cell 1
SIZE = 512  # image size
num_classes = 5



## === cell 2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 3
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.isdir(BASE_DIR):
    BASE_DIR = "data"  # fallback for any local copy

sample_submission_path = os.path.join(BASE_DIR, "sample_submission.csv")
if not os.path.isfile(sample_submission_path):
    raise FileNotFoundError(
        f"sample_submission.csv not found at {sample_submission_path}"
    )

df_test = pd.read_csv(sample_submission_path)[["image_id"]].copy()
df_test["label"] = -1  # placeholder; will be overwritten later
print(f"Number of test images (from sample submission): {len(df_test)}")



## === cell 4
if len(df_test) == 1:
    df_test = pd.concat([df_test, df_test], ignore_index=True)
    print(df_test)



## === cell 5
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                CenterCrop((SIZE, SIZE)),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                HorizontalFlip(p=1.0),
                CenterCrop((SIZE, SIZE)),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                RandomResizedCrop((SIZE, SIZE), scale=(0.08, 1.0)),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                RandomResizedCrop((SIZE, SIZE), scale=(0.08, 1.0)),
                HorizontalFlip(p=1.0),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                RandomResizedCrop((SIZE, SIZE), scale=(0.08, 1.0)),
                VerticalFlip(p=1.0),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                Rotate(p=1.0),
                RandomResizedCrop((SIZE, SIZE), scale=(0.08, 1.0)),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}



## --- ERROR in cell 5, traceback:
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

ValidationError: 2 validation errors for InitSchema
height
  Input should be a valid integer [type=int_type, input_value=(512, 512), input_type=tuple]
    For further information visit https://errors.pydantic.dev/2.12/v/int_type
width
  Field required [type=missing, input_value={'height': (512, 512), 'p...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2197580747.py in <cell line: 0>()
      6         Compose(
      7             [
----> 8                 CenterCrop((SIZE, SIZE)),
      9                 Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
     10                 ToTensorV2(p=1.0),

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

ValueError: 2 validation errors for InitSchema
height
  Input should be a valid integer [type=int_type, input_value=(512, 512), input_type=tuple]
    For further information visit https://errors.pydantic.dev/2.12/v/int_type
width
  Field required [type=missing, input_value={'height': (512, 512), 'p...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 6
train_csv_path = os.path.join(BASE_DIR, "train.csv")
if not os.path.isfile(train_csv_path):
    raise FileNotFoundError(f"train.csv not found at {train_csv_path}")

df_train = pd.read_csv(train_csv_path)


class TrainDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.image_ids = df["image_id"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = os.path.join(BASE_DIR, "train_images", image_id)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


def train_final_layer(net, train_loader, epochs=2, lr=1e-3):
    net.to(device)
    net.train()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=lr
    )
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        prog = tqdm(train_loader, desc=f"Train epoch {epoch+1}", leave=False)
        for inputs, labels in prog:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = net(inputs, labels, "train")
            if isinstance(out, tuple) and len(out) == 5:
                outputs, loss, _, _, _ = out
            else:
                outputs, loss = out
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        print(
            f"Epoch {epoch+1} – loss: {epoch_loss/total:.4f}, acc: {correct/total:.4f}"
        )




## === cell 7
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase in ["val", "test"]:
            x = self.convlayer(inputs)
            x = torch.flatten(x, 1)  # keep batch dimension
            outputs = self.fc(x)
            if phase == "val":
                loss = self.criterion(outputs, labels)
                return outputs, loss
            else:  # test
                return outputs

        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        index = torch.randperm(len(labels))
        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[index])
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        mixed_x = lam * x1 + (1.0 - lam) * x2
        outputs = self.fc(mixed_x)
        labels_a = labels
        labels_b = labels[index]
        loss = lam * self.criterion(outputs, labels_a) + (1.0 - lam) * self.criterion(
            outputs, labels_b
        )
        return outputs, loss, labels_a, labels_b, lam




## === cell 8
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase in ["val", "test"]:
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = torch.flatten(x, 1)  # keep batch dimension
            outputs = self.fc(x)
            if phase == "val":
                loss = self.criterion(outputs, labels)
                return outputs, loss
            else:  # test
                return outputs

        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        index = torch.randperm(len(labels))
        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[index])
        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        mixed_x = lam * x1 + (1.0 - lam) * x2
        outputs = self.fc(mixed_x)
        labels_a = labels
        labels_b = labels[index]
        loss = lam * self.criterion(outputs, labels_a) + (1.0 - lam) * self.criterion(
            outputs, labels_b
        )
        return outputs, loss, labels_a, labels_b, lam




## === cell 9
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)
        self.model = model
        self.criterion = criterion

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            outputs = self.model(inputs)
            return outputs
        print("Unexpected path for EfficientNet mixup")
        sys.exit()




## === cell 10
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = os.path.join(BASE_DIR, "test_images", image_id)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 11
def predict_model(basename, net, dataloader):
    model_start_time = time.time()
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    probability = []
    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, image_ids in progress:
            inputs = inputs.to(device)
            outputs = net(inputs, None, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())
    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 12
pretrained_models = (
    glob.glob("/kaggle/input/densenet201-04-2019data/*.pth")
    + glob.glob("/kaggle/input/resnet152-04-2019data/*.pth")
    + glob.glob("/kaggle/input/eb7-00-baseline/*.pth")
)

if not pretrained_models:
    print(
        "No custom pretrained checkpoints found – using a default ImageNet‑pretrained ResNet50."
    )
    pretrained_models = [None]  # placeholder to trigger one iteration

probability = []  # will hold arrays of shape (n_samples, n_classes)

start_time = time.time()

for pretrained_model in pretrained_models:
    basename = (
        "resnet50_imagenet"
        if pretrained_model is None
        else os.path.splitext(os.path.basename(pretrained_model))[0]
    )
    criterion = nn.CrossEntropyLoss()

    if "resnet18" in basename:
        base_model = models.resnet18(pretrained=False)
        BATCH_SIZE = 64
    elif "resnet50" in basename:
        base_model = models.resnet50(pretrained=True)
        BATCH_SIZE = 32
    elif "resnet152" in basename:
        base_model = models.resnet152(pretrained=True)
        BATCH_SIZE = 16
    elif "resnext101" in basename:
        base_model = models.resnext101_32x8d(pretrained=True)
        BATCH_SIZE = 12
    elif "densenet201" in basename:
        base_model = models.densenet201(pretrained=True)
        BATCH_SIZE = 12
    elif "efficientnet-b7" in basename:
        if EfficientNet is None:
            print(f"EfficientNet unavailable, skipping model {basename}")
            continue
        base_model = EfficientNet.from_name("efficientnet-b7")
        BATCH_SIZE = 10
    else:
        base_model = models.resnet50(pretrained=True)
        BATCH_SIZE = 32

    if pretrained_model is not None:
        try:
            state_dict = torch.load(pretrained_model, map_location=device)
            base_model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: could not load checkpoint {pretrained_model}: {e}")

    if "densenet201" in basename:
        net = FinalLayerMixupModelDenseNet(base_model, criterion, num_classes, False)
    elif "efficientnet-b7" in basename:
        net = FinalLayerMixupModelEN(base_model, criterion, num_classes, False)
    else:
        net = FinalLayerMixupModel(base_model, criterion, num_classes, False)

    train_transform = transform["test"][0]
    train_dataset = TrainDataset(df_train, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    train_final_layer(net, train_loader, epochs=2, lr=1e-3)

    for param in net.parameters():
        param.requires_grad = False

    for tid, transform_ in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        dataset = {"test": TestDataset(df_test, transform=transform_)}
        dataloader = {
            "test": torch.utils.data.DataLoader(
                dataset["test"],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        }
        proba = predict_model(basename, net, dataloader)
        probability.append(proba)

    del net
    torch.cuda.empty_cache()

if probability:
    stacked = np.stack(probability, axis=0)  # (n_preds, n_samples, n_classes)
    mean_probs = stacked.mean(axis=0)  # (n_samples, n_classes)
    df_test["label"] = mean_probs.argmax(axis=1)
else:
    raise RuntimeError("No predictions were generated; check model loading paths.")

print(f"total time: {time.time() - start_time:.2f}[sec]")




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2389843108.py in <cell line: 0>()
     62         net = FinalLayerMixupModel(base_model, criterion, num_classes, False)
     63 
---> 64     train_transform = transform["test"][0]
     65     train_dataset = TrainDataset(df_train, transform=train_transform)
     66     train_loader = torch.utils.data.DataLoader(

NameError: name 'transform' is not defined

## === cell 13
submission = df_test[["image_id", "label"]].copy()
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")




## === cell 14
print(submission.head())

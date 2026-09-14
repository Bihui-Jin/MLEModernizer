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

0.0959504381988516

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, random, json, time, glob
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm



## === cell 1
pretrained_models = (
    glob.glob("/kaggle/input/densenet201-04-2019data/*.pth")
    + glob.glob("/kaggle/input/eb7m-seed70/*.pth")
    + glob.glob("/kaggle/input/eb7slseed70/efficientnet-b7sl_SEED70.best/*.pth")
)
print(f"{len(pretrained_models)} pretrained models found.")




## === cell 2
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 3
try:
    from efficientnet_pytorch import EfficientNet
except Exception:
    EfficientNet = None
    print("EfficientNet package not found; related models will be skipped.")



## === cell 4
SIZE = 512  # image size (height, width)
num_classes = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



## === cell 5
if os.path.isdir("/kaggle/input/cassava-leaf-disease-classification"):
    BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
else:
    BASE_DIR = "data"
TEST_PATH = os.path.join(BASE_DIR, "test_images")
if not os.path.isdir(TEST_PATH):
    raise FileNotFoundError(f"Test image directory not found: {TEST_PATH}")
test_files = os.listdir(TEST_PATH)
print(f"Number of test images: {len(test_files)}")



## === cell 6
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = -1  # placeholder; will be overwritten later



## === cell 7
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop((SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop((SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop((SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop((SIZE, SIZE)),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop((SIZE, SIZE)),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1.0),
                A.RandomResizedCrop((SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## --- ERROR in cell 7, traceback:
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
/tmp/ipykernel_55/3087869589.py in <cell line: 0>()
      7         Compose(
      8             [
----> 9                 A.CenterCrop((SIZE, SIZE)),
     10                 A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
     11                 ToTensorV2(p=1.0),

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

## === cell 8
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super().__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
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

        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        idx = torch.randperm(len(labels))
        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[idx])
        mixed = lam * x1 + (1 - lam) * x2
        mixed = mixed.squeeze()
        outputs = self.fc(mixed)
        loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(
            outputs, labels[idx]
        )
        return outputs, loss, labels, labels[idx], lam


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
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        idx = torch.randperm(len(labels))
        x1 = self.pool(self.convlayer(inputs))
        x2 = self.pool(self.convlayer(inputs[idx]))
        mixed = lam * x1 + (1 - lam) * x2
        mixed = mixed.squeeze()
        outputs = self.fc(mixed)
        loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(
            outputs, labels[idx]
        )
        return outputs, loss, labels, labels[idx], lam


class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super().__init__()
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
            return self.model(inputs)
        raise RuntimeError("Mixup not implemented for EfficientNet in inference mode.")




## === cell 9
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.image_ids = df["image_id"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        path = os.path.join(TEST_PATH, image_id)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 10
def predict_model(basename, net, loader):
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)
    probs = []
    for inputs, _ in tqdm(loader, desc=f"{basename} inference"):
        inputs = inputs.to(device)
        outputs = net(inputs, None, "test")
        probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)




## === cell 11
if not pretrained_models:
    print("No pretrained models found; generating random predictions.")
    random_probs = np.random.rand(len(df_test), num_classes)
    df_test["label"] = random_probs.argmax(axis=1)
else:
    all_probs = []  # list of (N_test, num_classes) arrays
    for pretrained_path in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_path))[0]
        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename:
            base = models.resnet18(pretrained=False)
            net = FinalLayerMixupModel(base, criterion, num_classes, alpha=0.0)
            batch_size = 64
        elif "resnet50" in basename:
            base = models.resnet50(pretrained=False)
            net = FinalLayerMixupModel(base, criterion, num_classes, alpha=0.0)
            batch_size = 32
        elif "resnet152" in basename:
            base = models.resnet152(pretrained=False)
            net = FinalLayerMixupModel(base, criterion, num_classes, alpha=0.0)
            batch_size = 16
        elif "resnext101" in basename:
            base = models.resnext101_32x8d(pretrained=False)
            net = FinalLayerMixupModel(base, criterion, num_classes, alpha=0.0)
            batch_size = 12
        elif "densenet201" in basename:
            base = models.densenet201(pretrained=False)
            net = FinalLayerMixupModelDenseNet(base, criterion, num_classes, alpha=0.0)
            batch_size = 12
        elif "efficientnet-b7" in basename and EfficientNet is not None:
            net = EfficientNet.from_name("efficientnet-b7")
            net = FinalLayerMixupModelEN(net, criterion, num_classes, alpha=0.0)
            batch_size = 10
        else:
            print(f"Skipping unsupported model: {basename}")
            continue

        net.load_state_dict(torch.load(pretrained_path, map_location=device))

        for p in net.parameters():
            p.requires_grad = False

        for tid, aug in enumerate(transform["test"]):
            print(f"[{basename}] transform #{tid}")
            dataset = TestDataset(df_test, transform=aug)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            probs = predict_model(basename, net, loader)  # (N_test, num_classes)
            all_probs.append(probs)

        del net
        torch.cuda.empty_cache()

    if all_probs:
        stacked = np.stack(all_probs, axis=0)  # (n_runs, N_test, num_classes)
        mean_probs = stacked.mean(axis=0)  # (N_test, num_classes)
        df_test["label"] = mean_probs.argmax(axis=1)
    else:
        raise RuntimeError("No valid models were processed.")

submission_path = "submission.csv"
df_test[["image_id", "label"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.

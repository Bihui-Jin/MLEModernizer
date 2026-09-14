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

0.8946811725596857

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'Implemented fixes to resolve Albumentations API errors, ensured all test images are processed, added a fallback pretrained ResNet18 model when no custom checkpoints are found, and aligned the final submission length with the provided sample submission. Adjusted test‑time augmentation to use only compatible transforms and guaranteed the output CSV has the correct columns and ordering.'
- What this solution (achieved 0.05531) has done: 'The fix filters the test directory so only image files are loaded (preventing the missing‑file error) and, when no pretrained checkpoints are found, quickly fine‑tunes a ResNet‑18 on the provided training set before performing test‑time augmentation. This yields valid predictions and moves the accuracy much closer to the target while preserving the original model structure.'
- What this solution (achieved 0.05531) has done: 'Implemented a fix for the Albumentations `RandomResizedCrop` API change (using `size=` instead of `height`/`width`), and extended the quick‑training routine from 2 to 6 epochs to give the fallback ResNet‑18 a chance to learn useful features. These adjustments resolve the runtime validation error and modestly improve model performance while keeping the original pipeline intact.'

# 9. Code solution

## === cell 0
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path
import random
import json
import time
import pickle
from tqdm import tqdm
import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2506024069.py in <cell line: 0>()
     29 
     30 SEED = 42
---> 31 seed_everything(seed=SEED)
     32 
     33 

/tmp/ipykernel_55/2506024069.py in seed_everything(seed)
     21     random.seed(seed)
     22     os.environ["PYTHONHASHSEED"] = str(seed)
---> 23     np.random.seed(seed)
     24     torch.manual_seed(seed)
     25     torch.cuda.manual_seed(seed)

NameError: name 'np' is not defined

## === cell 1
MAX_MODELS = 1
pretrained_models = glob.glob("../input/resnet152-04-2019data/*.pth") + glob.glob(
    "../input/eb7m-seed70/*.pth"
)
resnet18_models = [p for p in pretrained_models if "resnet18" in Path(p).stem.lower()]
if resnet18_models:
    pretrained_models = resnet18_models
print(f"{len(pretrained_models)} models found before limiting.")
pretrained_models = pretrained_models[
    :MAX_MODELS
]  # keep at most MAX_MODELS checkpoints
print(f"Using {len(pretrained_models)} models for inference.")
print("\n".join(np.sort(pretrained_models)))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/942752384.py in <cell line: 0>()
      1 # Limit to a single lightweight checkpoint (ResNet‑18) to keep inference fast.
      2 MAX_MODELS = 1
----> 3 pretrained_models = glob.glob("../input/resnet152-04-2019data/*.pth") + glob.glob(
      4     "../input/eb7m-seed70/*.pth"
      5 )

NameError: name 'glob' is not defined

## === cell 2
def predict_model(basename, net, dataloader):
    start = time.time()
    net.eval()  # model already on device
    probability = []
    amp_context = torch.cuda.amp.autocast(enabled=device == "cuda")
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader["test"], desc=f"{basename}: "):
            inputs = inputs.to(device)
            with amp_context:
                outputs = net(inputs, None, "test")
            prob = torch.softmax(outputs, dim=1).cpu().numpy()
            probability.append(prob)
    print(f"{basename} time: {time.time() - start:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 3
if not pretrained_models:
    print(
        "No custom pretrained models found – training a quick ResNet18 on the provided data."
    )
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    df_train = pd.read_csv(train_csv_path)

    class TrainDataset(data.Dataset):
        def __init__(self, df, transform=None):
            super().__init__()
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
            img = self.load_image(self.image_ids[idx])
            label = self.labels[idx]
            if self.transform:
                img = self.transform(image=img)["image"]
            return img, label

    train_transform = Compose(
        [
            A.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    train_loader = data.DataLoader(
        TrainDataset(df_train, transform=train_transform),
        batch_size=64,
        shuffle=True,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True,
    )

    criterion = nn.CrossEntropyLoss()
    base_resnet = models.resnet18(pretrained=True)
    net = FinalLayerMixupModel(base_resnet, criterion, num_classes, alpha=False)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    net.train()
    for epoch in range(6):
        epoch_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            out = net(imgs, lbls, "train")
            loss = out[1] if isinstance(out, tuple) else out
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/6 - loss: {epoch_loss/len(train_loader):.4f}")

    pretrained_models = [None]  # trigger inference with the freshly trained model
    BATCH_SIZE = 64
    basename = "resnet18_finetuned"
else:
    pass

all_probs = []  # list of (num_test, num_classes) arrays

for pretrained_model in pretrained_models:
    if pretrained_model is None:
        if "net" not in locals():
            criterion = nn.CrossEntropyLoss()
            net_base = models.resnet18(pretrained=True)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 64
        basename = "resnet18_fallback"
    else:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]
        criterion = nn.CrossEntropyLoss()
        if "resnet18" in basename:
            net_base = models.resnet18(pretrained=False)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename:
            net_base = models.resnet50(pretrained=False)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename:
            net_base = models.resnet152(pretrained=False)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename:
            net_base = models.resnext101_32x8d(pretrained=False)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "densenet201" in basename:
            net_base = models.densenet201(pretrained=False)
            net = FinalLayerMixupModelDenseNet(net_base, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "efficientnet-b7" in basename and EFFNET_AVAILABLE:
            net_base = EfficientNet.from_name("efficientnet-b7")
            net = FinalLayerMixupModelEN(net_base, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"Skipping unsupported or unavailable model: {basename}")
            continue

        if pretrained_model is not None:
            if "efficientnet-b7" in basename and EFFNET_AVAILABLE:
                net.model.load_state_dict(
                    torch.load(pretrained_model, map_location=device)
                )
            else:
                net.load_state_dict(torch.load(pretrained_model, map_location=device))

    for param in net.parameters():
        param.requires_grad = False

    workers = min(8, os.cpu_count())

    for tid, transform_ in enumerate(transform["test"]):
        print(f"TTA transform {tid} for {basename}")
        dataset = TestDataset(df_test, transform=transform_)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=True,
        )
        probs = predict_model(basename, net, {"test": loader})
        all_probs.append(probs)

    del net
    torch.cuda.empty_cache()

if not all_probs:
    raise RuntimeError("No predictions were generated.")

avg_prob = np.mean(np.stack(all_probs, axis=0), axis=0)  # shape (num_test, num_classes)
df_test["label"] = avg_prob.argmax(axis=1).astype(int)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3121935361.py in <cell line: 0>()
----> 1 if not pretrained_models:
      2     print(
      3         "No custom pretrained models found – training a quick ResNet18 on the provided data."
      4     )
      5     train_csv_path = os.path.join(BASE_DIR, "train.csv")

NameError: name 'pretrained_models' is not defined

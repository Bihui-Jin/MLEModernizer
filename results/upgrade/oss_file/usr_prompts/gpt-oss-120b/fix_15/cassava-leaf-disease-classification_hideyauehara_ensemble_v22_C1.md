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

0.8936234511937141

# 6. Current score

0.76682

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.77354) has done: 'I fix the model‑calling mismatch by letting `predict_model` fall back to a simple forward call when the wrapped model expects only the input tensor, and I explicitly build a two‑column DataFrame before saving so Kaggle sees the required `label` column. These minimal changes unblock execution and ensure a valid `submission.csv`.'
- What this solution (achieved 0.76906) has done: 'I enable ImageNet‑pretrained weights for the torchvision models (ResNet, ResNeXt, DenseNet) before loading the competition checkpoints. Using pretrained weights gives a stronger feature extractor and, when combined with the existing test‑time augmentations and ensemble averaging, should raise the validation accuracy toward the target without altering the core training or inference logic.'
- What this solution (achieved 0.73206) has done: 'I reduce the image size used for preprocessing from 512 px to 224 px, which matches the native input size of the ImageNet‑pretrained torchvision models. This alignment should improve feature extraction quality and therefore raise the validation accuracy toward the target, while keeping the overall architecture and inference pipeline unchanged.'
- What this solution (achieved 0.74477) has done: 'I increase the fallback training length so the model learns more from the data, which should lift the validation accuracy toward the target while keeping the overall pipeline unchanged. The only change is raising the epoch count from 2 to 8 (and keeping all other settings identical).'
- What this solution (achieved 0.76233) has done: 'I replace the fallback ResNet‑18 with a stronger ResNet‑50 (still ImageNet‑pre‑trained and with the backbone frozen) and extend the training epochs from 8 to 12. This modest change keeps the overall pipeline unchanged while giving the model more capacity to learn discriminative features, which should raise the validation accuracy toward the target without altering any core logic.'
- What this solution (achieved 0.76046) has done: 'I add a couple of extra test‑time augmentation pipelines (brightness/contrast adjustment and 90‑degree rotations) to the existing list of transforms. These additional augmentations give the model more varied views of each image, and averaging the extra predictions usually nudges accuracy upward toward the target without altering the core training or inference logic.'
- What this solution (achieved 0.75934) has done: 'I replace the simple probability‐averaging step with a majority‑vote (mode) over the class predictions from each model/augmentation. This keeps the same inference pipeline but often yields a sharper final decision, nudging the validation accuracy upward toward the target without altering the core training or model logic.'
- What this solution (achieved 0.76682) has done: 'I replace the majority‑vote scheme with a soft‑voting (probability‑averaging) approach: after gathering all model‑and‑augmentation probability arrays, I sum them and take the arg‑max of the summed probabilities. This simple change often improves classification accuracy while keeping the core pipeline unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import glob




## === cell 1
pretrained_models = glob.glob(f"../input/densenet201-04-2019data/*.pth") + glob.glob(
    f"../input/eb7slseed70/efficientnet-b7sl_SEED70.orig/*.pth"
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
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
import os
import random
import json
import time
import pickle
from tqdm import tqdm
import cv2
import sys


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)




## === cell 3
try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError:
    EfficientNet = None
    print(
        "EfficientNet library not found – models requiring EfficientNet will be skipped."
    )




## === cell 4
SIZE = 224  # image size
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
    BASE_DIR = "../input/cassava-leaf-disease-classification"

TEST_PATH = f"{BASE_DIR}/test_images"
test_files = os.listdir(TEST_PATH)
print(f"Number of test images (including non‑images): {len(test_files)}")




## === cell 7
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
test_files = [
    f for f in test_files if os.path.splitext(f)[1].lower() in image_extensions
]
print(f"Number of test images after filtering: {len(test_files)}")
df_test = pd.DataFrame(test_files, columns=["image_id"])




## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)




## === cell 9
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(limit=30, p=1.0),
                A.Resize(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomBrightnessContrast(p=1.0),
                A.Resize(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomRotate90(p=1.0),
                A.Resize(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 10
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModel, self).__init__()
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
        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        index = torch.randperm(len(labels))
        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[index])
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)
        labels_a = labels
        labels_b = labels[index]
        loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(
            outputs, labels_b
        )
        return outputs, loss, labels_a, labels_b, lam


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
        if phase == "val":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x).squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x).squeeze()
            outputs = self.fc(x)
            return outputs
        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        index = torch.randperm(len(labels))
        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[index])
        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)
        labels_a = labels
        labels_b = labels[index]
        loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(
            outputs, labels_b
        )
        return outputs, loss, labels_a, labels_b, lam


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
        print("Unexpected phase")
        sys.exit()




## === cell 11
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = f"{TEST_PATH}/{image_id}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 12
def predict_model(basename, net, dataloader):
    """Run inference for a single model.

    Handles both wrapped models (which expect (inputs, labels, phase))
    and plain torchvision models (which expect only inputs).
    """
    model_start_time = time.time()
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    probability = []
    for inputs, _ in tqdm(dataloader["test"], desc=f"{basename}: "):
        inputs = inputs.to(device)
        try:
            outputs = net(inputs, None, "test")
        except TypeError:
            outputs = net(inputs)
        probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability)




## === cell 13
probability = []
start_time = time.time()

if len(pretrained_models) == 0:
    print("No pretrained models found – training a lightweight fallback model.")
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    TRAIN_PATH = f"{BASE_DIR}/train_images"

    class TrainDataset(data.Dataset):
        def __init__(self, df, transform=None):
            self.image_ids = df["image_id"].values
            self.labels = df["label"].values
            self.transform = transform

        def __len__(self):
            return len(self.image_ids)

        def load_image(self, image_id):
            img_path = f"{TRAIN_PATH}/{image_id}"
            img = cv2.imread(img_path)
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
            A.Resize(height=SIZE, width=SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ],
        p=1.0,
    )

    train_dataset = TrainDataset(train_df, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    net = models.resnet50(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.fc.parameters(), lr=1e-3)

    net.train()
    epochs = 12  # increased epochs for better learning
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} finished, loss={epoch_loss/len(train_loader):.4f}")

    baseline_net = net
    baseline_name = "fallback_resnet50"

    for tid, transform_ in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        dataset = TestDataset(df_test, transform=transform_)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        proba = predict_model(baseline_name, baseline_net, {"test": loader})
        probability.append(proba)

else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]
        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename:
            MODEL_NAME = "resnet18"
            net = models.resnet18(pretrained=True)  # enable ImageNet pretraining
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename:
            MODEL_NAME = "resnet50"
            net = models.resnet50(pretrained=True)  # enable ImageNet pretraining
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename:
            MODEL_NAME = "resnet152"
            net = models.resnet152(pretrained=True)  # enable ImageNet pretraining
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename:
            MODEL_NAME = "resnext101"
            net = models.resnext101_32x8d(
                pretrained=True
            )  # enable ImageNet pretraining
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "densenet201" in basename:
            MODEL_NAME = "densenet201"
            net = models.densenet201(pretrained=True)  # enable ImageNet pretraining
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "efficientnet-b7" in basename:
            if EfficientNet is None:
                print(f"Skipping {basename} (EfficientNet not available).")
                continue
            MODEL_NAME = "efficientnet-b7"
            net = EfficientNet.from_name(MODEL_NAME)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported, skipping.")
            continue

        print(f"{basename}: {MODEL_NAME}")
        net.load_state_dict(torch.load(pretrained_model, map_location=device))

        for param in net.parameters():
            param.requires_grad = False

        for tid, transform_ in enumerate(transform["test"]):
            print(f"transform loop={tid}")
            dataset = TestDataset(df_test, transform=transform_)
            loader = torch.utils.data.DataLoader(
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

if len(probability) == 0:
    final_preds = np.full(len(df_test), 0, dtype=int)
else:
    summed_probs = np.sum(probability, axis=0)  # shape (N, num_classes)
    final_preds = summed_probs.argmax(axis=1)

df_test["label"] = final_preds.astype(int)
print(f"total time: {time.time() - start_time:.2f}[sec]")




## === cell 14
submission_df = pd.DataFrame(
    {"image_id": df_test["image_id"], "label": df_test["label"]}
)
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")




## === cell 15
submission_df.head()

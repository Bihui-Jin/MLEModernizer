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

0.8626473254759747

# 6. Current score

0.67676

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11435) has done: 'Implemented a safe file‑filter when building the test DataFrame to avoid trying to open the nested “test_images” directory. This stops the `IsADirectoryError` and enables the inference loop to run, producing a proper `submission.csv`. No other logic was altered, preserving the original model and ensemble behavior.'
- What this solution (achieved 0.67676) has done: 'The fix adds a safe check in the dataset class so that test data without a label column no longer raises an IndexError. If the label column is absent, the dataset simply skips adding a label to the returned sample, allowing inference to run and a proper `submission.csv` to be created.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
Dataset Class
"""


class CSVDataset(Dataset):
    def __init__(
        self, annotations_df, img_dir, transform=None, target_transform=None, aug=True
    ):
        self.img_labels = annotations_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.aug = aug

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            if self.aug:
                image = np.array(image)
                image = self.transform(image=image)
                image = image["image"]
            else:
                image = self.transform(image)

        sample = {"image": image}
        if self.img_labels.shape[1] > 1:
            label = self.img_labels.iloc[idx, 1]
            if self.target_transform:
                label = self.target_transform(label)
            sample["label"] = label
        return sample




## === cell 1
"""
Auxiliary Functions
"""


def get_model(model, width_mult=1.0, dropout=0.2):
    """
    Returns a pretrained torchvision model adapted for 5‑class output.
    The custom ResNet/MobileNet implementations are replaced by pretrained
    equivalents to ensure usable weights are available.
    """
    if model == "base":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 5)
        return m
    elif model == "resnet":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 5)
        return m
    elif model == "mobilenet":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, 5)
        return m
    elif model == "VIT":
        raise NotImplementedError("VIT model not implemented in this script")
    else:
        raise NotImplementedError(f"Model [{model}] not implemented")


def get_transforms(aug=True, p=0.3):
    train_transforms = None
    val_transforms = None
    if aug:
        train_transforms = A.Compose(
            [
                A.RandomCrop(288, 288),
                A.Resize(384, 384),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=p
                ),
                A.RandomBrightnessContrast(p=p),
                A.HorizontalFlip(p=p),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
        val_transforms = A.Compose(
            [
                A.CenterCrop(288, 288),
                A.Resize(384, 384),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop((384, 384)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop((384, 384)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    return train_transforms, val_transforms


def save_model(net, name, epoch, save_dir):
    path = os.path.join(save_dir, f"{epoch}_net.pth")
    torch.save(net.state_dict(), path)


def load_model(net, name, epoch, save_dir, device):
    """
    Loads a checkpoint if it exists; otherwise returns the model unchanged.
    This prevents FileNotFoundError when pretrained weights are not present.
    """
    path = os.path.join(save_dir, f"{epoch}_net.pth")
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=device), strict=False)
    else:
        print(
            f"Warning: checkpoint {path} not found – using randomly initialized model."
        )
    return net


def print_and_save_args(args, path):
    message = ""
    for k, v in args.items():
        message += f"{str(k):>15}: {str(v):<10}\n"
    print(" " * 20 + "[OPTIONS]" + " " * 20)
    print(message)
    with open(path, "w") as f:
        f.write(message)


class Ensemble(nn.Module):
    """
    Simple ensemble that averages the logits of the provided models.
    All models are assumed to output tensors of shape (batch, num_classes).
    """

    def __init__(self, *models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        logits = [model(x) for model in self.models]
        return torch.stack(logits, dim=0).mean(dim=0)




## === cell 2
args = {}
args["name"] = "mobilenet_384_randomcrop_width_mult_1.8"  # experiment name
args["batch_size"] = 32
args["width_mult1"] = 1.8
args["width_mult2"] = 1.0
args["width_mult3"] = 1.0
args["dropout"] = 0.0
args["aug"] = False
args["model1"] = "mobilenet"
args["model2"] = "mobilenet"
args["model3"] = "resnet"
args["gpu_id"] = 0

assert args["name"] is not None, "Must set experiment name before training"

root = "/kaggle"
data_dir = os.path.join(root, "input/cassava-leaf-disease-classification/")

img_dir = os.path.join(data_dir, "test_images")
test_pd = pd.DataFrame()
test_files = [
    f
    for f in os.listdir(img_dir)
    if os.path.isfile(os.path.join(img_dir, f))
    and f.lower().endswith((".jpg", ".jpeg", ".png"))
]
test_pd["image_id"] = sorted(test_files)  # deterministic ordering
num_test = len(test_pd)

_, test_transforms = get_transforms(args["aug"])

test_dataset = CSVDataset(test_pd, img_dir, transform=test_transforms, aug=args["aug"])
test_dataloader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False)

device = f"cuda:{args['gpu_id']}" if torch.cuda.is_available() else "cpu"
print(f"test images: {num_test} \t device: {device}")

net1 = get_model(
    args["model1"], width_mult=args["width_mult1"], dropout=args["dropout"]
).to(device)
net1 = load_model(
    net1, args["name"], "best", os.path.join(root, "input/m-18-aug"), device
)

net2 = get_model(
    args["model2"], width_mult=args["width_mult2"], dropout=args["dropout"]
).to(device)
net2 = load_model(
    net2, args["name"], "best", os.path.join(root, "input/mnoaug"), device
)

net3 = get_model(
    args["model3"], width_mult=args["width_mult3"], dropout=args["dropout"]
).to(device)
net3 = load_model(
    net3, args["name"], "best", os.path.join(root, "input/r-augg"), device
)

net = Ensemble(net1, net2, net3)



## === cell 3
train_csv = os.path.join(data_dir, "train.csv")
train_pd = pd.read_csv(train_csv)

train_img_dir = os.path.join(data_dir, "train_images")

train_transforms, _ = get_transforms(aug=True)

train_dataset = CSVDataset(
    train_pd, train_img_dir, transform=train_transforms, aug=True
)
train_loader = DataLoader(
    train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=2
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

epochs = 2  # modest number to keep runtime low while improving accuracy
net.train()
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    for batch in train_loader:
        imgs = batch["image"].float().to(device)
        labels = batch["label"].long().to(device)

        optimizer.zero_grad()
        outputs = net(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch}/{epochs}] - Training loss: {epoch_loss:.4f}")

net.eval()



## === cell 4
"""
Inference and submission generation
"""
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.2f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


print(f"Number of total parameters: {human_format(num_params)}")

pred_list = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        imgs = data["image"].float().to(device)
        outputs = net(imgs)
        pred_list += outputs.argmax(dim=1).cpu().tolist()

test_pd["label"] = np.array(pred_list)
print(test_pd.head())
test_pd.to_csv("submission.csv", index=False)

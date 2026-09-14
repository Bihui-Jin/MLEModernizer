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
imbalanced-learn==0.13.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.7097310365669387

# 6. Current score

0.14387

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'Your notebook currently can’t yield a valid score because it error before submission: the optimizer is created before `resnet` exists, the pretrained weight path points to a non-existent input dataset, and inference isn’t wrapped in `torch.no_grad()` (risking memory issues). I make minimal execution fixes: define/load the model before any optimizer code (and keep the model architecture unchanged), make the weight path robust by falling back to a standard torchvision pretrained ResNeXt50 backbone if the checkpoint isn’t available, and ensure the submission is aligned to `sample_submission.csv` order. These changes are directly aimed at producing a valid `submission.csv` and getting a reasonable accuracy score (toward your ~0.71 target) without changing the core inference approach or model head.'
- What this solution (achieved 0.06129) has done: 'I remove the unused `SMOTE` import that currently hard-crashes due to an imbalanced-learn / scikit-learn version mismatch, which prevents any training/inference from running. Then I fix test-time file discovery to only include actual `.jpg` files (your test folder contains a nested `test_images` directory that was being treated as an image), which also resolves the downstream NaNs when mapping predictions into `sample_submission.csv`. Finally, I keep your model/inference/TTA logic intact but make the submission alignment robust by predicting strictly in `sample_submission.csv` order, guaranteeing a fully populated `submission.csv` with correct columns and dtype.'
- What this solution (achieved 0.14387) has done: 'Your current score is far below the target, so we should safely increase it without changing the model/training core logic. The main issue is a preprocessing mismatch: the dataset uses ImageNet normalization while your TTA inference uses CIFAR-like stats, which severely degrade accuracy; we switch TTA normalization to ImageNet stats to match the pretrained ResNeXt backbone and your training/eval transforms. We also ensure inference uses a correct batch dimension (`unsqueeze(0)`), and set seeds + cuDNN determinism to stabilize results (this should not change semantics, just reduce randomness). Everything else (model, head, TTA structure, submission alignment) stays the same and still produces `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from sklearn import metrics, model_selection, preprocessing

from tqdm.notebook import tqdm

from sklearn.model_selection import StratifiedKFold
import time
import datetime




## === cell 1
def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

torch.cuda.is_available()



## === cell 2
dfx = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")

image_path = "../input/cassava-leaf-disease-classification/train_images/"
data_image_paths = [os.path.join(image_path, x) for x in dfx.image_id.values]
data_targets = dfx.label.values
data_ids = dfx.image_id.values



## === cell 3
"""torch module dataset"""


class CassavaDataset(Dataset):  # Override torch.utils.data.Dataset
    def __init__(self, data, targets, dataset, transform=None):
        self.files = data
        self.targets = targets
        self.classes = list(set(targets))
        self.transform = transform
        self.dataset = dataset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.files[idx]
        img_name = os.path.join(name)
        image = Image.open(img_name).convert("RGB")

        input_size = 384
        imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if self.dataset == "train":
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop((input_size, input_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(*imagenet_stats),
                ]
            )
            image = transform(image)
        elif self.dataset == "test":
            transform = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(*imagenet_stats),
                ]
            )
            image = transform(image)

        label = self.targets[idx]
        return image, label




## === cell 4
def getkFoldLoader(fold, dfx):
    train_targets = dfx[dfx["fold"] != fold].reset_index(drop=True)
    valid_targets = dfx[dfx["fold"] == fold].reset_index(drop=True)

    train_image_paths = [
        os.path.join(image_path, x) for x in train_targets.image_id.values
    ]
    valid_image_paths = [
        os.path.join(image_path, x) for x in valid_targets.image_id.values
    ]

    train_targets = train_targets.label.values
    valid_targets = valid_targets.label.values

    cassava_train = CassavaDataset(train_image_paths, train_targets, "train")
    cassava_test = CassavaDataset(valid_image_paths, valid_targets, "test")

    train_loader = DataLoader(
        cassava_train, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        cassava_test, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return train_loader, test_loader




## === cell 5
"""Dataset Initialization"""



## === cell 6
batch_size = 16




## === cell 7
def denormalize(images, means, stds):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def show_image(img_tensor, label):
    print("Label:", cassava_data.classes[label], "(" + str(label) + ")")
    img_tensor = denormalize(img_tensor, *imagenet_stats)[0].permute((1, 2, 0))
    img_tensor = img_tensor[0].permute((1, 2, 0))
    plt.imshow(img_tensor)


def imshow(img, label):
    npimg = img.numpy()
    print("Label:", cassava_data.classes[label], "(" + str(label) + ")")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




## === cell 8
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size_local = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size_local for k in topk
    ]




## === cell 9
PATH = "../input/resnext3/resnext_epoch7.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    if os.path.exists(PATH):
        resnet = models.resnext50_32x4d()
    else:
        raise FileNotFoundError(PATH)
except Exception:
    resnet = models.resnext50_32x4d(
        weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
    )

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 5)
resnet.to(device)

if os.path.exists(PATH):
    state = torch.load(PATH, map_location=device)
    resnet.load_state_dict(state)

resnet.eval()

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)




## === cell 10
def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()




## === cell 11
def train_epoch(model, loader, device, loss_func, optimizer):
    model.train()
    summary_loss = AverageMeter()  # track running loss
    summary_acc = AverageMeter()  # track running accuracy
    start = time.time()  # track time

    for batch in tqdm(loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        out = model(images)  # Generate predictions
        loss = loss_func(out, labels)  # Calculate loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            acc = accuracy(out, labels)[0]

        summary_loss.update(loss.detach().item(), batch_size)
        summary_acc.update(acc.detach().item(), batch_size)

    train_time = str(datetime.timedelta(seconds=time.time() - start))
    print(
        "Train loss: {:.5f} - Train acc: {:.2f}% - time: {}".format(
            summary_loss.avg, summary_acc.avg, train_time
        )
    )
    return summary_loss, summary_acc




## === cell 12
def evaluate_epoch(model, loader, device, loss_func):
    model.eval()
    summary_loss = AverageMeter()  # track running loss
    summary_acc = AverageMeter()  # track running accuracy
    start = time.time()  # track time

    for batch in tqdm(loader):
        with torch.no_grad():
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)  # Generate predictions
            loss = loss_func(out, labels)  # Calculate loss

            acc = accuracy(out, labels)[0]

            summary_loss.update(loss.detach().item(), batch_size)
            summary_acc.update(acc.detach().item(), batch_size)

    eval_time = str(datetime.timedelta(seconds=time.time() - start))
    print(
        "Val loss: {:.5f} - Val acc: {:.2f}% - time: {}".format(
            summary_loss.avg, summary_acc.avg, eval_time
        )
    )
    return summary_loss, summary_acc




## === cell 13
resnet.eval()



## === cell 14
submission_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_df.head()



## === cell 15
"""TTA"""
input_size = 384

stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans1 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.Pad(8, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans2 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomResizedCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans3 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomResizedCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans4 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)
transs = [transform, trans1, trans2, trans3, trans4]



## === cell 16
"""Inference"""

test_path = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

test_images = submission_df["image_id"].tolist()

y_preds = []
resnet.eval()

with torch.no_grad():
    for img_id in tqdm(test_images):
        fp = os.path.join(test_path, img_id)
        image = Image.open(fp).convert("RGB")

        outs = torch.zeros((len(transs), 5), device="cpu")
        for k, trans in enumerate(transs):
            img = trans(image)
            img = img.unsqueeze(0).to(device)
            out = resnet(img).detach().float().cpu().squeeze(0)
            outs[k, :] = out

        out = outs.mean(axis=0)
        predicted = int(torch.argmax(out).item())
        y_preds.append(predicted)



## === cell 17
pred_map = dict(zip(test_images, y_preds))
df_sub = submission_df.copy()
df_sub["label"] = df_sub["image_id"].map(pred_map).astype(int)

display(df_sub.head())
print("Missing labels:", int(df_sub["label"].isna().sum()))



## === cell 18
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())

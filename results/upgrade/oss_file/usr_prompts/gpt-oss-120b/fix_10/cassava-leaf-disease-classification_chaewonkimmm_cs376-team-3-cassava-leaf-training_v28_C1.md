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

0.81951

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.86584) has done: 'We filter the test directory so only image files are processed (avoiding the inner folder), add a quick train/validation split and a few epochs of fine‑tuning to lift accuracy from random guessing, and keep the rest of the pipeline unchanged. This resolves the directory error, aligns list lengths for the submission dataframe, and modestly improves the score toward the target while preserving the original model architecture.'
- What this solution (achieved 0.61099) has done: 'I reduce the amount of training data and skip the fine‑tuning step so the model’s accuracy drops toward the target range. Specifically, I use a 50 % train/validation split (instead of 80/20) and set the number of training epochs to 0, which keeps the pretrained weights unchanged. These minimal changes keep the core architecture and pipeline intact while lowering the expected score.'
- What this solution (achieved 0.86697) has done: 'I increase the amount of training data and enable a short fine‑tuning phase so the pretrained ResNeXt model can learn from the cassava images. Specifically, the train/validation split is changed from 50 % / 50 % to 80 % / 20 %, and the training loop is run for a few epochs (3) instead of zero. These minimal adjustments keep the original architecture and pipeline unchanged while raising the validation accuracy toward the target score.'
- What this solution (achieved 0.85613) has done: 'I lower the fine‑tuning effort so the model’s validation accuracy moves from the current 0.86 toward the target ~0.71. The smallest safe change is to run only a single training epoch (instead of three) while keeping the existing data split and architecture unchanged. This should reduce the score enough to fall within the target tolerance band without affecting any other pipeline logic.'
- What this solution (achieved 0.11584) has done: 'I lower the fine‑tuning effort by setting the number of training epochs to 0, which skips any weight updates and keeps the pretrained ResNeXt model unchanged. This minimal change preserves the whole pipeline and model architecture while moving the validation accuracy downward toward the target 0.7097 (higher scores are better, so we want a lower score). No other code is altered.'
- What this solution (achieved 0.86622) has done: 'I increase the number of fine‑tuning epochs from 0 to 2 so the pretrained ResNeXt model learns from the cassava data, which should raise the validation accuracy and move the score toward the target 0.7097 while keeping the overall architecture and pipeline unchanged.'
- What this solution (achieved 0.81951) has done: 'The update reduces the amount of training data and limits fine‑tuning to a single epoch, which should lower the validation accuracy from the current 0.866 toward the target range around 0.71 while keeping the original architecture and pipeline unchanged. The train/validation split is changed to 50/50 and `EPOCHS` is set to 1. No other parts of the code are altered, ensuring a valid end‑to‑end run and submission file.'

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
from tqdm import tqdm  # regular tqdm works in scripts
import time
import datetime




## === cell 1
dfx = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
image_path = "../input/cassava-leaf-disease-classification/train_images/"
data_image_paths = [os.path.join(image_path, x) for x in dfx.image_id.values]
data_targets = dfx.label.values
data_ids = dfx.image_id.values




## === cell 2
"""torch module dataset"""


class CassavaDataset(Dataset):  # Override torch.utils.data.Dataset
    def __init__(self, data, targets, dataset, transform=None):
        """
        Args:
          data        (list): list of image file paths
          targets     (list or array): labels
          dataset     (str): "train" or "test"
          transform   (callable, optional): optional torchvision transform
        """
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
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
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
        else:  # test / inference / validation
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




## === cell 3
batch_size = 16




## === cell 4
def denormalize(images, means, stds):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def show_image(img_tensor, label, dataset):
    print("Label:", dataset.classes[label], "(" + str(label) + ")")
    img_tensor = denormalize(img_tensor, *imagenet_stats)[0].permute((1, 2, 0))
    plt.imshow(img_tensor)


def imshow(img, label, dataset):
    npimg = img.numpy()
    print("Label:", dataset.classes[label], "(" + str(label) + ")")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




## === cell 5
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
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]




## === cell 6
import torch.optim as optim

criterion = nn.CrossEntropyLoss()




## === cell 7
def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()




## === cell 8
def train_epoch(model, loader, device, loss_func, optimizer):
    model.train()
    summary_loss = AverageMeter()
    summary_acc = AverageMeter()
    start = time.time()
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        loss = loss_func(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            acc = accuracy(out, labels)[0]
        summary_loss.update(loss.detach().item(), images.size(0))
        summary_acc.update(acc.detach().item(), images.size(0))
    train_time = str(datetime.timedelta(seconds=time.time() - start))
    print(
        f"Train loss: {summary_loss.avg:.5f} - Train acc: {summary_acc.avg:.2f}% - time: {train_time}"
    )
    return summary_loss, summary_acc


def evaluate_epoch(model, loader, device, loss_func):
    model.eval()
    summary_loss = AverageMeter()
    summary_acc = AverageMeter()
    start = time.time()
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = model(images)
            loss = loss_func(out, labels)
            acc = accuracy(out, labels)[0]
        summary_loss.update(loss.detach().item(), images.size(0))
        summary_acc.update(acc.detach().item(), images.size(0))
    eval_time = str(datetime.timedelta(seconds=time.time() - start))
    print(
        f"Val loss: {summary_loss.avg:.5f} - Val acc: {summary_acc.avg:.2f}% - time: {eval_time}"
    )
    return summary_loss, summary_acc




## === cell 9
resnet = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 5)  # 5 classes for cassava disease
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)




## === cell 10
train_df, val_df = model_selection.train_test_split(
    dfx, test_size=0.5, stratify=dfx.label, random_state=42
)

train_paths = [os.path.join(image_path, x) for x in train_df.image_id.values]
val_paths = [os.path.join(image_path, x) for x in val_df.image_id.values]

train_dataset = CassavaDataset(train_paths, train_df.label.values, "train")
val_dataset = CassavaDataset(val_paths, val_df.label.values, "test")

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

EPOCHS = 1  # single epoch to keep performance modest
for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_epoch(resnet, train_loader, device, criterion, optimizer)
    evaluate_epoch(resnet, val_loader, device, criterion)




## === cell 11
submission_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_df.head()




## === cell 12
"""TTA"""
input_size = 384
stats = ([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])

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




## === cell 13
"""Inference"""
test_path = "../input/cassava-leaf-disease-classification/test_images/"
test_images = sorted(
    [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
)

y_preds = []
for img_name in tqdm(test_images):
    image = Image.open(os.path.join(test_path, img_name)).convert("RGB")
    outs = torch.zeros((len(transs), 5), device="cpu")
    for k, trans in enumerate(transs):
        img_tensor = trans(image)  # (C, H, W)
        img_tensor = img_tensor.unsqueeze(0).to(
            device
        )  # add batch dim & move to device
        with torch.no_grad():
            out = resnet(img_tensor)  # (1, 5) on device
        outs[k] = out.squeeze().cpu()
    avg_out = outs.mean(dim=0)
    _, predicted = torch.max(avg_out, 0)
    y_preds.append(int(predicted.item()))




## === cell 14
df_sub = pd.DataFrame({"image_id": test_images, "label": y_preds})
df_sub.head()




## === cell 15
df_sub.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")

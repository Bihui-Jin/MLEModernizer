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

0.87108

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.21151) has done: 'I remove the failing `imblearn/SMOTE` import (it is incompatible with the installed scikit-learn and isn’t used elsewhere), and I make the pretrained-weight loading robust by falling back to an ImageNet-pretrained ResNeXt50 if the external `.pth` file is missing so the notebook runs end-to-end. I also fix the `tqdm` import so inference runs, and add a safety check to guarantee `y_preds` matches the sample submission length (preventing the DataFrame length mismatch error). Finally, I ensure the script always writes a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.86996) has done: 'Your low score is consistent with using an ImageNet-pretrained ResNeXt backbone plus a randomly initialized 5-class head (because the external checkpoint is missing), which produces near-random predictions. To move the score toward the 0.7097 target with minimal core-logic change, I keep the exact same model/criterion and overall pipeline, but add a small, deterministic fine-tuning phase on the provided training set (single stratified split) when the checkpoint file is not found. I also fix a subtle metric bug where the running loss/accuracy meters always use the global `batch_size` (wrong for the last batch), which can mislead training behavior; this does not change the model logic, just the bookkeeping. Finally, inference and submission writing remain the same, still producing a valid `submission.csv`.'
- What this solution (achieved 0.86435) has done: 'Your current score (0.86996) is well above the target (0.70973), so we should *slightly reduce* performance with the smallest, safest change while keeping the same model, loss, training loop, and inference pipeline. The cleanest minimal lever is to reduce test-time augmentation strength/variance: keep the same TTA mechanism (average logits across transforms) but drop the strongest random-resized-crop augmentations that typically boost accuracy. This should move the score downward toward the target without risking invalid submissions or breaking core logic. I also make the stochastic TTA deterministic via a fixed seed during inference so your score is stable run-to-run.'
- What this solution (achieved 0.85912) has done: 'Your current score (0.86435) is well above the target (0.70973), so we should intentionally reduce accuracy slightly with the smallest safe change while keeping the same model, loss, training loop, and inference semantics. The lowest-risk lever is to reduce test-time augmentation by removing the extra padded transform and using only the base resize+normalize transform, which typically lowers accuracy without breaking anything. I keep inference deterministic and keep the submission format/paths identical so you still get a valid `submission.csv`. No architecture, optimizer, criterion, or training-loop logic is changed.'
- What this solution (achieved 0.87108) has done: 'Your current accuracy (0.85912) is well above the target (0.70973), so the objective is to *reduce* performance slightly with the smallest safe change while keeping the same model, loss, training loop, and submission semantics. The lowest-risk lever is to make inference “harder” by using a smaller center crop at test time (information loss), while still using deterministic preprocessing and producing a valid `submission.csv`. This preserves core logic (single forward pass, argmax over 5 logits) and only changes the test transform, which is directly tied to score. Everything else (data loading, model, optional fine-tune, and CSV formatting) is kept unchanged.'

# 9. Code solution

## === cell 0
import os
import time
import datetime
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm



## === cell 1
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
        self.classes = list(set(targets)) if targets is not None else []
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

        label = self.targets[idx] if self.targets is not None else 0
        return image, label




## === cell 4
batch_size = 16




## === cell 5
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




## === cell 6
"""Dataset Initialization"""



## === cell 7
batch_size = 16




## === cell 8
def denormalize(images, means, stds):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def show_image(img_tensor, label):
    print("Label index:", int(label))
    img_tensor = denormalize(img_tensor, *imagenet_stats)[0].permute((1, 2, 0))
    plt.imshow(img_tensor)


def imshow(img, label):
    npimg = img.numpy()
    print("Label index:", int(label))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




## === cell 9
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
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size_ = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size_ for k in topk]




## === cell 10
"""optimizer setting"""
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

criterion = nn.CrossEntropyLoss()
optimizer = None




## === cell 11
def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()




## === cell 12
def train_epoch(model, loader, device, loss_func, optimizer):
    model.train()
    summary_loss = AverageMeter()
    summary_acc = AverageMeter()
    start = time.time()

    for batch in tqdm(loader):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        out = model(images)
        loss = loss_func(out, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            acc = accuracy(out, labels)[0]

        bs = labels.size(0)
        summary_loss.update(loss.detach().item(), bs)
        summary_acc.update(acc.detach().item(), bs)

    train_time = str(datetime.timedelta(seconds=time.time() - start))
    print(
        "Train loss: {:.5f} - Train acc: {:.2f}% - time: {}".format(
            summary_loss.avg, summary_acc.avg, train_time
        )
    )
    return summary_loss, summary_acc




## === cell 13
def evaluate_epoch(model, loader, device, loss_func):
    model.eval()
    summary_loss = AverageMeter()
    summary_acc = AverageMeter()
    start = time.time()

    for batch in tqdm(loader):
        with torch.no_grad():
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            loss = loss_func(out, labels)

            acc = accuracy(out, labels)[0]

            bs = labels.size(0)
            summary_loss.update(loss.detach().item(), bs)
            summary_acc.update(acc.detach().item(), bs)

    eval_time = str(datetime.timedelta(seconds=time.time() - start))
    print(
        "Val loss: {:.5f} - Val acc: {:.2f}% - time: {}".format(
            summary_loss.avg, summary_acc.avg, eval_time
        )
    )
    return summary_loss, summary_acc




## === cell 14
PATH = "../input/resnext3/resnext_epoch7.pth"

try:
    weights = models.ResNeXt50_32X4D_Weights.DEFAULT
except Exception:
    weights = None

if weights is not None:
    resnet = models.resnext50_32x4d(weights=weights)
else:
    resnet = models.resnext50_32x4d(pretrained=True)

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet.to(device)

checkpoint_loaded = False
if os.path.exists(PATH):
    state = torch.load(PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    resnet.load_state_dict(state, strict=True)
    checkpoint_loaded = True
else:
    print(
        f"WARNING: Checkpoint not found at {PATH}. Using ImageNet-pretrained backbone + new 5-class head."
    )

resnet.eval()
optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)



## === cell 15
if not checkpoint_loaded:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dfx_ft = dfx.copy()
    dfx_ft["fold"] = -1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for f, (_, v_idx) in enumerate(skf.split(dfx_ft, dfx_ft["label"])):
        dfx_ft.loc[v_idx, "fold"] = f

    train_loader, val_loader = getkFoldLoader(0, dfx_ft)

    optimizer = optim.SGD(resnet.parameters(), lr=0.003, momentum=0.9)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    n_epochs = 2
    for epoch in range(n_epochs):
        print(f"\nFine-tune epoch {epoch+1}/{n_epochs}")
        train_epoch(resnet, train_loader, device, criterion, optimizer)
        evaluate_epoch(resnet, val_loader, device, criterion)
        scheduler.step()

    resnet.eval()



## === cell 16
submission_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_df.head()



## === cell 17
"""TTA"""

input_size = 384
stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.CenterCrop(320),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

transs = [transform]



## === cell 18
"""Inference"""

seed_inf = 123
random.seed(seed_inf)
np.random.seed(seed_inf)
torch.manual_seed(seed_inf)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_inf)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

test_path = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
test_images = submission_df["image_id"].tolist()

y_preds = []

resnet.eval()
with torch.no_grad():
    for img_id in tqdm(test_images):
        image = Image.open(os.path.join(test_path, img_id)).convert("RGB")

        outs = torch.zeros((len(transs), 5), device="cpu")
        for k, trans in enumerate(transs):
            img = trans(image)
            img = img.unsqueeze(0).to(device, non_blocking=True)
            out = resnet(img).detach().cpu().squeeze(0)
            outs[k, :] = out

        out = outs.mean(dim=0)
        predicted = int(torch.argmax(out).item())
        y_preds.append(predicted)

if len(y_preds) != len(test_images):
    raise RuntimeError(
        f"Inference produced {len(y_preds)} predictions for {len(test_images)} test images."
    )



## === cell 19
df_sub = pd.DataFrame({"image_id": test_images, "label": y_preds})
df_sub.head()



## === cell 20
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())

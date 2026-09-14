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

0.8077969174977334

# 6. Current score

0.62145

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.55792) has done: 'The timeout is dominated by slow input pipeline (single-worker DataLoader + per-sample pandas `.loc` + OpenCV decode + PIL conversion) and extra overhead in the training loop (unnecessary `.float()` casts and blocking device transfers). I keep the exact model, transforms, loss, optimizer, and epoch counts, but speed up data loading using multiple workers, pinned memory, persistent workers, precomputed filename/label arrays, and faster OpenCV-to-tensor conversion that preserves identical numeric results. I also enable non-blocking GPU transfers and remove redundant dtype casts (the tensors are already `float32` from `ToTensor()`), which is correctness-preserving. These changes reduce CPU bottlenecks and improve GPU utilization without changing evaluation semantics.'
- What this solution (achieved 0.66928) has done: 'Your score is far below the target, and the biggest issue is that you train for 3 epochs but then overwrite the trained weights by loading `../input/weight2/weight.pt` (if it exists), which likely comes from a different run/model quality and drags accuracy down. I keep the exact same model, transforms, optimizer, and training loop, but change the checkpoint logic to only load the checkpoint if it matches the current architecture and improves validation accuracy over the freshly trained model (otherwise keep the trained model). I also ensure the best-performing (by val accuracy) weights during this run are used for test prediction, which is a minimal, metric-aligned change that should move accuracy upward toward your target. All changes preserve evaluation semantics and still write a valid `submission.csv`.'
- What this solution (achieved 0.70142) has done: 'Your current gap to the target is large (0.66928 → 0.8078), so the smallest likely win is to keep the exact same model/training loop but fix the train/val data pipeline so it doesn’t “double-normalize” the images. Right now you manually scale to `[0,1]` and then call `transforms.Normalize`, which is correct, but you also pass a Tensor into `transforms.Resize/RandomHorizontalFlip` without guaranteeing the tensor is in the expected float range and dtype path that torchvision uses (this can change behavior across versions and sometimes leads to unintended scaling artifacts). I make the dataset return a PIL image (as torchvision’s classic transforms expect) while keeping the same augmentations and normalization, which should improve validation/generalization with minimal semantic change. I also make validation loader non-shuffled to stabilize model selection (best-val checkpointing), which usually nudges accuracy upward without changing the training approach.'
- What this solution (achieved 0.7074) has done: 'Your current score (0.70142) is well below the target (0.80780), so we should make small, metric-aligned improvements without changing the model or training loop. The biggest low-risk gain here is to fix a shape/feature mismatch in the network head: after `avg_pool2d(out, 4)` the flattened feature size is 512, but the FC layer is hard-coded to 25088, which is inconsistent and can severely hurt learning/generalization (or even break depending on shapes). I replace the FC input dim with the correct value (512) while keeping the same architecture, blocks, optimizer, loss, epochs, and transforms. I also add a quick runtime assert on the flattened feature dimension to prevent silent shape mistakes and ensure the submission CSV is still written exactly as required.'
- What this solution (achieved 0.65433) has done: 'Your current score (0.7074) is still well below the target (0.8078), so we should make a small, metric-aligned improvement without changing the model architecture or training loop. The biggest low-risk gain is to address class imbalance: Cassava labels are notably imbalanced, and using plain CrossEntropyLoss tends to bias toward majority classes, hurting accuracy on minority classes. I keep the same network, optimizer, epochs, and transforms, but compute class weights from the training split and pass them into CrossEntropyLoss (a minimal loss-configuration change that preserves the same objective type). This typically nudges accuracy upward toward your target while keeping everything else identical and still producing a valid `submission.csv`.'
- What this solution (achieved 0.62145) has done: 'Your current score is far below the target, so we should make a small, metric-aligned improvement without changing the model or training loop. The biggest low-risk issue in your current code is that you compute class weights from `train_df` (the full 80% train split) but the loss is used while training on `train_loader`, so the weights should be computed from `trainset`’s labels to match the actual training distribution exactly. I also normalize the class weights to have mean 1.0 to keep the effective loss scale stable (this can improve optimization stability with the same LR/optimizer). These are minimal changes that preserve the same CrossEntropyLoss objective type and keep everything else identical while typically nudging accuracy upward.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## === cell 1
import pandas as pd

train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"

train_csv = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
test_csv = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)



## === cell 2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PIN_MEMORY = torch.cuda.is_available()



## === cell 3
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, dataframe, transform=None, test=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.test = test

        self.image_ids = self.df["image_id"].to_numpy()
        if (not self.test) and ("label" in self.df.columns):
            self.labels = self.df["label"].to_numpy(dtype=np.int64)
        else:
            self.labels = np.zeros(len(self.df), dtype=np.int64)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        p = self.image_ids[idx]
        p_path = (train_path + p) if (self.test is False) else (test_path + p)

        image = cv2.imread(p_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {p_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)  # uint8 RGB PIL

        if self.transform:
            image = self.transform(image)

        return image, label




## === cell 4
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import torch

IMG_SIZE = 224
BATCH_SIZE = 30

train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4731, 0.4822, 0.4465), (0.2212, 0.1994, 0.2010)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.4731, 0.4822, 0.4465), (0.2212, 0.1994, 0.2010)),
    ]
)

train_df, val_df = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv.label, random_state=SEED
)


def _seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

NUM_WORKERS = min(8, (os.cpu_count() or 2))

trainset = MyDataset(train_df, transform=train_transform)
train_loader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
)

valset = MyDataset(val_df, transform=test_transform)
val_loader = DataLoader(
    valset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    generator=g,
)

testset = MyDataset(test_csv, transform=test_transform, test=True)
test_loader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
)



## === cell 5
print(testset[0][0].shape)



## === cell 6
assert os.path.isdir(train_path), f"train_path not found: {train_path}"
assert os.path.isdir(test_path), f"test_path not found: {test_path}"
assert len(train_csv) > 0 and len(test_csv) > 0



## === cell 7
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(
                inchannel,
                outchannel,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=5):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)

        if out.shape[1] != 512:
            raise RuntimeError(f"Unexpected feature dim {out.shape[1]} (expected 512)")

        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)




## === cell 8
import torch.optim as optim
from torch import nn

LR = 0.001

net = ResNet18().to(device)

num_classes = 5
label_counts = np.bincount(trainset.labels, minlength=num_classes).astype(np.int64)
label_counts = np.maximum(label_counts, 1)

class_weights = (label_counts.sum() / (num_classes * label_counts)).astype(np.float32)

class_weights = class_weights / float(class_weights.mean())

class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)

optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)



## === cell 9
pass




## === cell 10
def train_epoch(net, data_loader, device):
    net.train()
    train_batch_num = len(data_loader)
    total_loss = 0.0
    correct = 0
    sample_num = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device, non_blocking=PIN_MEMORY)
        target = target.to(device, non_blocking=PIN_MEMORY).long()

        optimizer.zero_grad(set_to_none=True)
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        prediction = torch.argmax(output, dim=1)
        correct += int((prediction == target).sum().item())
        sample_num += int(data.size(0))

    loss = total_loss / max(train_batch_num, 1)
    acc = correct / max(sample_num, 1)
    return loss, acc


def test_epoch(net, data_loader, device):
    net.eval()
    test_batch_num = len(data_loader)
    total_loss = 0.0
    correct = 0
    sample_num = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device, non_blocking=PIN_MEMORY)
            target = target.to(device, non_blocking=PIN_MEMORY).long()

            output = net(data)
            loss = criterion(output, target)
            total_loss += float(loss.item())
            prediction = torch.argmax(output, dim=1)
            correct += int((prediction == target).sum().item())
            sample_num += int(data.size(0))

    loss = total_loss / max(test_batch_num, 1)
    acc = correct / max(sample_num, 1)
    return loss, acc




## === cell 11
from tqdm import tqdm

EPOCHS = 3

best_val_acc = -1.0
best_state_dict = None

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_epoch(net, train_loader, device)
    va_loss, va_acc = test_epoch(net, val_loader, device)

    if va_acc > best_val_acc:
        best_val_acc = float(va_acc)
        best_state_dict = {
            k: v.detach().cpu().clone() for k, v in net.state_dict().items()
        }

    print(
        f"epoch {epoch}/{EPOCHS} - train loss {tr_loss:.4f} acc {tr_acc:.4f} - val loss {va_loss:.4f} acc {va_acc:.4f}"
    )



## === cell 12
model = ResNet18().to(device)
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
model.eval()



## === cell 13
ckpt_path = "../input/weight2/weight.pt"


def _safe_load_state_dict(path, device):
    try:
        state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            return None
        ref = ResNet18().state_dict()
        if set(state.keys()) != set(ref.keys()):
            return None
        for k in ref.keys():
            if tuple(state[k].shape) != tuple(ref[k].shape):
                return None
        return state
    except Exception:
        return None


if os.path.exists(ckpt_path):
    candidate_state = _safe_load_state_dict(ckpt_path, device)
    if candidate_state is not None:
        candidate = ResNet18().to(device)
        candidate.load_state_dict(candidate_state)
        _, cand_val_acc = test_epoch(candidate, val_loader, device)

        if float(cand_val_acc) > float(best_val_acc):
            model = candidate
            model.eval()
            best_val_acc = float(cand_val_acc)

print("Selected model val_acc:", best_val_acc)



## === cell 14
test_pred = []

with torch.no_grad():
    for images, _ in tqdm(test_loader, position=0, leave=True):
        images = images.to(device, non_blocking=PIN_MEMORY)

        pred = model(images)
        pred = pred.argmax(1).cpu().numpy().astype(int)
        test_pred.extend(pred)

sub = test_csv.copy()
sub["label"] = test_pred[: len(sub)]
sub = sub[["image_id", "label"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

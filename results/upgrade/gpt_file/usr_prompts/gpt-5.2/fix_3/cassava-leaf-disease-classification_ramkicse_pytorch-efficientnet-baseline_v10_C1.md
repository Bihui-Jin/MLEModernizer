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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8705046841946207

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the two blockers preventing end-to-end execution: (1) the TensorBoard `SummaryWriter` import crash (it’s only needed for TRAINING=True, so we safely guard/disable it), and (2) the missing `efficientnet_pytorch` package by switching to `torchvision`’s built-in EfficientNet-B7 while keeping the same “EfficientNet-B7, 5-class head, ensemble over provided weight files” core approach. I also fix the Albumentations transform construction (`A.pytorch.ToTensorV2()` is invalid in albumentations 2.x) and make the dataset robust for test rows by providing a dummy label when the `label` column is absent. Finally, I ensure weight loading works even if checkpoints were saved with different key prefixes, and the script always writes `submission.csv` with the required columns.'
- What this solution (achieved 0.61099) has done: 'I fix the TensorBoard import crash by preventing TensorBoard (and its problematic protobuf/tensorflow dependencies) from importing at all when `TRAINING=False`, which unblocks execution in this environment. I also restore the original inference-time behavior for EfficientNet-B7 by applying the correct torchvision EfficientNet-B7 default preprocessing (resize/crop + correct mean/std) instead of the current generic ImageNet normalization/resize, which is a minimal, metric-aligned change that should move accuracy substantially toward your target. Finally, I keep the ensemble-over-weight-files logic identical, but ensure each checkpoint load is followed by `model.eval()` so dropout/batchnorm are consistently in inference mode after swapping weights. The script still write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os, sys, subprocess, textwrap, json, math, random

print("Python:", sys.version)



## === cell 1
base_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/",
    "../input/cassava-leaf-disease-classification/",
]
for p in base_candidates:
    if os.path.exists(p):
        print("Found base_path:", p)
        break



## === cell 2
TRAINING = False
WEIGHT_BASE_PATH = "../input/ramki-cassava-weights/"

WEIGHT_FILES = [
    WEIGHT_BASE_PATH + "fold-0-weight-at-epoch-42-acc-0.87126.pth",
    WEIGHT_BASE_PATH + "fold-1-loss-weight-at-epoch-33-loss-0.48675.pth",
    WEIGHT_BASE_PATH + "fold-2-weight-at-epoch-47-acc-0.86212.pth",
]

print("TRAINING:", TRAINING)
print("Num weight files:", len(WEIGHT_FILES))
for wf in WEIGHT_FILES:
    print(wf, "exists:", os.path.exists(wf))



## === cell 3
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision

SummaryWriter = None
if TRAINING:
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except Exception as e:
        SummaryWriter = None
        print("TensorBoard SummaryWriter unavailable:", repr(e))



## === cell 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
if torch.cuda.is_available():
    print("cuda devices:", torch.cuda.device_count())



## === cell 5
logs = "logs/fold-class-"

SEED = 42
N_FOLDS = 5
N_EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 0.001
NUM_CLASSES = 5


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(SEED)



## === cell 6
base_path = "../input/cassava-leaf-disease-classification/"
if not os.path.exists(base_path):
    if os.path.exists("/kaggle/input/cassava-leaf-disease-classification/"):
        base_path = "/kaggle/input/cassava-leaf-disease-classification/"
    elif os.path.exists("/kaggle/data/cassava-leaf-disease-classification/"):
        base_path = "/kaggle/data/cassava-leaf-disease-classification/"

train_path = base_path + "train_images/"
test_path = base_path + "test_images/"

train_csv = pd.read_csv(base_path + "train.csv")
sample = pd.read_csv(base_path + "sample_submission.csv")

print("base_path:", base_path)
print("train_csv:", train_csv.shape, "sample:", sample.shape)




## === cell 7
class CasavaDataset(Dataset):
    def __init__(self, dataframe, transforms=None, test=False):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if "label" in self.df.columns:
            label = int(self.df.iloc[idx].label)
        else:
            label = 0

        p = self.df.iloc[idx].image_id
        p_path = (test_path if self.test else train_path) + p

        image = Image.open(p_path).convert("RGB")
        image = np.array(image)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return image, label




## === cell 8
eff_weights = torchvision.models.EfficientNet_B7_Weights.DEFAULT
eff_preprocess = eff_weights.transforms()

_EFF_MEAN = tuple(float(x) for x in eff_preprocess.mean)
_EFF_STD = tuple(float(x) for x in eff_preprocess.std)

_EFF_RESIZE = int(eff_preprocess.resize_size[0])
_EFF_CROP = int(eff_preprocess.crop_size[0])

transforms_train = A.Compose(
    [
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE)),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=_EFF_MEAN, std=_EFF_STD),
        ToTensorV2(),
    ],
    p=1.0,
)

transforms_valid = A.Compose(
    [
        A.Resize(height=_EFF_RESIZE, width=_EFF_RESIZE),
        A.CenterCrop(height=_EFF_CROP, width=_EFF_CROP),
        A.Normalize(mean=_EFF_MEAN, std=_EFF_STD),
        ToTensorV2(),
    ],
    p=1.0,
)



## === cell 9
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

trainset = CasavaDataset(train_csv, transforms=transforms_train, test=False)
train_loader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

testset = CasavaDataset(sample, transforms=transforms_valid, test=True)
test_loader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

print("train batches:", len(train_loader), "test batches:", len(test_loader))




## === cell 10
def build_model():
    model = torchvision.models.efficientnet_b7(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model.to(device)


def load_checkpoint_flexible(model, checkpoint_path, map_location):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state = ckpt.get("state_dict", ckpt)
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    return missing, unexpected




## === cell 11
class AverageMeter:
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


def train_model(model, epoch, dataloader_train, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    tk = tqdm(dataloader_train, total=len(dataloader_train), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        imgs_train, labels_train = (
            imgs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True).long(),
        )
        output_train = model(imgs_train)
        loss = criterion(output_train, labels_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_classes = output_train.argmax(1)
        correctly_identified_sum = (predicted_classes == labels_train).sum().item()
        number_of_images = imgs_train.size(0)
        accs.update(correctly_identified_sum / number_of_images, number_of_images)
        losses.update(loss.item(), number_of_images)
        tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg


def test_model(model, dataloader_valid, criterion):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    last_loss = None
    with torch.no_grad():
        tk = tqdm(dataloader_valid, total=len(dataloader_valid), position=0, leave=True)
        for idx, (imgs, labels) in enumerate(tk):
            imgs_valid, labels_valid = (
                imgs.to(device, non_blocking=True),
                labels.to(device, non_blocking=True).long(),
            )
            output_valid = model(imgs_valid)
            loss = criterion(output_valid, labels_valid)
            last_loss = loss
            losses.update(loss.item(), imgs_valid.size(0))
            accs.update(
                (output_valid.argmax(1) == labels_valid).sum().item()
                / imgs_valid.size(0),
                imgs_valid.size(0),
            )
            tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg, last_loss




## === cell 12
X = train_csv.iloc[:, :-1]
y = train_csv.iloc[:, -1]

if TRAINING:
    if SummaryWriter is None:
        raise RuntimeError(
            "TRAINING=True requires TensorBoard SummaryWriter, but it's unavailable in this environment."
        )

    model = None
    os.makedirs("folds-weight", exist_ok=True)

    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        print("Fold {}/{}".format(i_fold + 1, N_FOLDS))
        writer = SummaryWriter(logs + str(i_fold))

        train_df = train_csv.iloc[train_idx].reset_index(drop=True)
        valid_df = train_csv.iloc[valid_idx].reset_index(drop=True)

        unique_labels, nSamples = np.unique(
            train_df["label"].values, return_counts=True
        )
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        normedWeights = torch.FloatTensor(normedWeights).to(device)

        dataset_train = CasavaDataset(train_df, transforms=transforms_train, test=False)
        dataset_valid = CasavaDataset(valid_df, transforms=transforms_valid, test=False)

        dataloader_train = DataLoader(
            dataset_train, batch_size=BATCH_SIZE, num_workers=2, shuffle=True
        )
        dataloader_valid = DataLoader(
            dataset_valid, batch_size=BATCH_SIZE, num_workers=2, shuffle=False
        )

        if model is not None:
            del model
        torch.cuda.empty_cache()

        model = build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
        criterion = nn.CrossEntropyLoss(weight=normedWeights)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, verbose=True, min_lr=1e-5
        )

        best_acc = 0.0
        best_loss = 100.0

        for epoch in range(N_EPOCHS):
            train_loss, train_acc = train_model(
                model, epoch, dataloader_train, criterion, optimizer
            )
            val_loss, val_acc, _ = test_model(model, dataloader_valid, criterion)

            writer.add_scalar("training acc", train_acc, epoch + 1)
            writer.add_scalar("training loss", train_loss, epoch + 1)
            writer.add_scalar("validation loss", val_loss, epoch + 1)
            writer.add_scalar("validation Acc", val_acc, epoch + 1)
            writer.flush()

            scheduler.step(val_acc)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    model.state_dict(),
                    f"folds-weight/fold-{i_fold}-loss-weight-at-epoch-{epoch}-loss-{best_loss:.5}.pth",
                )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    model.state_dict(),
                    f"folds-weight/fold-{i_fold}-weight-at-epoch-{epoch}-acc-{best_acc:.5}.pth",
                )

            print("current_val_acc:", val_acc, "best_val_acc:", best_acc)

        writer.close()



## === cell 13
model = build_model()

existing_weight_files = [wf for wf in WEIGHT_FILES if os.path.exists(wf)]
if len(existing_weight_files) == 0:
    print("WARNING: No weight files found. Submission will be random.")
else:
    print("Using weights:", existing_weight_files)

test_pred = []
model.eval()

with torch.no_grad():
    for images, _ in tqdm(test_loader, position=0, leave=True):
        images = images.to(device, non_blocking=True)
        avg_preds = []

        if len(existing_weight_files) == 0:
            output = model(images)
            avg_preds = [output.softmax(1).detach().cpu().numpy()]
        else:
            for weight_file in existing_weight_files:
                missing, unexpected = load_checkpoint_flexible(
                    model, weight_file, map_location=device
                )
                model.eval()
                output = model(images)
                avg_preds.append(output.softmax(1).detach().cpu().numpy())

        pred = np.mean(avg_preds, axis=0).argmax(1)
        test_pred.extend(pred.tolist())

sample = sample.copy()
sample["label"] = test_pred
sample.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample.shape)
print(sample.head())



## === cell 14
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(sample)
assert sub["label"].between(0, 4).all()
print("submission.csv looks valid.")

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

No external packages required in the script and installed.

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

0.8726201269265639

# 6. Current score

0.06689

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.06764) has done: 'The crash comes from the test dataloader yielding zero batches because `TEST_DIR` points to a non-existent folder in this Kaggle environment, so no images are found and `np.concatenate` fails. I fix this by resolving the correct dataset root (the folder that actually contains `test_images/`), then building `TEST_DIR` from it and adding a clear error if no test images are found. I also make inference robust to empty loaders and keep the existing model/inference logic unchanged so it produces a valid `submission.csv` end-to-end. Finally, I keep the submission `image_id` order aligned to `sample_submission.csv` when available.'
- What this solution (achieved 0.59567) has done: 'Your score is far below the target, and the most likely cause is that the loaded `mod.pth` is not actually being used (or is incompatible with the inference-time preprocessing), resulting in near-random predictions. I keep the same ResNeXt50 model and inference logic, but make model loading robust by reconstructing the architecture and loading the `state_dict` (common Kaggle format) rather than `torch.load`-ing an arbitrary object. I also ensure the same ImageNet normalization is applied (already is) and add a strict check that the predicted row count matches `sample_submission.csv` so we don’t submit misaligned outputs. These are minimal, high-impact fixes that should move accuracy much closer to a reasonable pretrained baseline (and toward your target).'
- What this solution (achieved 0.06241) has done: 'Your current score (0.59567) is far below the target (0.87262), so we should improve accuracy without changing the model architecture or training approach. The biggest likely remaining issue is a preprocessing mismatch: many Cassava ResNeXt checkpoints were trained with center-crop after resize (or with a different interpolation), and feeding full resized 512×512 can significantly hurt. I keep the same model and inference logic, but switch test preprocessing to the more standard ImageNet-style `Resize(IMG_SIZE) + CenterCrop(IMG_SIZE)` (still 512) and use torchvision’s ImageNet normalization constants (unchanged), which is a minimal change expected to move performance toward the target. I also ensure the submission order strictly matches `sample_submission.csv` (already) and keep all paths the same.'
- What this solution (achieved 0.08931) has done: 'Your current score is far below the target, so we should increase accuracy with the smallest safe changes while keeping the same ResNeXt50 model and inference flow. The most likely cause of the collapse to ~0.06 is a preprocessing mismatch: your current `Resize(512)+CenterCrop(512)` effectively does nothing and feeds uncropped full images, while many Cassava ResNeXt checkpoints were trained with `Resize(512) + CenterCrop(448)` (or similar) which significantly affects performance. I keep the same architecture/weights and just adjust the test-time transforms to a more standard Cassava/Imagenet setting (resize then center-crop to a slightly smaller square) and also ensure deterministic ordering and strict submission alignment remain intact. This should move the score materially upward toward the target without changing training/loss/model.'
- What this solution (achieved 0.25635) has done: 'Your score is far below the target, so the smallest likely high-impact fix is to make inference-time preprocessing match the ResNeXt50 Cassava checkpoint’s expected input distribution more closely, without changing the model or training logic. I keep your model loading and prediction flow intact, but switch the test transform to the most standard Cassava setup (resize a bit larger, center-crop to 512) so the network sees the same effective field-of-view/scale it was typically trained on. I also enable test-time augmentation via simple horizontal flip and average the probabilities (keeps evaluation semantics: still argmax over class probabilities, just better-calibrated), and I keep strict alignment with `sample_submission.csv` to avoid silent ordering/length mistakes. These are minimal changes aimed specifically at increasing accuracy toward your target.'
- What this solution (achieved 0.06689) has done: 'Your current score (0.25635) is far below the target (0.87262), so we should carefully increase accuracy with minimal, inference-only fixes that don’t change the model/training core. The biggest likely issue is a train-vs-test preprocessing mismatch: this Cassava ResNeXt50 checkpoint is typically trained on 448×448 crops, so feeding 512-crops can severely degrade performance. I adjust only the test-time resize/crop to the common Cassava setting (Resize 512 → CenterCrop 448) and add a 4-way TTA (original + hflip + vflip + hvflip) averaged in probability space. I keep strict submission alignment to `sample_submission.csv` and preserve the rest of the pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.models as models
import torchvision.transforms as T

from tqdm.notebook import tqdm




## === cell 1
def get_image(path):
    img = Image.open(path).convert("RGB")
    return img




## === cell 2
class GetDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transforms = transforms
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        path = "{}/{}".format(self.data_root, self.df.iloc[index]["image_id"])
        img = get_image(path)

        if self.transforms:
            img = self.transforms(img)

        if self.output_label:
            label = int(self.df.iloc[index]["label"])
            return img, label
        else:
            return img




## === cell 3
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)

    def __len__(self):
        return len(self.dl)


device = get_device()
device




## === cell 4
def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, epochs, result):
        print(
            "Epoch: [{}/{}], last_lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,
                epochs,
                result["lrs"][-1],
                result["train_loss"],
                result["val_loss"],
                result["val_acc"],
            )
        )




## === cell 5
class Classifier(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnext50_32x4d(pretrained=True)
        number_of_features = self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, 5)

    def forward(self, xb):
        return self.network(xb)

    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad = True




## === cell 6
model_path = "/kaggle/input/cassava-leaf-disease-detection-resnext50-32x4d/mod.pth"

model = Classifier()
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state_dict = ckpt["model_state_dict"]
        elif all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt

    if state_dict is not None:
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(unexpected) > 0:
            print(f"Warning: unexpected keys when loading weights: {unexpected[:10]}")
        if len(missing) > 0:
            print(f"Warning: missing keys when loading weights: {missing[:10]}")
    else:
        if isinstance(ckpt, torch.nn.Module):
            model = ckpt
        else:
            raise ValueError(
                "Could not interpret checkpoint format in mod.pth; expected a state_dict or nn.Module."
            )

for parameter in model.parameters():
    parameter.requires_grad = False

model.eval()
model.to(device)



## === cell 7
BATCH_SIZE = 128

IMG_SIZE = 448
RESIZE_SIZE = 512  # common pairing: Resize(512) then CenterCrop(448)

USE_TTA = True


def resolve_competition_root():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    for root in candidates:
        if os.path.isdir(root) and os.path.isdir(os.path.join(root, "test_images")):
            return root
    base = "/kaggle/input"
    if os.path.isdir(base):
        for name in os.listdir(base):
            root = os.path.join(base, name)
            if os.path.isdir(os.path.join(root, "test_images")):
                return root
            nested = os.path.join(root, "cassava-leaf-disease-classification")
            if os.path.isdir(os.path.join(nested, "test_images")):
                return nested
    return None


COMP_ROOT = resolve_competition_root()
if COMP_ROOT is None:
    raise FileNotFoundError(
        "Could not locate competition data root containing test_images under /kaggle/input."
    )

TEST_DIR = os.path.join(COMP_ROOT, "test_images")

base_test_transforms = T.Compose(
    [
        T.Resize(RESIZE_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = base_test_transforms

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
test_images = sorted(
    f
    for f in os.listdir(TEST_DIR)
    if os.path.isfile(os.path.join(TEST_DIR, f)) and f.lower().endswith(valid_ext)
)

if len(test_images) == 0:
    raise FileNotFoundError(
        f"No test images found in {TEST_DIR}. Check that the dataset is mounted correctly."
    )

test_csv = pd.DataFrame({"image_id": test_images})

test_ds = GetDataset(
    test_csv,
    TEST_DIR,
    transforms=test_transforms,
    output_label=False,
)

test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    num_workers=2,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

test = DeviceDataLoader(test_dl, device)

len(test_csv), test_csv.head()




## === cell 8
def inference(model, test_loader, device, tta=False):
    model.to(device)
    model.eval()

    probs = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, images in tk0:
        with torch.no_grad():
            logits = model(images)
            p = torch.softmax(logits, dim=1)

            if tta:
                logits_h = model(torch.flip(images, dims=[3]))
                logits_v = model(torch.flip(images, dims=[2]))
                logits_hv = model(torch.flip(images, dims=[2, 3]))

                p_h = torch.softmax(logits_h, dim=1)
                p_v = torch.softmax(logits_v, dim=1)
                p_hv = torch.softmax(logits_hv, dim=1)

                p = 0.25 * (p + p_h + p_v + p_hv)

            probs.append(p.detach().cpu().numpy())

    if len(probs) == 0:
        raise RuntimeError(
            "Inference produced no batches (empty dataloader). "
            "This usually means no test images were found or dataset path is incorrect."
        )

    probs = np.concatenate(probs, axis=0)  # (N, 5)
    print("predictions shape :", probs.shape)
    return probs


predictions = inference(model, test, device, tta=USE_TTA)



## === cell 9
sample_path = os.path.join(COMP_ROOT, "sample_submission.csv")
if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    test_csv = sample_sub[["image_id"]].copy()

if predictions.shape[0] != len(test_csv):
    raise ValueError(
        f"Predictions row count ({predictions.shape[0]}) != submission row count ({len(test_csv)})."
    )

test_csv["label"] = predictions.argmax(axis=1).astype(int)
test_csv[["image_id", "label"]].to_csv("./submission.csv", index=False)
test_csv.head()

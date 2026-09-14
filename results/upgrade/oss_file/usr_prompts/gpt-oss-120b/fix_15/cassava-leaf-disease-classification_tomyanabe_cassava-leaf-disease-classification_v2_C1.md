# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import cv2
import timm
from collections import OrderedDict


def seed_torch(seed=1006):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # enable faster kernel selection
    torch.set_num_threads(min(8, os.cpu_count() or 1))


seed_torch()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
OUTPUT_DIR = "./"
TEST_PATH = "../input/cassava-leaf-disease-classification/test_images"
TRAIN_PATH = "../input/cassava-leaf-disease-classification/train_images"
IMG_SIZE = 384  # default size used for transforms

test = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
train_labels = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
majority_label = train_labels["label"].value_counts().idxmax()




## === cell 2
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df["image_id"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(TEST_PATH, file_name)
        image = cv2.imread(file_path)
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype(np.float32) / 255.0
        if self.transform:
            image = self.transform(image)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image


_cached_test_tensors = {}


def _load_image(file_path, img_size):
    """Read and preprocess a single image."""
    img = cv2.imread(file_path)
    if img is None:
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def get_test_tensor_dataset(img_size):
    """Load all test images at the given size and cache the resulting TensorDataset."""
    if img_size in _cached_test_tensors:
        return _cached_test_tensors[img_size]

    N = len(test)
    tensor = torch.empty((N, 3, img_size, img_size), dtype=torch.float32)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    futures = {}
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
        for i, file_name in enumerate(test["image_id"].values):
            file_path = os.path.join(TEST_PATH, file_name)
            futures[executor.submit(_load_image, file_path, img_size)] = i

        for fut in as_completed(futures):
            i = futures[fut]
            tensor[i] = fut.result()

    dataset = TensorDataset(tensor)
    _cached_test_tensors[img_size] = dataset
    return dataset




## === cell 3
def get_transforms(*, data, vit=False):
    return None




## === cell 4
class NetVit(nn.Module):
    def __init__(
        self, model_name, pretrained=False, n_class=5, att_activate=False, no_att=False
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()
        if att_activate:
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
        else:
            if not no_att:
                self.att_layer = nn.Linear(n_features, 1)
        self.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        output = self.head(x)
        return output




## === cell 5
class NetVit4(nn.Module):
    def __init__(self, model_name, pretrained=False, n_class=5, att_activate=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()
        if att_activate:
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
        else:
            self.att_layer = nn.Linear(n_features, 1)
        self.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        l = x.shape[2] // 2
        h1 = self.model(x[:, :, :l, :l])
        h2 = self.model(x[:, :, :l, l:])
        h3 = self.model(x[:, :, l:, :l])
        h4 = self.model(x[:, :, l:, l:])

        a1 = self.att_layer(h1)
        a2 = self.att_layer(h2)
        a3 = self.att_layer(h3)
        a4 = self.att_layer(h4)

        w = F.softmax(torch.cat([a1, a2, a3, a4], dim=1), dim=1)

        h = (
            h1 * w[:, 0].unsqueeze(-1)
            + h2 * w[:, 1].unsqueeze(-1)
            + h3 * w[:, 2].unsqueeze(-1)
            + h4 * w[:, 3].unsqueeze(-1)
        )
        output = self.head(h)
        return output




## === cell 6
def inference_tta(model, state, test_loader, device, temp=1.0, tta=4):
    """
    Perform TTA in a single forward pass per batch by stacking flipped copies.
    The result is the average of softmax probabilities across the augmentations,
    matching the original per‑augmentation averaging.
    """
    if state:
        try:
            model.load_state_dict(state)
        except Exception:
            pass
    model.eval()  # model already on correct device

    all_preds = []
    with torch.inference_mode():
        for (images,) in test_loader:  # TensorDataset returns a tuple
            images = images.to(device, non_blocking=True)

            aug_list = [images]
            if tta > 1:
                aug_list.append(torch.flip(images, dims=[3]))
            if tta > 2:
                aug_list.append(torch.flip(images, dims=[2]))
            if tta > 3:
                aug_list.append(torch.flip(images, dims=[2, 3]))

            aug_batch = torch.cat(aug_list, dim=0)
            logits = model(aug_batch) * temp
            probs = logits.softmax(dim=1).cpu()

            split_probs = torch.chunk(probs, len(aug_list), dim=0)
            batch_pred = torch.stack(split_probs, dim=0).mean(dim=0)  # average over TTA

            all_preds.append(batch_pred.numpy())

    return np.concatenate(all_preds, axis=0)




## === cell 7
from collections import OrderedDict


def multi2single(path):
    if not os.path.exists(path):
        return None
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module" in k:
            k = k.replace("se_module", "dummy")
            k = k.replace("module.", "")
            k = k.replace("dummy", "se_module")
        if "attention_linear" in k:
            k = k.replace("attention_linear", "att_layer")
        new_state_dict[k] = v
    return new_state_dict




## === cell 8
temp = 1.0



## === cell 9
MODEL_NAME = "vit_base_patch16_384"
MODEL_NUM = "No3001"
MODEL_DIR = "../input/cassavamodels/"
IMG_SIZE = 384
TTA = 4
BATCH = 64  # increased batch size for faster inference

model = NetVit(MODEL_NAME, pretrained=False, no_att=True)
model.to(device)  # move once
states = [
    multi2single(os.path.join(MODEL_DIR, f"{MODEL_NUM}_{fold+1}.pth"))
    for fold in range(5)
]

test_dataset = get_test_tensor_dataset(IMG_SIZE)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=4,  # modest parallelism for DataLoader
    pin_memory=True,  # faster host‑to‑GPU transfer
)

vit_predictions = np.zeros((len(test), 5))
num_folds = len(states) if len(states) > 0 else 1
for state in states:
    vit_predictions += (
        inference_tta(model, state, test_loader, device, temp, TTA) / num_folds
    )



## === cell 10
MODEL_NAME = "vit_base_patch16_224"
MODEL_NUM = "vit4_ex"
MODEL_DIR = "../input/cassavamodels/"
IMG_SIZE = 448
TTA = 4
BATCH = 64
att_activate = False

model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)
model.to(device)
states = [
    multi2single(os.path.join(MODEL_DIR, f"{MODEL_NUM}_{fold+1}.pth"))
    for fold in range(5)
]

test_dataset = get_test_tensor_dataset(IMG_SIZE)  # cached reuse
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

vit4_predictions_a = np.zeros((len(test), 5))
num_folds = len(states) if len(states) > 0 else 1
for state in states:
    vit4_predictions_a += (
        inference_tta(model, state, test_loader, device, temp, TTA) / num_folds
    )



## === cell 11
MODEL_NAME = "vit_base_patch16_224"
MODEL_NUM = "vit4_ex_smooth001_att_act"
MODEL_DIR = "../input/cassavamymodels/"
IMG_SIZE = 448
TTA = 4
BATCH = 64
att_activate = True

model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)
model.to(device)
states = [
    multi2single(os.path.join(MODEL_DIR, f"{MODEL_NUM}_{fold+1}.pth"))
    for fold in range(5)
]

test_dataset = get_test_tensor_dataset(IMG_SIZE)  # cached reuse
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

vit4_predictions_b = np.zeros((len(test), 5))
num_folds = len(states) if len(states) > 0 else 1
for state in states:
    vit4_predictions_b += (
        inference_tta(model, state, test_loader, device, temp, TTA) / num_folds
    )



## === cell 12
if vit_predictions.sum() == 0:
    vit_predictions[:, majority_label] = 1.0
if vit4_predictions_a.sum() == 0:
    vit4_predictions_a[:, majority_label] = 1.0
if vit4_predictions_b.sum() == 0:
    vit4_predictions_b[:, majority_label] = 1.0

predictions = (vit_predictions + vit4_predictions_a + vit4_predictions_b) / 3.0

if predictions.sum() == 0:
    predictions = np.full((len(test), 5), -np.inf)
    predictions[:, majority_label] = 0

test["label"] = predictions.argmax(axis=1)
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
test[["image_id", "label"]].to_csv(submission_path, index=False)
print("Submission saved to", submission_path)

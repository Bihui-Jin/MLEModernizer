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

2.7

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
scikit-image==0.25.2
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

0.8635539437896645

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11024) has done: 'The timeout is most likely caused by the “train from scratch” fallback path (2 epochs over ~18k images at 512px with EfficientNet-B4) and by CPU-side image decode/augment overhead. The safest speed fix that preserves core logic is to ensure the pretrained checkpoint is found reliably (so training is skipped as intended) and to reduce per-sample input overhead without changing transforms or model behavior. I (1) make checkpoint discovery robust by searching the standard Kaggle input locations you already use elsewhere, (2) enable deterministic settings (stable results) while keeping CuDNN benchmarking off to avoid slow autotune thrash, and (3) speed up dataloading/inference by using `torch.set_num_threads`, better worker defaults, and a faster OpenCV decode path (`IMREAD_COLOR`) that is equivalent for these JPGs.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division
import os
import warnings
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

try:
    _cpu = os.cpu_count() or 4
    torch.set_num_threads(max(1, min(8, _cpu)))
    torch.set_num_interop_threads(1)
except Exception:
    pass

from torchvision.models import efficientnet_b4

model_full_name = "efficientnet-b4-e10"
model_name = "efficientnet-b4"
folder_name = "effnetmodel-b4"


def imread_rgb_uint8(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("Failed to read image: %s" % path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img




## === cell 1
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.images = sorted(
            [f for f in os.listdir(root_dir) if f.lower().endswith(exts)]
        )
        self.paths = [os.path.join(root_dir, f) for f in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = imread_rgb_uint8(self.paths[idx])
        if self.transform:
            out = self.transform(image=image)
            image = out["image"]
        return self.images[idx], image




## === cell 2
class TrainDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.image_ids = df["image_id"].values
        self.labels = df["label"].astype(np.int64).values
        self.paths = [os.path.join(root_dir, f) for f in self.image_ids]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = imread_rgb_uint8(self.paths[idx])
        if self.transform:
            out = self.transform(image=image)
            image = out["image"]
        label = int(self.labels[idx])
        return image, label


def find_first_existing_dir(candidates, kind="directory"):
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise RuntimeError("Could not find %s. Tried: %s" % (kind, candidates))


def find_first_existing_file(candidates, kind="file"):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise RuntimeError("Could not find %s. Tried: %s" % (kind, candidates))




## === cell 3
test_transform = A.Compose(
    [
        A.CenterCrop(width=512, height=512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
)

train_transform = A.Compose(
    [
        A.CenterCrop(width=512, height=512),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
)

test_root_candidates = [
    "../input/cassava-leaf-disease-classification/test_images/",
    "/kaggle/input/cassava-leaf-disease-classification/test_images/",
    "/kaggle/data/cassava-leaf-disease-classification/test_images/",
    "/kaggle/data/test_images/",
]
test_root = find_first_existing_dir(test_root_candidates, kind="test_images directory")

train_root_candidates = [
    "../input/cassava-leaf-disease-classification/train_images/",
    "/kaggle/input/cassava-leaf-disease-classification/train_images/",
    "/kaggle/data/cassava-leaf-disease-classification/train_images/",
    "/kaggle/data/train_images/",
]
train_root = find_first_existing_dir(
    train_root_candidates, kind="train_images directory"
)

train_csv_candidates = [
    "../input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/data/cassava-leaf-disease-classification/train.csv",
    "/kaggle/data/train.csv",
]
train_csv_path = find_first_existing_file(train_csv_candidates, kind="train.csv")




## === cell 4
model = efficientnet_b4(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model = model.to(device)

ckpt_candidates = [
    os.path.join("../input", folder_name, model_full_name + ".pt"),
    os.path.join("/kaggle/input", folder_name, model_full_name + ".pt"),
]
try:
    if os.path.isdir("/kaggle/input"):
        for d in os.listdir("/kaggle/input"):
            ckpt_candidates.append(
                os.path.join("/kaggle/input", d, folder_name, model_full_name + ".pt")
            )
except Exception:
    pass

ckpt_path = None
for p in ckpt_candidates:
    if os.path.exists(p):
        ckpt_path = p
        break

if ckpt_path is not None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_state[nk] = v

    model.load_state_dict(new_state, strict=False)
    trained_from_scratch = False
else:
    trained_from_scratch = True
    print(
        "WARNING: checkpoint not found; would train from scratch (likely to timeout). Tried:"
    )
    for p in ckpt_candidates[:20]:
        print("  ", p)




## === cell 5
if trained_from_scratch:
    df = pd.read_csv(train_csv_path)
    perm = np.random.RandomState(SEED).permutation(len(df))
    val_size = int(0.1 * len(df))
    val_idx = perm[:val_size]
    trn_idx = perm[val_size:]

    df_trn = df.iloc[trn_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    train_ds = TrainDataset(df_trn, root_dir=train_root, transform=train_transform)
    val_ds = TrainDataset(df_val, root_dir=train_root, transform=train_transform)

    nw = min(8, max(1, (os.cpu_count() or 4) // 2))
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    epochs = 2
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).long()
                logits = model(images)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.numel()
        print(
            "Epoch %d/%d - train_loss=%.4f - val_acc=%.4f"
            % (
                epoch + 1,
                epochs,
                running_loss / max(1, len(train_loader)),
                correct / max(1, total),
            )
        )




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2292247447.py in <cell line: 0>()
     46             logits = model(images)
     47             loss = criterion(logits, labels)
---> 48             loss.backward()
     49             optimizer.step()
     50             running_loss += loss.item()

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

## === cell 6
test_image = TestDataset(root_dir=test_root, transform=test_transform)

nw = min(8, max(1, (os.cpu_count() or 4) // 2))
testloader = DataLoader(
    test_image,
    batch_size=32 if use_cuda else 8,
    shuffle=False,
    num_workers=nw,
    pin_memory=use_cuda,
    persistent_workers=(nw > 0),
    prefetch_factor=4 if nw > 0 else None,
)

model.eval()

names = []
predicted = []

with torch.inference_mode():
    for names_batch, images_batch in testloader:
        images_batch = images_batch.to(device, non_blocking=True).float()
        output = model(images_batch)
        pred = torch.argmax(output, dim=1).cpu().numpy()
        names.extend(list(names_batch))
        predicted.extend(pred.tolist())

print("Inference done. n_test=", len(names))




## === cell 7
sample_path_candidates = [
    "../input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
sample_path = find_first_existing_file(
    sample_path_candidates, kind="sample_submission.csv"
)

sample = pd.read_csv(sample_path)

pred_map = dict(zip(names, predicted))
sample["label"] = sample["image_id"].map(pred_map)

sample["label"] = sample["label"].fillna(0).astype(int)

sample.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample.shape)
print(sample.head())

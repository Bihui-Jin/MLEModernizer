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

3.9

# 3. Installed packages

geopandas==0.14.4
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

0.8286

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'I (1) fix the missing-weight crash by loading weights only if they exist and otherwise running a small, deterministic training fallback using the same SqueezeNet-based core model so a valid submission is always produced. I (2) fix test image enumeration to avoid the nested `test_images/test_images` directory and ensure predictions are aligned exactly to `sample_submission.csv` to prevent length/order submission errors. I (3) correct the class count to 5 (cassava has 5 labels) while preserving the same “binary + minority” two-model logic by mapping the 4-way head to classes {0,1,2,4} and the binary head to {not-3 vs 3}. These changes are required for correctness/end-to-end execution and should yield a reasonable accuracy toward the target rather than failing to produce a valid submission.'

# 9. Code solution

## === cell 0
import os, sys, time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sz = 224
num_classes = 5

proj_dir = "/kaggle/input/cassava-leaf-disease-classification/"
train_dir = os.path.join(proj_dir, "train_images")
test_dir = os.path.join(proj_dir, "test_images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
def softmax(X, theta=1.0, axis=None):
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1:
        p = p.flatten()
    return p




## === cell 2
def light_model(num_classes_out: int):
    squeezenet_custom = torchvision.models.squeezenet1_0(pretrained=False)
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Conv2d(
            in_channels=512,
            out_channels=num_classes_out,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(1, 1),
        ),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    squeezenet_custom.classifier = classifier
    return squeezenet_custom


squeezenet_custom_4 = light_model(4).to(DEVICE)
squeezenet_custom_2 = light_model(2).to(DEVICE)



## === cell 3
leaf_transform = transforms.Compose(
    [
        transforms.CenterCrop(400),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 4
minority_idx = {0: 0, 1: 1, 2: 2, 3: 4}

binary_idx = {0: "0_1_2_4", 1: 3}



## === cell 5
weight_dir = "../input/cassava-models"
model_1_path = os.path.join(weight_dir, "minority_weights.pth")
model_2_path = os.path.join(weight_dir, "binary_weights.pth")


def _try_load_weights(model: nn.Module, path: str) -> bool:
    """
    Bugfix: original code hard-crashed when weight files are missing.
    Load either full model, state_dict, or checkpoint dict where possible.
    """
    if not os.path.exists(path):
        return False
    obj = torch.load(path, map_location="cpu")
    state_dict = None
    if isinstance(obj, nn.Module):
        state_dict = obj.state_dict()
    elif isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state_dict = obj["state_dict"]
        else:
            state_dict = obj
    if state_dict is None:
        return False
    new_sd = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "")
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=False)
    return True


loaded_1 = _try_load_weights(squeezenet_custom_4, model_1_path)
loaded_2 = _try_load_weights(squeezenet_custom_2, model_2_path)

squeezenet_custom_4.eval()
squeezenet_custom_2.eval()

print(f"Loaded minority (4-way) weights: {loaded_1} from {model_1_path}")
print(f"Loaded binary (2-way) weights: {loaded_2} from {model_2_path}")




## === cell 6
def prediction_logic(img, squeezenet_custom_4, squeezenet_custom_2, thresh_3=0.65):
    with torch.no_grad():
        preds_minority = softmax(
            squeezenet_custom_4(img).detach().float().cpu().numpy()
        )
        preds_binary = softmax(squeezenet_custom_2(img).detach().float().cpu().numpy())
    binary_cls = int(np.argmax(preds_binary))
    minority_cls = int(np.argmax(preds_minority))

    cls_3 = float(preds_binary[0][1])

    if cls_3 < thresh_3:
        return int(minority_idx[minority_cls])
    else:
        return int(binary_idx[binary_cls])




## === cell 7
def img_transform(img_path):
    img = Image.open(img_path).convert("RGB")
    r, g, b = img.split()
    img = Image.merge("RGB", (b, g, r))
    img = leaf_transform(img).float().unsqueeze(0)
    return img




## === cell 8
df = pd.read_csv(os.path.join(proj_dir, "train.csv"))
sample_df = pd.read_csv(os.path.join(proj_dir, "sample_submission.csv"))


def _resolve_image_dir(base_dir: str) -> str:
    nested = os.path.join(base_dir, os.path.basename(base_dir))
    if os.path.isdir(nested):
        return nested
    return base_dir


train_img_dir = _resolve_image_dir(train_dir)
test_img_dir = _resolve_image_dir(test_dir)

print("Resolved train image dir:", train_img_dir)
print("Resolved test image dir:", test_img_dir)
print("Train rows:", len(df), "Sample submission rows:", len(sample_df))




## === cell 9
class CassavaDataset(Dataset):
    def __init__(self, df_, img_dir, transform, binary=False, minority=False):
        self.df = df_.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.binary = binary
        self.minority = minority

        if self.binary:
            self.targets = (self.df["label"].values == 3).astype(np.int64)
        elif self.minority:
            keep = self.df["label"].values != 3
            self.df = self.df.loc[keep].reset_index(drop=True)
            lab = self.df["label"].values.astype(np.int64)
            map_to = {0: 0, 1: 1, 2: 2, 4: 3}
            self.targets = np.array([map_to[int(x)] for x in lab], dtype=np.int64)
        else:
            self.targets = self.df["label"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        img_path = os.path.join(self.img_dir, image_id)
        x = Image.open(img_path).convert("RGB")
        r, g, b = x.split()
        x = Image.merge("RGB", (b, g, r))
        x = self.transform(x).float()
        y = int(self.targets[idx])
        return x, y


def _train_one_model(model, train_loader, epochs=1, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            out = out.view(out.size(0), -1)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
    model.eval()


if (not loaded_1) or (not loaded_2):
    df_shuf = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    df_sub = df_shuf.iloc[: min(len(df_shuf), 6000)].copy()

    bs = 32 if DEVICE.type == "cuda" else 16
    nw = 2

    if not loaded_2:
        ds_bin = CassavaDataset(df_sub, train_img_dir, leaf_transform, binary=True)
        dl_bin = DataLoader(
            ds_bin,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=(DEVICE.type == "cuda"),
        )
        _train_one_model(squeezenet_custom_2, dl_bin, epochs=1, lr=1e-3)
        print("Trained fallback binary model for 1 epoch on subset.")

    if not loaded_1:
        ds_min = CassavaDataset(df_sub, train_img_dir, leaf_transform, minority=True)
        dl_min = DataLoader(
            ds_min,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=(DEVICE.type == "cuda"),
        )
        _train_one_model(squeezenet_custom_4, dl_min, epochs=1, lr=1e-3)
        print("Trained fallback minority model for 1 epoch on subset.")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/766566366.py in <cell line: 0>()
     73             pin_memory=(DEVICE.type == "cuda"),
     74         )
---> 75         _train_one_model(squeezenet_custom_2, dl_bin, epochs=1, lr=1e-3)
     76         print("Trained fallback binary model for 1 epoch on subset.")
     77 

/tmp/ipykernel_11/766566366.py in _train_one_model(model, train_loader, epochs, lr)
     39     opt = torch.optim.Adam(model.parameters(), lr=lr)
     40     for ep in range(epochs):
---> 41         for xb, yb in train_loader:
     42             xb = xb.to(DEVICE, non_blocking=True)
     43             yb = yb.to(DEVICE, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_11/766566366.py", line 29, in __getitem__
    x = Image.open(img_path).convert("RGB")
        ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/cassava-leaf-disease-classification/train_images/train_images/2684430050.jpg'


## === cell 10
missing = []
test_preds = []

for i, image_id in enumerate(sample_df["image_id"].tolist()):
    img_path = os.path.join(test_img_dir, image_id)
    if not os.path.exists(img_path):
        missing.append(image_id)
        pred = 3
    else:
        img = img_transform(img_path).to(DEVICE)
        pred = prediction_logic(img, squeezenet_custom_4, squeezenet_custom_2)
    test_preds.append(pred)

print("Missing test images:", len(missing))
sub = pd.DataFrame(
    {
        "image_id": sample_df["image_id"].values,
        "label": np.array(test_preds, dtype=np.int64),
    }
)
sub.head()



## === cell 11
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote submission:", out_path, "rows:", len(sub), "cols:", list(sub.columns))
print(sub["label"].value_counts().sort_index())

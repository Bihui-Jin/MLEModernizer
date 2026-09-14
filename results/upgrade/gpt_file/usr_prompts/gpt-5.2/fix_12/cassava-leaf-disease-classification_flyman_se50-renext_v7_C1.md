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

albumentations==2.0.8
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
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

# 5. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd



## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import cv2
import timm
import albumentations as A




## === cell 2
def gem(x, p=3, eps=1e-5):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-5):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )




## === cell 3
class Net(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = timm.create_model("seresnext50_32x4d", pretrained=True)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, inputs):
        bs = inputs.size(0)
        x = self.model.forward_features(inputs)
        fm = self._avg_pooling(x)
        fm = fm.view(bs, -1)
        feature = self.dropout(fm)
        x = self._fc(feature)
        return x




## === cell 4
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DatasetTest(torch.utils.data.Dataset):
    def __init__(self, test_data_dir):
        self.ds = self.get_list(test_data_dir)
        self.root_dir = test_data_dir

        self.val_trans = A.Compose(
            [
                A.Resize(height=672, width=672, interpolation=cv2.INTER_LINEAR, p=1.0),
                A.CenterCrop(height=640, width=640, p=1.0),
            ]
        )

    def get_list(self, dir):
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
        out = []
        for name in os.listdir(dir):
            full = os.path.join(dir, name)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in valid_ext:
                out.append(name)
        out = sorted(out)
        return out

    def __len__(self):
        return len(self.ds)

    def preprocess_func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image1 = self.val_trans(image=image)["image"]
        image2 = self.val_trans(image=image)["image"]
        image3 = self.val_trans(image=image)["image"]
        image4 = self.val_trans(image=image)["image"]
        image5 = self.val_trans(image=image)["image"]
        image6 = self.val_trans(image=image)["image"]
        image7 = self.val_trans(image=image)["image"]
        image8 = self.val_trans(image=image)["image"]

        image_batch = np.stack(
            [image1, image2, image3, image4, image5, image6, image7, image8]
        ).astype(np.float32)

        image_batch = image_batch / 255.0
        image_batch = (
            image_batch - _IMAGENET_MEAN[None, None, None, :]
        ) / _IMAGENET_STD[None, None, None, :]

        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        return image_batch

    def __getitem__(self, item):
        fname = self.ds[item]
        image_path = os.path.join(self.root_dir, fname)
        image = cv2.imread(image_path, -1)
        if image is None:
            raise FileNotFoundError(
                f"Failed to read image with cv2.imread: {image_path}"
            )
        float_image = self.preprocess_func(image)
        return fname, torch.from_numpy(float_image)




## === cell 5
class DatasetTrain:
    def __init__(self, csv_path, img_dir, indices):
        self.df = pd.read_csv(csv_path).iloc[indices].reset_index(drop=True)
        self.img_dir = img_dir

        self.trans = A.Compose(
            [
                A.Resize(height=672, width=672, interpolation=cv2.INTER_LINEAR, p=1.0),
                A.CenterCrop(height=640, width=640, p=1.0),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        image_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(image_path, -1)
        if img is None:
            raise FileNotFoundError(
                f"Failed to read image with cv2.imread: {image_path}"
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.trans(image=img)["image"].astype(np.float32)

        img = img / 255.0
        img = (img - _IMAGENET_MEAN[None, None, :]) / _IMAGENET_STD[None, None, :]

        img = np.transpose(img, (2, 0, 1))  # CHW
        return torch.from_numpy(img), torch.tensor(y, dtype=torch.long)




## === cell 6
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = (
    True  # performance; does not introduce nondeterminism in inference
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kaggle_root = "/kaggle/input"

test_datadir = os.path.join(
    kaggle_root, "cassava-leaf-disease-classification/test_images"
)
if not os.path.isdir(test_datadir):
    test_datadir = os.path.join(
        kaggle_root,
        "cassava-leaf-disease-classification",
        "cassava-leaf-disease-classification",
        "test_images",
    )
assert os.path.isdir(test_datadir), f"Test image directory not found: {test_datadir}"

dataiter = DatasetTest(test_datadir)


def _find_weight_files(kaggle_root="/kaggle/input"):
    """
    Search common locations for .pth/.pt weights.
    """
    candidates = [
        os.path.join(kaggle_root, "cassva-models-se50-640"),
        os.path.join(kaggle_root, "cassava-models-se50-640"),
        os.path.join(kaggle_root, "cassava-leaf-disease-classification"),
    ]
    weight_files = []
    for d in candidates:
        if os.path.isdir(d):
            for f in os.listdir(d):
                lf = f.lower()
                if lf.endswith(".pth") or lf.endswith(".pt") or lf.endswith(".bin"):
                    weight_files.append(os.path.join(d, f))
    weight_files = sorted(weight_files)
    return weight_files


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    return ckpt


def _has_fc_weights(sd):
    if not isinstance(sd, dict):
        return False
    keys = set(sd.keys())
    return any(
        k.endswith("_fc.weight")
        or k.endswith("_fc.bias")
        or k.endswith("._fc.weight")
        or k.endswith("._fc.bias")
        for k in keys
    )




## === cell 7
def _train_fc_head_if_needed(
    model, kaggle_root="/kaggle/input", max_epochs=2, batch_size=16
):
    train_csv = os.path.join(
        kaggle_root, "cassava-leaf-disease-classification/train.csv"
    )
    if not os.path.isfile(train_csv):
        train_csv = os.path.join(kaggle_root, "train.csv")
    assert os.path.isfile(train_csv), f"train.csv not found at {train_csv}"

    train_dir = os.path.join(
        kaggle_root, "cassava-leaf-disease-classification/train_images"
    )
    if not os.path.isdir(train_dir):
        train_dir = os.path.join(
            kaggle_root,
            "cassava-leaf-disease-classification",
            "cassava-leaf-disease-classification",
            "train_images",
        )
    assert os.path.isdir(train_dir), f"Train image directory not found: {train_dir}"

    df = pd.read_csv(train_csv)
    n = len(df)

    labels = df["label"].astype(int).values
    rng = np.random.RandomState(42)
    idx_all = np.arange(n)

    tr_idx = []
    va_idx = []
    for c in np.unique(labels):
        cls_idx = idx_all[labels == c]
        rng.shuffle(cls_idx)
        split_c = int(0.9 * len(cls_idx))
        tr_idx.append(cls_idx[:split_c])
        va_idx.append(cls_idx[split_c:])
    tr_idx = np.concatenate(tr_idx)
    va_idx = np.concatenate(va_idx)
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)

    ds_tr = DatasetTrain(train_csv, train_dir, tr_idx)
    ds_va = DatasetTrain(train_csv, train_dir, va_idx)

    num_workers = min(4, os.cpu_count() or 2)

    g = torch.Generator()
    g.manual_seed(42)

    dl_tr = torch.utils.data.DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    dl_va = torch.utils.data.DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    for p in model.model.parameters():
        p.requires_grad = False
    for p in model._fc.parameters():
        p.requires_grad = True
    model.train()

    opt = torch.optim.AdamW(model._fc.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for xb, yb in dl_va:
                xb = xb.to(device, non_blocking=True).float()
                yb = yb.to(device, non_blocking=True)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        acc = correct / max(total, 1)
        print(f"fc-head epoch {epoch+1}/{max_epochs} val_acc={acc:.5f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model._fc.state_dict().items()
            }

    if best_state is not None:
        model._fc.load_state_dict(best_state)
    model.eval()
    return best_acc




## === cell 8
weights = _find_weight_files(kaggle_root)
print(f"Found {len(weights)} weight file(s).")


def predict_with_model(model, weights):
    merge_res_dict = {}

    sample_path = os.path.join(
        kaggle_root, "cassava-leaf-disease-classification/sample_submission.csv"
    )
    if not os.path.isfile(sample_path):
        sample_path = os.path.join(kaggle_root, "sample_submission.csv")
    sample = pd.read_csv(sample_path)

    if len(weights) == 0:
        print(
            "No finetuned model weights found under /kaggle/input. "
            "Training only the classifier head (`_fc`) on train.csv (backbone frozen)."
        )
        _train_fc_head_if_needed(
            model, kaggle_root=kaggle_root, max_epochs=6, batch_size=64
        )
        valid_weight_paths = [None]
    else:
        valid_weight_paths = []
        for w in weights:
            try:
                ckpt = torch.load(w, map_location="cpu")
                sd = _extract_state_dict(ckpt)
                sd = _strip_module_prefix(sd)
                if _has_fc_weights(sd):
                    valid_weight_paths.append(w)
            except Exception as e:
                print(f"Skipping unreadable checkpoint {w}: {repr(e)}")

        if len(valid_weight_paths) == 0:
            print(
                "Weight files were found but none contained '_fc' classifier weights compatible with this model. "
                "Training only the classifier head (`_fc`) on train.csv (backbone frozen)."
            )
            _train_fc_head_if_needed(
                model, kaggle_root=kaggle_root, max_epochs=6, batch_size=64
            )
            valid_weight_paths = [None]
        else:
            print(
                f"Using {len(valid_weight_paths)} checkpoint(s) that contain classifier head weights."
            )

    num_workers = min(4, os.cpu_count() or 2)
    test_loader = torch.utils.data.DataLoader(
        dataiter,
        batch_size=8,  # number of images per batch (each has 8 TTA crops internally)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    len_data = len(dataiter)

    for j, weight in enumerate(valid_weight_paths):
        if weight is not None:
            ckpt = torch.load(weight, map_location="cpu")
            state = _extract_state_dict(ckpt)
            state = _strip_module_prefix(state)
            model.load_state_dict(state, strict=False)

        model.eval()

        seen = 0
        for batch_idx, (fnames, float_images) in enumerate(test_loader):
            bsz = float_images.size(0)
            seen += bsz
            if seen == bsz or (seen % 200) < bsz or seen >= len_data:
                print(
                    f"weight {j+1}/{len(valid_weight_paths)}: data {min(seen, len_data)}/{len_data}"
                )

            x = float_images.to(device, non_blocking=True).float()
            x = x.view(-1, 3, x.size(-2), x.size(-1))  # [B*8,3,H,W] normalized

            with torch.inference_mode():
                output = model(x)  # [B*8,5]
                output = torch.nn.functional.softmax(output, dim=-1)  # [B*8,5]
                output = output.view(bsz, 8, -1).mean(dim=1)  # [B,5] average over 8 TTA

            out_np = output.detach().cpu().numpy()
            for k, fname in enumerate(fnames):
                fname = str(fname)
                if fname not in merge_res_dict:
                    merge_res_dict[fname] = out_np[k]
                else:
                    merge_res_dict[fname] += out_np[k]

    preds = []
    for fname in sample["image_id"].tolist():
        if fname in merge_res_dict:
            preds.append(int(np.argmax(merge_res_dict[fname])))
        else:
            preds.append(0)

    sub = pd.DataFrame({"image_id": sample["image_id"].tolist(), "label": preds})
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", sub.shape)
    print(sub.head())


model = Net().to(device)
predict_with_model(model, weights)

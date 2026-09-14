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
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = True




## === cell 1
def gem(x, p=3, eps=1e-5):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-5):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"




## === cell 2
class Net(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = timm.create_model("seresnext50_32x4d", pretrained=True)
        self._gem_pool = GeM(p=3.0, eps=1e-6)
        self.dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2048, num_classes, bias=True)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, inputs):
        x = inputs / 255.0
        x = (x - self.mean) / self.std
        bs = x.size(0)
        x = self.model.forward_features(x)
        fm = self._gem_pool(x).view(bs, -1)  # GeM pooling
        fm = self.dropout(fm)
        return self._fc(fm)




## === cell 3
class DatasetTest(Dataset):
    """Simple test‑set dataset that returns a filename and an 8‑fold TTA batch."""

    def __init__(self, test_data_dir):
        self.root_dir = test_data_dir
        self.ds = sorted(os.listdir(test_data_dir))

    def __len__(self):
        return len(self.ds)

    def _preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        aug = [
            image,
            np.rot90(image, 1),
            np.rot90(image, 2),
            np.rot90(image, 3),
            np.fliplr(image),
            np.rot90(np.fliplr(image), 1),
            np.rot90(np.fliplr(image), 2),
            np.rot90(np.fliplr(image), 3),
        ]
        batch = np.stack(aug)  # (8, H, W, C)
        batch = np.transpose(batch, (0, 3, 1, 2))  # (8, C, H, W)
        batch = torch.from_numpy(batch).float()
        return batch

    def __getitem__(self, idx):
        fname = self.ds[idx]
        img_path = os.path.join(self.root_dir, fname)
        image = cv2.imread(img_path, -1)
        if image is None:
            image = np.zeros((640, 640, 3), dtype=np.uint8)
        aug_batch = self._preprocess(image)  # (8, C, H, W) tensor
        return fname, aug_batch




## === cell 4
class CassavaTrainDataset(Dataset):
    """Training dataset – reads images, resizes to 640×640 and returns tensors."""

    def __init__(self, csv_path, images_dir):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image_id"])
        img = cv2.imread(img_path, -1)
        if img is None:
            img = np.zeros((640, 640, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        label = int(row["label"])
        return img, label




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kaggle_root = "/kaggle/input"

model_dir = os.path.join(kaggle_root, "cassava-models-se50-640")
weights = (
    [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".pth") or f.endswith(".pt")
    ]
    if os.path.isdir(model_dir)
    else []
)

train_csv = os.path.join(
    kaggle_root, "cassava-leaf-disease-classification", "train.csv"
)
train_imgs = os.path.join(
    kaggle_root, "cassava-leaf-disease-classification", "train_images"
)
train_dataset = CassavaTrainDataset(train_csv, train_imgs)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
for imgs, lbls in train_loader:
    imgs = imgs.to(device)
    lbls = lbls.to(device)
    optimizer.zero_grad()
    logits = model(imgs)
    loss = criterion(logits, lbls)
    loss.backward()
    optimizer.step()

test_dir = os.path.join(
    kaggle_root, "cassava-leaf-disease-classification", "test_images"
)
test_dataset = DatasetTest(test_dir)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,  # chosen to fit GPU memory while keeping high throughput
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


def predict_with_model(model, weight_paths, test_loader):
    """
    Batched TTA inference with optional ensemble over multiple weight files.
    """
    model.eval()
    if not weight_paths:
        weight_paths = [None]

    num_classes = model._fc.out_features
    num_samples = len(test_loader.dataset)
    agg_probs = torch.zeros(
        (num_samples, num_classes), dtype=torch.float32, device="cpu"
    )
    all_fnames = []

    with torch.no_grad():
        for wp in weight_paths:
            if wp:
                state = torch.load(wp, map_location=device)
                model.load_state_dict(state, strict=False)

            idx_offset = 0
            for fnames, aug_batch in test_loader:
                if not all_fnames:
                    all_fnames.extend(fnames)  # store order once

                B = aug_batch.size(0)  # batch size
                aug_batch = aug_batch.to(device)  # (B, 8, C, H, W)
                inp = aug_batch.view(B * 8, *aug_batch.shape[2:])  # (B*8, C, H, W)

                out = model(inp)  # (B*8, C)
                out = out.view(B, 8, -1)  # (B, 8, C)
                prob = torch.nn.functional.softmax(out, dim=-1).mean(dim=1)  # (B, C)

                agg_probs[idx_offset : idx_offset + B] += prob.cpu()
                idx_offset += B
    final_labels = agg_probs.argmax(dim=1).numpy()
    results = pd.DataFrame({"image_id": all_fnames, "label": final_labels})
    results.to_csv("submission.csv", index=False)


predict_with_model(model, weights, test_loader)

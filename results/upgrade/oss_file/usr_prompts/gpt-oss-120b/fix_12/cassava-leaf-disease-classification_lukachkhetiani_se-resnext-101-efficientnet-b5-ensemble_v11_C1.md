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
import os, glob, cv2, json, numpy as np, pandas as pd
import torch, torch.nn.functional as F
import timm
from torchvision import models

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True  # ensure cuDNN optimizations when GPU is present


def load_checkpoint(model, pattern):
    """Load checkpoint matching pattern and return True if found."""
    search_paths = [
        pattern,
        os.path.join("/kaggle/input", pattern),
        os.path.join("../input", pattern),
    ]
    for p in search_paths:
        for ckpt_path in glob.glob(p, recursive=True):
            if os.path.isfile(ckpt_path):
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state)
                print(f"Loaded checkpoint: {ckpt_path}")
                return True
    print(f"No checkpoint found for pattern: {pattern}")
    return False


try:
    from efficientnet_pytorch import EfficientNet

    efficient = EfficientNet.from_name("efficientnet-b5", num_classes=5)
except Exception:
    efficient = timm.create_model("tf_efficientnet_b5", pretrained=True, num_classes=5)

ckpt_eff = load_checkpoint(efficient, "**/eff_best.pth")
efficient.to(device).eval()

seresnext = timm.create_model("seresnext101_32x4d", pretrained=True, num_classes=5)
ckpt_ser = load_checkpoint(seresnext, "**/seresnext_best.pth")
seresnext.to(device).eval()

print("Models have been loaded...")




## === cell 1
def processor(image):
    """Resize to 224×224, convert BGR→RGB, normalize, and return a torch tensor."""
    img = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = (
        torch.tensor(img.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0).to(device)
    )
    return tensor




## === cell 2
def find_existing_path(candidates):
    for cand in candidates:
        if os.path.isdir(cand) or os.path.isfile(cand):
            return cand
    return None


train_csv_candidates = [
    "../input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "train.csv",
]
train_csv_path = find_existing_path(train_csv_candidates)
if train_csv_path is None:
    raise RuntimeError("train.csv not found.")

train_img_candidates = [
    "../input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "train_images",
]
train_img_dir = find_existing_path(train_img_candidates)
if train_img_dir is None:
    raise RuntimeError("train_images directory not found.")


class CassavaDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_dir, augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.augment and np.random.rand() < 0.5:
            img = cv2.flip(img, 1)  # horizontal flip

        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.tensor(
            img.transpose(2, 0, 1), dtype=torch.float
        )  # no batch dim yet
        label = int(row["label"])
        return tensor, label


if not (ckpt_eff and ckpt_ser) and device.type == "cpu":
    print(
        "No checkpoints found and running on CPU – skipping training to meet time limit."
    )
else:
    train_dataset = CassavaDataset(train_csv_path, train_img_dir, augment=True)

    num_workers = min(4, os.cpu_count() or 0) if device.type == "cuda" else 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_eff = torch.optim.Adam(efficient.parameters(), lr=1e-4)
    optimizer_ser = torch.optim.Adam(seresnext.parameters(), lr=1e-4)

    efficient.train()
    seresnext.train()
    num_epochs = 4
    for epoch in range(num_epochs):
        running_loss_eff = 0.0
        running_loss_ser = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer_eff.zero_grad()
            optimizer_ser.zero_grad()

            outputs_eff = efficient(imgs)
            outputs_ser = seresnext(imgs)

            loss_eff = criterion(outputs_eff, targets)
            loss_ser = criterion(outputs_ser, targets)

            loss_eff.backward()
            loss_ser.backward()

            optimizer_eff.step()
            optimizer_ser.step()

            running_loss_eff += loss_eff.item() * imgs.size(0)
            running_loss_ser += loss_ser.item() * imgs.size(0)

        epoch_loss_eff = running_loss_eff / len(train_loader.dataset)
        epoch_loss_ser = running_loss_ser / len(train_loader.dataset)
        print(
            f"EfficientNet Epoch [{epoch+1}/{num_epochs}] - Training loss: {epoch_loss_eff:.4f}"
        )
        print(
            f"SEResNeXt Epoch [{epoch+1}/{num_epochs}] - Training loss: {epoch_loss_ser:.4f}"
        )

efficient.eval()
seresnext.eval()




## === cell 3
test_dir_candidates = [
    "../input/cassava-leaf-disease-classification/test_images",
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "test_images",
]
test_dir = None
for cand in test_dir_candidates:
    if os.path.isdir(cand):
        test_dir = cand
        break
if test_dir is None:
    raise RuntimeError("Test images directory not found.")

files = sorted(glob.glob(os.path.join(test_dir, "*")))
batch_size = 32
names, labels = [], []

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_batch(img_paths):
    """Read, resize, normalize a list of image paths and return a torch tensor batch."""
    B = len(img_paths)
    batch_np = np.empty((B, 3, 224, 224), dtype=np.float32)
    for i, p in enumerate(img_paths):
        img = cv2.imread(p)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - mean) / std
        batch_np[i] = img.transpose(2, 0, 1)  # C, H, W
    return torch.tensor(batch_np, dtype=torch.float, device=device)


with torch.no_grad():
    for i in range(0, len(files), batch_size):
        batch_paths = files[i : i + batch_size]
        batch_tensor = preprocess_batch(batch_paths)  # shape (B, 3, 224, 224)

        se_out = seresnext(batch_tensor)
        eff_out = efficient(batch_tensor)

        se_prob = F.softmax(se_out, dim=1)
        eff_prob = F.softmax(eff_out, dim=1)
        avg_prob = (se_prob + eff_prob) / 2.0

        preds = torch.argmax(avg_prob, dim=1).cpu().numpy()
        names.extend([os.path.basename(p) for p in batch_paths])
        labels.extend(preds.tolist())

submission = pd.DataFrame({"image_id": names, "label": labels})
submission.to_csv("submission.csv", index=False, header=True)
print("Submission file written to submission.csv")

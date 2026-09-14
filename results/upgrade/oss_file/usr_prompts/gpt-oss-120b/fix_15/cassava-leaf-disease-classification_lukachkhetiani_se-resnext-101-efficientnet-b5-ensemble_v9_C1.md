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
import os, sys, random, glob, tqdm, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import cv2, pandas as pd
from torch.utils.data import Dataset, DataLoader

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError:
    EfficientNet = None

import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def try_load(path):
    if os.path.exists(path):
        try:
            return torch.load(path, map_location=device)
        except Exception as e:
            print(f"Could not load {path}: {e}")
    else:
        print(f"Checkpoint {path} not found.")
    return None


base_path = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
train_img_dir = os.path.join(base_path, "train_images")
test_img_dir = os.path.join(base_path, "test_images")

eff_path = "/kaggle/input/ensemble-2/eff_model_last.pth"
hr_path = "/kaggle/input/ensemble-2/seresnext_model_last.pth"

eff_state = try_load(eff_path)
hr_state = try_load(hr_path)

if eff_state is not None and EfficientNet is not None:
    efficient = EfficientNet.from_name("efficientnet-b5", num_classes=5)
    efficient.load_state_dict(eff_state)
    efficient = efficient.to(device).eval()
else:
    efficient = timm.create_model("efficientnet_b5", pretrained=True, num_classes=5)
    efficient = efficient.to(device).eval()

if hr_state is not None:
    hrnet = timm.create_model("seresnext101_32x4d", pretrained=False, num_classes=5)
    hrnet.load_state_dict(hr_state)
    hrnet = hrnet.to(device).eval()
else:
    hrnet = efficient  # fallback

extra_efficient = timm.create_model("efficientnet_b5", pretrained=True, num_classes=5)
extra_efficient = extra_efficient.to(device).eval()

model_ensemble = [efficient, hrnet, extra_efficient]

print("Model(s) loaded and ready for inference.")


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TARGET_SIZE = 456


def process(image):
    """Resize, apply CLAHE, convert to normalized torch tensor (CPU)."""
    img = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    tensor = (tensor - norm_mean) / norm_std
    return tensor


def augment(img):
    """Random lightweight augmentations for training."""
    if random.random() < 0.5:
        img = cv2.flip(img, 1)  # horizontal flip
    if random.random() < 0.5:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 90° rotation
    return img


if eff_state is None:
    print("No cassava checkpoint found – starting a quick fine‑tuning pass.")
    train_df = pd.read_csv(train_csv_path)

    class CassavaDataset(Dataset):
        def __init__(self, df, img_dir):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row["image_id"])
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
            img = augment(img)  # ← added augmentation
            tensor = process(img)  # CPU tensor
            label = int(row["label"])
            return tensor, label

    train_dataset = CassavaDataset(train_df, train_img_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
    )

    efficient.train()
    optimizer = optim.AdamW(efficient.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    max_epochs = 3  # increased from 1 to give the model more chance to learn
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for batch_tensors, batch_labels in tqdm.tqdm(
            train_loader, desc=f"Fine‑tuning epoch {epoch+1}/{max_epochs}"
        ):
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = efficient(batch_tensors)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_tensors.size(0)

        avg_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch+1} finished – average loss: {avg_loss:.4f}")

    efficient.eval()
    hrnet.eval()
    extra_efficient.eval()
else:
    print("Cassava checkpoint detected – skipping fine‑tuning.")




## === cell 1
files = glob.glob(os.path.join(test_img_dir, "**", "*.jpg"), recursive=True)

batch_size = 32
names, labels = [], []

batch_tensors = []
batch_names = []


def infer_batch(batch_tensor):
    """
    Perform inference with test‑time augmentations and a small ensemble.
    Optimized by stacking all augmentations and running one forward pass per model.
    Returns the predicted class indices for the batch.
    """
    aug_list = [
        batch_tensor,
        torch.flip(batch_tensor, dims=[3]),  # horizontal flip
        torch.flip(batch_tensor, dims=[2]),  # vertical flip
        torch.rot90(batch_tensor, k=1, dims=[2, 3]),  # 90° rotation
    ]
    aug_cat = torch.cat(aug_list, dim=0)

    with torch.no_grad():
        logits_sum = None
        for mdl in model_ensemble:
            out = mdl(aug_cat)  # (4*B, num_classes)
            B = batch_tensor.shape[0]
            out = out.view(4, B, -1)  # (4, B, num_classes)
            out = out.mean(dim=0)  # average over augmentations -> (B, num_classes)

            if logits_sum is None:
                logits_sum = out
            else:
                logits_sum += out

        avg_logits = logits_sum / len(model_ensemble)
        probs = F.softmax(avg_logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
    return preds.tolist()


for file in tqdm.tqdm(files, desc="Predicting"):
    img = cv2.imread(file)
    if img is None:
        continue
    tensor = process(img)  # CPU tensor
    batch_tensors.append(tensor)
    batch_names.append(os.path.basename(file))

    if len(batch_tensors) == batch_size:
        batch = torch.stack(batch_tensors).to(device, non_blocking=True)
        preds = infer_batch(batch)
        names.extend(batch_names)
        labels.extend(preds)
        batch_tensors, batch_names = [], []

if batch_tensors:
    batch = torch.stack(batch_tensors).to(device, non_blocking=True)
    preds = infer_batch(batch)
    names.extend(batch_names)
    labels.extend(preds)

print(f"Predicted {len(names)} images.")




## === cell 2
submission = pd.DataFrame({"image_id": names, "label": labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved as {submission_path} with {len(submission)} rows.")

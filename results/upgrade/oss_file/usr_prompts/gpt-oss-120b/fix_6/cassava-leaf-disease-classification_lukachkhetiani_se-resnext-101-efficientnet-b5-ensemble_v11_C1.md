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

0.8757932910244787

# 6. Current score

0.78587

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12182) has done: 'The script failed because the `efficientnet_pytorch` package is not available and the `crop_image` function referenced an undefined variable. I added a safe import that falls back to a timm EfficientNet model, introduced a device‑agnostic setup (CPU/GPU), corrected the variable name in `crop_image`, and updated all tensors to use the chosen device. The rest of the workflow stays unchanged, and the script now writes a proper `submission.csv` file.'
- What this solution (achieved 0.21263) has done: 'I switch the EfficientNet and SEResNeXt models to use pretrained weights (instead of random initialization) and fix the image preprocessing to correctly scale pixel values to the [0,1] range before normalisation. These small adjustments keep the original architecture and workflow unchanged while providing much more meaningful predictions, which should raise the accuracy from the current ~0.12 toward the target score.'
- What this solution (achieved 0.15209) has done: 'I simplify the preprocessing by removing the depth‑based cropping (which often discards useful leaf area) and resize every image to the standard 224 × 224 size expected by the ImageNet‑pretrained models. This keeps the same model architecture and ensemble logic but provides cleaner inputs, which should raise the accuracy toward the target score.'
- What this solution (achieved 0.11323) has done: 'Implemented robust checkpoint loading, corrected image channel ordering, added reproducibility seed, and refined preprocessing. These adjustments ensure the pretrained fine‑tuned weights are actually used (if present), improve input normalization, and keep the existing model architecture unchanged, moving the validation accuracy toward the target score.'
- What this solution (achieved 0.78587) has done: 'I add a lightweight fine‑tuning stage that trains the EfficientNet model on the provided training images for a couple of epochs before performing inference. This keeps the original inference pipeline and model architectures unchanged, but adapts the pretrained weights to the cassava dataset, which should raise the validation accuracy markedly and move the score closer to the target. The new cells locate the training CSV and image folder, define a simple `Dataset`, run a short training loop, and then keep the existing ensemble inference unchanged.'

# 9. Code solution

## === cell 0
import os, glob, cv2, json, numpy as np, pandas as pd
import torch, torch.nn.functional as F
import timm
from torchvision import models

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, pattern):
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
                return
    print(f"No checkpoint found for pattern: {pattern}")


try:
    from efficientnet_pytorch import EfficientNet

    efficient = EfficientNet.from_name("efficientnet-b5", num_classes=5)
except Exception:
    efficient = timm.create_model("tf_efficientnet_b5", pretrained=True, num_classes=5)

load_checkpoint(efficient, "**/eff_best.pth")
efficient.to(device).eval()

seresnext = timm.create_model("seresnext101_32x4d", pretrained=True, num_classes=5)
load_checkpoint(seresnext, "**/seresnext_best.pth")
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


train_dataset = CassavaDataset(train_csv_path, train_img_dir, augment=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(efficient.parameters(), lr=1e-4)

efficient.train()  # set to train mode for fine‑tuning
num_epochs = 2
for epoch in range(num_epochs):
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = efficient(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Training loss: {epoch_loss:.4f}")

efficient.eval()  # back to eval mode for inference



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

files = glob.glob(os.path.join(test_dir, "*"))
names, labels = [], []

for file_path in files:
    img = cv2.imread(file_path)
    if img is None:
        continue

    img_tensor = processor(img)

    with torch.no_grad():
        se_out = seresnext(img_tensor)
        eff_out = efficient(img_tensor)

        se_prob = F.softmax(se_out, dim=1)
        eff_prob = F.softmax(eff_out, dim=1)
        avg_prob = (se_prob + eff_prob) / 2.0

        pred_label = int(torch.argmax(avg_prob, dim=1).cpu().item())

    names.append(os.path.basename(file_path))
    labels.append(pred_label)

submission = pd.DataFrame({"image_id": names, "label": labels})
submission.to_csv("submission.csv", index=False, header=True)
print("Submission file written to submission.csv")

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

0.8874282260501662

# 6. Current score

0.7216

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.1222) has done: 'I fix the crash caused by trying to open a sub‑directory as an image. In the test data loading cell I filter the directory listing so only actual image files are included (skip directories). This prevents the `IsADirectoryError`, allows inference to run, and lets the script create the required `submission.csv` file.'
- What this solution (achieved 0.7216) has done: 'I add a quick fine‑tuning stage using the provided training CSV so the pretrained ResNeXt model adapts to the cassava data, which should raise the validation accuracy substantially and move the score toward the target. The image size is reduced to 224 px to keep training fast, and the existing inference and submission logic is kept unchanged apart from using the now‑trained model.'

# 9. Code solution

## === cell 0
import os, json, numpy as np, pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models, torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader




## === cell 1
def get_image(path):
    return Image.open(path).convert("RGB")




## === cell 2
class GetDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transforms = transforms
        self.output_label = output_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = os.path.join(self.data_root, self.df.iloc[index]["image_id"])
        img = get_image(path)
        if self.transforms:
            img = self.transforms(img)
        if self.output_label:
            label = int(self.df.iloc[index]["label"])
            return img, label
        return img




## === cell 3
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)


device = get_device()
print(f"Using device: {device}")




## === cell 4
def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return (preds == labels).float().mean()


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
            f"Epoch: [{epoch}/{epochs}], "
            f"lr: {result['lrs'][-1]:.6f}, "
            f"train_loss: {result['train_loss']:.4f}, "
            f"val_loss: {result['val_loss']:.4f}, "
            f"val_acc: {result['val_acc']:.4f}"
        )




## === cell 5
class Classifier(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnext50_32x4d(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 5)

    def forward(self, xb):
        return self.network(xb)

    def freeze(self):
        for p in self.network.parameters():
            p.requires_grad = False
        for p in self.network.fc.parameters():
            p.requires_grad = True

    def unfreeze(self):
        for p in self.network.parameters():
            p.requires_grad = True




## === cell 6
BATCH_SIZE = 64
IMG_SIZE = 224  # smaller size for faster fine‑tuning
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)

possible_train_dirs = [
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "./data/cassava-leaf-disease-classification/train_images",
    "./input/cassava-leaf-disease-classification/train_images",
    "./train_images",
]
TRAIN_DIR = None
for d in possible_train_dirs:
    if os.path.isdir(d):
        TRAIN_DIR = d
        break
if TRAIN_DIR is None:
    raise RuntimeError("Train images directory not found.")

possible_test_dirs = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "./data/cassava-leaf-disease-classification/test_images",
    "./input/cassava-leaf-disease-classification/test_images",
    "./test_images",
]
TEST_DIR = None
for d in possible_test_dirs:
    if os.path.isdir(d):
        TEST_DIR = d
        break
if TEST_DIR is None:
    raise RuntimeError("Test images directory not found.")

train_csv_paths = [
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "./data/cassava-leaf-disease-classification/train.csv",
    "./input/cassava-leaf-disease-classification/train.csv",
    "./train.csv",
]
train_csv = None
for p in train_csv_paths:
    if os.path.isfile(p):
        train_csv = p
        break
if train_csv is None:
    raise RuntimeError("train.csv not found.")
train_df = pd.read_csv(train_csv)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(0.8 * len(train_df))
df_train = train_df.iloc[:split_idx]
df_val = train_df.iloc[split_idx:]

train_transforms = T.Compose(
    [
        T.Resize(IMG_SHAPE),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_transforms = T.Compose(
    [
        T.Resize(IMG_SHAPE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_transforms = val_transforms

train_dataset = GetDataset(
    df_train, TRAIN_DIR, transforms=train_transforms, output_label=True
)
val_dataset = GetDataset(
    df_val, TRAIN_DIR, transforms=val_transforms, output_label=True
)

train_loader = DeviceDataLoader(
    DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    ),
    device,
)
val_loader = DeviceDataLoader(
    DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    ),
    device,
)

test_images = sorted(
    [
        f
        for f in os.listdir(TEST_DIR)
        if (not f.startswith(".")) and os.path.isfile(os.path.join(TEST_DIR, f))
    ]
)
test_df = pd.DataFrame({"image_id": test_images})
test_dataset = GetDataset(
    test_df, TEST_DIR, transforms=test_transforms, output_label=False
)
test_loader = DeviceDataLoader(
    DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    ),
    device,
)




## === cell 7
model = Classifier()
model.freeze()  # train only the final linear layer
model.to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)
num_epochs = 3

for epoch in range(1, num_epochs + 1):
    model.train()
    train_losses = []
    for batch in train_loader:
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())
    train_loss = np.mean(train_losses)

    model.eval()
    val_outputs = [model.validation_step(batch) for batch in val_loader]
    val_result = model.validation_epoch_end(val_outputs)

    result = {
        "train_loss": train_loss,
        "val_loss": val_result["val_loss"],
        "val_acc": val_result["val_acc"],
        "lrs": [optimizer.param_groups[0]["lr"]],
    }
    model.epoch_end(epoch, num_epochs, result)

print("Fine‑tuning completed.")




## === cell 8
def inference(model, loader, device):
    model.to(device)
    model.eval()
    all_probs = []
    tk0 = tqdm(enumerate(loader), total=len(loader))
    with torch.no_grad():
        for i, batch in tk0:
            if isinstance(batch, (list, tuple)):
                images = batch[0] if len(batch) == 1 else batch
            else:
                images = batch
            logits = model(images)
            probs = logits.softmax(dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    print("Predictions shape:", all_probs.shape)
    return all_probs


predictions = inference(model, test_loader, device)




## === cell 9
test_df["label"] = predictions.argmax(axis=1)
submission_path = "./submission.csv"
test_df[["image_id", "label"]].to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(test_df.head())

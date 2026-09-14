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

3.12

# 3. Installed packages

geopandas==0.14.4
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

0.8934723481414325

# 6. Current score

0.79671

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.16704) has done: 'Implemented minimal fixes so the script runs end‑to‑end and creates a valid `submission.csv`.  
- Removed unsupported `image_size` argument when building ViT models and set the model input size to the default 224.  
- Updated the global image‑size variables to 224 so the dataset resizing matches the model expectations.  
- These changes resolve the initialization error, allow inference to proceed, and ensure a correctly‑sized submission file is written.'
- What this solution (achieved 0.76383) has done: 'The update pre‑computes the frozen ViT model outputs once before training so the expensive forward passes are not repeated each epoch. The saved combined features are then used to train only the linear head for the required three epochs, preserving identical model logic and results while dramatically reducing runtime. Minor tweaks (persistent TensorDataset, moving tensors to device only when needed) keep memory usage low and maintain deterministic behavior.'
- What this solution (achieved 0.79484) has done: 'I increase the training epochs and slightly lower the learning rate to let the linear head learn more effectively, and I simplify the model‑output fusion to an equal average ( (out_a + out_b)/2 ) both when extracting features and during inference. These modest adjustments keep the original architecture untouched while expected to raise the accuracy toward the target.'
- What this solution (achieved 0.79297) has done: 'I adjust the way the two ViT model outputs are merged, giving a larger weight to the stronger vit_l model (0.6) and a smaller weight to vit_b (0.4). This modest change preserves the overall architecture and training loop while steering predictions toward a better accuracy, moving the score closer to the target.'
- What this solution (achieved 0.79372) has done: 'I equal‑weight the two frozen ViT models and turn on a lightweight test‑time augmentation (horizontal flip) that is applied consistently during inference. These minimal tweaks keep the original architecture and training loop untouched while typically improving classification accuracy enough to move the score into the target tolerance band.'
- What this solution (achieved 0.79596) has done: 'I slightly re‑weight the two ViT models to give more influence to the stronger vit_l model, extend the linear‑head training to 10 epochs and lower the learning rate to 2e‑4, and add a modest random‑horizontal‑flip augmentation during training. These minimal tweaks keep the overall architecture unchanged but are expected to raise the validation accuracy, moving the score closer to the target.'
- What this solution (achieved 0.79671) has done: 'We slightly increase the contribution of the stronger ViT‑L model, add a true identity transform to the test‑time augmentation list (so predictions are averaged over the original and flipped image), lower the learning rate and train a few more epochs – changes that keep the original architecture intact but are expected to raise validation accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
import torchvision




## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

model_a_img_size = 224
model_b_img_size = 224
batch_size = 16
num_workers = 4
num_classes = 5
tta = True  # enable lightweight test‑time augmentation

model_a = torchvision.models.vit_b_16(weights="IMAGENET1K_V1").to(device)
model_b = torchvision.models.vit_l_16(weights="IMAGENET1K_V1").to(device)
model_a.eval()
model_b.eval()
for param in model_a.parameters():
    param.requires_grad = False
for param in model_b.parameters():
    param.requires_grad = False

linear_head = torch.nn.Linear(1000, num_classes).to(device)

model_a_weight = 0.35
model_b_weight = 0.65




## === cell 2
class CassavaDataset(VisionDataset):
    """Dataset for inference (no labels)."""

    def __init__(self, data_dir, model_a_size, model_b_size, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )
        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, filename

    def __len__(self):
        return len(self.images)


class TrainCassavaDataset(VisionDataset):
    """Dataset for training (includes labels)."""

    def __init__(self, data_dir, df, model_a_size, model_b_size, transform=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = df["image_id"].tolist()
        self.labels = df["label"].tolist()
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        label = self.labels[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)

        if self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.images)




## === cell 3
train_transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),  # modest augmentation
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_df = pd.read_csv(train_csv_path)

train_dataset = TrainCassavaDataset(
    train_dir, train_df, model_a_img_size, model_b_img_size, transform=train_transforms
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,  # no need to shuffle for feature extraction
    num_workers=num_workers,
    pin_memory=True,
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(linear_head.parameters(), lr=1e-4)  # slightly smaller LR

model_a.eval()
model_b.eval()
combined_features = []
with torch.no_grad():
    for model_a_inputs, model_b_inputs, _ in train_loader:
        model_a_inputs = model_a_inputs.to(device)
        model_b_inputs = model_b_inputs.to(device)

        out_a = model_a(model_a_inputs)
        out_b = model_b(model_b_inputs)

        combined = model_a_weight * out_a + model_b_weight * out_b
        combined_features.append(combined.cpu())

combined_features = torch.cat(combined_features)  # shape (N, 1000)
labels_tensor = torch.tensor(train_df["label"].values, dtype=torch.long)

feature_dataset = torch.utils.data.TensorDataset(combined_features, labels_tensor)
feature_loader = DataLoader(
    feature_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

linear_head.train()
epochs = 12  # a few more epochs for better convergence
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    for feats, labels in feature_loader:
        feats = feats.to(device)
        labels = labels.to(device)

        logits = linear_head(feats)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * labels.size(0)

    avg_loss = epoch_loss / len(feature_dataset)
    print(f"Epoch {epoch}/{epochs} – Avg loss: {avg_loss:.4f}")

linear_head.eval()




## === cell 4
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomHorizontalFlip(p=0.0),  # identity (original)
        v2.RandomHorizontalFlip(p=1.0),  # guaranteed flip
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir, model_a_img_size, model_b_img_size, transform=test_transforms, ttas=ttas
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 5
all_names = []
all_preds = []

with torch.no_grad():
    for model_a_inputs, model_b_inputs, filenames in test_loader:
        batch_sz = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)

            out_a = model_a(model_a_inputs)
            out_b = model_b(model_b_inputs)

            a_split = torch.stack(torch.split(out_a, batch_sz), dim=0)
            b_split = torch.stack(torch.split(out_b, batch_sz), dim=0)

            a_mean = torch.mean(a_split, dim=0)
            b_mean = torch.mean(b_split, dim=0)

            combined_logits = model_a_weight * a_mean + model_b_weight * b_mean
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)

            out_a = model_a(model_a_inputs)
            out_b = model_b(model_b_inputs)

            combined_logits = model_a_weight * out_a + model_b_weight * out_b

        class_logits = linear_head(combined_logits)
        probs = normalizer(class_logits)
        pred_labels = torch.argmax(probs, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)




## === cell 6
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
my_submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
my_submission.head()

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

0.8981565427621638

# 6. Current score

0.78363

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.78214) has done: 'The fix replaces the manual tensor conversion with a proper PIL image load so the torchvision transforms (which expect a PIL image) work correctly. `Image` from Pillow is imported and both `CassavaDataset` and `TestDataset` now open images as PIL objects before applying the transforms that include `ToTensor`. This resolves the TypeError in the DataLoader and allows the script to run through training and generate a valid `submission.csv`.'
- What this solution (achieved 0.78288) has done: 'I pre‑compute the frozen ViT + EfficientNet features for the entire training and validation sets once, then train the linear head directly on those cached tensors. This removes the costly forward passes through the large backbones during each epoch while keeping the exact same feature representations, so model accuracy is unchanged. The evaluation loop is also switched to use the cached validation features. All other logic, model architecture, and I/O remain identical.'
- What this solution (achieved 0.78475) has done: 'I keep the overall architecture and feature‑caching pipeline unchanged but adjust the training hyper‑parameters to push validation accuracy higher and move the score toward the target. Specifically, I lower the learning rate, add a small weight‑decay, switch to a StepLR schedule (more aggressive decay) and increase the number of epochs so the linear head can better fit the frozen ViT + EfficientNet features. These tweaks are minimal, stay within the original logic, and are expected to raise the validation accuracy toward the target.'
- What this solution (achieved 0.77915) has done: 'I slightly adjust the training hyper‑parameters while keeping the overall architecture unchanged: use AdamW (a more suitable optimizer for weight‑decay), lower the learning rate a bit, decay it more slowly (step‑size 10 instead of 5) and train for more epochs (60). These modest changes are expected to let the linear head fit the frozen ViT + EfficientNet features better and raise validation accuracy toward the target score.'
- What this solution (achieved 0.78513) has done: 'I add a modest color‑jitter augmentation to the training pipeline, lower the learning rate and weight decay, switch to a cosine annealing scheduler, and extend training to 120 epochs. These small hyper‑parameter tweaks keep the original frozen‑backbone + linear‑head architecture unchanged while giving the linear head more epochs to fit the extracted features, which should raise validation accuracy toward the target score.'
- What this solution (achieved 0.77915) has done: 'I slightly raise the learning rate for the linear head, reduce weight decay a bit, and extend the cosine‑annealing period so the optimizer can keep improving the frozen‑backbone features over more epochs. These minimal hyper‑parameter tweaks stay within the original architecture and training loop but should move validation accuracy closer to the target score.'
- What this solution (achieved 0.78513) has done: 'I modestly improve the model by (1) adding a stronger augmentation (RandomResizedCrop) to the training transform so the frozen back‑bones see more varied inputs, (2) lowering the learning rate and increasing weight decay for the linear head to help it fit the richer features, and (3) extending training to 200 epochs (matching the cosine‑annealing schedule). These changes keep the original architecture and feature‑caching pipeline intact while giving the linear classifier more useful data and training time, which should raise validation accuracy toward the target.'
- What this solution (achieved 0.7799) has done: 'I slightly increase the training signal and give the model more opportunity to fit the data by (1) reducing the validation split from 10 % to 5 % so the linear head sees more training examples, (2) raising the learning rate of the AdamW optimizer to 5e‑4, and (3) extending training to 300 epochs while matching the cosine‑annealing schedule (T_max = 300). These minimal hyper‑parameter tweaks keep the exact architecture and feature‑caching pipeline unchanged but are expected to lift validation accuracy toward the target score.'
- What this solution (achieved 0.78363) has done: 'I lower the learning rate and weight decay slightly, extend the cosine‑annealing schedule and training length (to 500 epochs) so the linear head can fit the frozen backbone features more effectively. These minimal hyper‑parameter tweaks keep the architecture unchanged while giving the optimizer a smoother, longer training window, which is expected to raise validation accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os
import json
import glob
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import pandas as pd
from tqdm.auto import tqdm
from torch.backends import cudnn
from PIL import Image

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



## === cell 1
base_path = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
train_images_dir = os.path.join(base_path, "train_images")
test_images_dir = os.path.join(base_path, "test_images")
sample_submission_path = os.path.join(base_path, "sample_submission.csv")
label_map_path = os.path.join(base_path, "label_num_to_disease_map.json")

with open(label_map_path, "r") as f:
    label_map = json.load(f)


def load_models():
    try:
        vit = torch.load(
            "/kaggle/input/vit-v1-update/vit_v1_1.pt", map_location=device
        ).to(device)
        eff = torch.load(
            "/kaggle/input/efficient-net/vit_cont_3.pt", map_location=device
        ).to(device)
        head = torch.load(
            "/kaggle/input/linear-head/linear_cls.pt", map_location=device
        ).to(device)
        return vit, eff, head
    except FileNotFoundError:
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads = nn.Identity()
        vit = vit.to(device)

        eff = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        eff.classifier[1] = nn.Identity()
        eff = eff.to(device)

        head = nn.Linear(768 + 1280, 5).to(device)
        return vit, eff, head


vit_model, eff_model, linear_head = load_models()
for param in vit_model.parameters():
    param.requires_grad = False
for param in eff_model.parameters():
    param.requires_grad = False




## === cell 2
class CassavaDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_id"]
        label = self.df.iloc[idx]["label"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


train_transform = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

full_dataset = CassavaDataset(
    train_csv_path, train_images_dir, transform=train_transform
)

val_size = int(0.05 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(3407),
)
val_dataset.dataset.transform = val_transform

raw_train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)
raw_val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)


@torch.no_grad()
def extract_features(loader):
    feats_list = []
    labs_list = []
    for imgs, labels in tqdm(loader, desc="Extracting features"):
        imgs = imgs.to(device)
        vit_feat = vit_model(imgs)
        eff_feat = eff_model(imgs)
        feats = torch.cat([vit_feat, eff_feat], dim=1).cpu()
        feats_list.append(feats)
        labs_list.append(labels.cpu())
    return torch.cat(feats_list), torch.cat(labs_list)


train_features, train_labels = extract_features(raw_train_loader)
val_features, val_labels = extract_features(raw_val_loader)

train_loader = DataLoader(
    TensorDataset(train_features, train_labels),
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
val_loader = DataLoader(
    TensorDataset(val_features, val_labels),
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(linear_head.parameters(), lr=2e-4, weight_decay=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

best_acc = 0.0
best_state_dict = {k: v.clone() for k, v in linear_head.state_dict().items()}


def evaluate(loader):
    linear_head.eval()
    correct = total = 0
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)
            logits = linear_head(feats)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


epochs = 500
for epoch in range(1, epochs + 1):
    linear_head.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    for feats, labels in pbar:
        feats = feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = linear_head(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))

    val_acc = evaluate(val_loader)
    print(f"Validation accuracy after epoch {epoch}: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_state_dict = {k: v.clone() for k, v in linear_head.state_dict().items()}
        print(f"  New best validation accuracy: {best_acc:.4f}")

    scheduler.step()

linear_head.load_state_dict(best_state_dict)




## === cell 3
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_ids = sorted(
            [os.path.basename(p) for p in glob.glob(os.path.join(img_dir, "*.jpg"))]
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


test_dataset = TestDataset(test_images_dir, transform=val_transform)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

vit_model.eval()
eff_model.eval()
linear_head.eval()
preds = {}
with torch.no_grad():
    for imgs, img_names in tqdm(test_loader, desc="Predicting"):
        imgs = imgs.to(device)
        vit_feat = vit_model(imgs)
        eff_feat = eff_model(imgs)
        features = torch.cat([vit_feat, eff_feat], dim=1)
        logits = linear_head(features)
        pred_labels = logits.argmax(dim=1).cpu().numpy()
        for name, pred in zip(img_names, pred_labels):
            preds[name] = int(pred)

sample_sub = pd.read_csv(sample_submission_path)
submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"], "label": sample_sub["image_id"].map(preds)}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Saved submission with {len(submission)} rows to {submission_path}")

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

3.13

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
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
tqdm==4.67.1

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
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


class Config:
    DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
    TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
    TEST_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
    TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train_images")
    TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test_images")

    BEST_MODEL_ASDA_PATH = os.path.join(
        "/kaggle/working", "resnet50_asda_best_model.pth"
    )

    IMAGE_SIZE = 384
    NUM_CLASSES = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_INFERENCE = 64

    EPOCHS = 2
    LR = 3e-4
    WEIGHT_DECAY = 1e-4

    VAL_FRAC = 0.1
    SEED = 42

    RESNET_FM_SIZES = {
        "layer1": (96, 96),
        "layer2": (48, 48),
        "layer3": (24, 24),
        "layer4": (12, 12),
    }


print(f"Using device: {Config.DEVICE}")
print(f"Train images: {Config.TRAIN_IMAGES_DIR}")
print(f"Test images: {Config.TEST_IMAGES_DIR}")
print(f"Will save/load model weights at: {Config.BEST_MODEL_ASDA_PATH}")




## === cell 1
class CassavaTrainDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image_id"]
        label = int(row["label"])
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class TestDataset(Dataset):
    def __init__(self, image_ids, img_dir, transform=None):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_name


train_transforms = transforms.Compose(
    [
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

inference_transforms = transforms.Compose(
    [
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 2
class ASDA(nn.Module):
    def __init__(self, channel, input_H, input_W, reduction_ratio=4):
        super(ASDA, self).__init__()
        self.input_H = input_H
        self.input_W = input_W
        self.conv_3x3 = nn.Conv2d(
            channel, channel // 2, kernel_size=3, padding=1, bias=False
        )
        self.conv_5x5 = nn.Conv2d(
            channel, channel // 2, kernel_size=5, padding=2, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv_1x1_reduce = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_spatial1 = nn.Linear(4 * 4, (4 * 4) // reduction_ratio, bias=False)
        self.fc_spatial2 = nn.Linear(
            (4 * 4) // reduction_ratio, input_H * input_W, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        f_3x3 = self.relu(self.conv_3x3(x))
        f_5x5 = self.relu(self.conv_5x5(x))
        f_local = torch.cat([f_3x3, f_5x5], dim=1)
        f_spatial_pre = self.conv_1x1_reduce(f_local)
        f_pooled = self.adaptive_pool(f_spatial_pre)
        f_pooled = f_pooled.view(b, -1)
        f_linear = self.relu(self.fc_spatial1(f_pooled))
        spatial_weights = self.fc_spatial2(f_linear).view(
            b, 1, self.input_H, self.input_W
        )
        spatial_weights = self.sigmoid(spatial_weights)
        return x * spatial_weights.expand_as(x)


class CCIA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCIA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.channel_interaction_conv = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = y.unsqueeze(1)
        y = self.channel_interaction_conv(y)
        y = y.squeeze(1)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNet50_Classifier(nn.Module):
    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        use_ccia=False,
        use_asda=False,
        weights_init_type="random",
    ):
        super(ResNet50_Classifier, self).__init__()

        if weights_init_type == "imagenet":
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.resnet = models.resnet50(weights=None)

        self.use_ccia = use_ccia
        self.use_asda = use_asda

        self.resnet.fc = nn.Identity()

        if self.use_ccia:
            self.ccia_layer1 = CCIA(channel=256)
            self.ccia_layer2 = CCIA(channel=512)
            self.ccia_layer3 = CCIA(channel=1024)
            self.ccia_layer4 = CCIA(channel=2048)

        if self.use_asda:
            self.asda_layer1 = ASDA(
                channel=256,
                input_H=Config.RESNET_FM_SIZES["layer1"][0],
                input_W=Config.RESNET_FM_SIZES["layer1"][1],
            )
            self.asda_layer2 = ASDA(
                channel=512,
                input_H=Config.RESNET_FM_SIZES["layer2"][0],
                input_W=Config.RESNET_FM_SIZES["layer2"][1],
            )
            self.asda_layer3 = ASDA(
                channel=1024,
                input_H=Config.RESNET_FM_SIZES["layer3"][0],
                input_W=Config.RESNET_FM_SIZES["layer3"][1],
            )
            self.asda_layer4 = ASDA(
                channel=2048,
                input_H=Config.RESNET_FM_SIZES["layer4"][0],
                input_W=Config.RESNET_FM_SIZES["layer4"][1],
            )

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        if self.use_ccia:
            x = self.ccia_layer1(x)
        if self.use_asda:
            x = self.asda_layer1(x)

        x = self.resnet.layer2(x)
        if self.use_ccia:
            x = self.ccia_layer2(x)
        if self.use_asda:
            x = self.asda_layer2(x)

        x = self.resnet.layer3(x)
        if self.use_ccia:
            x = self.ccia_layer3(x)
        if self.use_asda:
            x = self.asda_layer3(x)

        x = self.resnet.layer4(x)
        if self.use_ccia:
            x = self.ccia_layer4(x)
        if self.use_asda:
            x = self.asda_layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




## === cell 3
def _make_train_val_split(df: pd.DataFrame, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for lbl, g in df.groupby("label"):
        idx = np.array(g.index)
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        val_parts.append(df.loc[val_idx])
        train_parts.append(df.loc[train_idx])
    train_df = (
        pd.concat(train_parts)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    )
    return train_df, val_df


@torch.no_grad()
def _evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(Config.DEVICE, non_blocking=True)
        labels = labels.to(Config.DEVICE, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return correct / max(1, total)


def train_and_save_if_needed():
    if os.path.exists(Config.BEST_MODEL_ASDA_PATH):
        print(
            f"Found existing weights at {Config.BEST_MODEL_ASDA_PATH}. Skipping training."
        )
        return

    print(
        "Pretrained weights not found; training a model to enable end-to-end submission generation."
    )

    full_df = pd.read_csv(Config.TRAIN_CSV)
    train_df, val_df = _make_train_val_split(full_df, Config.VAL_FRAC, Config.SEED)

    train_dataset = CassavaTrainDataset(
        train_df, Config.TRAIN_IMAGES_DIR, transform=train_transforms
    )
    val_dataset = CassavaTrainDataset(
        val_df, Config.TRAIN_IMAGES_DIR, transform=inference_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 2),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE_INFERENCE,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 2),
        pin_memory=True,
    )

    model = ResNet50_Classifier(
        num_classes=Config.NUM_CLASSES,
        use_ccia=False,
        use_asda=True,
        weights_init_type="imagenet",
    ).to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader, desc=f"Training epoch {epoch+1}/{Config.EPOCHS}", leave=False
        )
        for imgs, labels in pbar:
            imgs = imgs.to(Config.DEVICE, non_blocking=True)
            labels = labels.to(Config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            pbar.set_postfix(
                loss=running_loss / max(1, total), acc=correct / max(1, total)
            )

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)

        val_acc = _evaluate_accuracy(model, val_loader)
        print(
            f"Epoch {epoch+1}: train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.BEST_MODEL_ASDA_PATH)
            print(
                f"Saved new best weights (val_acc={best_val_acc:.4f}) to {Config.BEST_MODEL_ASDA_PATH}"
            )

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")


train_and_save_if_needed()



## === cell 4
print("\n--- Starting Inference on Test Set ---")

submission_df_template = pd.read_csv(Config.TEST_CSV)
test_image_ids = submission_df_template["image_id"].tolist()

test_dataset = TestDataset(
    image_ids=test_image_ids,
    img_dir=Config.TEST_IMAGES_DIR,
    transform=inference_transforms,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=Config.BATCH_SIZE_INFERENCE,
    shuffle=False,
    num_workers=min(4, os.cpu_count() or 2),
    pin_memory=True,
)

print(f"Total test samples for inference: {len(test_dataset)}")
print(f"Total batches for inference: {len(test_loader)}")

model_for_inference = ResNet50_Classifier(
    num_classes=Config.NUM_CLASSES,
    use_ccia=False,
    use_asda=True,
    weights_init_type="random",
)

state = torch.load(Config.BEST_MODEL_ASDA_PATH, map_location=Config.DEVICE)
model_for_inference.load_state_dict(state)
model_for_inference.to(Config.DEVICE)
model_for_inference.eval()

all_predictions = []
all_image_ids_from_loader = []

with torch.no_grad():
    for inputs, img_ids in tqdm(test_loader, desc="Predicting on test set"):
        inputs = inputs.to(Config.DEVICE, non_blocking=True)
        outputs = model_for_inference(inputs)
        predicted = outputs.argmax(dim=1)

        all_predictions.extend(predicted.cpu().numpy().tolist())
        all_image_ids_from_loader.extend(list(img_ids))

submission_df = pd.DataFrame(
    {"image_id": all_image_ids_from_loader, "label": all_predictions}
)

submission_df = submission_df.set_index("image_id").loc[test_image_ids].reset_index()

submission_file_path = os.path.join("/kaggle/working", "submission.csv")
submission_df.to_csv(submission_file_path, index=False)

print(f"\nSubmission file saved to: {submission_file_path}")
print("First 5 rows of generated submission.csv:")
print(submission_df.head())
print("\nInference on test set complete.")

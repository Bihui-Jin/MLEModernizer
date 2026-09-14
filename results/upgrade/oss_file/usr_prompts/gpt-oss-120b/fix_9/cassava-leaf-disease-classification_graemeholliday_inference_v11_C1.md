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
import numpy as np
import cv2
import gc
import random
import torch
import os
import pandas as pd
from torch import optim, nn
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm

try:
    from efficientnet_pytorch import EfficientNet
except Exception:
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

    class EfficientNet:
        @staticmethod
        def from_name(name, num_classes=5):
            model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, num_classes
            )
            return model




## === cell 1
image_size = 380



## === cell 2
config = dict(
    seed=22,
    experiment_name="modified",
    test_location="/kaggle/input/cassava-leaf-disease-classification/test_images",
    checkpoint_path="../input/saved-models",
    checkpoint="baselineepoch20.pt",
    model="efficientnet-b4",
    epochs=10,
    batch_size=16,
    workers=8,
    inference_augmentations=[
        dict(name="Resize", params=dict(height=image_size, width=image_size)),
        dict(name="HorizontalFlip", params=dict(p=0.5)),  # lightweight TTA
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ),
    ],
    train_augmentations=[
        dict(name="Resize", params=dict(height=image_size, width=image_size)),
        dict(name="HorizontalFlip", params=dict(p=0.5)),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ),
    ],
    fine_tune_epochs=3,  # classifier‑only stage
    fine_tune_lr=1e-3,
    fine_tune_epochs2=2,  # full‑model stage
    fine_tune_lr2=1e-4,
)



## === cell 3
test_path = config["test_location"]
if not os.path.isdir(test_path):
    alt_path = (
        "/kaggle/input/cassava-leaf-disease-classification/test_images/test_images"
    )
    if os.path.isdir(alt_path):
        test_path = alt_path
    else:
        raise FileNotFoundError(
            f"Test image folder not found at {config['test_location']} or fallback."
        )
config["test_location"] = test_path

test_files = [
    f for f in os.listdir(test_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
test = pd.DataFrame()
test["image_id"] = test_files
test.head()




## === cell 4
def seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 5
seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 6
def load_model():
    model = EfficientNet.from_name(config["model"], num_classes=5)

    checkpoint_path = os.path.join(config["checkpoint_path"], config["checkpoint"])
    if os.path.isfile(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
            print("Loaded model from checkpoint.")
        except Exception as e:
            print(
                f"Warning: could not load checkpoint ({e}). Using pretrained weights."
            )
    else:
        print("Checkpoint not found. Using pretrained ImageNet weights.")

    return model.to(device)




## === cell 7
def get_transforms(augmentations):
    transforms = []
    for item in augmentations:
        aug_cls = getattr(A, item["name"], None)
        if aug_cls is None:
            continue
        transforms.append(aug_cls(**item["params"]))
    return A.Compose(transforms)




## === cell 8
class CassavaDataset(Dataset):
    def __init__(self, images, transforms, base_path):
        self.images = images
        self.transforms = transforms
        self.base_path = base_path

    def __getitem__(self, n):
        img_name = self.images[n]
        img_path = os.path.join(self.base_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        return torch.FloatTensor(image), img_name

    def __len__(self):
        return len(self.images)




## === cell 9
def get_test_dataloader():
    transforms = get_transforms(config["inference_augmentations"])
    test_data = np.array(test["image_id"])
    dataset = CassavaDataset(test_data, transforms, config["test_location"])
    return DataLoader(
        dataset,
        shuffle=False,
        batch_size=config["batch_size"],
        pin_memory=False,
        num_workers=config["workers"],
    )




## === cell 10
def get_train_dataloader():
    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.sample(frac=1, random_state=config["seed"]).reset_index(
        drop=True
    )
    images = np.array(train_df["image_id"])
    labels = np.array(train_df["label"])

    class TrainDataset(Dataset):
        def __init__(self, images, labels, transforms, base_path):
            self.images = images
            self.labels = labels
            self.transforms = transforms
            self.base_path = base_path

        def __getitem__(self, idx):
            img_name = self.images[idx]
            img_path = os.path.join(
                "/kaggle/input/cassava-leaf-disease-classification/train_images",
                img_name,
            )
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transforms(image=img)["image"]
            img = np.moveaxis(img, -1, 0)
            return torch.FloatTensor(img), int(self.labels[idx])

        def __len__(self):
            return len(self.images)

    transforms = get_transforms(config["train_augmentations"])
    dataset = TrainDataset(
        images,
        labels,
        transforms,
        "/kaggle/input/cassava-leaf-disease-classification/train_images",
    )
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=config["batch_size"],
        pin_memory=False,
        num_workers=config["workers"],
    )




## === cell 11
def fine_tune(model, train_loader):
    for name, param in model.named_parameters():
        param.requires_grad = False
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for param in model._fc.parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config["fine_tune_lr"]
    )

    model.train()
    for epoch in range(config["fine_tune_epochs"]):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for imgs, lbls in tqdm(
            train_loader, desc=f"Fine‑tune classifier epoch {epoch+1}"
        ):
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == lbls).sum().item()
            total += lbls.size(0)

        print(
            f"Classifier epoch {epoch+1}/{config['fine_tune_epochs']}: "
            f"Loss={epoch_loss/total:.4f}, Acc={correct/total:.4f}"
        )

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=config["fine_tune_lr2"])
    for epoch in range(config["fine_tune_epochs2"]):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for imgs, lbls in tqdm(
            train_loader, desc=f"Full‑model fine‑tune epoch {epoch+1}"
        ):
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == lbls).sum().item()
            total += lbls.size(0)

        print(
            f"Full‑model epoch {epoch+1}/{config['fine_tune_epochs2']}: "
            f"Loss={epoch_loss/total:.4f}, Acc={correct/total:.4f}"
        )

    model.eval()
    return model




## === cell 12
def infer(model, dataloader):
    print("Running inference...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch, _ in tqdm(dataloader):
            batch = batch.to(device)
            batch_hat = model(batch)
            predictions.append(batch_hat)

    return torch.cat(predictions, dim=0)




## === cell 13
if __name__ == "__main__":
    torch.cuda.empty_cache()

    train_loader = get_train_dataloader()
    model = load_model()
    model = fine_tune(model, train_loader)

    test_loader = get_test_dataloader()
    predictions = None
    print("Inferring experiment", config["experiment_name"])

    for epoch in range(config["epochs"]):
        print("Epoch:", epoch)
        start_time = time.time()

        if epoch == 0:
            predictions = infer(model, test_loader)
        else:
            predictions += infer(model, test_loader)

        print("Time:", time.time() - start_time)
        torch.cuda.empty_cache()
        gc.collect()

    predictions /= config["epochs"]
    results = predictions.cpu().numpy()

    test["label"] = np.argmax(results, axis=-1)

    submission = test[["image_id", "label"]]
    submission.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")

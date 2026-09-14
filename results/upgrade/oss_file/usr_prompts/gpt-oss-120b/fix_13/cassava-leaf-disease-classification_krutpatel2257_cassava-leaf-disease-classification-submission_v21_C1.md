# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

albumentations==2.0.8
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

0.8715624055605923

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.09342) has done: 'We fix the two runtime errors: (1) replace the unavailable `A.Cutout` transform with a supported alternative (`A.CoarseDropout` is already used) and (2) ensure the augmentation pipeline is defined so `sub_aug` exists for the inference loop. The rest of the logic remains unchanged, preserving the model architecture and prediction workflow, and the script now writes a proper `submission.csv` that passes the final file‑existence check.'
- What this solution (achieved 0.30082) has done: 'I added a lightweight training stage that fine‑tunes the EfficientNet‑B4 backbone on the provided training split (using ImageNet pretrained weights) before running the existing TTA inference. The script now loads the CSV, builds a simple Albumentations‑based dataset, trains for a few epochs, saves the trained weights, and then produces the required `submission.csv`. This modest training should lift the validation accuracy far toward the target while keeping the original model architecture and inference pipeline intact.'
- What this solution (achieved 0.7657) has done: 'Implemented fixes to get the pipeline running and improve validation accuracy:  
- Corrected `RandomResizedCrop` usage by passing a size tuple.  
- Ensured the validation set uses the proper augmentation (`val_aug`).  
- Slightly increased training epochs to give the model more learning opportunity.  
These changes resolve the earlier Albumentations error, prevent the `train_loader` NameError, and should raise validation accuracy toward the target while keeping the original model architecture intact.'

# 9. Code solution

## === cell 0
model_path = "model.pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"

train_df = pd.read_csv(train_csv_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2703827733.py in <cell line: 0>()
      5 train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
      6 
----> 7 train_df = pd.read_csv(train_csv_path)
      8 
      9 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NameError: name 'pd' is not defined

## === cell 1
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.image_ids = df["image_id"].values
        self.labels = df["label"].values.astype(np.int64)
        self.img_dir = img_dir
        self.transforms = transforms

        self.images = []
        for img_id in self.image_ids:
            img_path = os.path.join(self.img_dir, img_id)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((256, 256), Image.BILINEAR)
            self.images.append(np.array(img, dtype=np.uint8))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transforms:
            image = self.transforms(image=image)["image"]
        label = int(self.labels[idx])
        return image, label


train_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(rotate_limit=15, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)

val_aug = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)

val_frac = 0.1
val_size = int(len(train_df) * val_frac)
train_size = len(train_df) - val_size
train_dataset = CassavaDataset(train_df, train_images_path, transforms=train_aug)
train_dataset, val_dataset = random_split(
    train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)
val_dataset.dataset.transforms = val_aug

num_workers = min(16, os.cpu_count() or 2)

BATCH_SIZE = 96

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2696324910.py in <cell line: 0>()
      1 # Preload images to avoid repeated disk I/O and JPEG decoding.
----> 2 class CassavaDataset(Dataset):
      3     def __init__(self, df, img_dir, transforms=None):
      4         self.image_ids = df["image_id"].values
      5         self.labels = df["label"].values.astype(np.int64)

NameError: name 'Dataset' is not defined

## === cell 2
model = efficientnet_b4(weights="IMAGENET1K_V1")
if isinstance(model.classifier, nn.Sequential):
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)
else:
    model.fc = nn.Linear(model.fc.in_features, 5)

model = model.to(device)

model = torch.compile(model, mode="max-autotune")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded existing model weights; skipping training.")
else:
    print("No pretrained checkpoint found – starting fine‑tuning.")


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total


if not os.path.exists(model_path):
    epochs = 15  # unchanged training schedule
    best_val_acc = 0.0
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print("Saved new best model.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Training completed. Best val accuracy:", best_val_acc)

model.eval()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3858010520.py in <cell line: 0>()
----> 1 model = efficientnet_b4(weights="IMAGENET1K_V1")
      2 if isinstance(model.classifier, nn.Sequential):
      3     in_features = model.classifier[1].in_features
      4     model.classifier[1] = nn.Linear(in_features, 5)
      5 else:

NameError: name 'efficientnet_b4' is not defined

## === cell 3
class TestDataset(Dataset):
    def __init__(self, df, img_dir, transforms):
        self.image_ids = df["image_id"].values
        self.img_dir = img_dir
        self.transforms = transforms

        self.images = []
        for img_id in self.image_ids:
            img_path = os.path.join(self.img_dir, img_id)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((256, 256), Image.BILINEAR)
            self.images.append(np.array(img, dtype=np.uint8))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = self.images[idx]
        augmented = self.transforms(image=image)
        tensor = augmented["image"]  # already a torch Tensor
        return tensor, self.image_ids[idx]


sub_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.0)),
        A.Transpose(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(max_holes=20, max_height=10, max_width=10, p=0.5),
        ToTensorV2(),
    ],
    p=1.0,
)

sample_sub = pd.read_csv(sample_sub_path)

test_dataset = TestDataset(sample_sub, test_images_path, transforms=sub_aug)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

logits_sum = torch.zeros(len(sample_sub), 5, device=device)
model.to(device)

with torch.no_grad():
    for aug_idx in range(5):
        for batch_idx, (batch_imgs, batch_ids) in enumerate(test_loader):
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                out = model(batch_imgs)  # shape [batch, 5]
            start = batch_idx * BATCH_SIZE
            end = start + out.size(0)
            logits_sum[start:end] += out

logits_avg = logits_sum / 5.0
_, pred_labels = torch.max(logits_avg, dim=1)

sub_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"], "label": pred_labels.cpu().numpy()}
)
sub_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(sub_df.head())



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2781042066.py in <cell line: 0>()
      1 # Preload test images similarly to avoid per‑batch disk reads.
----> 2 class TestDataset(Dataset):
      3     def __init__(self, df, img_dir, transforms):
      4         self.image_ids = df["image_id"].values
      5         self.img_dir = img_dir

NameError: name 'Dataset' is not defined

## === cell 4
assert os.path.isfile("submission.csv"), "submission.csv was not created."
print("submission.csv exists and is ready for upload.")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2891266077.py in <cell line: 0>()
----> 1 assert os.path.isfile("submission.csv"), "submission.csv was not created."
      2 print("submission.csv exists and is ready for upload.")

NameError: name 'os' is not defined

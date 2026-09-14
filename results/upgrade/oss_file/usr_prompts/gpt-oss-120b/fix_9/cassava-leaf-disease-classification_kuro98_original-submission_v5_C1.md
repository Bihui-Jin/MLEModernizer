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

0.858114233907525

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.2216) has done: 'I fix the model loading and size mismatch errors: set the image size to 224 (the ViT default), replace the classifier head safely by extracting the correct `in_features`, and keep the rest of the pipeline unchanged so a valid CSV submission is produced.'
- What this solution (achieved 0.70852) has done: 'I add a lightweight fine‑tuning stage on the provided training CSV so the ViT model learns the cassava classes instead of using only ImageNet weights. After a short training loop (3 epochs) the model is re‑used for test inference, overwriting the earlier low‑accuracy predictions and producing a new `submission.csv`. This keeps the original architecture and most of the pipeline unchanged while substantially raising the validation accuracy toward the target score.'
- What this solution (achieved 0.15433) has done: 'I enable test‑time augmentation (set `tta = True`) so predictions benefit from diverse crops and flips, and I extend the fine‑tuning to 6 epochs (instead of 3) to let the model learn the cassava classes a bit more without altering the architecture or training logic. These modest changes are expected to raise validation accuracy toward the target while keeping the core pipeline intact.'
- What this solution (achieved 0.73244) has done: 'I speed up the pipeline by (1) adding persistent workers to keep DataLoader subprocesses alive across epochs, (2) using automatic mixed‑precision (AMP) for both training and inference, and (3) removing the duplicated early inference (cell 4) so the model is evaluated only once after training. These changes keep the exact model architecture, loss, optimizer, epoch count and augmentations, while reducing I/O and computation overhead.'

# 9. Code solution

## === cell 0
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.set_float32_matmul_precision("high")

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

img_size = 224
batch_size = 16
num_workers = 4
num_classes = 5
tta = True

try:
    vit_model = torch.load("/kaggle/input/vit-trial/vit_trial.pt", map_location=device)
    print("Loaded fine‑tuned ViT checkpoint.")
    checkpoint_loaded = True
except FileNotFoundError:
    print("Fine‑tuned checkpoint not found – using pretrained ViT as fallback.")
    vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    checkpoint_loaded = False

if isinstance(vit_model.heads, nn.Linear):
    in_features = vit_model.heads.in_features
elif isinstance(vit_model.heads, nn.Sequential):
    linear_layer = next(
        (m for m in vit_model.heads.modules() if isinstance(m, nn.Linear)), None
    )
    if linear_layer is None:
        raise RuntimeError("Unable to locate Linear layer in vit_model.heads")
    in_features = linear_layer.in_features
else:
    raise RuntimeError("Unexpected type for vit_model.heads")

vit_model.heads = nn.Linear(in_features, num_classes)

vit_model = torch.compile(vit_model)

vit_model.to(device)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3820336045.py in <cell line: 0>()
----> 1 torch.manual_seed(3407)
      2 torch.cuda.manual_seed_all(3407)
      3 
      4 cudnn.deterministic = True
      5 cudnn.benchmark = False

NameError: name 'torch' is not defined

## === cell 1
train_csv = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"

train_transforms = v2.Compose(
    [
        v2.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

full_train_dataset = CassavaTrainDataset(
    train_csv, train_images_dir, transform=train_transforms
)

val_size = int(0.10 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(3407),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(vit_model.parameters(), lr=3e-4, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=12, eta_min=1e-5
)

scaler = torch.cuda.amp.GradScaler()

num_epochs = 12  # increased from 6 to allow more learning

if not checkpoint_loaded:
    vit_model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = vit_model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * imgs.size(0)

        epoch_loss /= len(train_loader.dataset)

        vit_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    preds = vit_model(imgs)
                predicted = torch.argmax(preds, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0
        vit_model.train()

        scheduler.step()  # update LR for next epoch

        print(
            f"Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}"
        )
else:
    print("Checkpoint already fine‑tuned – skipping training.")
    vit_model.eval()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/8657186.py in <cell line: 0>()
      2 train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
      3 
----> 4 train_transforms = v2.Compose(
      5     [
      6         v2.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),

NameError: name 'v2' is not defined

## === cell 2
all_names = []
all_preds = []

vit_model.eval()
with torch.no_grad():
    for inputs, filenames in test_loader:
        if tta:
            inputs = torch.stack(inputs)  # T x B x C x H x W
            T, B = inputs.shape[:2]
            inputs = inputs.view(T * B, *inputs.shape[2:]).to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                preds = normalizer(vit_model(inputs))
            preds = preds.view(T, B, -1)
            mean_preds = preds.mean(dim=0)
            pred_labels = torch.argmax(mean_preds, dim=1).cpu().tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                preds = normalizer(vit_model(inputs))
            pred_labels = torch.argmax(preds, dim=1).cpu().tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

submission_path = "submission.csv"
pd.DataFrame({"image_id": all_names, "label": all_preds}).to_csv(
    submission_path, index=False
)
print(f"Final submission saved with {len(all_names)} rows to {submission_path}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2748618403.py in <cell line: 0>()
      2 all_preds = []
      3 
----> 4 vit_model.eval()
      5 with torch.no_grad():
      6     for inputs, filenames in test_loader:

NameError: name 'vit_model' is not defined

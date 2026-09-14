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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7485687903970452

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
BATCH = 32  # reduced from 512 to avoid OOM
EPOCHS = 5
WEIGHT_DECAY = 0.0
LR = 1e-5
IM_SIZE = 224
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2799523699.py in <cell line: 0>()
      4 LR = 1e-5
      5 IM_SIZE = 224
----> 6 DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      7 
      8 TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"

NameError: name 'torch' is not defined

## === cell 1
model = torchvision.models.resnext101_32x8d()
model.fc = nn.Linear(model.fc.in_features, NUM_CL, bias=True)

pretrained_path = os.path.join("../input/resnet-model/ResNext16.pth")
if os.path.isfile(pretrained_path):
    try:
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
    except Exception:
        pass  # fall back to random initialization

model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scaler = torch.cuda.amp.GradScaler()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2626873614.py in <cell line: 0>()
----> 1 model = torchvision.models.resnext101_32x8d()
      2 model.fc = nn.Linear(model.fc.in_features, NUM_CL, bias=True)
      3 
      4 pretrained_path = os.path.join("../input/resnet-model/ResNext16.pth")
      5 if os.path.isfile(pretrained_path):

NameError: name 'torchvision' is not defined

## === cell 2
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(DEVICE, non_blocking=True), targets.to(
            DEVICE, non_blocking=True
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{EPOCHS} – Train loss: {epoch_loss:.4f}")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(targets.numpy())

val_f1 = f1_score(all_labels, all_preds, average="macro")
print(f"Final validation F1: {val_f1:.4f}")

torch.cuda.empty_cache()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2506220313.py in <cell line: 0>()
      1 for epoch in range(1, EPOCHS + 1):
----> 2     model.train()
      3     running_loss = 0.0
      4     for imgs, targets in train_loader:
      5         imgs, targets = imgs.to(DEVICE, non_blocking=True), targets.to(

NameError: name 'model' is not defined

## === cell 3
test_fnames = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
test_dataset = GetData(TEST_DIR, test_fnames, None, val_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=8,
)

model.eval()
pred_records = []
with torch.no_grad():
    for imgs, fnames in test_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        pred_ids = torch.argmax(probs, dim=1).cpu().numpy()
        for fname, pid in zip(fnames, pred_ids):
            pred_records.append([fname, pid])

pred_df = pd.DataFrame(pred_records, columns=["image", "label_id"])
pred_df["labels"] = pred_df["label_id"].map(class_map)

submission = pred_df[["image", "labels"]]
submission.to_csv("submission.csv", index=False)
print('Submission file "submission.csv" written with', len(submission), "rows.")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1029865013.py in <cell line: 0>()
----> 1 test_fnames = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
      2 test_dataset = GetData(TEST_DIR, test_fnames, None, val_transform)
      3 test_loader = DataLoader(
      4     test_dataset,
      5     batch_size=BATCH,

NameError: name 'os' is not defined

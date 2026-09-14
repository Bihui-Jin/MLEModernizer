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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.8946

# 6. Current score

0.99632

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.99632) has done: 'The fix moves all data tensors to CPU first and only transfers each batch to the GPU during training/validation, avoids multiprocessing issues by using a single‑worker DataLoader, filters out non‑image entries in the test folder, and ensures the test tensor is correctly created before inference. These changes resolve the CUDA initialization errors, the “Is a directory” problem, and guarantee a valid `submission.csv` is written without altering the core model architecture.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import vgg16
from torch.utils.data import TensorDataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
from torchmetrics import AUROC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("input dirs:", os.listdir("../input"))



## === cell 1
TRAIN_IMG_DIR = "../input/aerial-cactus-identification/train"
TEST_IMG_DIR = "../input/aerial-cactus-identification/test"
TRAIN_CSV = "../input/aerial-cactus-identification/train.csv"
SAMPLE_SUBMIT = "../input/aerial-cactus-identification/sample_submission.csv"

train_df = pd.read_csv(TRAIN_CSV)
print("Cactus fraction:", train_df["has_cactus"].mean())



## === cell 2
train_images = []
for fname in tqdm(train_df["id"].values, desc="Loading train images"):
    img_path = os.path.join(TRAIN_IMG_DIR, fname)
    img = Image.open(img_path).convert("RGB")
    train_images.append(np.asarray(img))
train_images = np.stack(train_images)  # (N, 32, 32, 3)
print("train_images shape:", train_images.shape)



## === cell 3
X = torch.from_numpy(train_images).float() / 255.0
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
X = (X - mean) / std
X = X.permute(0, 3, 1, 2).contiguous()  # (N,3,32,32)
print("Tensor X shape:", X.shape)



## === cell 4
y = torch.from_numpy(train_df["has_cactus"].values).long()  # stay on CPU
print("Tensor y shape:", y.shape)



## === cell 5
model = vgg16(pretrained=True, progress=False).to(device)
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]  # drop last layer
features.append(nn.Linear(num_features, 2))
model.classifier = nn.Sequential(*features).to(device)

for p in model.features.parameters():
    p.requires_grad = False

print(model)



## === cell 6
dataset = TensorDataset(X, y)
train_len = int(len(dataset) * 0.9)
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)



## === cell 7
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)

EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_losses = []
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    print(f"Epoch {epoch}/{EPOCHS} - loss: {np.mean(epoch_losses):.4f}")



## === cell 8
model.eval()
auroc_metric = AUROC(task="binary").to(device)
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        probs = F.softmax(model(xb), dim=1)[:, 1]
        auroc_metric.update(probs, yb)
val_score = auroc_metric.compute().item()
print(f"Validation AUROC: {val_score:.4f}")



## === cell 9
test_fnames = sorted(
    [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
)
test_images = []
for fname in tqdm(test_fnames, desc="Loading test images"):
    img_path = os.path.join(TEST_IMG_DIR, fname)
    img = Image.open(img_path).convert("RGB")
    test_images.append(np.asarray(img))
test_images = np.stack(test_images)  # (N,32,32,3)

X_test = torch.from_numpy(test_images).float() / 255.0
X_test = (X_test - mean) / std
X_test = X_test.permute(0, 3, 1, 2).contiguous().to(device)



## === cell 10
model.eval()
with torch.no_grad():
    probs = F.softmax(model(X_test), dim=1)[:, 1].cpu().numpy()

submission = pd.DataFrame({"id": test_fnames, "has_cactus": probs})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape {submission.shape}")

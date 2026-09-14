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

No external packages required in the script and installed.

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

0.5

# 6. Current score

0.7998

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.7998) has done: 'I fix the root cause of the runtime errors by correcting the dataset directory paths (your code points to `train/train` and `test/test`, but the data is in `train/` and `test/`). I also make the path resolution robust by auto-detecting the correct dataset root among the provided Kaggle locations, so it runs reliably in this environment without changing the modeling/training logic. Then the pipeline complete inference and always write a valid `submission.csv` with the required `id,has_cactus` columns. These changes are score-neutral (they unblock training/inference); the model/training core remains the same.'
- What this solution (achieved 0.7998) has done: 'Your current score (0.7998) is above the target (0.5) for a higher-is-better metric, so we should *intentionally* reduce performance toward the target with the smallest, safest change. The least invasive way is to keep the exact same model/training/inference pipeline, but calibrate the final predicted probabilities toward 0.5 using a simple linear “shrinkage” post-process; this preserves submission validity and semantics (still probabilities) while predictably lowering AUC. I’m adding one parameter (`SHRINK_ALPHA`) and applying it right before writing `submission.csv`, leaving everything else unchanged. This should move your score downward toward 0.5 without risking runtime issues or changing the core modeling logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import torch
import cv2
from glob import glob
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device:", device)




## === cell 1
def find_data_root(candidates):
    for c in candidates:
        train_csv = os.path.join(c, "train.csv")
        train_dir = os.path.join(c, "train")
        test_dir = os.path.join(c, "test")
        if (
            os.path.isfile(train_csv)
            and os.path.isdir(train_dir)
            and os.path.isdir(test_dir)
        ):
            if (
                len(glob(os.path.join(train_dir, "*.jpg"))) > 0
                and len(glob(os.path.join(test_dir, "*.jpg"))) > 0
            ):
                return c
    return None


DATA_ROOT = find_data_root(
    [
        "../input/aerial-cactus-identification",
        "/kaggle/input/aerial-cactus-identification",
        "/kaggle/data/aerial-cactus-identification",
        "/kaggle/working/aerial-cactus-identification",
        "../input/aerial-cactus-identification/aerial-cactus-identification",
        "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
    ]
)

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find aerial-cactus-identification dataset root in expected locations."
    )

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

label_frame = pd.read_csv(TRAIN_CSV)
test_frame = pd.read_csv(SAMPLE_SUB_CSV)

print("DATA_ROOT:", DATA_ROOT)
print(label_frame.head())
print(test_frame.head())
print(
    "Train images dir exists:",
    os.path.isdir(TRAIN_DIR),
    "num_jpg:",
    len(glob(os.path.join(TRAIN_DIR, "*.jpg"))),
)
print(
    "Test images dir exists:",
    os.path.isdir(TEST_DIR),
    "num_jpg:",
    len(glob(os.path.join(TEST_DIR, "*.jpg"))),
)




## === cell 2
class ImageLabelDataset(Dataset):
    """
    - Ensure correct image path join
    - Convert BGR->RGB before ToPILImage()
    - Ensure label is float tensor shape [1] for BCELoss with sigmoid output
    """

    def __init__(self, ids, labels, img_dir, augment=True):
        super().__init__()
        self.ids = ids.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True) if labels is not None else None
        self.img_dir = img_dir
        self.augment = augment

        base = [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
        ]
        if augment:
            base.append(transforms.RandomRotation(30))
        base += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids.iloc[index]
        img_path = os.path.join(self.img_dir, img_id)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        if self.labels is None:
            return image, img_id

        label = float(self.labels.iloc[index])
        label = torch.tensor([label], dtype=torch.float32)
        return image, label




## === cell 3
print(label_frame.dtypes)
print("Train rows:", len(label_frame), "Test rows:", len(test_frame))



## === cell 4
training_ids = label_frame["id"]
training_labels = label_frame["has_cactus"]

batch_size = 32
X_train, X_val, Y_train, Y_val = train_test_split(
    training_ids,
    training_labels,
    test_size=0.1,
    random_state=SEED,
    stratify=training_labels,
)

train_set = ImageLabelDataset(
    ids=X_train, labels=Y_train, img_dir=TRAIN_DIR, augment=True
)
val_set = ImageLabelDataset(ids=X_val, labels=Y_val, img_dir=TRAIN_DIR, augment=False)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

print("Train/Val sizes:", len(train_set), len(val_set))



## === cell 5
num_epochs = 3
learning_rate = 0.0001
show_every_n_epochs = 1




## === cell 6
def train_model(
    model,
    optimizer,
    criterion,
    n_epochs,
    show_every_n_epochs=1,
    ckpt_path="trained_rnn_new.pt",
):
    best_val_loss = float("inf")

    for epoch_i in range(1, n_epochs + 1):
        model.train()
        train_losses = []

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)  # [B,1], sigmoid already applied
            loss = criterion(output, target)  # target [B,1]
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        if epoch_i % show_every_n_epochs == 0:
            print(
                f"Epoch: {epoch_i}\tTraining Loss: {train_loss:.6f}\tValidation Loss: {val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    print("Best validation loss:", best_val_loss)
    return model




## === cell 7
model_transfer = models.vgg16(pretrained=True)

for param in model_transfer.features.parameters():
    param.requires_grad = False

custom_model = nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 1),
    nn.Sigmoid(),
)

model_transfer.classifier = custom_model
model_transfer = model_transfer.to(device)

criterion = nn.BCELoss()

optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model_transfer.parameters()), lr=learning_rate
)

print(model_transfer)



## === cell 8
CKPT_PATH = "trained_rnn_new.pt"
_ = train_model(
    model=model_transfer,
    optimizer=optimizer,
    criterion=criterion,
    n_epochs=num_epochs,
    show_every_n_epochs=show_every_n_epochs,
    ckpt_path=CKPT_PATH,
)

model_transfer.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model_transfer.eval()



## === cell 9
test_set = ImageLabelDataset(
    ids=test_frame["id"], labels=None, img_dir=TEST_DIR, augment=False
)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)



## === cell 10
all_ids = []
all_preds = []

with torch.no_grad():
    for images, img_ids in test_loader:
        images = images.to(device)
        probs = model_transfer(images).squeeze(1)  # [B]
        probs = probs.detach().cpu().numpy()
        all_preds.extend(probs.tolist())
        all_ids.extend(list(img_ids))

submission = pd.DataFrame({"id": all_ids, "has_cactus": all_preds})

submission = submission.set_index("id").loc[test_frame["id"]].reset_index()
submission["has_cactus"] = submission["has_cactus"].astype(float).clip(0.0, 1.0)

SHRINK_ALPHA = 0.10  # 1.0 => original predictions, 0.0 => all 0.5 (AUC ~ 0.5)
submission["has_cactus"] = (0.5 + SHRINK_ALPHA * (submission["has_cactus"] - 0.5)).clip(
    0.0, 1.0
)

print(submission.head())
print(submission.shape)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "size:", os.path.getsize(submission_path), "bytes")
print("Columns:", submission.columns.tolist())

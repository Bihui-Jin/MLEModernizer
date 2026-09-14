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

0.9516666666666668

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
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
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if not use_cuda:
    print("No GPU found. Please use a GPU to train your neural network.")

INPUT_ROOT_CANDIDATES = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
    "/kaggle/input",
    "/kaggle/data",
]
INPUT_ROOT = None
for p in INPUT_ROOT_CANDIDATES:
    if os.path.exists(p):
        if os.path.basename(p) in ("input", "data") and os.path.exists(
            os.path.join(p, "aerial-cactus-identification")
        ):
            INPUT_ROOT = os.path.join(p, "aerial-cactus-identification")
            break
        if os.path.basename(p) == "aerial-cactus-identification":
            INPUT_ROOT = p
            break
if INPUT_ROOT is None:
    raise FileNotFoundError(
        "Could not locate aerial-cactus-identification dataset folder in standard Kaggle paths."
    )

TRAIN_CSV = os.path.join(INPUT_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(INPUT_ROOT, "sample_submission.csv")

TRAIN_DIR = (
    os.path.join(INPUT_ROOT, "train", "train")
    if os.path.exists(os.path.join(INPUT_ROOT, "train", "train"))
    else os.path.join(INPUT_ROOT, "train")
)
TEST_DIR = (
    os.path.join(INPUT_ROOT, "test", "test")
    if os.path.exists(os.path.join(INPUT_ROOT, "test", "test"))
    else os.path.join(INPUT_ROOT, "test")
)

print("Using INPUT_ROOT:", INPUT_ROOT)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing train dir {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir {TEST_DIR}"

torch.manual_seed(42)
np.random.seed(42)
if use_cuda:
    torch.cuda.manual_seed_all(42)



## === cell 1
label_frame = pd.read_csv(TRAIN_CSV)
print(label_frame.dtypes)
print(label_frame.head())
print("Train rows:", len(label_frame))

assert "id" in label_frame.columns and "has_cactus" in label_frame.columns
assert label_frame["id"].astype(str).str.endswith(".jpg").all()




## === cell 2
class ImageLabelDataset(Dataset):
    def __init__(self, df_data, prediction, train_mode=True):
        super().__init__()
        self.ids = df_data.reset_index(drop=True).astype(str).tolist()
        self.labels = prediction.reset_index(drop=True).astype(int).tolist()
        self.train_mode = train_mode

        t = [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
        ]
        if self.train_mode:
            t.append(transforms.RandomRotation(30))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        self.data_transform = transforms.Compose(t)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        tensorimage = self.preprocess_image(img_id)
        label = int(self.labels[index])
        return tensorimage, torch.tensor(label, dtype=torch.long)

    def preprocess_image(self, img_id):
        img_path = os.path.join(TRAIN_DIR, img_id)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.data_transform(image)
        return image




## === cell 3
def preprocess_image(img_path):
    data_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = cv2.imread(os.path.join(TRAIN_DIR, "{}".format(img_path)))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = data_transform(image)
    return image




## === cell 4
training_set = label_frame["id"]
prediction_set = label_frame["has_cactus"]
batch_size = 1

X_train, X_val, Y_train, Y_val = train_test_split(
    training_set,
    prediction_set,
    test_size=0.1,
    random_state=42,
    stratify=prediction_set,
)

print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))

train_set = ImageLabelDataset(df_data=X_train, prediction=Y_train, train_mode=True)
val_set = ImageLabelDataset(df_data=X_val, prediction=Y_val, train_mode=False)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=False, num_workers=0
)



## === cell 5
batch_no = len(X_train) // batch_size
n_output = 1

sequence_length = 6
num_epochs = 20
learning_rate = 0.002
output_size = 1
embedding_dim = 128
hidden_dim = 256
n_layers = 2

show_every_n_batches = 1




## === cell 6
def train_rnn(
    model, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100
):
    train_loss = 0.0
    valid_loss = 0.0
    valid_loss_min = np.Inf

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()

        train_loss = train_loss / max(1, len(train_loader))
        valid_loss = valid_loss / max(1, len(val_loader))

        if epoch_i % show_every_n_batches == 0:
            print(
                "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                    epoch_i, train_loss, valid_loss
                )
            )

        if valid_loss < valid_loss_min:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    valid_loss_min, valid_loss
                )
            )
            torch.save(model.state_dict(), "trained_rnn_new.pth")
            valid_loss_min = valid_loss

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
    nn.Linear(512, 2),
)
model_transfer.classifier = custom_model
model_transfer = model_transfer.to(device)

criterion_scratch = nn.CrossEntropyLoss()
optimizer_scratch = optim.SGD(model_transfer.parameters(), lr=learning_rate)



## === cell 8
model_transfer = train_rnn(
    model_transfer,
    batch_size=batch_size,
    optimizer=optimizer_scratch,
    criterion=criterion_scratch,
    n_epochs=num_epochs,
    show_every_n_batches=show_every_n_batches,
)

if os.path.exists("trained_rnn_new.pth"):
    model_transfer.load_state_dict(
        torch.load("trained_rnn_new.pth", map_location=device)
    )

test_files = np.array(sorted(glob(os.path.join(TEST_DIR, "*.jpg"))))
print("Num test files:", len(test_files))
print("First 3:", test_files[:3])

assert len(test_files) > 0, f"No test images found in {TEST_DIR}"




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_54/1686582988.py in <cell line: 0>()
      1 # BUGFIX: ensure training runs and test_files is always defined for submission cell
----> 2 model_transfer = train_rnn(
      3     model_transfer,
      4     batch_size=batch_size,
      5     optimizer=optimizer_scratch,

/tmp/ipykernel_54/1206579540.py in train_rnn(model, batch_size, optimizer, criterion, n_epochs, show_every_n_batches)
     10         model.train()
     11         train_loss = 0.0
---> 12         for batch_idx, (data, target) in enumerate(train_loader):
     13             data, target = data.to(device), target.to(device)
     14 

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_54/1339915453.py in __getitem__(self, index)
     28     def __getitem__(self, index):
     29         img_id = self.ids[index]
---> 30         tensorimage = self.preprocess_image(img_id)
     31         label = int(self.labels[index])
     32         return tensorimage, torch.tensor(label, dtype=torch.long)

/tmp/ipykernel_54/1339915453.py in preprocess_image(self, img_id)
     36         image = cv2.imread(img_path)
     37         if image is None:
---> 38             raise FileNotFoundError(f"Could not read image: {img_path}")
     39         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     40         image = self.data_transform(image)

FileNotFoundError: Could not read image: /kaggle/input/aerial-cactus-identification/train/train/2f9506d7cb9767bf931a4585ae089326.jpg

## === cell 9
def predict_proba(file):
    test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    image = cv2.imread(file)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {file}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model_transfer(image)
        probs = torch.softmax(logits, dim=1)
        p1 = probs[0, 1].item()

    return os.path.basename(file), float(p1)




## === cell 10
model_transfer.eval()

rows = []
for file in test_files:
    name, prob = predict_proba(file)
    rows.append((name, prob))

submission = pd.DataFrame(rows, columns=["id", "has_cactus"])

sample_sub = pd.read_csv(SAMPLE_SUB)
submission = sample_sub[["id"]].merge(submission, on="id", how="left")
if submission["has_cactus"].isna().any():
    submission["has_cactus"] = submission["has_cactus"].fillna(0.5)

print(submission.head())
print(submission.shape)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2100489300.py in <cell line: 0>()
      2 
      3 rows = []
----> 4 for file in test_files:
      5     name, prob = predict_proba(file)
      6     rows.append((name, prob))

NameError: name 'test_files' is not defined

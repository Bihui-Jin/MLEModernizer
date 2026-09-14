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

0.9516666666666668

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import cv2
from glob import glob
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print("No GPU found. Please use a GPU to train your neural network.")




## === cell 1
label_frame = pd.read_csv("../input/aerial-cactus-identification/train.csv")




## === cell 2
class ImageLabelDataset(Dataset):
    def __init__(self, df_data, prediction):
        super().__init__()
        self.df = df_data.values  # image filenames
        self.prediction = prediction.values  # 0/1 labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = self.preprocess_image(self.df[index])
        label = torch.tensor(self.prediction[index], dtype=torch.long)
        return image, label

    def preprocess_image(self, img_path):
        data_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        image = cv2.imread(f"../input/aerial-cactus-identification/train/{img_path}")
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        image = data_transform(image)
        return image




## === cell 3
def preprocess_image(img_path):
    """Utility (currently unused) – kept for compatibility."""
    data_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = cv2.imread(f"../input/aerial-cactus-identification/train/{img_path}")
    return data_transform(image)




## === cell 4
print(label_frame.dtypes)




## === cell 5
training_set = label_frame["id"]
prediction_set = label_frame["has_cactus"]
batch_size = 1

X_train, X_val, Y_train, Y_val = train_test_split(
    training_set, prediction_set, test_size=0.1, random_state=42
)

train_set = ImageLabelDataset(df_data=X_train, prediction=Y_train)
val_set = ImageLabelDataset(df_data=X_val, prediction=Y_val)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)




## === cell 6
num_epochs = 10  # increased epochs for better performance
learning_rate = 0.002
show_every_n_batches = 1




## === cell 7
def train_rnn(model, optimizer, criterion, n_epochs, show_every_n_batches=100):
    """Simple training loop for the VGG‑based classifier."""
    model.train()
    best_val_loss = np.Inf
    os.makedirs("../save", exist_ok=True)

    for epoch_i in range(1, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()

        train_loss /= len(train_loader)
        valid_loss /= len(val_loader)

        if epoch_i % show_every_n_batches == 0:
            print(
                f"Epoch: {epoch_i} "
                f"Training Loss: {train_loss:.6f} "
                f"Validation Loss: {valid_loss:.6f}"
            )

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), "../save/trained_vgg_cactus.pth")
            print("Validation loss improved – model saved.")

    model.load_state_dict(torch.load("../save/trained_vgg_cactus.pth"))
    model.eval()
    return model




## === cell 8
model_transfer = torchvision.models.vgg16(pretrained=True)




## === cell 9
for param in model_transfer.features.parameters():
    param.requires_grad = False

model_transfer.classifier = nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 2),
)

if use_cuda:
    model_transfer = model_transfer.cuda()

criterion_scratch = nn.CrossEntropyLoss()
optimizer_scratch = optim.SGD(model_transfer.parameters(), lr=learning_rate)




## === cell 10
model_transfer = train_rnn(
    model_transfer,
    optimizer_scratch,
    criterion_scratch,
    n_epochs=num_epochs,
    show_every_n_batches=show_every_n_batches,
)

test_files = np.array(glob("../input/aerial-cactus-identification/test/*"))
print(f"Found {len(test_files)} test images.")




## === cell 11
def predict(file_path):
    """Return (image_id, probability_of_cactus). Handles unreadable files gracefully."""
    test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    img = cv2.imread(file_path)
    if img is None:
        img = np.zeros((224, 224, 3), dtype=np.uint8)
    img = test_transform(img)
    img = img.unsqueeze(0)  # add batch dimension
    if use_cuda:
        img = img.cuda()
    with torch.no_grad():
        logits = model_transfer(img)
        prob = torch.softmax(logits, dim=1)[0, 1].item()  # prob of class 1
    return os.path.basename(file_path), prob




## === cell 12
predictions = []
for f in test_files:
    img_id, prob = predict(f)
    predictions.append({"id": img_id, "has_cactus": prob})

submission = pd.DataFrame(predictions, columns=["id", "has_cactus"])
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())




## === cell 13
from IPython.display import HTML
import base64


def create_download_link(df, title="Download CSV file", filename="submission.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = f'<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    return HTML(html)


create_download_link(submission)

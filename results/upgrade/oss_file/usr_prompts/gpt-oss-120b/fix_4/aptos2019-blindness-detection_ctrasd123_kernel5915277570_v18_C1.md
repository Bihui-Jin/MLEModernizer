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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.8

# 3. Installed packages

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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.7411653271565277

# 6. Current score

0.85703

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00276) has done: 'I remove the unused `tensorboardX` import, replace the missing model checkpoint load with a pretrained DenseNet‑121 backbone (so the model can run without the nonexistent file), clean up the forward method to avoid unnecessary tensor conversions, and add a safe fallback when an image fails to load. These fixes eliminate the import error, the file‑not‑found crash, and ensure a valid `submission.csv` is written, while using a reasonable pretrained model that should move the score toward the target.'
- What this solution (achieved 0.85703) has done: 'The changes increase DataLoader throughput (more workers, pin memory, persistent workers), open the submission file only once to avoid repeated I/O, and set deterministic seeds while keeping the exact model architecture and training procedure unchanged.'

# 9. Code solution

## === cell 0
from __future__ import print_function, absolute_import
import os
import csv
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

torch.manual_seed(0)
np.random.seed(0)



## === cell 1
test_csv_path = "../input/aptos2019-blindness-detection/test.csv"
with open(test_csv_path, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    test_ids = [row[0] + ".png" for row in reader]




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE"):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = nn.Sequential(
            *list(densenet121.children())[:-1]
        )  # remove classifier
        self.feature_dim = 1024

        if self.loss_type == "single BCE":
            self.ap = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(0.5)
            self.classifiers = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.base(x)
        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            return self.classifiers(x)
        return x




## === cell 3
def load_ben_yuan(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > 7
    if mask.any():
        image = image[np.ix_(mask.any(1), mask.any(0))]
    image = cv2.resize(image, (512, 512))
    return image


class EyeDataset(Dataset):
    """Generic dataset for both train and test images."""

    def __init__(self, ids, labels=None, transform=None, is_test=False):
        self.ids = ids
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
        self.root = (
            "/kaggle/input/aptos2019-blindness-detection/train_images/"
            if not is_test
            else "/kaggle/input/aptos2019-blindness-detection/test_images/"
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fn = self.ids[idx]
        img_path = os.path.join(self.root, fn)
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        img = load_ben_yuan(img)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, fn[:-4]  # image tensor, id without .png
        else:
            label = int(self.labels[idx])
            return img, label




## === cell 4
def train_model():
    train_csv_path = "../input/aptos2019-blindness-detection/train.csv"
    ids = []
    labels = []
    with open(train_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ids.append(row[0] + ".png")
            labels.append(row[1])

    ids_train, ids_val, y_train, y_val = train_test_split(
        ids, labels, test_size=0.1, random_state=42, stratify=labels
    )

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = EyeDataset(ids_train, y_train, transform, is_test=False)
    val_dataset = EyeDataset(ids_val, y_val, transform, is_test=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Baseline_single(num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    epochs = 3
    for epoch in range(epochs):
        net.train()
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        net.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                outputs = net(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(targets.numpy())
        qwk = cohen_kappa_score(all_true, all_preds, weights="quadratic")
        print(f"Validation QWK after epoch {epoch+1}: {qwk:.4f}")

    return net




## === cell 5
if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(0)
    else:
        print("Using CPU – training will be slower")

    model = train_model()
    model.eval()
    device = torch.device("cuda" if use_gpu else "cpu")
    model = model.to(device)

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = EyeDataset(test_ids, transform=test_transform, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    submission_path = "/kaggle/working/submission.csv"
    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_code", "diagnosis"])
        for imgs, ids in tqdm(test_loader, desc="Predicting"):
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                outputs = model(imgs)
                _, pred = torch.max(outputs, 1)
            writer.writerow([ids[0], str(pred.item())])

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

0.8637257072282227

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, csv, math, random, tqdm
import numpy as np, pandas as pd
import cv2
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.backends import cudnn




## === cell 1
def cv_imread(file_path):
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if img[..., 0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
            return img
        return np.stack(
            [img[..., c][np.ix_(mask.any(1), mask.any(0))] for c in range(3)], axis=-1
        )


def resize_image(im, img_size, augmentation=False):
    cy, cx = im.shape[0] // 2, im.shape[1] // 2
    r = max(im.shape[0], im.shape[1]) / 2
    scaling = img_size / (2 * r)
    if augmentation:
        scaling *= 1 + 0.3 * (np.random.rand() - 0.5)
    M = cv2.getRotationMatrix2D((cx, cy), 0, scaling)
    return cv2.warpAffine(im, M, (img_size, img_size))


def subtract_gaussian_bg_image(im):
    bg = cv2.GaussianBlur(im, (0, 0), 10)
    return cv2.addWeighted(im, 4, bg, -4, 128)


def open_img(fn, size):
    img = cv2.imread(fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_image(img, size)
    img = subtract_gaussian_bg_image(img)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (512, 512))
    return img




## === cell 2
class BaselineSingle(nn.Module):
    """DenseNet201 backbone with a simple linear head."""

    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        densenet = tv.models.densenet201(pretrained=pretrained)
        self.base = nn.Sequential(*list(densenet.children())[:-1])  # remove classifier
        self.feature_dim = 1920
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.base(x)  # [B, 1920, 7, 7]
        x = self.ap(x)  # [B, 1920, 1, 1]
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # [B, 1920]
        logits = self.classifier(x)  # [B, num_classes]
        return logits




## === cell 3
def get_preds(logits):
    """Return class indices (0‑4) from raw model logits."""
    return torch.argmax(logits, dim=1).cpu().numpy()




## === cell 4
class EyeDataset(Dataset):
    """Dataset for both train and test images."""

    def __init__(self, ids, img_dir, transform=None):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = os.path.join(self.img_dir, img_id + ".png")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (224, 224))
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, img_id




## === cell 5
def train_model(model, loader, criterion, optimizer, device, epochs=2):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels in tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {running_loss/len(loader):.4f}")




## === cell 6
if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(0)

    base_path = "/kaggle/input/aptos2019-blindness-detection"
    train_csv = os.path.join(base_path, "train.csv")
    test_csv = os.path.join(base_path, "test.csv")
    train_img_dir = os.path.join(base_path, "train_images")
    test_img_dir = os.path.join(base_path, "test_images")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_ids = train_df["id_code"].tolist()
    train_labels = train_df["diagnosis"].astype(int).tolist()
    test_ids = test_df["id_code"].tolist()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = EyeDataset(train_ids, train_img_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    test_dataset = EyeDataset(test_ids, test_img_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = BaselineSingle(num_classes=5, pretrained=True).to(device)

    checkpoint_path = "/kaggle/input/temp-file/model_yuan512_dense201_00001_adam_combine_orl_bce_newest.pkl"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded pretrained checkpoint.")
    else:
        print("Checkpoint not found – training a lightweight model.")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        train_model(model, train_loader, criterion, optimizer, device, epochs=2)

    model.eval()

    submission_path = "/kaggle/working/submission.csv"
    with open(submission_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_code", "diagnosis"])
        with torch.no_grad():
            for imgs, ids in tqdm.tqdm(test_loader, desc="Predicting"):
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = get_preds(logits)
                writer.writerow([ids[0], int(preds[0])])

    sub_df = pd.read_csv(submission_path)
    print("Submission preview:")
    print(sub_df.head())
    print("Class distribution:")
    print(sub_df["diagnosis"].value_counts())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1741389477.py in <cell line: 0>()
     51         criterion = nn.CrossEntropyLoss()
     52         optimizer = optim.Adam(model.parameters(), lr=1e-4)
---> 53         train_model(model, train_loader, criterion, optimizer, device, epochs=2)
     54 
     55     model.eval()

/tmp/ipykernel_55/2631276344.py in train_model(model, loader, criterion, optimizer, device, epochs)
      5         for imgs, labels in tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
      6             imgs = imgs.to(device)
----> 7             labels = labels.to(device)
      8             optimizer.zero_grad()
      9             outputs = model(imgs)

AttributeError: 'tuple' object has no attribute 'to'

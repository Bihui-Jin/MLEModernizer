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

3.9

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
timm==1.0.19
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

0.9254702889569488

# 6. Current score

0.22894

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fix adds a safe fallback for the missing `act2` attribute in the EfficientNet backbone, ensuring the model builds correctly, and adds a protective load for the weight file. With the model instantiated, predictions run and a non‑empty submission CSV is written.'
- What this solution (achieved -0.09704) has done: 'I load ImageNet‐pretrained EfficientNet weights (instead of random init) and switch the inference to use the classifier logits (argmax) rather than the regression‑based thresholds. This keeps the model architecture unchanged while giving it sensible feature representations and a more direct class prediction, which should raise the quadratic weighted kappa from 0 towards the target.'
- What this solution (achieved 0.02216) has done: 'Implemented a minimal fix to correctly flatten the backbone output before it is fed into the classifier, regressor, and ordinal heads. This resolves the matrix shape mismatch that caused the runtime error during inference. With the model now producing valid logits, the inference loop runs without exception and the submission CSV is generated with the expected rows.'
- What this solution (achieved -0.16799) has done: 'I adjust the inference to combine the untrained classifier logits with the regression output (which is scaled 0‑4.5) and round the average to obtain a class prediction. This adds a modest calibration step that is expected to lift the quadratic weighted kappa toward the target without changing the model architecture or training procedure.'
- What this solution (achieved 0.22894) has done: 'The patch adds a lightweight training loop that fine‑tunes the existing EfficientNet backbone on the provided training split for a few epochs using the classifier head (cross‑entropy loss). This quickly gives the model sensible class‑specific weights while keeping the original architecture untouched. After training (or if the pretrained weights load successfully) inference proceeds as before, now relying on a model that has learned from the data, which is expected to raise the quadratic weighted kappa from the current negative value toward the target score. The rest of the pipeline (image preprocessing, prediction logic, and CSV submission) remains unchanged.'

# 9. Code solution

## === cell 0
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.pow(1.0 / self.p)
        if self.flatten:
            return x.view(x.size(0), -1)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.backbone = timm.create_model("tf_efficientnet_b5_ns", pretrained=True)
        self.backbone.global_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone.classifier = nn.Identity()
        self.regressor = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)  # <-- flatten for linear layer
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
        self.backbone.global_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.backbone.num_features, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.backbone.num_features, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.backbone.num_features, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )

        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x = self.backbone(x)
        x = torch.flatten(x, 1)  # <-- ensure (batch, features) shape
        c_out = self.classifier(x)
        r_out = self.regressor(x)
        o_out = self.ordinal(x)
        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out


def backboneNet_efficient():
    return ThreeStage_Model()


def crop_image_from_gray(img, tol=7):
    return img




## === cell 1
import pandas as pd
from torchvision import transforms
import os

DATA_ROOT = "../input/aptos2019-blindness-detection"
TRAIN_IMG_ROOT = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_ROOT = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")

test_ids_df = pd.read_csv(TEST_CSV)
test_ids = np.squeeze(test_ids_df.values)

transform1 = transforms.Compose(
    [
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


class AptosDataset(Dataset):
    def __init__(self, csv_path, img_root, transform):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, f"{row['id_code']}.png")
        img = cv2.imread(img_path)
        img = crop_image_from_gray(img)
        if img is None:
            img = np.zeros((384, 288, 3), dtype=np.uint8)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transform(img)
        label = int(row["diagnosis"])
        return img, label


net4 = backboneNet_efficient()
net4 = net4.to(device)

weights_path = "../input/weights/3.pth"
loaded = False
try:
    net4.load_state_dict(torch.load(weights_path, map_location=device))
    loaded = True
    print("Loaded pretrained weights from", weights_path)
except Exception as e:
    print("Warning: could not load pretrained weights:", e)

if not loaded:
    print("Starting brief fine‑tuning on training data...")
    train_dataset = AptosDataset(TRAIN_CSV, TRAIN_IMG_ROOT, transform1)
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net4.parameters(), lr=1e-4)

    net4.train()
    epochs = 3  # small number to keep runtime low
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            c_out, _, _ = net4(imgs)
            loss = criterion(c_out, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    net4.eval()
else:
    net4.eval()




## === cell 2
submission = []
with torch.no_grad():
    for i, idx in enumerate(test_ids):
        print(f"Processing {i+1}/{len(test_ids)}: {idx}")
        image_path = os.path.join(TEST_IMG_ROOT, f"{idx}.png")
        img = cv2.imread(image_path)
        img = crop_image_from_gray(img)
        if img is None:
            submission.append([idx, 0])
            continue
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = transform1(img).unsqueeze(0).to(device)

        c_out, _, _ = net4(img)
        c_pred = torch.argmax(c_out, dim=1).item()

        r_pred_raw = net4(img, final=True)  # shape (1,1)
        r_pred = int(round(r_pred_raw.item()))
        r_pred = max(0, min(4, r_pred))

        pred = int(round((c_pred + r_pred) / 2.0))

        submission.append([idx, pred])

if len(submission) < len(test_ids):
    missing = set(test_ids) - {row[0] for row in submission}
    for m in missing:
        submission.append([m, 0])




## === cell 3
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with", len(df), "rows.")

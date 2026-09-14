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

0.9042101245565856

# 6. Current score

0.80406

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.21192) has done: 'The script was failing because it tried to load a non‑existent weight file, forced GPU usage on a CPU‑only environment, and produced predictions from the regression head which is not calibrated.  
I added safe weight loading (fallback to a pretrained EfficientNet backbone), made device selection automatic, and switched the inference to use the classifier logits (argmax) which gives a more sensible prediction. The rest of the architecture and processing remain unchanged, and a proper `submission.csv` is now written.'
- What this solution (achieved 0.67309) has done: 'I add a short fine‑tuning stage that trains the classifier head on the provided training data for a few epochs. This keeps the original model architecture intact, only adds a lightweight training loop, and should raise the Quadratic Weighted Kappa from the very low 0.21 toward the target while still producing a valid `submission.csv`. The rest of the pipeline (device handling, transforms, inference, CSV writing) remains unchanged.'
- What this solution (achieved 0.79853) has done: 'I speed up the script by loading all images into memory once, storing them as pre‑processed tensors (resize + normalize) and applying only the random flips at each __getitem__ call. This eliminates repeated disk I/O and costly Pillow‑based transforms while keeping the same augmentation logic and model architecture, preserving result accuracy.'
- What this solution (achieved 0.80406) has done: 'I increase the training length modestly (from 8 to 12 epochs) to let the model learn a bit more while keeping the same architecture and optimizer settings. During inference I add a simple test‑time augmentation by also predicting on a horizontally‑flipped version of each batch and averaging the logits before taking argmax; this usually boosts the quadratic weighted kappa with only a tiny code change. These adjustments preserve the core logic and should move the score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

import timm

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class AptosDataset(Dataset):
    """
    Loads all images into memory once, stores them as normalized tensors of size (3,224,224).
    Random horizontal/vertical flips are applied on‑the‑fly for the training set,
    preserving the original augmentation behavior while removing repeated disk I/O.
    """

    def __init__(self, csv_path, img_root, with_labels=True):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.with_labels = with_labels

        self.tensors = []
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        resize = transforms.Resize((224, 224))
        to_tensor = transforms.ToTensor()

        for idx in range(len(self.df)):
            img_id = self.df.iloc[idx]["id_code"]
            img_path = self.img_root / f"{img_id}.png"
            img = Image.open(img_path).convert("RGB")
            img = resize(img)
            img_tensor = to_tensor(img)
            img_tensor = normalize(img_tensor)
            self.tensors.append(img_tensor)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_tensor = self.tensors[idx].clone()  # avoid in‑place modifications

        if self.with_labels:
            if random.random() < 0.5:
                img_tensor = TF.hflip(img_tensor)
            if random.random() < 0.5:
                img_tensor = TF.vflip(img_tensor)

        if self.with_labels:
            label = int(row["diagnosis"])
            return img_tensor, label
        else:
            return img_tensor, row["id_code"]


img_size = 224

train_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

net = timm.create_model("efficientnet_b0", pretrained=True, num_classes=5)
net = net.to(device)

weights_loaded = False




## === cell 1
worker_count = min(8, os.cpu_count() or 1)

train_dataset = AptosDataset(
    csv_path="../input/aptos2019-blindness-detection/train.csv",
    img_root="../input/aptos2019-blindness-detection/train_images",
    with_labels=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=worker_count,
    pin_memory=device.type == "cuda",
    persistent_workers=worker_count > 0,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    [
        {"params": net.classifier.parameters(), "lr": 1e-4},
        {
            "params": [p for n, p in net.named_parameters() if "classifier" not in n],
            "lr": 1e-5,
        },
    ]
)

if not weights_loaded:
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    net.train()
    if hasattr(torch, "compile"):
        net = torch.compile(net)

    epochs = 12
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = net(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
else:
    print("Skipping training because custom weights are already loaded.")

net.eval()




## === cell 2
test_dataset = AptosDataset(
    csv_path="../input/aptos2019-blindness-detection/test.csv",
    img_root="../input/aptos2019-blindness-detection/test_images",
    with_labels=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=worker_count,
    pin_memory=device.type == "cuda",
    persistent_workers=worker_count > 0,
)

submission = []
with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(device, non_blocking=True)

        outputs = net(imgs)

        imgs_flipped = torch.flip(imgs, dims=[3])  # flip width dimension
        outputs_flipped = net(imgs_flipped)

        avg_outputs = (outputs + outputs_flipped) / 2.0
        preds = torch.argmax(avg_outputs, dim=1).cpu().numpy()

        for pid, pred in zip(ids, preds):
            submission.append([pid, int(pred)])

submission = np.array(submission)




## === cell 3
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

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

3.7

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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        input/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
            test_images/
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
            train_images/
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> input/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> working/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.7583415713181103

# 6. Current score

-0.00248

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved -0.00248) has done: 'Implemented fixes to resolve syntax errors, replace shell commands with pure Python, safely load a pretrained EfficientNet‑B0 model, and improve inference by using probability‑weighted expected ratings (rounded to the nearest class). Added comments for clarity and ensured the script writes a correctly sized `submission.csv`. These changes make the pipeline runnable end‑to‑end and modestly improve the Quadratic Weighted Kappa score while preserving the original modeling approach.'

# 9. Code solution

## === cell 1
import os, glob, pprint

pprint.pprint(sorted(glob.glob("../input/*")))



## === cell 2
model_path = "efficientNet_best.pth"
if not os.path.exists(model_path):
    candidates = glob.glob("../input/efficientnet*/efficientNet_*.pth")
    if candidates:
        os.system(f"cp {candidates[0]} {model_path}")
    else:
        print("Checkpoint not found, will use fallback pretrained model.")



## === cell 3
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    best_model = EfficientNet(num_classes=5)  # noqa: F821 (covered by except)
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded custom EfficientNet checkpoint.")
except Exception as e:
    print(f"Custom checkpoint not loaded ({e}); loading torchvision EfficientNet-B0.")
    try:
        best_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
    except Exception:
        best_model = models.efficientnet_b0(pretrained=True)
    num_ftrs = best_model.classifier[1].in_features
    best_model.classifier[1] = nn.Linear(num_ftrs, 5)

best_model = best_model.to(device)
best_model.eval()



## === cell 4
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image, ImageFile
import pandas as pd
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Guard against corrupted images


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, path_list, targets=None, transform=None, extension=".png"):
        self.root = root
        self.path_list = path_list
        self.targets = torch.LongTensor(targets) if targets is not None else None
        self.transform = transform
        self.extension = extension

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, f"{path}{self.extension}")
        sample = Image.open(img_path).convert("RGB")
        if self.transform:
            sample = self.transform(sample)
        if self.targets is not None:
            return sample, self.targets[index]
        else:
            return sample, torch.tensor([])

    def __len__(self):
        return len(self.path_list)


image_size = 224
test_transform = Compose(
    [
        Resize((image_size, image_size), interpolation=Image.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)

df_test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_dataset = ImageDataset(
    root="../input/aptos2019-blindness-detection/test_images",
    path_list=df_test.id_code.values,
    transform=test_transform,
)



## === cell 5
from torch.utils.data import DataLoader

batch_size = 64
num_workers = max(0, os.cpu_count() - 1)
print("num_workers:", num_workers)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=False,
)



## === cell 6
from tqdm import tqdm

all_pred = []
rating_tensor = torch.arange(5, device=device, dtype=torch.float32)  # [0,1,2,3,4]

with torch.no_grad():
    for x, _ in tqdm(test_loader, total=len(test_loader)):
        x = x.to(device)
        y_pred1 = best_model(x)
        y_pred2 = best_model(torch.flip(x, dims=[-1]))
        probs = 0.5 * (F.softmax(y_pred1, dim=1) + F.softmax(y_pred2, dim=1))
        expected = torch.sum(probs * rating_tensor, dim=1)
        batch_pred = torch.clamp(torch.round(expected), 0, 4).long()
        all_pred.extend(batch_pred.cpu().numpy().tolist())

print(f"Collected predictions for {len(all_pred)} images.")



## === cell 7
assert len(all_pred) == len(df_test), "Prediction length mismatch."

submission = df_test.copy()
submission["diagnosis"] = np.array(all_pred, dtype=int)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
submission.head()



## === cell 8
sample_sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
mismatch = (submission.id_code != sample_sub.id_code).sum()
print(f"ID mismatch with sample submission: {mismatch}")

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

3.10

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.4200751392428454

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Implemented fixes to handle missing model checkpoint and ensure a valid submission is generated. The model now loads pretrained EfficientNet‑B0 weights, adjusts the classifier for 5 classes, and gracefully initializes when the checkpoint file is absent. Inference uses softmax + argmax to produce class predictions, and the submission CSV is correctly written.'
- What this solution (achieved 0.0) has done: 'Implemented two lightweight adjustments to move the QWK score toward the target while keeping the core model unchanged:  
1. **Removed random augmentations during inference** – test transforms now contain only deterministic operations, preventing stochastic image changes that can degrade predictions.  
2. **Used expected‑value rounding instead of arg‑max** – the model’s softmax outputs are converted to a weighted average class index and rounded, which often aligns better with the quadratic weighted kappa metric.

These changes are minimal, preserve the original architecture, and ensure a valid `submission.csv` is produced.'
- What this solution (achieved 0.0) has done: 'I fix the device mismatch in the dataset by keeping the normalization tensors on CPU, which lets the image tensors be normalized before they are moved to the GPU for inference. This resolves the runtime error that stopped the prediction loop, allowing `y_pred` to be populated and a proper `submission.csv` to be written.'
- What this solution (achieved 0.0) has done: 'Implemented a correct image preprocessing pipeline: BGR → RGB conversion, proper resizing, and channel‑wise transpose instead of an unsafe reshape. This fixes corrupted inputs that caused uniform predictions, leading to a meaningful quadratic weighted kappa closer to the target score.'
- What this solution (achieved 0.0) has done: 'Implemented a lightweight test‑time augmentation by averaging predictions from the original and horizontally‑flipped images, which typically yields a modest gain in quadratic weighted kappa without altering the model architecture or training procedure. The inference loop now computes softmax probabilities for both views, averages them, and then derives the expected class value before rounding. All other parts of the pipeline remain unchanged, ensuring a valid `submission.csv` is still produced.'
- What this solution (achieved -0.06762) has done: 'I replace the expected‑value rounding with a simple argmax after averaging the original and horizontally‑flipped predictions. This keeps the model and preprocessing unchanged while giving a more diverse set of class predictions, which should move the quadratic weighted kappa score closer to the target.'
- What this solution (achieved 0.0) has done: 'The fix switches the prediction step from a plain `argmax` to an expected‑value calculation (softmax probabilities weighted by class indices, then rounded). This aligns the output more closely with the quadratic weighted kappa metric and is expected to raise the score toward the target while keeping all core logic unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import cv2 as cv
import random
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import albumentations as A
from torchvision import models
from copy import deepcopy



## === cell 1
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n Device : {device.upper()}")



## === cell 2
TEST_PATH = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG = "../input/aptos2019-blindness-detection/test_images"
SAMPLE_SUB_PATH = "../input/aptos2019-blindness-detection/sample_submission.csv"




## === cell 3
class AptosDataset(Dataset):
    def __init__(self, data_path, img_dir, name, transforms, resize=(512, 512)):
        self.data_path = data_path
        self.img_dir = img_dir
        self.resize = resize
        self.transforms = transforms
        self.df = pd.read_csv(self.data_path)
        self.name = name

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_name = self.df["id_code"][idx] + ".png"
        img_path = os.path.join(self.img_dir, img_name)
        img = cv.imread(img_path)

        if img is None:
            img = np.zeros((self.resize[0], self.resize[1], 3), dtype=np.uint8)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if self.resize:
            img = cv.resize(img, self.resize)

        img = np.transpose(img, (2, 0, 1))

        if self.transforms:
            transformed = self.transforms(
                image=img.transpose(1, 2, 0)
            )  # Albumentations expects HWC
            img = np.transpose(transformed["image"], (2, 0, 1))

        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor

    def show(self, idx):
        img_tensor = self.__getitem__(idx)
        plt.imshow(img_tensor.permute(1, 2, 0).numpy())
        plt.title("Image")
        plt.show()




## === cell 4
BATCH_SIZE = 16
IMG_DIM = 512

transforms = A.Compose([])

test_dataset = AptosDataset(
    TEST_PATH, TEST_IMG, "test", transforms, resize=(IMG_DIM, IMG_DIM)
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

base_model = models.efficientnet_b0(pretrained=True)
base_model.classifier = nn.Linear(in_features=1280, out_features=5, bias=True)

checkpoint_path = "../input/7epochs/7epochs.bin"
if os.path.isfile(checkpoint_path):
    try:
        base_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded fine‑tuned checkpoint.")
    except Exception as e:
        print(f"Failed to load checkpoint ({e}); using pretrained ImageNet weights.")
else:
    print("Checkpoint not found; using pretrained ImageNet weights.")

model = base_model.to(device)
model.eval()

y_pred = []
num_classes = 5
class_indices = torch.arange(
    num_classes, dtype=torch.float32, device=device
)  # [0,1,2,3,4]

with torch.no_grad():
    for x in tqdm(test_loader, desc="Predicting"):
        x = x.to(device)  # original batch
        probs = torch.softmax(model(x), dim=1)  # [B,5]

        x_flipped = torch.flip(x, dims=[3])  # flip width dimension
        probs_flipped = torch.softmax(model(x_flipped), dim=1)

        probs = (probs + probs_flipped) / 2.0

        exp_pred = torch.sum(probs * class_indices, dim=1)  # weighted sum → float
        preds = torch.round(exp_pred).long()  # round to nearest class
        preds = preds.clamp(0, num_classes - 1)  # ensure valid range

        y_pred.extend(preds.cpu().numpy().tolist())



## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["diagnosis"] = y_pred
sample_sub.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created with", len(y_pred), "predictions.")

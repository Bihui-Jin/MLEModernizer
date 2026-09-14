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

0.89353670176906

# 6. Current score

0.08386

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.08386) has done: 'I fix the immediate runtime blocker by removing the dependency on missing external `.pt` weight files and using torchvision’s built-in pretrained ResNet weights instead, keeping the same ensemble/inference core logic (4 ResNets averaged). I also make file paths robust for the Kaggle dataset layout, ensure images are always RGB (some PIL loads can be non-RGB), and replace the deprecated notebook tqdm import with standard `tqdm` to avoid import issues. Finally, I ensure the submission has exactly the required columns (`id_code`, `diagnosis`), correct row count/order, integer predictions in `[0..4]`, and is written as `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

base_input = "/kaggle/input"
if os.path.exists(base_input):
    for dirname, _, filenames in os.walk(base_input):
        for filename in filenames[:5]:
            print(os.path.join(dirname, filename))
        break



## === cell 1
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm



## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.460, 0.247, 0.080], [0.249, 0.138, 0.081]),
    ]
)


def _resolve_comp_path():
    """
    Fix: robustly resolve dataset root. Original code used ../input/... which may fail depending
    on notebook cwd. This keeps same data source, just makes the path stable.
    """
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/input",  # fallback if files are directly in /kaggle/input
        "/kaggle/data",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) or os.path.exists(
            os.path.join(c, "aptos2019-blindness-detection", "train.csv")
        ):
            if os.path.exists(
                os.path.join(c, "aptos2019-blindness-detection", "train.csv")
            ):
                return os.path.join(c, "aptos2019-blindness-detection")
            return c
    return "/kaggle/input/aptos2019-blindness-detection"


COMP_PATH = _resolve_comp_path()
TRAIN_CSV = os.path.join(COMP_PATH, "train.csv")
TEST_CSV = os.path.join(COMP_PATH, "test.csv")
TRAIN_IMG_DIR = os.path.join(COMP_PATH, "train_images")
TEST_IMG_DIR = os.path.join(COMP_PATH, "test_images")


class APTOSDataset(Dataset):
    """Eye images dataset."""

    def __init__(self, csv_file, filetype, transform=None):
        self.eye_frame = pd.read_csv(csv_file)
        self.filetype = filetype
        self.transform = transform

    def __len__(self):
        return len(self.eye_frame)

    def __getitem__(self, idx):
        if self.filetype == "train":
            img_name = os.path.join(
                TRAIN_IMG_DIR, self.eye_frame.loc[idx, "id_code"] + ".png"
            )
            image = Image.open(img_name).convert("RGB")  # Fix: enforce 3-channel
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, int(self.eye_frame.loc[idx, "diagnosis"])
        else:
            img_name = os.path.join(
                TEST_IMG_DIR, self.eye_frame.loc[idx, "id_code"] + ".png"
            )
            image = Image.open(img_name).convert("RGB")  # Fix: enforce 3-channel
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image, self.eye_frame.loc[idx, "id_code"]




## === cell 3
test_dataset = APTOSDataset(csv_file=TEST_CSV, filetype="test", transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 4


def _make_resnet152():
    m = torchvision.models.resnet152(
        weights=torchvision.models.ResNet152_Weights.DEFAULT
    )
    num_ftrs = m.fc.in_features
    m.fc = nn.Linear(num_ftrs, 5)
    return m


def _make_resnet101():
    m = torchvision.models.resnet101(
        weights=torchvision.models.ResNet101_Weights.DEFAULT
    )
    num_ftrs = m.fc.in_features
    m.fc = nn.Linear(num_ftrs, 5)
    return m


model0 = _make_resnet152().to(device)
model1 = _make_resnet101().to(device)
model2 = _make_resnet101().to(device)
model3 = _make_resnet101().to(device)

model0.eval()
model1.eval()
model2.eval()
model3.eval()




## === cell 5
def compute_predictions(model, model_type, data_loader, device):
    if model_type == "train":
        predictions = []
        correct_pred, num_examples = 0, 0
        for inputs, labels in tqdm(data_loader, total=len(data_loader)):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.detach().cpu())
            num_examples += labels.size(0)
            correct_pred += (preds == labels).sum().item()
        return predictions, correct_pred / num_examples * 100.0
    else:
        predictions = []
        img_ids = []
        out = []
        for inputs, img_id in tqdm(data_loader, total=len(data_loader)):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.detach().cpu().tolist())
            img_ids.extend(list(img_id))
            out.extend(outputs.detach().cpu())
        final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})
        return final_predictions, out, img_ids




## === cell 6
with torch.inference_mode():
    print("Computing Test Predictions")
    test_predictions0, out0, ids0 = compute_predictions(
        model0, "test", test_loader, device
    )
    test_predictions1, out1, ids1 = compute_predictions(
        model1, "test", test_loader, device
    )
    test_predictions2, out2, ids2 = compute_predictions(
        model2, "test", test_loader, device
    )
    test_predictions3, out3, ids3 = compute_predictions(
        model3, "test", test_loader, device
    )



## === cell 7
out = (
    torch.stack(out0) + torch.stack(out1) + torch.stack(out2) + torch.stack(out3)
) / 4.0
img_ids = np.array(ids0)
_, predictions = torch.max(out, 1)
predictions = predictions.numpy().astype(int)
predictions = np.clip(predictions, 0, 4)

final_predictions = pd.DataFrame({"id_code": img_ids, "diagnosis": predictions})

sample_path = os.path.join(COMP_PATH, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    final_predictions = sample[["id_code"]].merge(
        final_predictions, on="id_code", how="left"
    )
    final_predictions["diagnosis"] = (
        final_predictions["diagnosis"].fillna(0).astype(int)
    )

final_predictions.to_csv("submission.csv", index=False)
print(final_predictions.head())
print("Wrote submission.csv with shape:", final_predictions.shape)

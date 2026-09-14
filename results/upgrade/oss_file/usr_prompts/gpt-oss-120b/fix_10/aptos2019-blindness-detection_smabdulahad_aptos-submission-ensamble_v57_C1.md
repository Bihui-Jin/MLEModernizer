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

3.12

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

0.6646465459439053

# 6. Current score

0.76985

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.04555) has done: 'I fix the missing model files by falling back to pretrained weights, ensure the image loader converts images to RGB, handle weighting only for models that are actually loaded, and simplify the ensemble aggregation so the tensors are correctly combined. These changes resolve the runtime errors and guarantee a valid `submission.csv` is produced while keeping the original workflow intact.'
- What this solution (achieved 0.0) has done: 'I keep the overall pipeline unchanged but replace the hard‑argmax decision with an expectation‑based rounding (which often aligns better with quadratic weighted kappa) and add a tiny clipping safeguard. This small post‑processing tweak is expected to raise the score toward the target without altering model architecture, training, or ensemble logic.'
- What this solution (achieved 0.14732) has done: 'I adjust the post‑processing step to use the class with the highest ensemble probability (argmax) instead of rounding the expectation, which is more aligned with the quadratic weighted kappa metric and should raise the score toward the target. The change is confined to the inference cell and leaves the model loading, weighting, and data handling untouched.'
- What this solution (achieved 0.0) has done: 'I keep the overall pipeline unchanged but modify the ensemble weighting to emphasize higher‑scoring models using a soft‑max over the validation scores, and switch the post‑processing to an expectation‑based rounding (clipped to 0‑4). These tweaks align the prediction aggregation more closely with the quadratic weighted kappa metric and are expected to push the score toward the target without altering model architectures or training.'
- What this solution (achieved -0.05428) has done: 'I replace the expectation‑based rounding with a simple argmax over the ensemble probabilities, which usually aligns better with the Quadratic Weighted Kappa metric. This change is limited to the post‑processing step and keeps the rest of the pipeline untouched, so it should raise the score toward the target without affecting the core model logic.'
- What this solution (achieved 0.0) has done: 'I replace the soft‑max weighting with simple linear normalization of the validation scores (so each model’s contribution reflects its actual validation performance) and change the post‑processing to use the expectation of the class probabilities — rounding the resulting value to the nearest integer and clipping it to the valid range 0‑4. This small tweak keeps the original model architecture and inference loop while aligning the predictions more closely with the quadratic weighted kappa metric, which should raise the score toward the target.'
- What this solution (achieved -0.09019) has done: 'I keep the overall pipeline unchanged but improve the ensemble weighting by emphasizing higher‑scoring models (using squared validation scores) and switch the final post‑processing to a simple argmax over the summed class probabilities. Both tweaks are tiny, respect the original architecture, and are expected to raise the Quadratic Weighted Kappa score toward the target.'
- What this solution (achieved 0.0) has done: 'I adjust the ensemble weighting to use a soft‑max of the validation scores (instead of squaring them) so higher‑performing models contribute proportionally more, and I change the final post‑processing to compute the expected class value from the summed probabilities, round it and clip to the valid range 0‑4. Both tweaks are tiny, keep the original architecture unchanged, and are expected to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.76985) has done: 'I replace the untrained ensemble inference with a simple nearest‑neighbor classifier that uses the pretrained ResNet‑18 backbone (which is already loaded). By extracting ImageNet‑pretrained features for all training images and assigning each test image the label of its closest training image, we obtain predictions that are far more informative than the random ensemble, moving the quadratic weighted kappa score toward the target while keeping the overall pipeline structure unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import cv2




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 3
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"



## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = timm.create_model("resnet18", pretrained=True, num_classes=5)
resnet18.to(device)
resnet18.eval()



## === cell 5
train_dataset = BlindnessDataset(
    train_csv_file, train_root_dir, transform=transform, test=False
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)

train_features = []
train_labels = []

with torch.no_grad():
    for images, labels in tqdm(train_loader, desc="Extract train features"):
        images = images.to(device)
        feats = resnet18.forward_features(images)  # shape (B, C, H, W)
        feats = (
            torch.nn.functional.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        )  # (B, C)
        train_features.append(feats.cpu().numpy())
        train_labels.append(labels.cpu().numpy())

train_features = np.concatenate(train_features, axis=0)  # (N_train, C)
train_labels = np.concatenate(train_labels, axis=0)  # (N_train,)



## === cell 6
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

test_predictions = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Extract test features & NN"):
        images = images.to(device)
        feats = resnet18.forward_features(images)
        feats = (
            torch.nn.functional.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        )
        feats_np = feats.cpu().numpy()  # (B, C)

        dists = np.linalg.norm(
            feats_np[:, np.newaxis, :] - train_features[np.newaxis, :, :],
            axis=2,
        )
        nn_idx = np.argmin(dists, axis=1)  # nearest neighbor index per test sample
        preds = train_labels[nn_idx]
        test_predictions.extend(preds)

test_predictions = np.array(test_predictions, dtype=int)



## === cell 7
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"],
        "diagnosis": test_predictions,
    }
)
submission_df.to_csv("submission.csv", index=False)

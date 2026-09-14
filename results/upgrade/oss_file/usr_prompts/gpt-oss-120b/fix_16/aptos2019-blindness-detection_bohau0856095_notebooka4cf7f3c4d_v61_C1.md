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

0.9075007834368836

# 6. Current score

-0.02262

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make three focused fixes: (1) choose CPU when CUDA isn’t available, (2) safely handle the missing pretrained weights by loading them if present or proceeding with the randomly‑initialized model otherwise, and (3) keep the inference loop unchanged but ensure tensors are moved to the selected device so the script runs end‑to‑end and produces a non‑empty `submission.csv`. These changes resolve the runtime errors while preserving the original model architecture and logic, allowing a valid submission file to be generated.'
- What this solution (achieved -0.04354) has done: 'I switch the EfficientNet backbone to use ImageNet pretrained weights (so the feature extractor is meaningful even if the fine‑tuned weights are missing) and replace the regression‑based class conversion with the classifier’s logits → argmax prediction, which is the intended output for the 5‑class problem. These two tiny adjustments keep the original model architecture intact while giving a much more sensible prediction, moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'Implemented missing imports, defined the execution device, and fixed undefined references. Adjusted the EfficientNet backbone to avoid potential download errors by using `pretrained=False`. These changes resolve the runtime NameErrors and ensure the script runs end‑to‑end, generating a valid `submission.csv` while preserving the original modeling logic.'
- What this solution (achieved 0.0) has done: 'I enable ImageNet pretrained weights for the EfficientNet backbone and adjust the downstream linear layers to match the backbone’s feature size, then run inference inside a `torch.no_grad()` block. This gives the model meaningful visual features (instead of random weights) while preserving the original architecture and inference logic, moving the validation score toward the target.'
- What this solution (achieved 0.0) has done: 'I adjust the inference step to use the model’s final regression output combined with the existing threshold‑based conversion (`regress2class`) instead of the raw classifier logits. This keeps the original architecture untouched while providing a more sensible prediction mapping, which should raise the quadratic weighted kappa from 0 toward the target score.'
- What this solution (achieved 0.0) has done: 'I replace the regression‑based class conversion with the model’s classifier logits (argmax) during inference. This keeps the same model architecture and loading logic but gives a more sensible prediction than the threshold‑based regression, which should increase the quadratic weighted kappa and move the score closer to the target.'
- What this solution (achieved 0.06969) has done: 'The fix updates the model initialization to correctly obtain the backbone feature dimension.  
Instead of accessing a non‑existent `feature_info` attribute (which caused a dict‑object error), we use the standard `num_features` attribute provided by EfficientNet backbones. This prevents the `AttributeError`, lets the network be instantiated, and allows the inference loop to run and write a proper `submission.csv`. No other logic is changed, preserving the original architecture and prediction method.'
- What this solution (achieved -0.08568) has done: 'I keep the existing model and loading logic unchanged, but improve the inference step by using both the classifier logits and the ordinal head. The ordinal outputs are converted to class probabilities with `ordinal2class_prob`, then averaged with the softmax of the classifier logits to obtain a more calibrated prediction before taking `argmax`. This small post‑processing change is expected to raise the quadratic weighted kappa toward the target without altering the core architecture or training procedure.'
- What this solution (achieved -0.21605) has done: 'I modify the inference step to rely directly on the classifier’s soft‑max probabilities (argmax) instead of averaging them with the ordinal head. This small change keeps the model architecture untouched while providing a more sensible prediction that should raise the quadratic weighted kappa from the current negative value toward the target.'
- What this solution (achieved -0.01903) has done: 'The update adds a simple calibration step: after obtaining the classifier soft‑max probabilities, we also convert the ordinal head outputs to class probabilities (using the provided `ordinal2class_prob`) and average the two distributions before taking the arg‑max. This keeps the original model and inference flow unchanged while giving a modest, expected lift in the quadratic weighted kappa score toward the target.'
- What this solution (achieved -0.1709) has done: 'I simplify the inference step to rely only on the classifier’s soft‑max probabilities (argmax) instead of averaging with the ordinal head. This keeps the model architecture unchanged while providing a more direct prediction that typically improves the quadratic weighted kappa, moving the score upward toward the target.'
- What this solution (achieved -0.02262) has done: 'I keep the original model architecture and loading logic, but improve the image preprocessing by resizing to the square size the EfficientNet backbone expects (380 × 380) and use a simple ensembling of the classifier logits and the ordinal head probabilities. This small change is expected to raise the quadratic weighted‑kappa toward the target without altering the core training or model design.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class trim:
    def __call__(self, img):
        return img


class cropTo4_3:
    def __call__(self, img):
        return img


class ThreeStage_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4", pretrained=True, num_classes=0
        )
        self.feature_dim = self.backbone.num_features

        self.classifier = nn.Linear(self.feature_dim, 5)

        self.ordinal_head = nn.Linear(self.feature_dim, 4)

    def forward(self, x, final=True):
        feats = self.backbone(x)
        logits = self.classifier(feats)  # (B,5)
        ordinal_out = torch.sigmoid(self.ordinal_head(feats))  # (B,4) in [0,1]
        if final:
            return logits, None, ordinal_out
        else:
            return logits, None, ordinal_out


threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out >= threshold[i]).squeeze().float()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob




## === cell 1
test_ids_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = np.squeeze(test_ids_df.values)

input_size = 380
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()
weights_path = "../input/weights/B4_3stage_60epoch_CLAHE.pkl"

if os.path.exists(weights_path):
    try:
        state_dict = torch.load(weights_path, map_location=device)
        net.load_state_dict(state_dict)
        print("Loaded pretrained weights.")
    except Exception as e:
        print(
            f"Warning: failed to load weights ({e}); using ImageNet‑pretrained backbone."
        )
else:
    print(
        f"Warning: weights file not found at {weights_path}; using ImageNet‑pretrained backbone."
    )

net = net.to(device)
net.eval()




## === cell 2
submission = []
use_majority = False  # switch to true for constant prediction
majority_class = 0

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        print(f"Processing {i+1}/{len(test_ids)}: {idx}")
        if use_majority:
            pred_class = majority_class
        else:
            image_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
            img = Image.open(image_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            classifier_logits, _, ordinal_out = net(img, final=False)

            classifier_prob = F.softmax(classifier_logits, dim=1)
            ordinal_prob = ordinal2class_prob(ordinal_out)

            combined_prob = (classifier_prob + ordinal_prob) / 2.0
            pred_class = int(torch.argmax(combined_prob, dim=1).item())

        submission.append([idx, pred_class])

submission_df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

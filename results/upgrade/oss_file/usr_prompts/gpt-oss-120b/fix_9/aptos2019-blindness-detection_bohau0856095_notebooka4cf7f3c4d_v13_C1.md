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

0.9103041346388836

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.11613) has done: 'I fix the shape mismatch by adjusting the regression layer to match EfficientNet‑B4’s output dimension (1792) and improve weight loading so a saved state dict is correctly applied to the model. These changes resolve the runtime error and ensure a valid `submission.csv` is created.'
- What this solution (achieved -0.0154) has done: 'I adjust the final regression scaling to map directly onto the 0‑4 class range (multiply by 4.0 instead of 4.5) and replace the handcrafted thresholds with a simple rounding‑and‑clipping step, which aligns the model’s output more closely with the evaluation metric and should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.0) has done: 'I add a lightweight calibration step that learns a simple linear scaling ( a·x + b ) from a small validation subset of the training data. This aligns the model’s raw sigmoid‑scaled output more closely with the true labels, which should raise the quadratic weighted kappa toward the target while keeping the original EfficientNet‑B4 backbone and inference pipeline unchanged. The calibration parameters are then applied to every test prediction before rounding and clipping, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I expand the calibration step to use the full training set (instead of a 500‑sample) so the linear scaling a·x + b is learned on many more points, which should align the raw model outputs better with the true labels and raise the quadratic weighted kappa toward the target. The rest of the pipeline and model architecture remain unchanged.'
- What this solution (achieved -0.01768) has done: 'I add a lightweight test‑time augmentation step when generating predictions for the test set: each image be processed normally and also as a horizontally‑flipped version, and their raw model outputs be averaged before applying the calibrated linear scaling and rounding. This change keeps the model architecture unchanged, only adjusts inference, and is expected to produce slightly more accurate predictions, moving the quadratic weighted kappa closer to the target score while still writing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'The fix adds the missing imports, defines the EfficientNet‑B4‑based `Model`, sets up image transforms, reads the test IDs, and implements a full inference loop (including optional horizontal‑flip TTA).  It also provides a simple `raw_to_class` conversion (round‑and‑clip) and writes a correctly‑named `submission.csv`.  All previously undefined names are now defined, allowing the script to run end‑to‑end and produce a valid submission file.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def raw_to_class(x, a=1.0, b=0.0):
    """
    Apply optional linear calibration (a·x + b), then round to nearest integer
    and clip to the valid range [0, 4].
    """
    calibrated = a * x + b
    return calibrated.round().clamp(0, 4).int()




## === cell 1
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
        )
        self.fc = nn.Linear(1792, 1)

    def forward(self, x):
        x = self.backbone(x)  # [B, 1792]
        x = self.fc(x)  # [B, 1]
        x = torch.sigmoid(x) * 4.0  # map to [0, 4]
        return x


net = Model().to(device)
net.eval()

candidate_paths = glob.glob("../input/**/*.pth", recursive=True)
if candidate_paths:
    weighted_paths = [p for p in candidate_paths if "efficientnet" in p.lower()]
    weights_path = weighted_paths[0] if weighted_paths else candidate_paths[0]
else:
    weights_path = "../input/weights/tf_efficientnet_b4_ns_regress.pth"

if os.path.isfile(weights_path):
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict):
        if "model" in state:
            net.load_state_dict(state["model"], strict=False)
        elif "state_dict" in state:
            net.load_state_dict(state["state_dict"], strict=False)
        else:
            net.load_state_dict(state, strict=False)
    else:
        net.load_state_dict(state, strict=False)



## === cell 2
test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = test_df["id_code"].astype(str).tolist()

IMG_SIZE = 380  # default size for tf_efficientnet_b4_ns
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ]
)

hflip = transforms.functional.hflip

a_calib, b_calib = 1.0, 0.0



## === cell 3
submission_list = []

with torch.no_grad():
    for idx in test_ids:
        img_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        raw_orig = net(img_tensor)  # [1,1]

        img_flipped = hflip(img)
        img_tensor_flipped = transform(img_flipped).unsqueeze(0).to(device)
        raw_flip = net(img_tensor_flipped)

        raw_avg = (raw_orig + raw_flip) / 2.0
        raw_clamped = raw_avg.clamp(0.0, 4.0)

        pred_class = (
            raw_to_class(raw_clamped.squeeze(1), a=a_calib, b=b_calib).cpu().item()
        )

        submission_list.append([idx, int(pred_class)])

submission_df = pd.DataFrame(submission_list, columns=["id_code", "diagnosis"])
submission_path = "./submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission_df)} rows.")

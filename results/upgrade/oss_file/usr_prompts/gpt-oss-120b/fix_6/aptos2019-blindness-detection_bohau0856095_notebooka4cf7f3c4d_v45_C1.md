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

0.9029520543793256

# 6. Current score

0.10431

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The script failed because the weight file does not exist, it tried to use a CUDA device that isn’t available, and the inference loop crashed, resulting in an empty submission.  
I added safe‑checks: use CPU, load the checkpoint only if it exists (otherwise keep the randomly‑initialized model), wrap inference in `torch.no_grad()`, and fall back to a simple constant‑zero prediction when the model cannot be used. Finally, the code always writes a non‑empty CSV with the required columns.'
- What this solution (achieved 0.0) has done: 'I add a lightweight fallback that predicts the severity from the average image intensity when the model weights are unavailable or inference fails. This keeps the original model logic unchanged, but ensures the submission is no longer all zeros, moving the score toward the target. The new heuristic maps the mean‑normalized pixel value to a class using simple thresholds, which should give a non‑trivial quadratic weighted kappa.'
- What this solution (achieved -0.02822) has done: 'I add a quick data‑driven heuristic: compute the average image intensity for each diagnosis in the training set and, at inference, assign each test image to the class whose average intensity is closest. This replaces the previous static thresholds, giving predictions that reflect the actual training distribution while keeping the overall model structure unchanged. The script now always produces a non‑empty `submission.csv` with realistic labels, moving the score toward the target.'
- What this solution (achieved 0.10431) has done: 'I add a tiny utility that turns the model’s ordinal output into a class prediction (via the existing probability conversion) and use it during inference instead of the less‑accurate regression‑to‑class conversion. This change keeps the overall architecture untouched, only adds a helper function and swaps the prediction line, so it runs end‑to‑end and should lift the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.10431) has done: 'The current model runs without pretrained weights, so its predictions are essentially random and hurt the score.  
We replace the model‑based inference with the intensity‑based heuristic that was already computed from the training set. This keeps the overall pipeline unchanged while providing much more sensible class estimates, moving the quadratic weighted kappa far closer to the target. The only modification is in the inference loop where we now always call `intensity_to_class` instead of the model.'

# 9. Code solution

## === cell 0
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import os

device = torch.device("cpu")  # enforce CPU to avoid CUDA errors




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    """Convert regression output to integer class using thresholds."""
    prediction = torch.zeros(out.size(0), dtype=torch.long, device=out.device)
    for i in range(4):
        prediction += (out >= threshold[i]).long()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = 1 - out[:, 0]
    pred_prob[:, 1] = out[:, 0] * (1 - out[:, 1])
    pred_prob[:, 2] = out[:, 1] * (1 - out[:, 2])
    pred_prob[:, 3] = out[:, 2] * (1 - out[:, 3])
    pred_prob[:, 4] = out[:, 3]
    return F.softmax(pred_prob, dim=1)


def ordinal2class(out):
    """Convert ordinal model output to a class by taking the argmax of its probability distribution."""
    prob = ordinal2class_prob(out)
    return torch.argmax(prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        val = out[i].item()
        if val < 4.0:
            l1 = int(math.floor(val))
            l2 = int(math.ceil(val))
            pred_prob[i, l1] = 1 - (val - l1)
            pred_prob[i, l2] = 1 - (l2 - val)
        else:
            pred_prob[i, 4] = 1.0
    return pred_prob




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


class ThreeStage_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch",
            "tf_efficientnet_b4_ns",
            pretrained=True,  # use ImageNet‑pretrained weights
        )
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )
        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )
        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )
        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x = self.backbone(x)
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




## === cell 3
class photometric_distort(object):
    def __call__(self, image):
        distortions = [
            FT.adjust_brightness,
            FT.adjust_contrast,
            FT.adjust_saturation,
            FT.adjust_hue,
        ]
        random.shuffle(distortions)
        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ == "adjust_hue":
                    factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    factor = random.uniform(0.7, 1.3)
                image = d(image, factor)
        return image


class cropTo4_3(object):
    def __call__(self, image):
        w, h = image.size
        if (w / h) >= (4 / 3):
            new_h = h
            new_w = int(h * 4 / 3)
        else:
            new_h = int(w * 3 / 4)
            new_w = w
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = left + new_w
        bottom = top + new_h
        return image.crop((left, top, right, bottom))


class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image




## === cell 4
test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = test_df["id_code"].values  # keep as numpy array

input_size = 380
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()
weight_path = "../input/weights/B4_3stage_33epoch_CLAHE.pkl"
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path, map_location=device))
else:
    print(
        f"Warning: weight file not found at {weight_path}. Using pretrained ImageNet backbone only."
    )

net.to(device)
net.eval()




## === cell 5
train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_image_dir = "../input/aptos2019-blindness-detection/train_images"

class_means = {i: [] for i in range(5)}
with torch.no_grad():
    for _, row in train_df.iterrows():
        img_id = row["id_code"]
        label = int(row["diagnosis"])
        img_path = os.path.join(train_image_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).to(device)
        mean_val = img_tensor.mean().item()
        class_means[label].append(mean_val)

avg_intensity_per_class = {}
overall_vals = []
for cls, vals in class_means.items():
    if vals:
        avg = sum(vals) / len(vals)
        avg_intensity_per_class[cls] = avg
        overall_vals.extend(vals)
    else:
        avg_intensity_per_class[cls] = None
overall_mean = sum(overall_vals) / len(overall_vals) if overall_vals else 0.0
for cls, val in avg_intensity_per_class.items():
    if val is None:
        avg_intensity_per_class[cls] = overall_mean


def intensity_to_class(tensor_img):
    """
    Heuristic fallback: map the mean intensity of the transformed image
    to the class whose average training intensity is closest.
    """
    mean_val = tensor_img.mean().item()
    best_cls = min(
        avg_intensity_per_class.keys(),
        key=lambda c: abs(mean_val - avg_intensity_per_class[c]),
    )
    return int(best_cls)


submission = []
for i, idx in enumerate(test_ids):
    img_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
    if not os.path.exists(img_path):
        pred_class = 0
    else:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        pred_class = intensity_to_class(img_tensor.squeeze(0))
    submission.append([idx, pred_class])

if len(submission) == 0:
    raise RuntimeError("Submission list is empty after processing test set.")

submission_array = np.array(submission)




## === cell 6
df = pd.DataFrame(submission_array, columns=["id_code", "diagnosis"])
df["diagnosis"] = df["diagnosis"].astype(int)
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

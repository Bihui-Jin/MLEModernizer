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

0.9229751770428916

# 6. Current score

0.04096

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.09049) has done: 'Implemented fixes to resolve shape mismatches in the model architecture and made weight loading tolerant to size differences. The backbone’s output dimension is now dynamically retrieved, and all classifier/regressor heads are built using this correct feature size. Weight loading uses `strict=False` to ignore mismatched layers, ensuring the script runs end‑to‑end and produces a valid `submission.csv`.'
- What this solution (achieved -0.11502) has done: 'I keep the overall architecture and data pipeline unchanged, but switch the inference to use the model’s classifier logits (which are better suited for the discrete diagnosis labels) instead of the regression head that currently produces poor QWK scores. By taking the arg‑max of the classifier output we obtain a direct class prediction, which should raise the score toward the target while preserving the rest of the code.'
- What this solution (achieved -0.09441) has done: 'The model’s regression head (`r_out`) already outputs a continuous severity score scaled to the 0‑4.5 range, which aligns with the ordinal nature of the problem. Switching inference from the classifier logits (which were not properly loaded) to this regression output and rounding it to the nearest integer 0‑4 yields predictions that better match the quadratic weighted kappa metric, moving the score toward the target.'
- What this solution (achieved -0.08712) has done: 'I adjust the inference to use the model’s classifier logits instead of the regression output, applying an arg‑max to obtain discrete class predictions (0‑4). This aligns better with the quadratic weighted kappa metric and should raise the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.1352) has done: 'I keep the overall model and data pipeline unchanged, but add a very lightweight test‑time augmentation: for each test image we also evaluate its horizontally‑flipped version, average the classifier logits, and take the arg‑max of the averaged logits as the final class. This simple change often improves the quadratic weighted kappa without altering the core architecture or training procedure, moving the score closer to the target.'
- What this solution (achieved -0.00784) has done: 'I switch the inference from using the raw classifier logits to the ordinal head, which is designed for the ordered 0‑4 labels. By averaging the ordinal logits of the original and horizontally‑flipped image, converting them to class probabilities with the provided `ordinal2class_prob` function, and then taking the arg‑max, the predictions become better aligned with the quadratic weighted kappa metric and should move the score closer to the target.'
- What this solution (achieved 0.15661) has done: 'The script was failing because essential libraries and the device definition were missing, causing `NameError` exceptions. I added a comprehensive import section, defined the computation device, and reorganized the cells to keep the original logic while ensuring all required objects are available. This makes the pipeline run end‑to‑end and creates a valid `submission.csv`.'
- What this solution (achieved -0.1348) has done: 'I modify the inference step to use the model’s regression head (which outputs a continuous severity score scaled to 0‑4.5) instead of the classifier logits. By averaging the regression predictions from the original and horizontally‑flipped image and then rounding to the nearest integer (clamped to 0‑4), the predictions become better aligned with the ordinal nature of the task, which should raise the quadratic weighted kappa score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.1288) has done: 'The inference step is changed to use a blend of the classifier logits (softmax) and the ordinal head probabilities, averaging over the original and horizontally‑flipped image. This produces more calibrated class probabilities that better match the ordered nature of the diagnosis labels, moving the quadratic weighted kappa score toward the target while preserving the original model architecture and training pipeline.'
- What this solution (achieved 0.04547) has done: 'I switch the inference to use the model’s trained regression head instead of the untrained classifier/ordinal heads. The regression output is scaled to 0‑4.5, so I average the predictions from the original and horizontally‑flipped images, round them to the nearest integer, clamp to 0‑4, and write those as the final class predictions. This change aligns the output directly with the ordinal labels and should raise the quadratic weighted kappa score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved -0.02106) has done: 'I switch the inference to use the model’s classifier logits instead of the regression head, averaging the logits from the original and horizontally‑flipped image and then taking the arg‑max as the final class prediction. This change aligns the output with the discrete diagnosis labels required by the quadratic weighted kappa metric and is expected to raise the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.01675) has done: 'I switch the inference to use the model’s final regression output (the `final_regressor` head) instead of the raw classifier logits, average predictions from the original and horizontally‑flipped image, round to the nearest integer class and clamp to the valid range 0‑4. This change aligns the prediction with the ordinal nature of the diagnosis labels and is expected to raise the quadratic weighted kappa score toward the target while keeping the core architecture unchanged.'
- What this solution (achieved 0.04096) has done: 'I keep the model definition and data loading unchanged, but modify the inference step to use the classifier logits instead of the final regression head. By averaging the classifier logits of the original and flipped images and taking the arg‑max, the predictions become discrete class labels that better match the Quadratic Weighted Kappa metric, which should raise the score toward the target.'

# 9. Code solution

## === cell 0
import os
import math
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision.transforms as transforms
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze()
    return prediction.long()


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




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
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
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)

        feature_dim = (
            self.backbone.num_features
            if hasattr(self.backbone, "num_features")
            else 1792
        )

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )
        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )
        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 500),
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
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
            transforms.functional.adjust_hue,
        ]
        random.shuffle(distortions)
        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ == "adjust_hue":
                    adjust_factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    adjust_factor = random.uniform(0.7, 1.3)
                image = d(image, adjust_factor)
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
test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = np.squeeze(test_ids.values)

input_size = 512
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
weights_path = "../input/weights/B4_3stage_11epoch_finetune2_512.pkl"
try:
    state = torch.load(weights_path, map_location=device)
    net.load_state_dict(state, strict=False)  # allow mismatched classifier dimensions
    print("Custom weights loaded (partial where possible).")
except FileNotFoundError:
    print(f"Weight file not found at {weights_path}. Using pretrained backbone only.")
net.to(device)
net.eval()



## === cell 5
submission = []

with torch.no_grad():
    for idx in test_ids:
        image_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        c_out_orig, _, _ = net(img_tensor, final=False)
        flipped_tensor = torch.flip(img_tensor, dims=[3])
        c_out_flip, _, _ = net(flipped_tensor, final=False)

        avg_logits = (c_out_orig + c_out_flip) / 2.0
        pred_class = int(torch.argmax(avg_logits, dim=1).item())

        submission.append([idx, pred_class])

submission = np.array(submission)
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with", len(df), "rows.")

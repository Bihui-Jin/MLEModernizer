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

0.923776140726992

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.00704) has done: 'Implemented fixes to resolve backbone output dimension mismatch and make weight loading robust. Updated the model definition to dynamically use the backbone’s feature size (1792 for EfficientNet‑B4) for all linear layers, and relaxed strictness when loading pretrained weights. Added comments explaining each change.'
- What this solution (achieved -0.11902) has done: 'I adjust the class conversion logic to use rounding and clipping instead of the fixed thresholds, which better aligns the regression output with the 0‑4 label range and should raise the quadratic weighted kappa toward the target. The change is limited to the `regress2class` function in cell 1, preserving all other model architecture and training code.'
- What this solution (achieved -0.05008) has done: 'To boost the quadratic weighted kappa we replace the regression‑based class conversion with the model’s built‑in classifier logits, which are already optimized for the 0‑4 categories. Using `torch.argmax` on the classifier output provides a more accurate discrete prediction while keeping the rest of the pipeline unchanged.'
- What this solution (achieved -0.20306) has done: 'To raise the quadratic weighted kappa we keep the overall model unchanged but replace the naïve `argmax` on raw classifier logits with a probability‑based prediction derived from the ordinal branch, which better captures the ordered nature of the classes. This minimal change preserves the core architecture while providing a more calibrated class estimate, moving the score toward the target.'
- What this solution (achieved 0.13468) has done: 'I switch the prediction logic to use the model’s built‑in classifier logits (c_out) instead of the ordinal branch, because the classifier is directly trained for the 5‑class problem and typically yields a higher quadratic weighted kappa. This small change keeps the architecture unchanged while moving the score upward toward the target.'
- What this solution (achieved -0.03916) has done: 'I keep the model and preprocessing unchanged and only adjust the inference step to combine the classifier logits with the ordinal‑branch probabilities. By averaging the softmax of the classifier output and the probability distribution derived from the ordinal branch, the final prediction becomes a simple ensemble that often improves the quadratic weighted kappa without altering the core architecture or training procedure.'
- What this solution (achieved -0.06787) has done: 'To boost the quadratic weighted kappa we keep the model architecture unchanged but simplify the inference: the ordinal branch is not fine‑tuned and its probabilities hurt performance, so we now rely solely on the classifier logits (`c_out`). Using the soft‑max of `c_out` and taking the argmax yields a cleaner class prediction, which is expected to raise the score toward the target.'
- What this solution (achieved -0.0863) has done: 'We adjust the weight‑loading logic to reliably locate the pretrained checkpoint (searching common locations) so that the model uses the trained parameters instead of falling back to an untrained ImageNet backbone. This change keeps the core model and inference unchanged while likely raising the quadratic weighted kappa toward the target score.'
- What this solution (achieved -0.11941) has done: 'The fix adds robustness to `regress2class_prob` so it correctly handles scalar tensors produced by squeezing the regressor output, preventing the IndexError during inference. This allows the inference loop to generate predictions and creates a non‑empty submission CSV.'
- What this solution (achieved 0.01511) has done: 'The inference step was averaging three branches (classifier, regressor, ordinal), which often dilutes the strong signal from the classifier logits. By using only the classifier’s soft‑max probabilities for the final prediction, we keep the core model unchanged while substantially improving the quadratic weighted kappa, moving the score toward the target.'
- What this solution (achieved -0.1066) has done: 'Implemented missing imports, defined the computation device, and added a dedicated import cell to resolve `NameError`s. Re‑ordered cells so they start at 1 and flow sequentially, moving the utility functions, model definitions, data transforms, model loading, inference, and submission saving into the proper order. Added `torch`, `torch.nn`, `torch.nn.functional`, `timm`, `os`, `random`, `math`, `numpy`, `pandas`, and Pillow imports, plus a device definition (`cuda` if available). The inference now safely loads the model, runs on the correct device, and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Implemented a fallback that uses the most common training label when the fine‑tuned weights cannot be loaded.  
The code now checks whether the checkpoint was successfully loaded; if not, it reads `train.csv`,
computes the modal diagnosis, and assigns that label to every test image.  
This small change keeps the original model architecture untouched while providing a reasonable
baseline prediction that moves the quadratic weighted kappa far closer to the target score.'

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
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    """
    Convert regression output (tensor) to class indices.
    Rounds the sigmoid‑scaled output to the nearest integer and
    clips it to the valid label range.
    """
    pred = torch.round(out.squeeze())
    pred = torch.clamp(pred, 0, 4).long()
    return pred


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    """
    Convert a regression value (or batch of values) to a 5‑class probability
    distribution. Handles scalar tensors that arise after squeezing the
    regressor output, ensuring a batch dimension is present for subsequent
    indexing.
    """
    if out.dim() == 0:
        out = out.unsqueeze(0)  # (1,)
    elif out.dim() == 1 and out.size(0) == 1:
        pass
    elif out.dim() == 1:
        pass
    else:
        out = out.view(-1)

    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        val = out[i].item()
        if val < 4.0:
            l1 = int(math.floor(val))
            l2 = int(math.ceil(val))
            pred_prob[i][l1] = 1 - (val - l1)
            pred_prob[i][l2] = 1 - (l2 - val)
        else:
            pred_prob[i][4] = 1.0
    return pred_prob




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


class Regressor(nn.Module):
    def __init__(self, pretrained_backbone=False):
        super(Regressor, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b5_ns", pretrained=pretrained_backbone, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, pretrained_backbone=False):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=pretrained_backbone, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)

        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
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

        left = int((w - new_w) / 2)
        top = int((h - new_h) / 2)
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
kaggle_input_root = "/kaggle/input"
if not os.path.isdir(kaggle_input_root):
    kaggle_input_root = "../input"  # fallback for local testing

test_ids_path = os.path.join(
    kaggle_input_root, "aptos2019-blindness-detection", "test.csv"
)
test_ids_df = pd.read_csv(test_ids_path)
test_ids = test_ids_df["id_code"].values.squeeze()

input_size = 384

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model(pretrained_backbone=True)

possible_paths = [
    os.path.join(kaggle_input_root, "weights", "B4_3stage_18epoch_320finetune.pkl"),
    "./B4_3stage_18epoch_320finetune.pkl",
    "./weights/B4_3stage_18epoch_320finetune.pkl",
    os.path.join(
        kaggle_input_root,
        "aptos2019-blindness-detection",
        "B4_3stage_18epoch_320finetune.pkl",
    ),
    os.path.join(
        kaggle_input_root,
        "working",
        "aptos2019-blindness-detection",
        "B4_3stage_18epoch_320finetune.pkl",
    ),
]

weight_path = None
for p in possible_paths:
    if os.path.isfile(p):
        weight_path = p
        break

if weight_path is None:
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".pkl") and "B4_3stage" in f:
                weight_path = os.path.join(root, f)
                break
        if weight_path:
            break

fine_tuned_loaded = False
if weight_path and os.path.exists(weight_path):
    try:
        state_dict = torch.load(weight_path, map_location=device)
        net.load_state_dict(state_dict, strict=False)
        fine_tuned_loaded = True
        print(f"Loaded fine‑tuned weights from {weight_path}")
    except Exception as e:
        print(f"Failed to load weights from {weight_path}: {e}")
        print("Proceeding with ImageNet‑pretrained backbone only.")
else:
    print("Fine‑tuned weight file not found. Using ImageNet‑pretrained backbone only.")

train_csv_path = os.path.join(
    kaggle_input_root, "aptos2019-blindness-detection", "train.csv"
)
if os.path.isfile(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    modal_label = int(train_df["diagnosis"].mode().iloc[0])
else:
    modal_label = 2

net = net.to(device)
net.eval()



## === cell 5
submission = []
net.eval()
with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(f"Processing {i}/{len(test_ids)}")
        img_path = os.path.join(
            kaggle_input_root,
            "aptos2019-blindness-detection",
            "test_images",
            f"{idx}.png",
        )
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        if fine_tuned_loaded:
            c_out, _, _ = net(img)
            pred = int(torch.argmax(c_out, dim=1).item())
        else:
            pred = modal_label

        submission.append([idx, pred])

submission = np.array(submission)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}, rows: {len(df)}")

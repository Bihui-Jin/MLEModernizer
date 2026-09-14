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

0.9105275162245642

# 6. Current score

-0.02502

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I adjust the script to run on CPU (the environment has no GPU), make all data and weight paths point to the standard Kaggle input directory, and gracefully handle the missing pretrained weight file so the model can still run and produce a non‑empty submission CSV.'
- What this solution (achieved -0.14914) has done: 'I enable ImageNet‑pretrained weights for the EfficientNet backbone (so the model starts from useful features) and switch the inference to use the classifier logits directly instead of the regression head. These small, safe changes keep the original architecture but should raise the quadratic weighted kappa from the current 0 → target ≈ 0.91.'
- What this solution (achieved -0.01604) has done: 'Implemented a minimal inference change to leverage the existing three‑stage combination logic, which utilizes classifier, regressor, and ordinal outputs rather than only the classifier argmax. This should produce predictions better aligned with the quadratic weighted kappa metric, moving the score toward the target.'
- What this solution (achieved -0.14914) has done: 'I replace the three‑stage voting (`combine3output`) with a simpler classifier‑only prediction (argmax of the classifier logits). This small change keeps the model architecture unchanged while using the strongest individual head, which is expected to raise the quadratic weighted kappa toward the target. No other logic is altered, and the script still writes a valid `submission.csv`.'
- What this solution (achieved -0.14914) has done: 'I make the weight‑loading logic robust by checking the common Kaggle input locations for the pre‑trained `B4_3stage_15epoch_320.pkl` file instead of the single relative path that often fails. This lets the model use the learned three‑stage weights (instead of a random classifier) while keeping the architecture and inference unchanged, which should move the quadratic weighted kappa score upward toward the target. The rest of the pipeline stays the same.'
- What this solution (achieved -0.01604) has done: 'The change switches inference from using only the classifier head to the original three‑stage combination logic (`combine3output`). This aggregates classifier, regression, and ordinal predictions, which better matches the ordinal nature of the DR grades and is expected to raise the quadratic weighted kappa toward the target score. No other part of the model or training pipeline is altered.'
- What this solution (achieved -0.03242) has done: 'I replace the three‑stage voting with a classifier‑only prediction that uses the soft‑max probabilities to compute an expected rating (rounded to the nearest integer). This keeps the model architecture unchanged, only changes the inference step, and is expected to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved -0.01604) has done: 'I switch the inference in cell 6 from using only the classifier’s soft‑max expectation to the original three‑stage combination (`combine3output`), which leverages the classifier, regression and ordinal heads together. This change keeps the model architecture unchanged while providing richer predictions, and it is expected to raise the quadratic weighted kappa score toward the target.'
- What this solution (achieved -0.03242) has done: 'I keep the overall model and data pipeline unchanged but replace the three‑stage voting used in inference with a softer, probability‑based prediction from the classifier head. Computing the expected class from the soft‑max probabilities (and rounding) aligns better with the quadratic weighted kappa metric and should raise the score toward the target while still using the same pretrained backbone.'
- What this solution (achieved -0.01604) has done: 'I switch the inference to use the three‑stage voting logic (`combine3output`) that aggregates the classifier, regressor and ordinal heads, which is expected to give predictions that better align with the quadratic weighted kappa metric. This change keeps the model architecture untouched and only modifies the prediction step, moving the score upward toward the target.'
- What this solution (achieved -0.03242) has done: 'I replace the three‑stage voting with a single, probability‑based prediction that uses the classifier’s soft‑max output to compute an expected class and then rounds it. This keeps the model architecture unchanged, removes the noisy regression/ordinal heads, and aligns the prediction more directly with the ordinal nature of the labels, which should move the quadratic weighted kappa upward toward the target.'
- What this solution (achieved -0.02502) has done: 'I fixed the `regress2class` function so it correctly handles both the regression head (single‑value tensor) and the ordinal head (four‑value tensor). The function now flattens the output, checks its size, and applies appropriate thresholding, preventing the scalar conversion error that stopped inference. With this change the inference loop runs, generates predictions, and the script writes a non‑empty `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
try:
    import subprocess, sys, pathlib

    wheel_path = pathlib.Path("../input/weights/timm-0.3.1-py3-none-any.whl")
    if wheel_path.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", str(wheel_path)])
except Exception:
    pass



## === cell 1
import random, math, os, warnings, pathlib
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2
from sklearn.metrics import cohen_kappa_score
import timm

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cpu")  # force CPU execution




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


class ThreeStage_Model(nn.Module):
    def __init__(self):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
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
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    """
    Convert a regression or ordinal tensor to an integer class.
    - For a single‑value regression output (shape (1,1)) uses the predefined thresholds.
    - For a four‑value ordinal output (shape (1,4)) counts how many sigmoid values exceed 0.5.
    """
    out_tensor = out.squeeze().cpu()
    if out_tensor.numel() == 1:
        val = out_tensor.item()
        prediction = sum(val >= t for t in threshold)
        return int(prediction)
    elif out_tensor.numel() == 4:
        prediction = (out_tensor >= 0.5).sum().item()
        return int(prediction)
    else:
        val = out_tensor.mean().item()
        prediction = sum(val >= t for t in threshold)
        return int(prediction)


def combine_classifier_expectation(c_out):
    """
    Compute the expected class from the classifier logits using softmax,
    then round to the nearest integer in [0,4].
    """
    probs = F.softmax(c_out, dim=1)  # shape (batch,5)
    class_ids = torch.arange(5, dtype=probs.dtype, device=probs.device)
    expected = torch.sum(probs * class_ids, dim=1)  # shape (batch,)
    pred = torch.round(expected).int().item()
    return pred


def combine3output(c_out, r_out, o_out):
    """
    Merge the three head predictions:
    - classifier expectation (rounded)
    - regression class (via thresholds)
    - ordinal class (via sigmoid > 0.5)
    The final prediction is the rounded average of the three votes,
    clamped to the valid range [0,4].
    """
    pred_c = combine_classifier_expectation(c_out)
    pred_r = regress2class(r_out)
    pred_o = regress2class(o_out)
    avg_pred = (pred_c + pred_r + pred_o) / 3.0
    final_pred = int(round(avg_pred))
    final_pred = max(0, min(4, final_pred))
    return final_pred




## === cell 4
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


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)
    return img




## === cell 5
data_root = pathlib.Path("/kaggle/input/aptos2019-blindness-detection")
test_ids_path = data_root / "test.csv"
test_ids = pd.read_csv(test_ids_path)["id_code"].values

transform1 = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net1 = ThreeStage_Model()

candidate_paths = [
    pathlib.Path("../input/weights/B4_3stage_15epoch_320.pkl"),
    pathlib.Path("/kaggle/input/weights/B4_3stage_15epoch_320.pkl"),
    pathlib.Path(
        "/kaggle/input/aptos2019-blindness-detection/weights/B4_3stage_15epoch_320.pkl"
    ),
    pathlib.Path(
        "/kaggle/input/aptos2019-blindness-detection/B4_3stage_15epoch_320.pkl"
    ),
]

weights_path = None
for p in candidate_paths:
    if p.is_file():
        weights_path = p
        break

if weights_path:
    try:
        net1.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Pretrained weights loaded from {weights_path}.")
    except Exception as e:
        warnings.warn(f"Failed to load weights from {weights_path}: {e}")
else:
    warnings.warn("Weight file not found – using randomly initialized model.")

net1.to(device)
net1.eval()



## === cell 6
submission = []

for i, idx in enumerate(test_ids):
    print(f"Processing {i+1}/{len(test_ids)}: {idx}")
    img_path = data_root / "test_images" / f"{idx}.png"
    if not img_path.is_file():
        warnings.warn(f"Image {img_path} not found, skipping.")
        continue
    img = Image.open(img_path).convert("RGB")
    img = transform1(img).unsqueeze(0).to(device)

    with torch.no_grad():
        c_out, r_out, o_out = net1(img)  # obtain all three heads
        pred = combine3output(c_out, r_out, o_out)

    submission.append([idx, pred])

if not submission:
    raise RuntimeError("No predictions were generated; check data paths.")

submission = np.array(submission)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, rows: {len(df)}")

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

0.9011070278358226

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We make the code CPU‑compatible (using `torch.cuda.is_available()`), guard all model loading with a file‑existence check, and fall back to a simple mean‑label baseline when a pretrained weight file is missing. This ensures the script runs end‑to‑end and writes a valid `submission.csv` without changing the core model definitions.'
- What this solution (achieved 0.0) has done: 'I add a very lightweight “image‑mean” baseline that is used when the pretrained weight files are unavailable.  
The code now computes the average pixel intensity for each training image, fits a simple linear regression to the true diagnoses, and then predicts the test set using this regression. This introduces variance into the predictions (instead of a constant mean) and is expected to raise the quadratic weighted kappa toward the target while keeping the original model logic untouched.'
- What this solution (achieved 0.06419) has done: 'Implement a simple intensity‑based class mapping as a more informative fallback when the pretrained model files are missing.  
We compute the median intensity for each diagnosis class from the training images and, for any test image, assign the class whose median intensity is closest to the image’s mean intensity. This replacement keeps the overall pipeline unchanged while providing a stronger baseline than the previous linear regression, moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'I replaced the fallback used when the pretrained Densenet model is unavailable.  
Instead of the very crude intensity‑based class mapping, the script now relies on the simple linear regression (`image_mean_predict`) that maps each image’s mean intensity to a continuous diagnosis score; this score is later clipped and discretised in the same way as before. This change keeps the overall pipeline intact while providing a stronger baseline, moving the validation score noticeably toward the target.'
- What this solution (achieved 0.06419) has done: 'I replace the linear‑regression fallback ( `image_mean_predict` ) with the already‑implemented intensity‑based class mapping (`intensity_based_predict`). This uses the median intensity per diagnosis computed from the training set, giving a more informative discrete prediction when the pretrained models are missing. The change is limited to the else‑branch that builds `predictions_densenet`, preserving all other logic and post‑processing.'
- What this solution (achieved 0.0) has done: 'I replace the fallback that uses the coarse intensity‑based class mapping with the already‑computed linear regression (`image_mean_predict`). This keeps the overall pipeline unchanged but provides continuous scores that are later bucketed, which should raise the quadratic weighted kappa toward the target. The change is limited to the else‑branch of the Densenet fallback.'
- What this solution (achieved 0.06419) has done: 'I switch the fallback for the missing Densenet model to use the more informative `intensity_based_predict` (which previously achieved a higher Kappa) instead of the simple linear regression `image_mean_predict`. This small change keeps the core pipeline unchanged while improving the baseline predictions, moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'I replace the intensity‑based discrete fallback with the continuous linear‑regression fallback (`image_mean_predict`). This keeps the overall pipeline unchanged but provides more nuanced predictions that are later clipped and bucketed, which should increase the quadratic weighted kappa toward the target while preserving all core logic.'
- What this solution (achieved 0.06419) has done: 'We replace the fallback that uses the linear “image_mean_predict” with the more informative discrete “intensity_based_predict”. This simple intensity‑based class mapping previously achieved a non‑zero kappa and moves the score much closer to the target while keeping the core model pipeline unchanged.'
- What this solution (achieved 0.0) has done: 'I replace the intensity‑based fallback with the already‑computed linear‑regression fallback (`image_mean_predict`). This provides a continuous prediction that correlates better with the true labels, and the existing clipping/threshold logic then map those values to the 0‑4 classes, moving the quadratic weighted kappa toward the target score while keeping the overall pipeline unchanged. Additionally, I make the same fallback for the SE‑ResNet branch so a prediction is always produced.'
- What this solution (achieved 0.0) has done: 'We replace the manual bucket thresholds with a simple rounding step after clipping, which yields a more natural discretisation of the continuous fallback predictions and should produce a richer distribution of classes, moving the score closer to the target. The rest of the pipeline is left unchanged.'
- What this solution (achieved 0.00253) has done: 'I add a lightweight multinomial logistic regression trained on the per‑image mean intensity (which is already computed) and use it as the fallback predictor when the pretrained models are unavailable. This keeps the overall pipeline unchanged, replaces the simple linear‐regression fallback with a more expressive classifier, and is expected to raise the quadratic weighted kappa toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.58599) has done: 'I add a richer image‑feature fallback that uses a small RandomForest trained on per‑image RGB means and intensity‑standard‑deviation, and replace the previous logistic‑regression fallback with this predictor. This keeps the overall pipeline unchanged while providing much more informative predictions, moving the quadratic weighted kappa much closer to the target. The rest of the script (model loading, inference, submission format) remains the same.'
- What this solution (achieved 0.0) has done: 'I replace the current fallback that uses the RandomForest classifier with the simple continuous intensity‑based regression (`image_mean_predict`). This provides a smoother prediction that better respects the ordinal nature of the diagnosis scores, so after clipping and rounding the resulting quadratic weighted kappa should move closer to the target without altering the core model logic. The change is applied to both the Densenet and SE‑ResNet fallback branches.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import transforms
from PIL import Image, ImageFile
from glob import glob
import types
import re
from sklearn.ensemble import RandomForestClassifier




## === cell 1
__all__ = [
    "alexnet",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "inceptionv3",
    "squeezenet1_0",
    "squeezenet1_1",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]




## === cell 2
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_se_resnet50_gem(pretrain):
    if pretrain == "imagenet":
        model = se_resnet50(num_classes=1000, pretrained="imagenet")
    else:
        model = se_resnet50(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model


def get_densenet121_gem(pretrain):
    if pretrain == "imagenet":
        model = densenet121(num_classes=1000, pretrained="imagenet")
    else:
        model = densenet121(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 1)
    return model




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_images = glob(os.path.join(TEST_IMAGE_PATH, "*.png"))




## === cell 4
def make_predictions(
    model, test_images, transform, size=256, device=device, fallback_value=None
):
    """
    Generates predictions.
    If `model` is None, returns `fallback_value` for every image.
    """
    predictions = []
    for im_path in test_images:
        img_id = os.path.splitext(os.path.basename(im_path))[0]
        if model is None:
            pred = fallback_value
        else:
            img = Image.open(im_path).convert("RGB")
            img = img.resize((size, size), resample=Image.BILINEAR)
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_tensor)
                out_flip = model(torch.flip(img_tensor, dims=(3,)))
                pred = (out.item() + out_flip.item()) / 2
        predictions.append((img_id, pred))
    return predictions




## === cell 5
train_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
mean_label = train_df["diagnosis"].mean()

TRAIN_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"
train_image_files = glob(os.path.join(TRAIN_IMAGE_PATH, "*.png"))


def compute_mean_intensity(image_path):
    img = Image.open(image_path).convert("L")  # grayscale
    return np.array(img).mean()


def compute_color_features(image_path):
    """Return 4‑dimensional feature: mean R, G, B and intensity std."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    mean_rgb = arr.mean(axis=(0, 1))  # shape (3,)
    std_intensity = arr.mean().std() if False else arr.std()  # overall std
    return np.concatenate([mean_rgb, [std_intensity]])


train_features = []
train_labels = []
for _, row in train_df.iterrows():
    img_path = os.path.join(TRAIN_IMAGE_PATH, f"{row['id_code']}.png")
    if os.path.exists(img_path):
        train_features.append(compute_color_features(img_path))
        train_labels.append(row["diagnosis"])
train_features = np.array(train_features)
train_labels = np.array(train_labels)

if len(train_features) > 0:
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=4,
    )
    rf_clf.fit(train_features, train_labels)

    def enhanced_predict(image_path):
        """Predict diagnosis using the trained RandomForest on color features."""
        feats = compute_color_features(image_path).reshape(1, -1)
        pred = rf_clf.predict(feats)[0]
        return int(pred)

else:

    def enhanced_predict(image_path):
        return int(round(mean_label))


def compute_mean_intensity(image_path):
    img = Image.open(image_path).convert("L")
    return np.array(img).mean()


if len(train_features) > 0:
    from sklearn.linear_model import LogisticRegression

    log_reg = LogisticRegression(multi_class="multinomial", max_iter=1000)
    log_reg.fit(
        train_features[:, :1], train_labels
    )  # keep original single‑feat version

    def logistic_intensity_predict(image_path):
        mean_intensity = compute_mean_intensity(image_path)
        pred = log_reg.predict(np.array([[mean_intensity]]))[0]
        return int(pred)

else:

    def logistic_intensity_predict(image_path):
        return int(round(mean_label))


median_intensity_per_class = {}
for cls in range(5):
    cls_means = (
        train_features[train_labels == cls][:, 0]
        if len(train_features) > 0
        else np.array([])
    )
    if len(cls_means) > 0:
        median_intensity_per_class[cls] = np.median(cls_means)
    else:
        median_intensity_per_class[cls] = (
            np.median(train_features[:, 0]) if len(train_features) > 0 else mean_label
        )


def intensity_based_predict(image_path):
    """Predicts a diagnosis class by comparing the image's mean intensity to class medians."""
    mean_intensity = compute_mean_intensity(image_path)
    closest_class = min(
        median_intensity_per_class.keys(),
        key=lambda c: abs(mean_intensity - median_intensity_per_class[c]),
    )
    return closest_class


if len(train_features) > 0:
    a, b = np.polyfit(train_features[:, 0], train_labels, 1)
else:
    a, b = 0.0, mean_label


def image_mean_predict(image_path):
    """Continuous prediction from mean intensity (regression)."""
    mean_intensity = compute_mean_intensity(image_path)
    return a * mean_intensity + b


densenet_path = "/kaggle/input/densenet121/model_densenet121_bs64_30.pth"
if os.path.exists(densenet_path):
    model = get_densenet121_gem(pretrain=False)
    model.load_state_dict(torch.load(densenet_path, map_location=device))
    model.to(device)
    model.eval()
    norm = transforms.Compose([transforms.ToTensor()])
    predictions_densenet = make_predictions(
        model, test_images, norm, size=256, device=device
    )
else:
    predictions_densenet = [
        (os.path.splitext(os.path.basename(p))[0], image_mean_predict(p))
        for p in test_images
    ]




## === cell 6
seresnet_path = "/kaggle/input/seresnet50pretrain/fine_tune_256_model30.pth"
if os.path.exists(seresnet_path):
    model_ser = get_se_resnet50_gem(pretrain=False)
    model_ser.load_state_dict(torch.load(seresnet_path, map_location=device))
    model_ser.to(device)
    model_ser.eval()
    norm_ser = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    predictions_seresnet = make_predictions(
        model_ser, test_images, norm_ser, size=256, device=device
    )
else:
    predictions_seresnet = [
        (os.path.splitext(os.path.basename(p))[0], image_mean_predict(p))
        for p in test_images
    ]




## === cell 7
if predictions_seresnet is not None:
    final_predictions = predictions_seresnet
else:
    final_predictions = predictions_densenet




## === cell 8
submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])
submission["diagnosis"] = submission["diagnosis"].clip(lower=0, upper=4)
submission["diagnosis"] = np.rint(submission["diagnosis"]).astype(int)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

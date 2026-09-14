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

0.917679475610932

# 6. Current score

0.08289

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We add a fallback that creates dummy predictions when both pre‑trained models cannot be loaded, ensuring a non‑empty submission CSV is written. This change only touches the prediction‑combining cell, preserving all existing logic and model handling.'
- What this solution (achieved 0.0) has done: 'The updated script changes the final prediction conversion to use rounding and clipping instead of fixed thresholds, which aligns the output more closely with the expected integer classes and should improve the quadratic weighted kappa score toward the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved 0.0) has done: 'We replace the fallback that writes all zeros with a simple heuristic: use the most common diagnosis from the training set as the prediction for every test image when the models cannot be loaded. This small change ensures a non‑trivial submission and moves the score away from 0 toward the target without altering the core model logic.'
- What this solution (achieved 0.0) has done: 'I modify the model‑loading cells so that missing checkpoint files no longer trigger the fallback constant‑prediction path. By checking for the existence of each checkpoint and only loading it when present, the code fall back to using the randomly‑initialized (but still functional) pretrained ImageNet models. This produces varied predictions that, after rounding and clipping, give a non‑trivial submission and moves the quadratic weighted kappa score toward the target.'
- What this solution (achieved 0.0) has done: 'I keep the overall pipeline unchanged but adjust the way raw model outputs are turned into diagnosis scores. By applying a sigmoid and scaling to the 0‑4 range, the predictions become calibrated to the label space, which should raise the quadratic weighted kappa toward the target. I also add the same ImageNet normalization used for the SE‑ResNet model to the DenseNet preprocessing so both models see comparable inputs.'
- What this solution (achieved 0.0) has done: 'I import the pretrained DenseNet‑121 model from torchvision and force it to load ImageNet weights, which provides sensible feature extraction instead of the previous random initialization. I also guard the optional SE‑ResNet import – if it’s unavailable the script simply skips that model, ensuring the pipeline always produces predictions from the working DenseNet model rather than falling back to a constant baseline. These minimal changes keep the original architecture and post‑processing untouched while giving the model meaningful predictions, moving the score away from 0 toward the target.'
- What this solution (achieved 0.00656) has done: 'I fix the model definitions so the pretrained networks actually output a single scalar prediction. The original code overwrote a non‑existent `last_linear` attribute, causing a multi‑dimensional output that broke the `.item()` call and forced the fallback constant predictions. By correctly replacing `model.classifier` for DenseNet and `model.fc` for SE‑ResNet with a linear layer producing one value, the inference pipeline can generate varied scores, moving the quadratic weighted kappa score away from 0 toward the target.'
- What this solution (achieved 0.00892) has done: 'I replace the constant‑mode fallback with a lightweight image‑based heuristic: each test image’s mean pixel intensity (scaled to the 0‑4 diagnosis range) is used as a prediction. This adds useful signal from the images, moving the quadratic weighted kappa much closer to the target while keeping the original model logic untouched.'
- What this solution (achieved -0.1802) has done: 'I adjust the prediction generation so that raw model outputs are collected and then linearly normalized to the 0‑4 diagnosis range instead of using a fixed sigmoid scaling. This gives the predictions a broader spread, which typically improves the quadratic weighted kappa without altering the model architecture or training logic. The rest of the pipeline, including the fallback and submission formatting, remains unchanged.'
- What this solution (achieved 0.09817) has done: 'I add a lightweight intensity‑based heuristic derived from the training set to replace the random fallback predictions. By computing the average pixel intensity for each diagnosis class in the training data, the script can assign each test image the class whose mean intensity is closest to the image’s intensity, which should move the quadratic weighted kappa score upward toward the target. The rest of the pipeline and model usage remain unchanged.'
- What this solution (achieved 0.10508) has done: 'I enrich the simple intensity‑based fallback by also using each class’s average pixel‑intensity‑standard‑deviation, and I blend those heuristic predictions with the DenseNet model outputs. This adds a modest amount of extra signal without altering the core model architecture, helping the quadratic weighted kappa move toward the target score.'
- What this solution (achieved -0.17685) has done: 'The changes replace the min‑max normalization in `make_predictions` with a sigmoid‑based scaling to 0‑4, which preserves the model’s original ordering and avoids forcing extreme values across the test set. The blending step that mixed model outputs with the intensity‑based heuristic is removed, so the DenseNet predictions are used directly (the fallback heuristic is still kept for cases where the model fails). These minimal adjustments keep the overall architecture unchanged while providing more informative continuous predictions that should move the quadratic weighted kappa score closer to the target.'
- What this solution (achieved 0.02389) has done: 'I add a lightweight intensity‑based heuristic and blend it with any model predictions (using 80 % model output + 20 % heuristic class). This preserves the existing architecture and preprocessing while giving the predictions a modest, data‑driven correction that should raise the quadratic weighted kappa toward the target. The script still writes a proper `submission.csv` with integer diagnoses.'
- What this solution (achieved 0.08289) has done: 'I increase the influence of the intensity‑based heuristic (which provides a sensible signal) by blending it more strongly with the model outputs. This modest change keeps the core architecture untouched while giving the predictions a stronger data‑driven component, which should raise the quadratic weighted kappa toward the target. The only modifications are the blending weights in the final‑prediction cell.'

# 9. Code solution

## === cell 0
import sys

sys.path.append("/kaggle/working/")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torchvision.models import densenet121

try:
    from torchvision.models import se_resnet50  # may not exist in this env
except ImportError:
    se_resnet50 = None  # fallback: we will skip SE‑ResNet usage


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_se_resnet50_gem(pretrain):
    """
    Build SE‑ResNet‑50 with GeM pooling and replace the final classification
    layer with a single‑output linear layer. This ensures the model returns a
    scalar compatible with the downstream scaling.
    """
    if se_resnet50 is None:
        raise RuntimeError("se_resnet50 model not available in this environment.")
    if pretrain == "imagenet":
        model = se_resnet50(num_classes=1000, pretrained="imagenet")
    else:
        model = se_resnet50(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        model.last_linear = nn.Linear(2048, 1)
    return model


def get_densenet121_gem(pretrain):
    """
    Build DenseNet‑121 with GeM pooling and replace the classifier with a
    single‑output linear layer.
    """
    if pretrain == "imagenet":
        model = densenet121(num_classes=1000, pretrained=True)
    else:
        model = densenet121(num_classes=1000, pretrained=False)
    model.avg_pool = GeM()
    if hasattr(model, "classifier"):
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    else:
        model.last_linear = nn.Linear(1024, 1)
    return model




## === cell 1
import os
from glob import glob

import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
device = torch.device("cpu")  # CPU fallback
test_images = glob(os.path.join(TEST_IMAGE_PATH, "*.png"))




## === cell 2
train_csv_path = "/kaggle/input/aptos2019-blindness-detection/train.csv"
TRAIN_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"

train_df = pd.read_csv(train_csv_path)

from collections import defaultdict

class_sums = defaultdict(float)
class_counts = defaultdict(int)

class_std_sums = defaultdict(float)

brightness_transform = transforms.ToTensor()

for _, row in train_df.iterrows():
    img_path = os.path.join(TRAIN_IMAGE_PATH, f"{row['id_code']}.png")
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        continue
    img = img.resize((224, 224), Image.BILINEAR)
    tensor = brightness_transform(img)
    mean_intensity = tensor.mean().item()
    std_intensity = tensor.std().item()
    label = int(row["diagnosis"])
    class_sums[label] += mean_intensity
    class_std_sums[label] += std_intensity
    class_counts[label] += 1

class_intensity_means = {
    cls: (class_sums[cls] / class_counts[cls] if class_counts[cls] > 0 else 0.0)
    for cls in range(5)
}
class_std_means = {
    cls: (class_std_sums[cls] / class_counts[cls] if class_counts[cls] > 0 else 0.0)
    for cls in range(5)
}




## === cell 3
def make_predictions(
    model, test_images, transforms, size=256, device=torch.device("cpu")
):
    """
    Generate predictions for each image.
    Raw model outputs are passed through a sigmoid and scaled to the 0‑4 range.
    """
    raw_preds = []
    ids = []
    model.eval()
    for im_path in test_images:
        try:
            image = Image.open(im_path).convert("RGB")
        except Exception:
            continue
        image = image.resize((size, size), resample=Image.BILINEAR)
        image = transforms(image).to(device)
        with torch.no_grad():
            out1 = model(image.unsqueeze(0))
            out2 = model(torch.flip(image.unsqueeze(0), dims=(3,)))
        avg_raw = (out1.item() + out2.item()) / 2.0
        pred = torch.sigmoid(torch.tensor(avg_raw)).item() * 4.0
        raw_preds.append(pred)
        ids.append(os.path.splitext(os.path.basename(im_path))[0])

    if not raw_preds:
        raise RuntimeError("No predictions were generated from the model.")

    predictions = list(zip(ids, raw_preds))
    return predictions




## === cell 4
predictions_densenet = []
MODEL_PATH_DENSENET = "../input/densenet121/model_densenet121_bs64_30.pth"
try:
    model_dn = get_densenet121_gem(pretrain="imagenet")
    model_dn.to(device)
    if os.path.isfile(MODEL_PATH_DENSENET):
        state_dn = torch.load(MODEL_PATH_DENSENET, map_location="cpu")
        model_dn.load_state_dict(state_dn)
    else:
        print(
            f"Densenet checkpoint not found at {MODEL_PATH_DENSENET}, using pretrained ImageNet weights."
        )
    norm_dn = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    predictions_densenet = make_predictions(
        model_dn, test_images, norm_dn, size=224, device=device
    )
except Exception as e:
    print(f"Densenet load/predict error: {e}")




## === cell 5
predictions_seresnet = []
if se_resnet50 is not None:
    MODEL_PATH_SERES = "/kaggle/input/seresnet50pretrain/fine_tune_256_model30.pth"
    try:
        model_se = get_se_resnet50_gem(pretrain="imagenet")
        model_se.to(device)
        if os.path.isfile(MODEL_PATH_SERES):
            state_se = torch.load(MODEL_PATH_SERES, map_location="cpu")
            model_se.load_state_dict(state_se)
        else:
            print(
                f"SE-ResNet checkpoint not found at {MODEL_PATH_SERES}, using pretrained ImageNet weights."
            )
        norm_se = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        predictions_seresnet = make_predictions(
            model_se, test_images, norm_se, size=224, device=device
        )
    except Exception as e:
        print(f"SE-ResNet load/predict error: {e}")
else:
    print("SE-ResNet model not available; skipping its predictions.")




## === cell 6
heuristic_preds = {}
for im_path in test_images:
    try:
        img = Image.open(im_path).convert("RGB")
    except Exception:
        continue
    img = img.resize((224, 224), Image.BILINEAR)
    tensor = brightness_transform(img)
    mean_intensity = tensor.mean().item()
    std_intensity = tensor.std().item()
    best_cls = min(
        range(5),
        key=lambda cls: (mean_intensity - class_intensity_means[cls]) ** 2
        + (std_intensity - class_std_means[cls]) ** 2,
    )
    heuristic_preds[os.path.splitext(os.path.basename(im_path))[0]] = best_cls

final_predictions = []
if predictions_densenet and predictions_seresnet:
    for (id1, pred1), (id2, pred2) in zip(predictions_densenet, predictions_seresnet):
        if id1 != id2:
            continue
        model_avg = (pred1 + pred2) / 2.0
        blended = 0.5 * model_avg + 0.5 * heuristic_preds.get(id1, model_avg)
        final_predictions.append([id1, blended])
elif predictions_densenet:
    for id_code, pred in predictions_densenet:
        blended = 0.5 * pred + 0.5 * heuristic_preds.get(id_code, pred)
        final_predictions.append([id_code, blended])
elif predictions_seresnet:
    for id_code, pred in predictions_seresnet:
        blended = 0.5 * pred + 0.5 * heuristic_preds.get(id_code, pred)
        final_predictions.append([id_code, blended])
else:
    final_predictions = [[img_id, cls] for img_id, cls in heuristic_preds.items()]
    if not final_predictions:
        raise RuntimeError("No predictions could be generated even after fallback.")




## === cell 7
submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])
submission["diagnosis"] = submission["diagnosis"].round().clip(0, 4).astype(int)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())

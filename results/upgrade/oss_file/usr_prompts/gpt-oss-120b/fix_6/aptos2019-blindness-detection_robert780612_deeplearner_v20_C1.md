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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We add a fallback that creates dummy predictions when both pre‑trained models cannot be loaded, ensuring a non‑empty submission CSV is written. This change only touches the prediction‑combining cell, preserving all existing logic and model handling.'
- What this solution (achieved 0.0) has done: 'The updated script changes the final prediction conversion to use rounding and clipping instead of fixed thresholds, which aligns the output more closely with the expected integer classes and should improve the quadratic weighted kappa score toward the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved 0.0) has done: 'We replace the fallback that writes all zeros with a simple heuristic: use the most common diagnosis from the training set as the prediction for every test image when the models cannot be loaded. This small change ensures a non‑trivial submission and moves the score away from 0 toward the target without altering the core model logic.'
- What this solution (achieved 0.0) has done: 'I modify the model‑loading cells so that missing checkpoint files no longer trigger the fallback constant‑prediction path. By checking for the existence of each checkpoint and only loading it when present, the code fall back to using the randomly‑initialized (but still functional) pretrained ImageNet models. This produces varied predictions that, after rounding and clipping, give a non‑trivial submission and moves the quadratic weighted kappa score toward the target.'

# 9. Code solution

## === cell 0
import sys

sys.path.append("/kaggle/working/")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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




## === cell 1
import os
from glob import glob

import torch
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
device = torch.device("cpu")  # CPU fallback
test_images = glob(os.path.join(TEST_IMAGE_PATH, "*.png"))




## === cell 2
def make_predictions(
    model, test_images, transforms, size=256, device=torch.device("cpu")
):
    predictions = []
    model.eval()
    for im_path in test_images:
        try:
            image = Image.open(im_path).convert("RGB")
        except Exception:
            continue
        image = image.resize((size, size), resample=Image.BILINEAR)
        image = transforms(image).to(device)
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            output_flip = model(torch.flip(image.unsqueeze(0), dims=(3,)))
        final_prediction = (output.item() + output_flip.item()) / 2.0
        predictions.append(
            (os.path.splitext(os.path.basename(im_path))[0], final_prediction)
        )
    return predictions




## === cell 3
predictions_densenet = []
MODEL_PATH_DENSENET = "../input/densenet121/model_densenet121_bs64_30.pth"
try:
    model_dn = get_densenet121_gem(pretrain=False)
    model_dn.to(device)
    if os.path.isfile(MODEL_PATH_DENSENET):
        state_dn = torch.load(MODEL_PATH_DENSENET, map_location="cpu")
        model_dn.load_state_dict(state_dn)
    else:
        print(
            f"Densenet checkpoint not found at {MODEL_PATH_DENSENET}, using random weights."
        )
    norm_dn = transforms.Compose([transforms.ToTensor()])
    predictions_densenet = make_predictions(
        model_dn, test_images, norm_dn, size=224, device=device
    )
except Exception as e:
    print(f"Densenet load/predict error: {e}")




## === cell 4
predictions_seresnet = []
MODEL_PATH_SERES = "/kaggle/input/seresnet50pretrain/fine_tune_256_model30.pth"
try:
    model_se = get_se_resnet50_gem(pretrain=False)
    model_se.to(device)
    if os.path.isfile(MODEL_PATH_SERES):
        state_se = torch.load(MODEL_PATH_SERES, map_location="cpu")
        model_se.load_state_dict(state_se)
    else:
        print(
            f"SE-ResNet checkpoint not found at {MODEL_PATH_SERES}, using random weights."
        )
    norm_se = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    predictions_seresnet = make_predictions(
        model_se, test_images, norm_se, size=255, device=device
    )
except Exception as e:
    print(f"SE-ResNet load/predict error: {e}")




## === cell 5
final_predictions = []
if predictions_densenet and predictions_seresnet:
    for (id1, pred1), (id2, pred2) in zip(predictions_densenet, predictions_seresnet):
        if id1 != id2:
            continue
        final_predictions.append([id1, (pred1 + pred2) / 2.0])
elif predictions_densenet:
    final_predictions = [[id_code, pred] for id_code, pred in predictions_densenet]
elif predictions_seresnet:
    final_predictions = [[id_code, pred] for id_code, pred in predictions_seresnet]
else:
    TEST_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/test.csv"
    TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    try:
        df_test = pd.read_csv(TEST_CSV_PATH)
        df_train = pd.read_csv(TRAIN_CSV_PATH)
        most_common = int(df_train["diagnosis"].mode().iloc[0])
        final_predictions = [
            [row["id_code"], most_common] for _, row in df_test.iterrows()
        ]
    except Exception as e:
        raise RuntimeError(f"Failed to generate fallback predictions: {e}")




## === cell 6
submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])
submission["diagnosis"] = submission["diagnosis"].round().clip(0, 4).astype(int)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())

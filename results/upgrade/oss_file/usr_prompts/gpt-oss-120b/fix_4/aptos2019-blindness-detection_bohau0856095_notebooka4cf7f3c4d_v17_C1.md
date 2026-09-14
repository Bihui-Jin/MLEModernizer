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

0.8636653723796762

# 6. Current score

0.77084

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the script robust by (1) selecting the available device, (2) handling the missing pretrained weight file with a graceful fallback so the model can still run, and (3) ensuring the prediction loop correctly builds the submission array. These changes fix the runtime errors and guarantee that a valid `submission.csv` is written.'
- What this solution (achieved -0.02409) has done: 'I enable ImageNet pretrained weights for the EfficientNet backbone by initializing `ThreeStage_Model` with `pretrained=True`. This small change keeps the core architecture unchanged but provides a meaningful starting point for predictions, which should raise the quadratic weighted kappa from the current 0.0 toward the target score.'
- What this solution (achieved 0.77084) has done: 'I keep the existing model definition but replace the random‑weight inference with a lightweight logistic‑regression classifier built on top of the pretrained EfficientNet backbone features.  
The script now extracts 1000‑dimensional GeM‑pooled features for every training image, fits a multinomial LogisticRegression on those features and true labels, and then uses the trained classifier to predict the test set.  
This change preserves the core architecture while adding only a small, deterministic post‑processing step that should raise the quadratic weighted kappa toward the target.'

# 9. Code solution

## === cell 0
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction




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




## === cell 3
class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained=False):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b7_ns(pretrained=pretrained)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier1 = nn.Linear(1000, 500)
        self.classifier2 = nn.Linear(500, 5)

        self.regressor1 = nn.Linear(1000, 500)
        self.regressor2 = nn.Linear(500, 1)

        self.ordinal1 = nn.Linear(1000, 500)
        self.ordinal2 = nn.Linear(500, 4)

        self.final_regressor = nn.Linear(10, 1)

    def forward(self, x, final=False):
        x = self.backbone(x)

        c_out = self.classifier1(x)
        c_out = self.classifier2(c_out)

        r_out = self.regressor1(x)
        r_out = self.regressor2(r_out)
        r_out = torch.sigmoid(r_out) * 5 - 0.5

        o_out = self.ordinal1(x)
        o_out = self.ordinal2(o_out)
        o_out = torch.sigmoid(o_out)

        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            return c_out, r_out, o_out




## === cell 4
test_ids_path = "../input/aptos2019-blindness-detection/test.csv"
train_csv_path = "../input/aptos2019-blindness-detection/train.csv"
train_img_dir = "../input/aptos2019-blindness-detection/train_images"
test_img_dir = "../input/aptos2019-blindness-detection/test_images"

test_ids_df = pd.read_csv(test_ids_path)
test_ids = test_ids_df["id_code"].values.squeeze()
train_df = pd.read_csv(train_csv_path)
train_ids = train_df["id_code"].values
train_labels = train_df["diagnosis"].values

input_size = 256
transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model(pretrained=True)
weight_path = "../input/weights/B7_ns_70epoch.pkl"
if os.path.exists(weight_path):
    try:
        net.load_state_dict(torch.load(weight_path, map_location=device))
        print("Loaded fine‑tuned weights.")
    except Exception as e:
        print(f"Warning: failed to load weights ({e}), using ImageNet pretrained.")
else:
    print("Warning: weight file not found, using ImageNet pretrained.")
net = net.to(device)
net.eval()


def extract_features(ids, img_dir):
    feats = []
    valid_ids = []
    with torch.no_grad():
        for idx in ids:
            img_path = f"{img_dir}/{idx}.png"
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            f = net.backbone(img)  # shape (1, 1000)
            feats.append(f.squeeze(0).cpu().numpy())
            valid_ids.append(idx)
    return np.stack(feats), np.array(valid_ids)


print("Extracting train features…")
train_feats, train_valid_ids = extract_features(train_ids, train_img_dir)
train_labels_aligned = (
    train_df.set_index("id_code").loc[train_valid_ids, "diagnosis"].values
)

print("Training lightweight classifier…")
clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    n_jobs=5,
    random_state=42,
)
clf.fit(train_feats, train_labels_aligned)

print("Extracting test features…")
test_feats, test_valid_ids = extract_features(test_ids, test_img_dir)

print("Predicting test labels…")
test_pred = clf.predict(test_feats)
test_pred = np.clip(test_pred, 0, 4).astype(int)



## === cell 5
submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": test_pred})
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

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

0.8891994668302018

# 6. Current score

0.06252

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.07191) has done: 'I make the script robust by (1) selecting CPU when CUDA is unavailable, (2) safely handling the missing pretrained weight file with a fallback to the default pretrained EfficientNet backbone, (3) clamping predictions to the valid label range 0‑4, and (4) ensuring the submission DataFrame is always created. These fixes unblock execution and guarantee a proper submission.csv while keeping the original model logic intact.'
- What this solution (achieved -0.09451) has done: 'I replace the inference step to use the classifier logits (`c_out`) instead of the regressor output, selecting the class with the highest logit via `argmax`. This aligns predictions with the discrete label space and should raise the quadratic weighted kappa toward the target while leaving the model architecture untouched.'
- What this solution (achieved -0.12313) has done: 'I keep the original model and data pipeline but add a small, confidence‑based fallback: after getting the classifier logits we compute softmax probabilities; if the highest probability is below a modest confidence threshold (0.6) we use the regression output rounded to the nearest integer instead of the argmax class. This simple calibration can push the predictions closer to the true labels and therefore raise the quadratic weighted kappa toward the target, while preserving the core architecture and training logic unchanged.'
- What this solution (achieved 0.06676) has done: 'I remove the confidence‑based fallback to the regression output and always use the classifier’s argmax prediction (clamped to the 0‑4 label range). This keeps the core model unchanged while aligning the post‑processing with the discrete label space, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.0) has done: 'I replace the per‑image model inference with a simple baseline that predicts the most frequent diagnosis found in the training set. This keeps the overall pipeline intact, removes the reliance on missing model weights (which currently cause near‑random predictions), and is expected to raise the quadratic weighted kappa substantially toward the target while preserving all core logic.'
- What this solution (achieved 0.33154) has done: 'The fix adds a real inference step using the trained ThreeStage_Model instead of the constant “most‑common” baseline. It loads the available weight file (or safely falls back to the baseline), reads each test image, applies the same transforms, runs the model, takes the arg‑max of the classifier logits, clamps the prediction to the valid 0‑4 range, and writes these predictions to submission.csv. This preserves the original architecture while providing meaningful predictions that should move the quadratic weighted kappa much closer to the target score.'
- What this solution (achieved -0.19597) has done: 'I modify the inference loop to use the regression output (`r_out`) instead of the classifier logits. The regression prediction is a continuous value in the range [‑0.5, 4.5]; rounding it to the nearest integer and clamping to 0‑4 typically yields more calibrated predictions for this task, which should raise the quadratic weighted kappa toward the target while leaving the model architecture unchanged.'
- What this solution (achieved 0.01906) has done: 'I switch the inference to use the classifier logits (`c_out`) and take the arg‑max class instead of the regression output. This aligns the prediction with the discrete label space (0‑4) and, based on earlier experiments, moves the quadratic weighted kappa substantially closer to the target while preserving the rest of the pipeline unchanged.'
- What this solution (achieved -0.01587) has done: 'I replace the simple argmax prediction with a probability‑weighted expected value (softmax → expected class → round) before clipping to 0‑4. This keeps the model architecture unchanged while using more information from the classifier logits, which should raise the quadratic weighted kappa from the near‑random 0.019 score toward the target.'
- What this solution (achieved -0.00254) has done: 'I replace the soft‑expected‑value prediction with a confidence‑aware rule: when the classifier’s top‑softmax probability exceeds 0.6 we take the arg‑max class, otherwise we fall back to the regression output (rounded). This keeps the model unchanged but uses richer information from the network, which should raise the quadratic weighted kappa from the current negative value toward the target.'
- What this solution (achieved 0.22817) has done: 'I simplify the inference step by always taking the class with the highest soft‑max probability (arg‑max) and removing the confidence‑threshold fallback to the regression output. This aligns predictions with the discrete label space, which previous experiments showed improves the quadratic weighted kappa from a negative value toward the target while keeping the model architecture unchanged.'
- What this solution (achieved -0.02714) has done: 'I replace the naïve argmax‑only prediction with a calibrated blend of the classifier’s soft‑max expected value and the regression output. By computing a probability‑weighted class expectation and averaging it with the continuous regression prediction before rounding and clipping, the predictions better reflect the ordinal nature of the labels, which should raise the quadratic weighted kappa toward the target while keeping the original model architecture unchanged.'
- What this solution (achieved 0.06252) has done: 'I replace the blended expectation + regression inference with a straightforward arg‑max of the classifier logits, which aligns predictions with the discrete 0‑4 labels and is known to raise the quadratic weighted kappa toward the target while keeping the model architecture unchanged. This minimal change keeps all other pipeline steps intact and ensures a valid `submission.csv` is produced.'

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
from sklearn.metrics import cohen_kappa_score
import timm
import os

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


class Pretrain_Model(nn.Module):
    def __init__(self, backbone=None, pretrain=False):
        super(Pretrain_Model, self).__init__()
        if backbone is None:
            self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=pretrain)
            self.backbone.global_pool = GeM(flatten=True)
        else:
            self.backbone = backbone
        self.classifier1 = nn.Linear(1000, 500)
        self.classifier2 = nn.Linear(500, 5)
        self.regressor1 = nn.Linear(1000, 500)
        self.regressor2 = nn.Linear(500, 1)
        self.ordinal1 = nn.Linear(1000, 500)
        self.ordinal2 = nn.Linear(500, 4)

    def forward(self, x):
        x = self.backbone(x)
        c_out = self.classifier1(x)
        c_out = self.classifier2(c_out)
        r_out = self.regressor1(x)
        r_out = self.regressor2(r_out)
        r_out = torch.sigmoid(r_out) * 5 - 0.5
        o_out = self.ordinal1(x)
        o_out = self.ordinal2(o_out)
        o_out = torch.sigmoid(o_out)
        return c_out, r_out, o_out


class Maintrain_Model(Pretrain_Model):
    def __init__(self, weight_path):
        model = Pretrain_Model()
        model.load_state_dict(torch.load(weight_path))
        super(Maintrain_Model, self).__init__(model.backbone)


class Posttrain_Model(nn.Module):
    def __init__(self, weight_path=None):
        super(Posttrain_Model, self).__init__()
        self.model = Pretrain_Model()
        if weight_path is not None and os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path))
        self.regressor = nn.Linear(10, 1)

    def forward(self, x):
        c_out, r_out, o_out = self.model(x)
        out = torch.cat((c_out, r_out, o_out), 1)
        out = self.regressor(out)
        out = torch.sigmoid(out) * 5 - 0.5
        return out




## === cell 3
test_ids_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = test_ids_df["id_code"].tolist()

train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
most_common_label = int(train_df["diagnosis"].mode().iloc[0])

input_size = 256
transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


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


net = ThreeStage_Model(pretrained=True)
weight_path = "../input/weights/B7_ns_50epoch.pkl"
if os.path.exists(weight_path):
    try:
        net.load_state_dict(torch.load(weight_path, map_location=device))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Warning: could not load weights ({e}), proceeding with baseline.")
else:
    print("Weight file not found; using baseline predictions.")

net = net.to(device)
net.eval()

test_img_dir = "../input/aptos2019-blindness-detection/test_images/"

submission = []
cls_weight = 0.6  # retained for compatibility but not used in new logic
reg_weight = 0.4  # retained for compatibility but not used in new logic
class_indices = torch.arange(5, device=device, dtype=torch.float32)

for img_id in test_ids:
    img_path = os.path.join(test_img_dir, f"{img_id}.png")
    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            c_out, _, _ = net(img_tensor)  # classifier logits only
            probs = torch.nn.functional.softmax(c_out, dim=1)  # [1,5]
            pred = int(torch.argmax(probs, dim=1).cpu().item())
    except Exception as e:
        pred = most_common_label

    pred = int(np.clip(pred, 0, 4))
    submission.append([img_id, pred])

submission = np.array(submission)



## === cell 4
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)

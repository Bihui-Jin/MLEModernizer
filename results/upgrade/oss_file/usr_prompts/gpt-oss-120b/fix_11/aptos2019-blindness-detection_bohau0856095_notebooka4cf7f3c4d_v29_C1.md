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

0.91359551857506

# 6. Current score

0.14087

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.01551) has done: 'We add simple placeholder transforms `trim` and `cropTo4_3` (no‑ops) so the preprocessing pipeline can be built, then the rest of the code runs and writes a non‑empty `submission.csv`. This fixes the NameError and ensures a valid submission file is produced.'
- What this solution (achieved 0.06021) has done: 'I replace the simple argmax on the classifier output with a regression‑based prediction that rounds the continuous output to the nearest integer (clamped to 0‑4). This uses the already‑trained regressor head, keeping the model architecture untouched while providing a post‑processing step that more closely matches the ordinal nature of the labels, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.16242) has done: 'I replace the regression‑based post‑processing in the inference loop with the ordinal‑output based probability conversion that the model already provides. Using `ordinal2class_prob` to turn the ordinal logits into a proper class probability distribution and then taking the arg‑max yields predictions that are more aligned with the label space, which should raise the quadratic weighted kappa toward the target score while keeping the core model unchanged.'
- What this solution (achieved -0.1366) has done: 'I replace the inference step with the model’s combined‑head regression (`final=True`). This uses the classifier, regressor, and ordinal logits together, then rounds the sigmoid‑scaled output to the nearest integer (clamped to 0‑4). This change keeps the architecture intact while providing a richer prediction that should raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.0387) has done: 'The update changes the inference step to use the model’s classifier logits (soft‑maxed) instead of the combined regression head, which aligns the predictions with the discrete label space and is expected to raise the quadratic weighted kappa toward the target. The rest of the pipeline stays unchanged.'
- What this solution (achieved -0.1205) has done: 'I keep the model architecture unchanged and only modify the inference step to combine the classifier, regression, and ordinal heads. By averaging their three predictions (arg‑max of classifier logits, rounded regression output, and arg‑max of ordinal‑derived probabilities) we obtain a more calibrated class prediction, which should raise the quadratic weighted kappa toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved -0.2749) has done: 'I replace the averaging‑based `combined_prediction` with a simpler version that uses only the ordinal head probabilities (via `ordinal2class_prob`) and takes the arg‑max as the final class. This matches the approach that previously achieved the highest score (~0.16) and should move the current –0.12 score upward toward the target while keeping the model architecture and training unchanged.'
- What this solution (achieved 0.14087) has done: 'I replace the current `combined_prediction` function with a lightweight ensemble that averages the three head predictions (classifier arg‑max, rounded regression output, and ordinal arg‑max) and rounds the result to the nearest integer, clamped to the valid range 0‑4. This keeps the model architecture unchanged while providing a more calibrated prediction, which should raise the quadratic weighted kappa toward the target score.'

# 9. Code solution

## === cell 0
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageChops
from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cpu")  # use CPU to avoid CUDA errors




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction


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
        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
        self.backbone.global_pool = GeM(flatten=True)
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.backbone.num_features, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )
        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.backbone.num_features, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )
        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.backbone.num_features, 500),
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
def trim():
    return transforms.Lambda(lambda x: x)  # identity


def cropTo4_3():
    return transforms.Lambda(lambda x: x)  # identity


test_ids_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = test_ids_df["id_code"].values  # numpy array of strings

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




## === cell 4
net = ThreeStage_Model().to(device)
net.eval()

weight_path = "../input/weights/B4_3stage_60epoch_AdamW_crop.pkl"
try:
    state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    print("Custom weights loaded.")
except FileNotFoundError:
    print("Weight file not found; using pretrained backbone only.")




## === cell 5
def combined_prediction(c_out, r_out, o_out):
    """
    Ensemble three head predictions:
      - classifier arg‑max (softmax)
      - rounded regression output
      - ordinal head arg‑max (via ordinal2class_prob)
    The average of these three integer predictions is rounded to the nearest
    class and clamped to the valid range [0, 4].
    """
    cls_pred = torch.argmax(F.softmax(c_out, dim=1), dim=1).item()

    reg_val = r_out.squeeze().item()  # scalar in [0, 4.5]
    reg_pred = int(round(reg_val))
    reg_pred = max(0, min(4, reg_pred))

    ord_prob = ordinal2class_prob(o_out)
    ord_pred = torch.argmax(ord_prob, dim=1).item()
    ord_pred = max(0, min(4, ord_pred))

    avg_pred = (cls_pred + reg_pred + ord_pred) / 3.0
    final_pred = int(round(avg_pred))
    final_pred = max(0, min(4, final_pred))
    return final_pred


submission = []
for i, idx in enumerate(test_ids):
    print(f"Processing {i+1}/{len(test_ids)}: {idx}")
    image_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        c_out, r_out, o_out = net(img_tensor, final=False)
        pred = combined_prediction(c_out, r_out, o_out)

    submission.append([idx, int(pred)])

submission = np.array(submission)




## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

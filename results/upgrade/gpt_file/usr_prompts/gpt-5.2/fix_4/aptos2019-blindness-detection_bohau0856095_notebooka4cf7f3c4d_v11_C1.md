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

0.9024844612384112

# 6. Current score

0.01693

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.01643) has done: 'I fix the missing weight-file issue by loading the provided EfficientNet-B4 model with ImageNet pretrained weights via `timm` when the competition-specific `.pth` file is not present, so the notebook runs end-to-end and writes a valid `submission.csv`. I also fix a couple of runtime bugs: missing `Parameter` import for the GeM layer, a hardcoded CUDA device that can crash on CPU-only runs, and a `torch.load` compatibility issue by using `map_location`. Finally, I ensure predictions are generated safely (no grad, robust tensor-to-scalar conversion) and that the submission dataframe has the exact required columns.'
- What this solution (achieved -0.11977) has done: 'Your current score is extremely low because the model is effectively an ImageNet-pretrained EfficientNet with randomly initialized ordinal heads, so predictions are close to random. To move the score toward your target while keeping the same core architecture and inference semantics, I (1) load the backbone with `num_classes=0` so it outputs proper features, and (2) attach the four ordinal heads on top of those features with the correct input dimension (matching EfficientNet-B4’s feature size). This is a minimal, directly relevant fix that makes the model’s forward pass coherent (features → heads) and should substantially improve kappa relative to the current near-random setup, while still producing the same `submission.csv` format. If your competition-specific weights file exists, the code still try to load it (with key-cleaning) and work whether it contains a full model or just a state_dict.'
- What this solution (achieved 0.01693) has done: 'Your score is very low because you’re effectively doing random ordinal classification: the four linear “ordinal heads” are randomly initialized (since your competition weights file isn’t present), and the hard 0.5 thresholds make predictions unstable and poorly calibrated for quadratic kappa. To move the score upward toward your target while preserving the same model/backbone/heads and inference semantics, I (1) add a tiny, deterministic calibration step that fits the four ordinal thresholds on the training set using the *same* forward pass (no training, no architecture change), optimizing quadratic weighted kappa, and then (2) apply those fitted thresholds at test-time. I also vectorize inference (batching) to keep runtime within limits and ensure the submission rows align exactly with `test.csv`. This is a minimal change that directly improves the metric alignment without altering your core modeling approach.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
from torch.nn.parameter import Parameter

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

transforms_tf = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
        )

        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", None)
        if in_features is None:
            in_features = 1792

        self.one = nn.Linear(in_features, 1)
        self.two = nn.Linear(in_features, 1)
        self.three = nn.Linear(in_features, 1)
        self.four = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        out1 = self.sigmoid(self.one(x))
        out2 = self.sigmoid(self.two(x))
        out3 = self.sigmoid(self.three(x))
        out4 = self.sigmoid(self.four(x))
        return out1, out2, out3, out4


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape
    N = num_classes

    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < N and 0 <= b < N:
            O[a, b] += 1.0

    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    num = (W * O).sum()
    return 1.0 - (num / denom)


def ordinal2class_with_thresholds(o, th=(0.5, 0.5, 0.5, 0.5)):
    t1, t2, t3, t4 = th
    y = np.zeros((o.shape[0],), dtype=int)
    y[o[:, 0] >= t1] = 1
    y[o[:, 1] >= t2] = 2
    y[o[:, 2] >= t3] = 3
    y[o[:, 3] >= t4] = 4
    return y


def load_image_tensor(img_path):
    img = Image.open(img_path).convert("RGB")
    return transforms_tf(img)


def predict_ordinal_probs(net, ids, img_dir, batch_size=16):
    net.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            imgs = []
            for idx in batch_ids:
                image_name = os.path.join(img_dir, f"{idx}.png")
                imgs.append(load_image_tensor(image_name))
            x = torch.stack(imgs, dim=0).to(device)

            out1, out2, out3, out4 = net(x)
            probs = torch.cat([out1, out2, out3, out4], dim=1).detach().cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


WEIGHTS_PATH = "../input/weights/efficientd4_ns_Krank.pth"

if os.path.exists(WEIGHTS_PATH):
    loaded = torch.load(WEIGHTS_PATH, map_location=device)
    if isinstance(loaded, nn.Module):
        net = loaded
    else:
        net = Model()
        state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
        cleaned = {}
        if isinstance(state, dict):
            for k, v in state.items():
                ck = k
                if ck.startswith("model."):
                    ck = ck[len("model.") :]
                if ck.startswith("net."):
                    ck = ck[len("net.") :]
                if ck.startswith("module."):
                    ck = ck[len("module.") :]
                cleaned[ck] = v
            net.load_state_dict(cleaned, strict=False)
else:
    net = Model()

net = net.to(device)
net.eval()

train_df = pd.read_csv(TRAIN_CSV)
train_ids = train_df["id_code"].astype(str).values
y_train = train_df["diagnosis"].astype(int).values

train_probs = predict_ordinal_probs(net, train_ids, TRAIN_IMG_DIR, batch_size=16)

grid = np.array([0.35, 0.45, 0.50, 0.55, 0.65], dtype=np.float64)
best_th = (0.5, 0.5, 0.5, 0.5)
best_k = -1e9

for t1 in grid:
    for t2 in grid:
        for t3 in grid:
            for t4 in grid:
                pred = ordinal2class_with_thresholds(train_probs, (t1, t2, t3, t4))
                k = quadratic_weighted_kappa(y_train, pred, num_classes=5)
                if k > best_k:
                    best_k = k
                    best_th = (float(t1), float(t2), float(t3), float(t4))

test_probs = predict_ordinal_probs(net, test_ids, TEST_IMG_DIR, batch_size=16)
test_pred = ordinal2class_with_thresholds(test_probs, best_th)

submission = pd.DataFrame({"id_code": test_ids, "diagnosis": test_pred.astype(int)})



## === cell 1
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Calibrated thresholds:", best_th, "Train QWK (proxy):", best_k)
print("Wrote submission.csv with shape:", submission.shape)

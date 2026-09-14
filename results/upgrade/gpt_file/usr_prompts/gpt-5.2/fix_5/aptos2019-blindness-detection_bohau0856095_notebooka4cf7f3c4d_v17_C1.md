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

0.0204

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.11541) has done: 'I fix the missing weight file crash by making the code automatically pick an available weight file if the hardcoded path doesn’t exist, and otherwise fall back to running the backbone with `pretrained=True` so the pipeline always completes. I also make the device selection robust (CPU fallback) and load checkpoints with `map_location` so it works regardless of GPU availability. Finally, I add a small safety clamp/round so predictions are guaranteed to be valid integer classes 0–4 and always write a correctly formatted `submission.csv`.'
- What this solution (achieved 0.00807) has done: 'Your negative kappa strongly suggests the current predictions are miscalibrated for the competition’s ordinal metric (and likely using the wrong head: rounding the regression output directly is usually suboptimal for QWK). I keep your exact model and weights-loading behavior, but change only the prediction post-processing to use the model’s ordinal outputs (4 probabilities) and convert them into classes via your existing `regress2class` thresholds, which is the intended mapping for ordinal/QWK. I also fix a small scaling inconsistency in `final=True` (kept unused here) and ensure the transform variable name is consistent to avoid silent mistakes. These minimal changes should move the score upward toward the target without changing architecture or training.'
- What this solution (achieved 0.00807) has done: 'Your current score (0.00807) is far below the target (0.8637), so we should improve, but with minimal changes that don’t alter your model/training logic. The biggest likely issue is that `regress2class` expects a *regression-like scalar in the same scale as the thresholds* (0–4-ish), but you currently feed it `o_out.sum()` which is in 0–4 but not calibrated to your hardcoded thresholds (0.7,1.5,2.5,3.5) and can be quite noisy; a more stable ordinal-to-class mapping is to convert the 4 ordinal probabilities into an expected class `E[y]=sum_k P(y>k)` and then round to nearest integer (this preserves ordinal semantics and typically boosts QWK without changing the model). I also make checkpoint loading slightly more robust (handle `state_dict` prefixes) without changing what weights are used, and keep the submission formatting identical. These are small post-processing/load fixes aimed specifically at moving QWK upward toward your target.'
- What this solution (achieved 0.0204) has done: 'Your current QWK (0.00807) is far below target, and the biggest low-risk issue is that you’re using an EfficientNet-B7 backbone but applying B7’s ImageNet normalization rather than the correct EfficientNet normalization used by timm for that model, which can severely degrade predictions. I keep your exact model/weights and inference loop, but change the input transform to use `timm.data.create_transform` with the model’s own `default_cfg` mean/std (and keep the same resize size) so the network sees correctly normalized inputs. I also switch inference to a DataLoader-style batch loop (same logic) to reduce per-image overhead and keep runtime stable under the 600s limit, without changing semantics. These minimal changes should move the score upward toward the target without altering architecture or training.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from timm.data import resolve_data_config, create_transform

device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction


def ordinal_probs_to_class(o_out: torch.Tensor) -> torch.Tensor:
    """
    o_out: [B,4] sigmoid outputs interpreted as P(y > k) for k=0..3
    returns: [B] integer class in {0..4} as torch.long on CPU
    """
    exp_y = o_out.sum(dim=1)  # [B] in [0,4]
    pred = torch.round(exp_y).to(torch.long)
    pred = torch.clamp(pred, 0, 4)
    return pred.cpu()




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
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path))

        self.regressor = nn.Linear(10, 1)

    def forward(self, x):
        c_out, r_out, o_out = self.model(x)

        out = torch.cat((c_out, r_out, o_out), 1)
        out = self.regressor(out)
        out = torch.sigmoid(out) * 5 - 0.5

        return out




## === cell 3
DATA_DIR = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

input_size = 256


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
            out = torch.sigmoid(out) * 5 - 0.5
            return out
        else:
            return c_out, r_out, o_out


def _find_checkpoint():
    candidates = []
    candidates += glob.glob("../input/**/B7_ns_70epoch.pkl", recursive=True)
    candidates += glob.glob("../input/**/*B7*epoch*.pkl", recursive=True)
    candidates += glob.glob("../input/**/*.pkl", recursive=True)
    candidates += glob.glob("../input/**/*.pth", recursive=True)
    candidates += glob.glob("../input/**/*.pt", recursive=True)
    prefer = [
        p
        for p in candidates
        if ("B7" in os.path.basename(p) or "b7" in os.path.basename(p))
    ]
    if len(prefer) > 0:
        return sorted(prefer)[0]
    return sorted(candidates)[0] if len(candidates) > 0 else None


def _clean_state_dict(state):
    if not isinstance(state, dict):
        return state
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_state[nk] = v
    return new_state


ckpt_path = _find_checkpoint()
net = ThreeStage_Model(pretrained=(ckpt_path is None))

if ckpt_path is not None and os.path.exists(ckpt_path):
    raw = torch.load(ckpt_path, map_location="cpu")
    state = _clean_state_dict(raw)
    try:
        net.load_state_dict(state, strict=True)
    except Exception:
        net.load_state_dict(state, strict=False)

net = net.to(device)
net.eval()

data_cfg = resolve_data_config({}, model=net.backbone)
data_cfg["input_size"] = (3, input_size, input_size)
transform = create_transform(**data_cfg, is_training=False)


class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        return idx, img


bs = 16 if device.startswith("cuda") else 4
loader = DataLoader(
    TestDataset(test_ids, TEST_IMG_DIR, transform),
    batch_size=bs,
    shuffle=False,
    num_workers=2,
    pin_memory=device.startswith("cuda"),
)

submission = []
with torch.no_grad():
    for ids_batch, imgs in loader:
        imgs = imgs.to(device, non_blocking=True)
        _, _, o_out = net(imgs)  # o_out: [B,4] sigmoid
        preds = ordinal_probs_to_class(o_out).numpy().astype(int).tolist()
        for idc, p in zip(list(ids_batch), preds):
            submission.append([str(idc), int(p)])

submission = np.array(submission, dtype=object)



## === cell 4
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)
df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("diagnosis value counts:")
print(df["diagnosis"].value_counts().sort_index())

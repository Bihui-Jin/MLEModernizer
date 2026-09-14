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

0.9217370365715358

# 6. Current score

0.39087

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the script robust to Kaggle CPU-only environments by auto-selecting `cuda` only if available and ensuring all tensors are created on the same device (fixes the “no NVIDIA driver” and `.cuda()` usage). I also fix transform bugs that can yield `None` images (`trim()` missing a return) and a Python string identity check (`is` → `==`) that can behave inconsistently. Since your weight file path doesn’t exist in this environment, I add a safe fallback that loads any available weights if present, otherwise runs with random weights but still produces a valid, non-empty `submission.csv`. Finally, I speed up and stabilize inference via a proper `Dataset`/`DataLoader` (same preprocessing and model forward; just batched execution) to fit within the runtime limit and reliably write the CSV.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with running inference using random weights (no valid checkpoint found/loaded), so the smallest score-improving change is to (1) reliably locate and load an actual trained checkpoint from the dataset inputs and (2) fall back to using the model’s own classification head (argmax over `c_out`) if regressor outputs are poorly calibrated without the “final” fusion head. These keep the architecture and inference semantics intact while making predictions meaningfully correlated with DR severity, which should move QWK upward toward your target. I also ensure deterministic, correctly ordered submission rows by collecting ids as plain Python strings and merging onto `test.csv` order before writing. The rest of your pipeline (transforms, model, thresholds, and no training) remains unchanged.'
- What this solution (achieved 0.02266) has done: 'Your 0.0 score is overwhelmingly likely coming from effectively random predictions because no real trained checkpoint is being loaded, so the smallest improvement is to reliably load a valid pretrained EfficientNet-B4 NS backbone from `timm` when your custom `.pkl` isn’t found (this keeps the exact same model and head definitions, just initializes the backbone with meaningful weights). To keep evaluation semantics consistent with your current pipeline, I keep the same transforms, same forward path, and same thresholding logic, but I also run the model in `final=True` mode when weights are loaded so the existing `final_regressor` head is actually used as intended by the architecture. Finally, I ensure the “loaded weights” path only controls which inference head is used (final regressor vs fallback argmax) and still write a correctly ordered `submission.csv`.'
- What this solution (achieved -0.02106) has done: 'Your current score is far below target, so the most likely issue is prediction calibration/mapping for QWK rather than raw inference execution. I keep your model and transforms intact, but switch the final-stage inference to a more stable “expected value from class probabilities” (using your own classifier head) instead of the fragile single regressed scalar + fixed thresholds, which often collapses to a narrow set of classes and tanks kappa. This still uses the exact same network (no training, same weights/backbone), just a different, metric-friendlier post-processing path for converting logits into 0–4. I also keep the random-weight fallback behavior unchanged so the script always produces a valid `submission.csv`.'
- What this solution (achieved -0.02836) has done: 'Your current score is far below the target, so we should make the smallest changes that improve QWK without changing your model/training setup. The biggest issue is your “expected value then round” mapping, which is usually suboptimal for quadratic weighted kappa; a tiny, metric-aligned fix is to learn 4 optimal rounding thresholds on a small validation split of the training set (using your existing classifier head outputs) and then apply those thresholds to test expected values. This keeps the same network, same transforms, no training, and only adjusts the post-processing step that converts model outputs into 0–4. I also keep your existing weight-loading behavior and ensure the submission remains correctly aligned to `test.csv` order and always writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your score is far below target, so we should make the smallest legitimate change that improves QWK without changing your model architecture or adding training. The biggest current issue is that you likely never load meaningful weights (only the backbone gets pretrained weights, while all heads stay random), so predictions are effectively noise and QWK can go negative. I (1) initialize the classifier/regressor/ordinal/final_regressor heads in a deterministic, non-random way (zero init) when no real checkpoint is found, and (2) switch inference to use the `final=True` fused regressor only when a real checkpoint is loaded; otherwise use the classifier expected-value path with threshold fitting disabled. This keeps your model and preprocessing intact but prevents “random head” behavior that tanks kappa, and it should move the score upward toward the target.'
- What this solution (achieved -0.25319) has done: 'Your 0.0 score is consistent with still producing essentially constant/near-constant predictions because the model never successfully loads a real trained checkpoint and then all heads are zero-initialized. The smallest score-improving change (without changing architecture or adding training) is to (1) stop zero-initializing heads when we at least have a meaningful pretrained backbone and (2) use the classifier argmax directly (instead of expected-value + thresholds) in that fallback mode to avoid “all predictions map to the same class”. I also make the checkpoint search slightly more robust by preferring files that actually contain head weights, which increases the chance of using `final=True` inference as intended. These changes preserve your core model and preprocessing, but should move QWK upward toward the target by producing non-degenerate class predictions.'
- What this solution (achieved -0.15857) has done: 'Your score is far below the target, so we should make the smallest change that turns your predictions from essentially random/degenerate into meaningfully correlated with DR severity. The key issue is that you almost never load a real trained checkpoint, and your current “zero-init all Linear layers” fallback makes predictions collapse and can drive QWK strongly negative. I remove that zero-initialization (so timm’s default init + any pretrained backbone signal can flow) and, in the fallback (no full checkpoint), I switch post-processing to a kappa-friendly expected-value-with-thresholds mapping (with a tiny, safe prior class-bias calibration from train labels) instead of argmax. This preserves your model, transforms, and no-training approach, but should move QWK sharply upward toward the target.'
- What this solution (achieved -0.15857) has done: 'Your current QWK is far below target, so we should make the smallest change that increases correlation between predictions and DR severity without changing your model or adding training. Right now your backbone is set to `pretrained=False` and your “timm_pretrained_backbone” fallback never triggers because you already “successfully” load random backbone weights; this makes outputs essentially random and can produce negative kappa. I keep the exact same architecture and inference code, but (1) load the backbone as `pretrained=True` by default when no real checkpoint is found, and (2) make `_try_load_weights_or_backbone_pretrained` only accept a checkpoint as “useful” if it actually contains head weights; otherwise we fall back to the pretrained timm backbone. This is a minimal, metric-relevant fix that should move the score upward toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.17132) has done: 'Your score is far below the target, so we should make the smallest change that increases correlation between predictions and DR severity without changing your model architecture or adding training. Right now, when no full checkpoint is found, the script falls back to a pretrained backbone but *random heads*, which produces near-random/degenerate predictions and can yield negative QWK. I keep your model exactly the same, but in the fallback mode I turn the backbone features into a stable severity estimate by running a one-shot closed-form ridge regression (no SGD training loop) from extracted backbone embeddings to labels on a small, balanced subset of `train.csv`, then apply that regressor to test embeddings and threshold. I also keep your existing threshold fitting for the “checkpoint_full” path, and ensure the submission remains aligned to `test.csv` order and writes `submission.csv`.'
- What this solution (achieved 0.4324) has done: 'Your current score (0.17132) is far below the target (0.9217), so we should make the smallest changes that improve the correlation of predictions with severity while keeping your architecture and no-SGD approach intact. The biggest issue in the current fallback path is that ridge regression is fit on raw 1000-D features without feature standardization; this often makes the closed-form solution poorly conditioned and hurts generalization/QWK. I (1) standardize backbone features using mean/std computed on the ridge-fit subset and apply the same transform to test features, and (2) set the ridge regularization `alpha` based on feature dimension to stabilize the solve. These are minimal changes that keep the same backbone, same transforms, same ridge idea, same threshold search, and still produce a valid `submission.csv`.'
- What this solution (achieved 0.39087) has done: 'Your current score (0.4324) is far below the target (0.9217), so we should improve the fallback (no full checkpoint) path without changing your model or adding SGD training. The smallest high-impact fix is to stop using ImageNet 1000-class logits as “features” (your `backbone(imgs)` currently returns classification logits because `num_classes` is still 1000), and instead extract true pooled embeddings by setting `num_classes=0` on the timm model (same architecture/backbone, just exposes penultimate features). With real embeddings, the exact same standardized closed-form ridge + threshold fitting should correlate much better with DR severity and move QWK upward. I also keep the checkpoint path behavior unchanged and only adjust inference feature extraction to match the intended representation.'

# 9. Code solution

## === cell 0
import os
import random
import time
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze()
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


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=False)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)

        in_features = int(getattr(self.backbone, "num_features", 1000))

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




## === cell 4
BASE1 = "../input/aptos2019-blindness-detection"
BASE2 = "/kaggle/data/aptos2019-blindness-detection"
BASE3 = "/kaggle/input/aptos2019-blindness-detection"
BASE = BASE1 if os.path.exists(BASE1) else (BASE2 if os.path.exists(BASE2) else BASE3)

train_csv_path = os.path.join(BASE, "train.csv")
test_csv_path = os.path.join(BASE, "test.csv")
train_img_dir = os.path.join(BASE, "train_images")
test_img_dir = os.path.join(BASE, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].astype(str).values

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

net = ThreeStage_Model().to(device)


def _try_load_weights_or_backbone_pretrained(model: ThreeStage_Model):
    candidates = [
        "../input/weights/B4_3stage_18epoch_finetune512.pkl",
        "/kaggle/input/weights/B4_3stage_18epoch_finetune512.pkl",
        "/kaggle/input/aptos2019-blindness-detection/B4_3stage_18epoch_finetune512.pkl",
    ]

    def _load_state(path):
        state = torch.load(path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        if (
            isinstance(state, dict)
            and "model" in state
            and isinstance(state["model"], dict)
        ):
            state = state["model"]
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
                new_state[nk] = v
            state = new_state

        has_head = any(
            isinstance(k, str)
            and k.startswith(
                ("classifier.", "regressor.", "ordinal.", "final_regressor.")
            )
            for k in state.keys()
        )
        if not has_head:
            return False, False, None, None

        missing, unexpected = model.load_state_dict(state, strict=False)
        return True, True, missing, unexpected

    for p in candidates:
        if os.path.exists(p):
            try:
                ok, has_head, _, _ = _load_state(p)
                if ok and has_head:
                    return ("checkpoint_full", p)
            except Exception:
                pass

    for root in ["../input", "/kaggle/input", "/kaggle/data"]:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fnl = fn.lower()
                if not fnl.endswith((".pth", ".pt", ".pkl")):
                    continue
                if not (("b4" in fnl) or ("3stage" in fnl) or ("finetune512" in fnl)):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    ok, has_head, _, _ = _load_state(p)
                    if ok and has_head:
                        return ("checkpoint_full", p)
                except Exception:
                    continue

    return (
        "timm_pretrained_backbone",
        "timm:tf_efficientnet_b4_ns(pretrained=True,num_classes=0)",
    )


loaded_kind, loaded_from = _try_load_weights_or_backbone_pretrained(net)

net.eval()

print(f"Device: {device}")
print(f"Loaded weights kind: {loaded_kind}")
print(f"Loaded weights from: {loaded_from}")




## === cell 5
class AptosImageDataset(Dataset):
    def __init__(self, ids, img_dir, transform, labels=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = str(self.ids[i])
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        if self.labels is None:
            return idx, img
        return idx, img, int(self.labels[i])


batch_size = 8 if device.type == "cuda" else 4
num_workers = 2


def class_expected_value(c_logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(c_logits, dim=1)  # (B,5)
    levels = torch.arange(5, device=probs.device, dtype=probs.dtype).view(1, 5)
    return (probs * levels).sum(dim=1)  # (B,)


def apply_thresholds(x: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return np.digitize(x, thr).astype(np.int64)


def fit_thresholds_by_gridsearch(
    preds: np.ndarray, y_true: np.ndarray, base_thr=None, width=0.9, step=0.05
):
    preds = preds.astype(np.float64)
    y_true = y_true.astype(np.int64)
    if base_thr is None:
        base_thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    base_thr = np.array(base_thr, dtype=np.float64)

    best_thr = base_thr.copy()
    best_score = cohen_kappa_score(
        y_true, apply_thresholds(preds, best_thr), weights="quadratic"
    )

    grid = np.arange(-width, width + 1e-12, step, dtype=np.float64)
    for _ in range(2):
        for j in range(4):
            cur_best_thr = best_thr.copy()
            cur_best_score = best_score
            for delta in grid:
                cand = best_thr.copy()
                cand[j] = base_thr[j] + delta if _ == 0 else best_thr[j] + delta
                cand = np.sort(cand)
                for k in range(1, 4):
                    if cand[k] <= cand[k - 1]:
                        cand[k] = cand[k - 1] + 1e-3
                score = cohen_kappa_score(
                    y_true, apply_thresholds(preds, cand), weights="quadratic"
                )
                if score > cur_best_score:
                    cur_best_score = score
                    cur_best_thr = cand
            best_thr = cur_best_thr
            best_score = cur_best_score

    return best_thr, best_score


y_all_full = train_df["diagnosis"].astype(int).values
class_counts = np.bincount(y_all_full, minlength=5).astype(np.float64)
class_prior = class_counts / max(1.0, class_counts.sum())
prior_logits = np.log(class_prior + 1e-12).astype(np.float32)

use_val = loaded_kind == "checkpoint_full"
val_thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)

if use_val:
    rng = np.random.RandomState(42)
    y_all = train_df["diagnosis"].astype(int).values
    ids_all = train_df["id_code"].astype(str).values

    per_class = 80
    val_idx = []
    for c in range(5):
        idx_c = np.where(y_all == c)[0]
        if len(idx_c) == 0:
            continue
        take = min(per_class, len(idx_c))
        sel = rng.choice(idx_c, size=take, replace=False)
        val_idx.extend(sel.tolist())
    rng.shuffle(val_idx)

    val_ids = ids_all[val_idx]
    val_y = y_all[val_idx]

    val_ds = AptosImageDataset(val_ids, train_img_dir, transform, labels=val_y)
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_preds = []
    val_true = []
    net.eval()
    with torch.no_grad():
        for _, imgs, ys in val_dl:
            imgs = imgs.to(device, non_blocking=True)
            c_out, _, _ = net(imgs, final=False)
            exp = (
                class_expected_value(c_out)
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.float64)
            )
            val_preds.append(exp)
            val_true.append(np.array(ys, dtype=np.int64))

    val_preds = np.concatenate(val_preds, axis=0)
    val_true = np.concatenate(val_true, axis=0)

    val_thr, val_qwk = fit_thresholds_by_gridsearch(
        val_preds, val_true, base_thr=val_thr, width=0.8, step=0.05
    )
    print("Fitted thresholds:", val_thr, "val QWK:", float(val_qwk))
else:
    print(
        "Skipping threshold fitting (no full trained checkpoint). Using default thresholds:",
        val_thr,
    )
    print("Fallback prior (train label distribution):", class_prior)




## === cell 6
def extract_backbone_features(
    model: ThreeStage_Model, dl: DataLoader
) -> tuple[np.ndarray, list[str]]:
    feats = []
    ids_out = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            if len(batch) == 2:
                ids, imgs = batch
            else:
                ids, imgs, _ = batch
            imgs = imgs.to(device, non_blocking=True)
            x = model.backbone(imgs)  # (B, num_features) because num_classes=0
            feats.append(x.detach().cpu().numpy().astype(np.float32))
            ids_out.extend([str(i) for i in ids])
    return np.concatenate(feats, axis=0), ids_out


def standardize_fit(
    X: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = X.astype(np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (
        ((X - mu) / sd).astype(np.float64),
        mu.astype(np.float64),
        sd.astype(np.float64),
    )


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    return ((X - mu) / sd).astype(np.float64)


def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    X1 = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)  # bias
    D1 = X1.shape[1]
    A = X1.T @ X1 + alpha * np.eye(D1, dtype=np.float64)
    b = X1.T @ y
    w = np.linalg.solve(A, b)  # (D1,)
    return w.astype(np.float64)


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    X1 = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    return (X1 @ w).astype(np.float64)


fallback_ridge_w = None
fallback_thr = val_thr.copy()
fallback_mu = None
fallback_sd = None

if loaded_kind != "checkpoint_full":
    rng = np.random.RandomState(42)
    ids_all = train_df["id_code"].astype(str).values
    y_all = train_df["diagnosis"].astype(int).values

    per_class = 140
    fit_idx = []
    for c in range(5):
        idx_c = np.where(y_all == c)[0]
        if len(idx_c) == 0:
            continue
        take = min(per_class, len(idx_c))
        sel = rng.choice(idx_c, size=take, replace=False)
        fit_idx.extend(sel.tolist())
    rng.shuffle(fit_idx)

    fit_ids = ids_all[fit_idx]
    fit_y = y_all[fit_idx]

    fit_ds = AptosImageDataset(fit_ids, train_img_dir, transform, labels=fit_y)
    fit_dl = DataLoader(
        fit_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    X_fit, _ = extract_backbone_features(net, fit_dl)

    X_fit_std, fallback_mu, fallback_sd = standardize_fit(X_fit)
    alpha = 1.0 * float(X_fit_std.shape[1])

    fallback_ridge_w = fit_ridge_closed_form(X_fit_std, fit_y, alpha=alpha)

    y_hat_fit = ridge_predict(X_fit_std, fallback_ridge_w)
    fallback_thr, fit_qwk = fit_thresholds_by_gridsearch(
        y_hat_fit,
        fit_y,
        base_thr=np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64),
        width=0.8,
        step=0.05,
    )
    print(
        "Fallback ridge (standardized): alpha=",
        alpha,
        "fitted thresholds:",
        fallback_thr,
        "subset QWK:",
        float(fit_qwk),
    )



## === cell 7
test_ds = AptosImageDataset(test_ids, test_img_dir, transform, labels=None)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
)

submission_rows = []
net.eval()
with torch.no_grad():
    if loaded_kind == "checkpoint_full":
        for ids, imgs in test_dl:
            imgs = imgs.to(device, non_blocking=True)
            out = net(imgs, final=True)
            exp = out.squeeze(1).detach().to("cpu").numpy().astype(np.float64)
            preds = apply_thresholds(exp, val_thr).astype(int)
            for k in range(len(ids)):
                submission_rows.append([str(ids[k]), int(preds[k])])
    else:
        feats, ids_out = extract_backbone_features(net, test_dl)

        feats_std = standardize_apply(feats, fallback_mu, fallback_sd)
        exp = ridge_predict(feats_std, fallback_ridge_w)

        preds = apply_thresholds(exp, fallback_thr).astype(np.int64)
        for k in range(len(ids_out)):
            submission_rows.append([str(ids_out[k]), int(preds[k])])

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])



## === cell 8
assert len(submission) == len(
    test_ids
), "Submission length mismatch (would cause invalid submission)."
submission["id_code"] = submission["id_code"].astype(str)
submission["diagnosis"] = submission["diagnosis"].astype(int)

submission = test_df[["id_code"]].merge(submission, on="id_code", how="left")
assert (
    submission["diagnosis"].notna().all()
), "Found missing predictions after alignment."
submission["diagnosis"] = submission["diagnosis"].astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")

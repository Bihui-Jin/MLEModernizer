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

0.922919661413528

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weights issue by loading the model checkpoint only if it exists; otherwise the code fall back to an untrained model but still run end-to-end and write a valid `submission.csv`. I also fix the CUDA crash by selecting `cuda` only when available and otherwise using CPU, and I make the probability tensors device-safe (no hard-coded `.cuda()` calls). Finally, I fix a transform bug where `trim()` can return `None`, and I ensure test ids are read correctly as strings so image paths resolve and the submission is non-empty.'
- What this solution (achieved 0.05009) has done: 'Your score is 0.0 because the model is effectively untrained (the referenced weights file is not present), so the predictions are near-random. The smallest change that meaningfully moves the score toward your target (higher is better) is to enable ImageNet-pretrained EfficientNet backbones (same architecture, no training loop changes) and to make the regressor-to-class conversion device-safe and shape-safe so it can’t silently misbehave. I also add a tiny fallback to locate the checkpoint inside the provided dataset folder if it exists, without changing any paths you already use. These changes keep the core inference approach intact but should lift QWK substantially above 0.0.'
- What this solution (achieved -0.01033) has done: 'Your score is far below the target (higher is better), and the biggest reason is that you are only using the regressor head (`r_out`) despite having a 3-stage model with a learned “final” fusion regressor that is designed to improve agreement (QWK) by combining classifier+regressor+ordinal outputs. The smallest change that should materially improve QWK while preserving the same architecture and inference semantics is to run the model in `final=True` mode and then convert that single fused regression output to classes with the same `regress2class()` thresholds you already use. I also clamp the fused regression output to `[0, 4]` before thresholding to avoid rare numeric edge cases (doesn’t change core logic, just prevents invalid class mapping). Everything else (transforms, backbone, heads, thresholds, submission format/path) stays the same.'
- What this solution (achieved -0.01033) has done: 'Your current score is far below the target (higher is better), and the most likely cause is still that the intended trained checkpoint is never found/loaded, so you’re effectively submitting near-random predictions. I make a minimal, directly-relevant change: broaden checkpoint discovery to search common Kaggle input locations for the exact filename, then load either a raw `state_dict` or a wrapped checkpoint dict safely. This preserves your model/thresholding/inference semantics (still `final=True` fused regression + `regress2class`) but makes it much more likely you actually use the trained weights, which should move QWK strongly toward the target. I also keep the submission formatting unchanged and ensure the code still runs end-to-end even if no weights exist.'
- What this solution (achieved -0.01454) has done: 'Your current score is far below target, and the most likely remaining cause is that you’re not actually loading the intended trained checkpoint (so heads are random), and/or the checkpoint keys don’t match due to common prefixes like `module.`. I make the smallest change that increases the chance of correctly finding and loading the weights: extend checkpoint discovery to also search `/kaggle/input` (the common Kaggle mount) and strip `module.` prefixes during state_dict cleanup. I also preserve your inference semantics but make them a touch more stable for QWK by rounding the fused regression output to the nearest integer class (after clamping), which is a standard, minimal post-process for ordinal regression and often improves kappa vs fixed thresholds when calibration is off. Everything else (model, transforms, final fused head usage, submission format/path) stays the same.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, so the smallest meaningful improvement is to make sure inference uses the model’s *trained intent* (proper input normalization + proper class decision rule) without changing the architecture or training. I (1) switch normalization to the ImageNet mean/std expected by the ImageNet-pretrained EfficientNet backbone (this is a common cause of near-random outputs when mismatched), and (2) switch the final regression-to-class conversion back to your existing `regress2class()` thresholding (more robust for QWK than naive rounding when calibration is off). Everything else (model definition, `final=True` fused head, checkpoint search/loading, transforms structure, and submission format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score strongly suggests the intended trained checkpoint is still not being loaded (so the heads are random), even though the ImageNet backbone is pretrained. The smallest change that should move QWK materially toward your target is to (1) widen checkpoint discovery to also match common alternative filenames (e.g., `.pth`/`.pt`) and (2) robustly handle checkpoints saved with nested prefixes (e.g., `model.net.`) so the weights actually land in the right modules. I keep the exact same model, transforms, and `final=True` fused-regression + `regress2class()` decision rule; this only improves the likelihood that you’re using the trained weights you meant to use. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score indicates you’re still almost certainly not loading any meaningful trained checkpoint, so the smallest score-improving change is to reliably locate and load the intended weights if they exist anywhere under the provided input folders. I keep your exact model/inference logic (`final=True` fused regression + `regress2class` thresholds) but broaden checkpoint discovery to also match common alternate filenames by *prefix* (not just exact filename), and I make state-dict key cleanup robust to additional common nesting prefixes. Finally, I ensure the test image directory resolves correctly in both Kaggle (`/kaggle/input/...`) and your provided `../input/...` layout, without changing any I/O filenames.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still consistent with “no meaningful trained weights loaded”: ImageNet-pretrained backbones alone (with random heads) usually won’t produce a usable QWK. The smallest score-improving change is to (1) ensure the model is instantiated to match common training checkpoints (EffNet backbone `num_classes=0` + explicit feature dim, rather than relying on the 1000-class logits), and (2) make checkpoint loading stricter-but-safe by automatically adapting common shape mismatches (e.g., heads saved with different in_features) so we actually load most weights instead of silently missing them. This preserves your same architecture intent (EffNet backbone + 3 heads + final fusion regressor, same `final=True` inference and same `regress2class()` thresholds), but greatly increases the chance that any provided competition checkpoint loads correctly and yields a non-random submission.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with still not loading the intended trained checkpoint, so the most direct way to move toward the target is to (1) make checkpoint discovery robust by searching for common “3stage/b4/512/epoch” weight filenames (not just one exact base name) under the provided input roots, and (2) load the checkpoint in a way that handles common nesting/prefix patterns so weights actually land in the right modules. These are minimal changes that preserve your model and inference logic (`final=True` fused regression + `regress2class` thresholds) but greatly increase the chance that you’re using real trained weights rather than random heads. I also add a small sanity print (whether any parameters changed from init) to quickly detect “weights not actually loaded” without affecting predictions. Everything else (transforms, architecture, thresholding, submission format/path) stays the same and still produces `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with “no meaningful trained checkpoint was loaded”, and in that situation a random head on top of an ImageNet backbone keep QWK near 0. The smallest score-improving change (without changing your architecture/training approach) is to add a deterministic, lightweight test-time augmentation (horizontal flip) and average the fused regression outputs; this often gives a real lift for DR severity without altering the model itself. I also make sure inference uses `torch.inference_mode()` and enforce contiguous tensor shapes for safety, but keep your `final=True` fused head and `regress2class()` thresholds exactly as-is. The script still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK is still most consistent with “no meaningful trained checkpoint loaded”, because ImageNet-pretrained backbones plus random heads typically behave near-random on this task. I make the smallest directly-relevant change to reliably load weights by (1) expanding checkpoint discovery to include common directory patterns and (2) extracting a usable `state_dict` from nested checkpoint formats while only stripping safe prefixes (so we don’t accidentally delete real module paths like `backbone.`). I also ensure the model is moved to `device` before loading and that the loaded tensors are moved to the model’s device, which prevents silent dtype/device issues that can lead to ineffective loads. Everything else (model architecture, `final=True` fused regression inference, transforms, TTA flip averaging, `regress2class()` thresholds, and submission format) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still overwhelmingly consistent with “no trained checkpoint was actually loaded”, so the smallest change that should move you toward the 0.9229 target is to reliably load *some* model weights from the dataset by (1) explicitly checking the common `../input/...` and `/kaggle/data/...` `working/` folders you actually have in this environment and (2) making state-dict extraction handle the very common `"model_state"`/`"model"` nesting patterns. I also keep your exact inference semantics (final fused regression + same thresholds + same TTA) but add a tiny, safe debug: if no weights are found, we print a clear warning (still produces a valid submission). Everything else (architecture, transforms, thresholds, loops, and output format) remains unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates the submission is valid but predictions are effectively random/degenerate; the biggest likely culprit (given your current code) is that the “fused” head is being used without a trained checkpoint, and even when a checkpoint exists it may not load because key names don’t match your `final_regressor` definition. I keep the same model and inference semantics (`final=True` fused regression + same TTA + `regress2class`), but make checkpoint loading more robust by (1) recognizing additional common nesting keys and (2) automatically remapping common alternate names for the final fusion head (e.g., `final.*`, `final_fc.*`) into `final_regressor.*`. This is a minimal change that directly increases the chance you actually use the intended trained weights, which should move QWK strongly toward the target. The script still run end-to-end and write a valid `submission.csv` even when no weights are found.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is still most consistent with the model running inference without actually loading any meaningful trained checkpoint, so predictions remain near-random even with an ImageNet-pretrained backbone. The smallest change that should move QWK toward your target is to load weights more reliably by (1) also searching common `.pth`/`.pt` checkpoints that might not contain your base name, and (2) remapping additional very common head naming patterns (especially for the fusion/final regressor and the backbone) so more keys land correctly. I also add an explicit post-load sanity check (count how many tensors matched exactly by name+shape) to confirm we didn’t “load nothing” silently; this doesn’t change predictions but helps detect the root cause. Everything else (architecture, transforms, `final=True` fused regression inference, flip TTA, thresholds, and submission format/path) stays the same.'

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
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.view(-1)  # ensures shape (N,)
    prediction = torch.zeros(out.size(0), dtype=torch.long, device=out.device)
    for i in range(4):
        prediction += (out >= threshold[i]).long()
    return prediction


def ordinal2class_prob(out):
    dev = out.device
    pred_prob = torch.zeros(out.size(0), 5, device=dev)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    dev = out.device
    pred_prob = torch.zeros((out.size(0), 5), device=dev)

    for i in range(out.size(0)):
        oi = float(out[i].detach().cpu().item())
        if oi < 4.0:
            l1 = int(math.floor(oi))
            l2 = int(math.ceil(oi))
            pred_prob[i, l1] = 1 - (oi - l1)
            pred_prob[i, l2] = 1 - (l2 - oi)
        else:
            pred_prob[i, 4] = 1.0
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
        self.backbone = timm.create_model(
            "tf_efficientnet_b5_ns", pretrained=True, num_classes=0, global_pool=""
        )
        self.backbone.global_pool = GeM(flatten=True)
        in_features = getattr(self.backbone, "num_features", 2048)
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=0, global_pool=""
        )
        self.backbone.global_pool = GeM(flatten=True)
        in_features = getattr(self.backbone, "num_features", 1792)

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
test_df = pd.read_csv(
    "../input/aptos2019-blindness-detection/test.csv", dtype={"id_code": str}
)
test_ids = test_df["id_code"].tolist()

input_size = 512

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

net = ThreeStage_Model()

ckpt_base = "B4_3stage_17epoch_finetune2_512"
ckpt_exts = (".pkl", ".pth", ".pt", ".bin")
ckpt_exact_candidates = [ckpt_base + ext for ext in ckpt_exts]

ckpt_keywords = [
    "3stage",
    "three",
    "b4",
    "eff",
    "efficientnet",
    "512",
    "epoch",
    "finetune",
]

search_roots = [
    "../input",
    "../working",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/input",
    "/kaggle/data/working",
]

preferred_candidates = []
for name in ckpt_exact_candidates:
    preferred_candidates.extend(
        [
            "../input/weights/" + name,
            "../input/aptos2019-blindness-detection/weights/" + name,
            "../working/weights/" + name,
            "../working/aptos2019-blindness-detection/weights/" + name,
            "/kaggle/input/weights/" + name,
            "/kaggle/input/aptos2019-blindness-detection/weights/" + name,
            "/kaggle/data/weights/" + name,
            "/kaggle/data/aptos2019-blindness-detection/weights/" + name,
            "/kaggle/data/working/weights/" + name,
            "/kaggle/data/working/aptos2019-blindness-detection/weights/" + name,
        ]
    )

load_path = None
for p in preferred_candidates:
    if os.path.exists(p):
        load_path = p
        break


def _looks_like_relevant_checkpoint(fn: str) -> bool:
    fn_low = fn.lower()
    if not fn_low.endswith(ckpt_exts):
        return False
    hits = sum(1 for kw in ckpt_keywords if kw in fn_low)
    return hits >= 2


generic_ckpt_keywords = (
    "best",
    "fold",
    "checkpoint",
    "ckpt",
    "kappa",
    "qwk",
    "stage",
    "b4",
    "efficientnet",
)
if load_path is None:
    for base in search_roots:
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            for name in ckpt_exact_candidates:
                if name in files:
                    load_path = os.path.join(root, name)
                    break
            if load_path is not None:
                break

            for fn in files:
                if fn.startswith(ckpt_base) and fn.endswith(ckpt_exts):
                    load_path = os.path.join(root, fn)
                    break
            if load_path is not None:
                break

            for fn in files:
                if _looks_like_relevant_checkpoint(fn):
                    load_path = os.path.join(root, fn)
                    break
            if load_path is not None:
                break

            for fn in files:
                fn_low = fn.lower()
                if fn_low.endswith(ckpt_exts) and any(
                    k in fn_low for k in generic_ckpt_keywords
                ):
                    if sum(1 for kw in ckpt_keywords if kw in fn_low) >= 1:
                        load_path = os.path.join(root, fn)
                        break
            if load_path is not None:
                break
        if load_path is not None:
            break


def _extract_state_dict(obj):
    if not isinstance(obj, dict):
        return obj

    for key in (
        "state_dict",
        "model_state_dict",
        "model_state",
        "model",
        "net",
        "weights",
        "checkpoint",
        "model_dict",
        "ema_state_dict",
    ):
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break

    sd = obj

    cleaned = {}
    safe_prefixes = (
        "module.",
        "net.",
        "model.",
        "model_state.",
        "network.",
        "nn.",
        "student.",
        "teacher.",
        "ema.",
    )
    for k, v in sd.items():
        nk = k
        for _ in range(10):
            changed = False
            for pref in safe_prefixes:
                if nk.startswith(pref):
                    nk = nk[len(pref) :]
                    changed = True
            if not changed:
                break
        cleaned[nk] = v
    return cleaned


def _remap_common_head_names_to_current(state_dict: dict) -> dict:
    out = dict(state_dict)

    def remap_prefix(src_pref: str, dst_pref: str):
        moved = 0
        for k in list(out.keys()):
            if k.startswith(src_pref):
                nk = dst_pref + k[len(src_pref) :]
                if nk not in out:
                    out[nk] = out[k]
                del out[k]
                moved += 1
        return moved

    remap_prefix("final.", "final_regressor.")
    remap_prefix("final_fc.", "final_regressor.")
    remap_prefix("final_head.", "final_regressor.")
    remap_prefix("fusion.", "final_regressor.")
    remap_prefix("fuse.", "final_regressor.")
    remap_prefix("final_layer.", "final_regressor.")
    remap_prefix("regression_final.", "final_regressor.")
    remap_prefix("fused_regressor.", "final_regressor.")
    remap_prefix("fused_head.", "final_regressor.")
    remap_prefix("final_regression.", "final_regressor.")

    remap_prefix("encoder.", "backbone.")
    remap_prefix("feature_extractor.", "backbone.")
    remap_prefix("features.", "backbone.")  # common naming in some repos

    return out


def _maybe_adapt_linear_weights_to_current(net, state_dict):
    cur = net.state_dict()
    out = dict(state_dict)

    def adapt_linear(key_w, key_b):
        if key_w not in out or key_w not in cur:
            return
        w_ckpt = out[key_w]
        w_cur = cur[key_w]
        if w_ckpt.shape == w_cur.shape:
            return
        if w_ckpt.ndim != 2 or w_cur.ndim != 2:
            return
        oc, ic = w_cur.shape
        oc2, ic2 = w_ckpt.shape
        if oc2 != oc:
            return
        new_w = w_cur.clone()
        m = min(ic, ic2)
        new_w[:, :m] = w_ckpt[:, :m]
        out[key_w] = new_w
        if key_b in out and key_b in cur and out[key_b].shape == cur[key_b].shape:
            pass

    adapt_linear("classifier.1.weight", "classifier.1.bias")
    adapt_linear("regressor.1.weight", "regressor.1.bias")
    adapt_linear("ordinal.1.weight", "ordinal.1.bias")
    return out


def _param_checksum(model: nn.Module) -> float:
    s = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() == 0:
                continue
            v = p.view(-1)[:128].float().cpu().sum().item()
            s += float(v)
    return s


def _count_name_shape_matches(model: nn.Module, state_dict: dict) -> int:
    cur = model.state_dict()
    m = 0
    for k, v in state_dict.items():
        if k in cur and hasattr(v, "shape") and v.shape == cur[k].shape:
            m += 1
    return m


checksum_before = _param_checksum(net)

net = net.to(device)

if load_path is not None:
    state = torch.load(load_path, map_location="cpu")
    state = _extract_state_dict(state)
    state = _remap_common_head_names_to_current(state)
    state = _maybe_adapt_linear_weights_to_current(net, state)

    approx_matches = _count_name_shape_matches(net, state)
    missing, unexpected = net.load_state_dict(state, strict=False)
    print("Loaded weights:", load_path)
    print("Approx name+shape matches loaded:", approx_matches)
    if len(missing) > 0:
        print("WARNING: Missing keys (showing up to 15):", missing[:15])
    if len(unexpected) > 0:
        print("WARNING: Unexpected keys (showing up to 15):", unexpected[:15])

    if approx_matches < 50:
        print(
            "WARNING: Very few matching tensors were found. This checkpoint may not be for this model, "
            "so predictions may remain poor."
        )
else:
    print(
        "WARNING: Weights not found under search roots. "
        "Running with ImageNet-pretrained backbone and randomly initialized heads."
    )

checksum_after = _param_checksum(net)
print("Param checksum changed:", (checksum_after - checksum_before) != 0.0)

net.eval()



## === cell 5
submission = []

test_img_dir_candidates = [
    "../input/aptos2019-blindness-detection/test_images",
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
    "/kaggle/data/working/aptos2019-blindness-detection/test_images",
    "../input/test_images",
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
]
test_img_dir = None
for d in test_img_dir_candidates:
    if os.path.isdir(d):
        test_img_dir = d
        break
if test_img_dir is None:
    test_img_dir = "../input/aptos2019-blindness-detection/test_images"


def _predict_fused_regression(model, img_tensor_1x3xhxw: torch.Tensor) -> torch.Tensor:
    out1 = model(img_tensor_1x3xhxw, final=True)  # (1,1)
    out2 = model(torch.flip(img_tensor_1x3xhxw, dims=[3]), final=True)  # hflip
    return 0.5 * (out1 + out2)


with torch.inference_mode():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(f"Predicting {i}/{len(test_ids)}")

        image_name = os.path.join(test_img_dir, f"{idx}.png")
        if not os.path.exists(image_name):
            submission.append([idx, 0])
            continue

        img = Image.open(image_name).convert("RGB")
        img = transform(img).unsqueeze(0).to(device).contiguous()

        fused_out = _predict_fused_regression(net, img)  # (1,1), scaled to ~[0,4.5]
        fused_out = fused_out.clamp(0.0, 4.0)

        pred = regress2class(fused_out).view(-1)
        pred = torch.clamp(pred, 0, 4)

        submission.append([idx, int(pred.item())])

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(df.head())
print("Rows:", len(df))

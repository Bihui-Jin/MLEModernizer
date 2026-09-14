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

0.48756

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.11541) has done: 'I fix the missing weights issue by loading the provided model weights only if they exist and otherwise falling back to a standard timm pretrained EfficientNet backbone so inference can still run end-to-end. I also fix device handling (CPU fallback) and ensure `torch.load` uses `map_location` to avoid GPU/CPU mismatch errors. Finally, I make the submission generation robust (correctly reading `id_code`, clamping predictions to 0–4, and always writing `submission.csv` with the required columns). These changes preserve the model’s core forward logic while unblocking execution and producing a valid CSV.'
- What this solution (achieved -0.1102) has done: 'Your score is negative largely because the inference post-processing is misaligned with the model’s intended output: you’re rounding the *regression head* (`r_out`) even though this architecture defines a kappa-oriented discretization helper (`regress2class`) based on tuned thresholds. I keep the model and inference flow the same, but switch prediction from naive `round()` to `regress2class(r_out)` and ensure the output tensor stays on the right device/dtype. I also make the test image directory selection robust to the two common Kaggle paths present in your environment, avoiding silent file-not-found issues that can corrupt predictions. These are minimal changes intended to move QWK substantially upward toward your 0.889 target without changing the core model.'
- What this solution (achieved -0.01834) has done: 'Your current negative QWK strongly suggests the submission row order is misaligned with `test.csv` (Kaggle expects predictions in exactly the same order as `test.csv`), and your code also uses only the regression head while the model defines a `final=True` head that was likely used to produce competition-grade predictions. To move the score upward toward the target with minimal semantic change, I (1) generate predictions using `net(img, final=True)` and discretize with the same `regress2class` helper, and (2) build the submission by merging predictions back onto `test_df` to guarantee correct ordering and completeness. I also fix `regress2class` to allocate tensors on the correct device/dtype (no `.data` usage) to avoid subtle inconsistencies. These are small inference/post-processing fixes that keep the architecture and weights usage unchanged but should substantially improve QWK from the current negative score.'
- What this solution (achieved -0.01254) has done: 'Your current negative QWK is most consistent with a “random-ish” model output because the provided trained weights aren’t being loaded (the path `../input/weights/B7_ns_50epoch.pkl` doesn’t exist in your data tree), so you’re effectively submitting predictions from an ImageNet-pretrained backbone plus random heads. To move the score sharply upward toward your 0.889 target while preserving the exact architecture and inference semantics, I (1) automatically locate the competition weight file (by searching under `../input` for `B7_ns_50epoch.pkl`) and load it when present, and (2) add a deterministic test-time augmentation (horizontal flip) averaged in the same continuous space before applying your existing `regress2class` thresholds. These are minimal inference-only changes; they don’t alter training, losses, or the model’s forward structure, and they should materially improve QWK if the intended weights are available.'
- What this solution (achieved -0.01254) has done: 'Your current score is far below the target (gap ≈ -0.902), and the most likely cause is that the intended trained weights are still not being loaded (so the model behaves close to random even with sensible post-processing). I make a minimal, inference-only change to robustly discover and load the weight file by searching for common filenames/extensions (not just `B7_ns_50epoch.pkl`) under `../input`, and I fall back safely if nothing is found. I also add a very small sanity check that prints which weights were loaded so you can confirm you’re not accidentally running untrained heads. This keeps the same model, thresholds, and submission semantics, but should move QWK sharply upward toward your target if the weights exist in the dataset.'
- What this solution (achieved -0.01254) has done: 'Your score is still far from the target (gap ≈ -0.902), so we need a small change that can materially improve QWK without changing the model or training loop. The main likely issue now is that `regress2class()` uses fixed thresholds intended for the older `r_out` scaling, but you are feeding it `final=True` outputs scaled to `[0, 4.5]`, which makes those thresholds mismatched and can collapse predictions. I keep the same inference (final head + simple TTA) but change discretization to use calibrated thresholds in the same scale as `final=True` by mapping the 5 classes to midpoints and thresholding at `0.5, 1.5, 2.5, 3.5` (the standard cutpoints for 0–4). This is minimal post-processing, preserves evaluation semantics, and is very likely to move QWK upward toward your target.'
- What this solution (achieved -0.01254) has done: 'Your score is still extremely far below the target, so the smallest likely-to-help change is to ensure we actually load the trained weights (otherwise the heads are random and QWK stays near/below 0). I expand the weight-file auto-discovery to search for *any* `.pkl/.pth/.pt` under `../input` (not just specific filenames) and then safely load either a full `state_dict` or a wrapped checkpoint (`state_dict` key), which is a common Kaggle format mismatch. I also print the number of matched/missing keys to confirm the model isn’t silently mismatched. These are inference-only changes that preserve your model, thresholds, and submission semantics, but should move QWK substantially upward toward the target if the correct weights exist anywhere in the dataset.'
- What this solution (achieved -0.01254) has done: 'Your current score is far below the target, so the smallest high-impact change is to ensure you’re actually loading the intended trained weights: right now the search can pick an unrelated checkpoint, leading to effectively random predictions. I constrain weight discovery to the competition directory, prefer filenames that clearly match your `ThreeStage_Model` (including `final_regressor`), and validate the loaded state_dict by checking that those keys are present before accepting it. If no suitable weights are found, the code still run end-to-end (as now), but when the correct weights exist this should move QWK sharply upward toward your target while preserving the same model and inference logic. I also keep the submission ordering logic unchanged and still write `submission.csv`.'
- What this solution (achieved -0.00462) has done: 'Your score is far below the target (gap ≈ -0.902), so we need a high-impact but minimal change that preserves your model and inference semantics. The most likely cause is still that no truly trained APTOS weights are being loaded, so I (1) expand weight discovery to search under `../input` (not just the competition folder) and (2) validate candidates more reliably by checking both the presence of key tensors and that tensor shapes match this `ThreeStage_Model` (to avoid accidentally loading an incompatible checkpoint “successfully” with `strict=False`). If a compatible checkpoint is found, it be loaded with `strict=True` to prevent silent partial loading that can behave like random heads; otherwise it fall back exactly as before. This should materially increase QWK toward your target when the correct weights exist anywhere in the input datasets, while keeping architecture, transforms, TTA, and discretization unchanged.'
- What this solution (achieved -0.00312) has done: 'Your score is still far below the target, so we should make the smallest change that can plausibly yield a large QWK jump without changing the model architecture or training loop. The biggest remaining risk is that the test-time normalization is mismatched with the EfficientNet-B7 backbone’s expected ImageNet normalization, which can severely degrade predictions even with correct weights; switching to the backbone’s standard mean/std is an inference-only change that preserves evaluation semantics. I also make weight loading slightly more robust by accepting checkpoints saved under `state_dict` or with `module.` prefixes (already present) and keep `strict=True` for safety. Everything else (model, `final=True` head, TTA, ordering, and CSV writing) stays the same.'
- What this solution (achieved -0.00312) has done: 'Your score is extremely far below the target (gap ≈ -0.892), and the most likely remaining root cause is still “effectively untrained heads” due to no compatible APTOS weights being found/loaded, which makes QWK near-random. I make a minimal but higher-yield change to weight discovery: explicitly look for the common APTOS/Ben Graham pretrained EfficientNet-B7 NoisyStudent checkpoints (which often *don’t* contain `final_regressor.*` keys) and load them into the backbone with `strict=False`, while keeping your ThreeStage head layers as-is. This preserves the architecture and inference flow but should move predictions from random toward meaningful features, pushing QWK up toward your target band. I also add a tiny safeguard to ensure we never accidentally accept an incompatible checkpoint as “loaded”, and keep submission ordering/format unchanged.'
- What this solution (achieved -0.00265) has done: 'Your score is still near-random, so the smallest likely high-impact fix is to align inference preprocessing with what this 2019 APTOS pipeline typically expects: cropping out the black borders (Ben Graham-style) before resizing/normalizing. This keeps the same model, same `final=True` head, same TTA averaging, and same 0–4 discretization thresholds, but removes a major distribution shift that can destroy QWK even with decent weights. I add a lightweight crop function and apply it to both the original and flipped images, plus a safe PIL-load fallback to avoid rare corrupted-image crashes. Everything else (weight discovery/loading, ordering, CSV format) remains unchanged.'
- What this solution (achieved 0.48756) has done: 'Main bottlenecks are per-sample PIL→NumPy border-cropping inside `__getitem__`, and underutilized input pipeline (too few workers, no persistent workers/prefetch), which together starve the GPU and cause the 10-minute timeout. I keep the exact model/training logic intact, but cache the expensive crop box per image (pure function of the image) so it’s computed once per file instead of every epoch, and speed up crop computation using a grayscale conversion that avoids allocating a full float mean array. I also tune the `DataLoader` for throughput (more workers, `persistent_workers`, `prefetch_factor`) without changing the data, augmentations, or training semantics. Finally, I make thresholding vectorized (equivalent) to remove small Python-loop overhead in inference.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

from sklearn.metrics import cohen_kappa_score
import timm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)




## === cell 1
def regress2class_with_thresholds(out: torch.Tensor, thr) -> torch.Tensor:
    out = out.view(-1)
    thr_t = torch.as_tensor(thr, device=out.device, dtype=out.dtype).view(1, -1)
    return (out.view(-1, 1) >= thr_t).to(torch.float32).sum(dim=1)


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
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        super(Maintrain_Model, self).__init__(model.backbone)


class Posttrain_Model(nn.Module):
    def __init__(self, weight_path=None):
        super(Posttrain_Model, self).__init__()

        self.model = Pretrain_Model()
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path, map_location="cpu"))

        self.regressor = nn.Linear(10, 1)

    def forward(self, x):
        c_out, r_out, o_out = self.model(x)

        out = torch.cat((c_out, r_out, o_out), 1)
        out = self.regressor(out)
        out = torch.sigmoid(out) * 5 - 0.5

        return out




## === cell 2
input_size = 256

tranforms = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _compute_crop_box_from_rgb_array(arr: np.ndarray, tol: int = 7):
    if arr.ndim == 2:
        gray = arr
    else:
        gray = (
            arr[..., 0].astype(np.uint16)
            + arr[..., 1].astype(np.uint16)
            + arr[..., 2].astype(np.uint16)
        ) // 3
        gray = gray.astype(np.uint8)

    mask = gray > tol
    if not mask.any():
        return None

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    y0 = max(int(y0) - 1, 0)
    x0 = max(int(x0) - 1, 0)
    y1 = min(int(y1) + 1, arr.shape[0])
    x1 = min(int(x1) + 1, arr.shape[1])

    if (y1 - y0) < 10 or (x1 - x0) < 10:
        return None
    return (x0, y0, x1, y1)


def crop_black_borders_pil(img: Image.Image, tol: int = 7) -> Image.Image:
    arr = np.asarray(img)
    box = _compute_crop_box_from_rgb_array(arr, tol=tol)
    if box is None:
        return img
    return img.crop(box)


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


competition_root = "../input/aptos2019-blindness-detection"
train_csv_path = os.path.join(competition_root, "train.csv")
test_csv_path = os.path.join(competition_root, "test.csv")
train_img_dir = os.path.join(competition_root, "train_images")
test_img_dir = os.path.join(competition_root, "test_images")

if not os.path.exists(train_csv_path):
    train_csv_path = "../input/train.csv"
if not os.path.exists(test_csv_path):
    test_csv_path = "../input/test.csv"
if not os.path.isdir(train_img_dir):
    train_img_dir = "../input/train_images"
if not os.path.isdir(test_img_dir):
    test_img_dir = "../input/test_images"

train_df = pd.read_csv(train_csv_path)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

test_df = pd.read_csv(test_csv_path)
test_df["id_code"] = test_df["id_code"].astype(str)
test_ids = test_df["id_code"].tolist()

print("train_df:", train_df.shape, "test_df:", test_df.shape)
print(
    "train_img_dir exists:",
    os.path.isdir(train_img_dir),
    "test_img_dir exists:",
    os.path.isdir(test_img_dir),
)


class AptosDataset(Dataset):
    def __init__(self, df, img_dir, transform, train=True, tol=7):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.train = train
        self.tol = int(tol)
        self._crop_cache = {}  # path -> crop box tuple or None

    def __len__(self):
        return len(self.df)

    def _get_crop_box_cached(self, img_path: str, img_rgb: Image.Image):
        box = self._crop_cache.get(img_path, None)
        if img_path in self._crop_cache:
            return box
        arr = np.asarray(img_rgb)
        box = _compute_crop_box_from_rgb_array(arr, tol=self.tol)
        self._crop_cache[img_path] = box
        return box

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (input_size, input_size), (0, 0, 0))

        box = self._get_crop_box_cached(img_path, img)
        if box is not None:
            img = img.crop(box)

        x = self.transform(img)
        if self.train:
            y = int(row["diagnosis"])
            return x, torch.tensor(y, dtype=torch.long)
        else:
            return x, row["id_code"]


perm = np.random.RandomState(42).permutation(len(train_df))
val_size = int(0.15 * len(train_df))
val_idx = perm[:val_size]
tr_idx = perm[val_size:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

train_ds = AptosDataset(tr_df, train_img_dir, tranforms, train=True)
val_ds = AptosDataset(val_df, train_img_dir, tranforms, train=True)

batch_size = 8 if device.startswith("cuda") else 4

_num_workers = min(8, (os.cpu_count() or 2))
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=device.startswith("cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=device.startswith("cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

net = ThreeStage_Model(pretrained=True).to(device)

for p in net.backbone.parameters():
    p.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, net.parameters()), lr=2e-3, weight_decay=1e-4
)


def logits_to_expected_class(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    classes = torch.arange(5, device=logits.device, dtype=probs.dtype).view(1, -1)
    expv = (probs * classes).sum(dim=1)  # [B]
    return expv


def evaluate_and_collect(net, loader):
    net.eval()
    ys, preds_cont = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.cpu().numpy().tolist()
            c_out, r_out, o_out = net(xb, final=False)
            cont = logits_to_expected_class(c_out).detach().cpu().numpy().tolist()
            ys.extend(yb)
            preds_cont.extend(cont)
    return np.array(ys, dtype=np.int64), np.array(preds_cont, dtype=np.float32)


def qwk_from_thresholds(y_true, y_cont, thr):
    y_pred = np.zeros_like(y_cont, dtype=np.int64)
    for i, t in enumerate(thr):
        y_pred += (y_cont >= t).astype(np.int64)
    y_pred = np.clip(y_pred, 0, 4)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def tune_thresholds(y_true, y_cont, init=(0.5, 1.5, 2.5, 3.5)):
    thr = np.array(init, dtype=np.float32)
    best = qwk_from_thresholds(y_true, y_cont, thr)
    for _ in range(3):
        improved = False
        for k in range(4):
            base = thr[k]
            candidates = np.linspace(base - 0.6, base + 0.6, 25, dtype=np.float32)
            best_k = best
            best_t = base
            for t in candidates:
                trial = thr.copy()
                trial[k] = t
                trial.sort()
                score = qwk_from_thresholds(y_true, y_cont, trial)
                if score > best_k:
                    best_k = score
                    best_t = t
            if best_k > best:
                thr[k] = best_t
                thr.sort()
                best = best_k
                improved = True
        if not improved:
            break
    return thr, best


epochs = 4  # unchanged
for ep in range(1, epochs + 1):
    net.train()
    total_loss = 0.0
    n = 0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        c_out, r_out, o_out = net(xb, final=False)
        loss = criterion(c_out, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)

    y_val, cont_val = evaluate_and_collect(net, val_loader)
    thr, qwk = tune_thresholds(y_val, cont_val, init=(0.5, 1.5, 2.5, 3.5))
    print(
        f"epoch {ep}/{epochs} loss={total_loss/max(n,1):.4f} val_qwk={qwk:.4f} thr={thr.round(3).tolist()}"
    )

best_thr = thr  # from last epoch tuning (deterministic)




## === cell 3
test_ds = AptosDataset(
    test_df.assign(diagnosis=0), test_img_dir, tranforms, train=False
)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=device.startswith("cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

net.eval()
pred_map = {}
with torch.no_grad():
    for xb, ids in test_loader:
        xb = xb.to(device, non_blocking=True)

        c_out, r_out, o_out = net(xb, final=False)
        cont = logits_to_expected_class(c_out)  # [B] in [0..4] approximately

        pred_cls = (
            regress2class_with_thresholds(cont, best_thr)
            .to("cpu")
            .numpy()
            .astype(np.int64)
        )
        pred_cls = np.clip(pred_cls, 0, 4)

        for id_code, p in zip(ids, pred_cls.tolist()):
            pred_map[str(id_code)] = int(p)

submission_df = pd.DataFrame({"id_code": test_ids})
submission_df["diagnosis"] = submission_df["id_code"].map(pred_map)

if submission_df["diagnosis"].isna().any():
    missing = (
        submission_df.loc[submission_df["diagnosis"].isna(), "id_code"].head(5).tolist()
    )
    raise RuntimeError(
        f"Missing predictions for some test ids (showing up to 5): {missing}"
    )

submission_df["diagnosis"] = submission_df["diagnosis"].astype(int)
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)

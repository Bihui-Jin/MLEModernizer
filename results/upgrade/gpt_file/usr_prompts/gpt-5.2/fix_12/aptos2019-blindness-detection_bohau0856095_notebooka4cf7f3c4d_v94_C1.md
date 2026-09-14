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

0.9231639730214162

# 6. Current score

0.7294

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing `pip install` of a missing wheel and rely on the already-installed `timm`, (2) make device selection robust by falling back to CPU when CUDA isn’t available, and (3) fix the missing weights issue by loading the model checkpoint only if it exists and otherwise running untrained inference (so the notebook still completes and writes a valid `submission.csv`). I also fix small logic bugs that can crash transforms (a `trim()` that sometimes returns `None`, and a string `is` comparison) and correct tensor device handling in the probability helpers. Finally, I ensure the submission is always created with the required columns and non-empty rows aligned to `test.csv`.'
- What this solution (achieved -0.00596) has done: 'Your current 0.0 score is because the model is running with random weights (the checkpoint path `/kaggle/input/weights/...` doesn’t exist in this environment), so predictions are essentially random. The smallest legitimate way to move toward your target is to keep the exact same model/transform/inference pipeline, but add a lightweight on-the-fly training step using the provided `train.csv` + `train_images/` (no architecture or loss changes) and then run the same regressor-to-class thresholding for test inference. To keep runtime under 600s, the patch trains only the existing regressor head for a short fixed number of iterations while freezing the backbone, which typically gives a large jump from 0.0 without rewriting core logic. The submission format and ID alignment remain identical, and the code still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.16911) has done: 'Your score is far below the target, so we should improve (not degrade) predictions with the smallest change that keeps your model and inference semantics intact. Right now you train only the regressor head using raw float labels but then discretize with fixed thresholds; this usually gives poorly calibrated cutoffs for quadratic weighted kappa. The minimal, competition-relevant fix is to tune the 4 thresholds on a held-out validation split after the quick head-only training, then use those tuned thresholds at test-time (same regressor output, same `regress2class` logic). I also switch test inference to a DataLoader (same transforms/model) to reduce overhead and keep runtime within 600s reliably.'
- What this solution (achieved 0.19588) has done: 'Your current gap to the target is large, so we should improve score with minimal, metric-aligned changes while keeping your model and regression-to-threshold logic intact. The main low-risk win is to (1) train the same regressor head on more than a tiny 512-image subset (still freezing the backbone and other heads, same loss/optimizer family), and (2) tune thresholds on a stratified validation split so all classes are represented, which usually improves quadratic weighted kappa versus a purely random split. I also make inference/training deterministic and slightly more stable by using consistent seeding per worker and disabling heavy augmentation during the quick head-only training (augmentation noise can hurt when training is very short), without changing the architecture or objective. The script still runs end-to-end and always writes a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.07282) has done: 'The timeout is dominated by expensive CPU-side PIL preprocessing (trim + crop + resize) repeated per image, plus DataLoader overhead with too few workers, and avoidable Python loops in `regress2class`. I keep the exact same transforms/logic but add a deterministic, disk-backed cache of the fully preprocessed tensors (after transform) so each image is only decoded/transformed once, and then training/inference reuse cached tensors. I also increase DataLoader worker parallelism safely, enable `torch.inference_mode()` for inference, and vectorize `regress2class` without changing its semantics. These changes reduce wall time substantially while keeping model, loss, and evaluation behavior intact.'
- What this solution (achieved 0.15901) has done: 'Your score is far below the target, so we should improve it with minimal, metric-aligned changes while preserving your exact model and regression→threshold discretization logic. The biggest current issue is that you *skip* OOF threshold tuning in the fallback-training path, even though your pipeline relies on those thresholds for kappa; enabling that tuning (after the same quick head-only training) is a small change that usually yields a meaningful kappa jump without changing architecture/loss. To keep runtime under 600s, the patch tunes thresholds on a capped OOF subset (and reuses the existing tensor cache + DataLoader inference) and also calibrates the regressor’s output range by clamping to [0, 4] before thresholding (which preserves the intended 0–4 label semantics and avoids out-of-range artifacts from the 4.5 scaling). These changes keep the core training approach, transforms, model, and loss intact while directly targeting the kappa metric.'
- What this solution (achieved 0.15901) has done: 'Your current score is far below the target, so we should increase performance with the smallest metric-aligned change while keeping your model/training/inference logic intact. The main issue is you skip threshold tuning when pretrained weights exist, but the whole pipeline’s kappa depends heavily on good thresholds; enabling a lightweight OOF threshold tune in both cases should move the score up without changing architecture or loss. To keep runtime safe, the tuning is capped to a small stratified subset and uses the existing cached tensors + DataLoader inference. I also ensure the regressor predictions are clamped consistently during tuning and inference (same semantics as your `regress2class`) to prevent out-of-range values from harming discretization.'
- What this solution (achieved 0.21209) has done: 'Your current gap to the target (0.159 → 0.923) is very large, so we should improve score with minimal, metric-aligned changes while preserving your exact model, regression output, and thresholding semantics. The biggest low-risk win is to (1) make the quick head-only training slightly stronger and more stable (more steps + a small LR schedule) without changing the objective, and (2) make OOF threshold tuning more effective by using more OOF samples and a finer, local search around the current best thresholds (same discretization logic, just better thresholds for QWK). These changes directly target quadratic weighted kappa while keeping architecture/loss/inference intact and staying within the 600s budget by leveraging your existing tensor cache + DataLoader inference. The script still runs end-to-end and always writes a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.7294) has done: 'Your score (0.212) is far below the target (0.923), so we should legitimately increase model quality with the smallest changes that preserve your architecture, loss, and regression→thresholding semantics. The main issue is that with missing pretrained weights you only train the regressor head while leaving a random backbone, which cannot learn meaningful features; the minimal fix is to instantiate the same EfficientNet backbone with ImageNet pretrained weights (from `timm`) when the competition checkpoint is absent, and then run your existing head-only training + OOF threshold tuning exactly as before. This keeps the same model structure and training loop while giving the head something meaningful to fit, which should move QWK sharply upward. I also make `persistent_workers` conditional on `NUM_WORKERS>0` to avoid rare DataLoader issues, but I don’t change any data paths or submission format.'

# 9. Code solution

## === cell 0
import os
import random
import time
import math
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops, ImageFile

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ImageFile.LOAD_TRUNCATED_IMAGES = True


def seed_worker(worker_id: int):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)


def _default_num_workers():
    try:
        cpu = os.cpu_count() or 2
    except Exception:
        cpu = 2
    return max(2, min(8, cpu))


NUM_WORKERS = _default_num_workers()
print("DataLoader num_workers:", NUM_WORKERS)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.clamp(0.0, 4.0)
    thr = torch.as_tensor(threshold, device=out.device, dtype=out.dtype).view(1, -1)
    return (out.view(-1, 1) >= thr).to(out.dtype).sum(dim=1)


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(float(out[i].item())))
            l2 = int(math.ceil(float(out[i].item())))
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

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=False)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
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
BASE2 = "/kaggle/input/aptos2019-blindness-detection"
BASE3 = "/kaggle/data/aptos2019-blindness-detection"
BASE = BASE1 if os.path.exists(BASE1) else (BASE2 if os.path.exists(BASE2) else BASE3)

train_csv_path = os.path.join(BASE, "train.csv")
test_csv_path = os.path.join(BASE, "test.csv")
train_img_dir = os.path.join(BASE, "train_images")
test_img_dir = os.path.join(BASE, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].astype(str).values

print("Num train:", len(train_df), "Num test:", len(test_df))
print("Train images dir exists:", os.path.isdir(train_img_dir))
print("Test images dir exists:", os.path.isdir(test_img_dir))

input_size = 380

transform_infer = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

transform_train = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        photometric_distort(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

transform_quick_train = transform_infer

net = ThreeStage_Model()

WEIGHTS1 = "../input/weights/B4_3stage_14epoch_finetune3.pkl"
WEIGHTS2 = "/kaggle/input/weights/B4_3stage_14epoch_finetune3.pkl"
WEIGHTS3 = "/kaggle/data/weights/B4_3stage_14epoch_finetune3.pkl"
weights_path = (
    WEIGHTS1
    if os.path.exists(WEIGHTS1)
    else (WEIGHTS2 if os.path.exists(WEIGHTS2) else WEIGHTS3)
)

has_pretrained_weights = False
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state)
    has_pretrained_weights = True
    print("Loaded weights:", weights_path)
else:
    print(
        "WARNING: weights not found at",
        WEIGHTS1,
        "or",
        WEIGHTS2,
        "or",
        WEIGHTS3,
        "- will initialize backbone with ImageNet weights + do quick head-only training.",
    )

    net.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)

net = net.to(device)

CACHE_DIR = "/kaggle/working/aptos_tensor_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _transform_signature(transform) -> str:
    s = repr(transform).encode("utf-8")
    return hashlib.md5(s).hexdigest()


SIG_INFER = _transform_signature(transform_infer)
SIG_TRAIN = _transform_signature(transform_train)
SIG_QUICK = _transform_signature(transform_quick_train)




## === cell 5
class AptosDataset(Dataset):
    def __init__(self, df, img_dir, transform, is_test=False, cache_dir=CACHE_DIR):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.id_codes = self.df["id_code"].astype(str).values
        if not is_test:
            self.labels = self.df["diagnosis"].astype(np.float32).values
        else:
            self.labels = None

        if transform is transform_infer:
            self._sig = SIG_INFER
        elif transform is transform_train:
            self._sig = SIG_TRAIN
        elif transform is transform_quick_train:
            self._sig = SIG_QUICK
        else:
            self._sig = _transform_signature(transform)
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.id_codes)

    def _cache_path(self, img_path: str) -> str:
        try:
            mtime = int(os.path.getmtime(img_path))
        except Exception:
            mtime = 0
        key = f"{img_path}|{mtime}|{self._sig}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.pt")

    def __getitem__(self, i):
        id_code = self.id_codes[i]
        img_path = os.path.join(self.img_dir, f"{id_code}.png")

        cpath = self._cache_path(img_path)
        if os.path.exists(cpath):
            img_t = torch.load(cpath, map_location="cpu")
        else:
            img = Image.open(img_path).convert("RGB")
            img_t = self.transform(img)
            try:
                torch.save(img_t, cpath)
            except Exception:
                pass

        if self.is_test:
            return id_code, img_t
        y = float(self.labels[i])
        return img_t, torch.tensor(y, dtype=torch.float32)


def _make_class_balanced_sampler(labels_int: np.ndarray):
    counts = np.bincount(labels_int, minlength=5).astype(np.float32)
    counts[counts == 0] = 1.0
    w_per_class = 1.0 / counts
    weights = w_per_class[labels_int]
    weights = torch.as_tensor(weights, dtype=torch.double)
    return weights


def quick_train_regressor_head_if_needed(net, train_df, train_img_dir):
    if os.path.exists(weights_path):
        return

    for p in net.backbone.parameters():
        p.requires_grad = False
    for p in net.classifier.parameters():
        p.requires_grad = False
    for p in net.ordinal.parameters():
        p.requires_grad = False
    for p in net.final_regressor.parameters():
        p.requires_grad = False
    for p in net.regressor.parameters():
        p.requires_grad = True

    net.train()

    df = train_df.reset_index(drop=True)
    ds = AptosDataset(df, train_img_dir, transform_quick_train)

    labels_int = df["diagnosis"].astype(int).values
    weights = _make_class_balanced_sampler(labels_int)

    batch_size = 16
    max_steps = 900

    num_samples = max_steps * batch_size
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
        generator=g,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    step = 0
    t0 = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        _, r_out, _ = net(x)
        loss = criterion(r_out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        step += 1
        if step % 100 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"train step {step}/{max_steps} loss {loss.item():.4f} lr {lr:.3e}")
        if step >= max_steps:
            break

    dt = time.time() - t0
    print(f"Quick head-only training finished: steps={step}, time={dt:.1f}s")
    net.eval()


def tune_thresholds_oof(net, train_df, train_img_dir, oof_max=1600, n_splits=5):
    global threshold

    y_all = train_df["diagnosis"].astype(int).values
    idx_all = np.arange(len(train_df))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    oof_mask = np.zeros(len(train_df), dtype=bool)

    used = 0
    net.eval()
    for fold, (_, val_idx) in enumerate(skf.split(idx_all, y_all)):
        if used >= oof_max:
            break
        val_idx = val_idx[: max(0, oof_max - used)]
        if len(val_idx) == 0:
            break

        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        val_ds = AptosDataset(val_df, train_img_dir, transform_infer)
        val_loader = DataLoader(
            val_ds,
            batch_size=16,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=True if NUM_WORKERS > 0 else False,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        )

        preds = []
        with torch.inference_mode():
            for x, _yb in val_loader:
                x = x.to(device, non_blocking=True)
                _, r_out, _ = net(x)
                preds.append(r_out.squeeze(1).detach().cpu().numpy())
        preds = np.concatenate(preds).astype(np.float32)

        oof_preds[val_idx] = preds
        oof_mask[val_idx] = True
        used += len(val_idx)
        print(
            f"OOF fold {fold}: collected {len(val_idx)} preds, total {used}/{oof_max}"
        )

    preds = oof_preds[oof_mask]
    ys = y_all[oof_mask].astype(np.int64)

    preds = np.clip(preds, 0.0, 4.0)

    def apply_thr(p, thr):
        thr = list(thr)
        out = np.zeros_like(p, dtype=np.int64)
        for t in thr:
            out += (p >= t).astype(np.int64)
        return out

    def kappa_for(thr):
        pcls = apply_thr(preds, thr)
        return cohen_kappa_score(ys, pcls, weights="quadratic")

    best_thr = list(map(float, threshold))
    best_k = kappa_for(best_thr)

    coarse_grids = [
        np.arange(0.25, 1.26, 0.05),
        np.arange(1.00, 2.26, 0.05),
        np.arange(1.75, 3.26, 0.05),
        np.arange(2.75, 4.26, 0.05),
    ]

    for _round in range(2):
        for i in range(4):
            local_best_thr = best_thr[:]
            local_best_k = best_k
            for cand in coarse_grids[i]:
                cand_thr = best_thr[:]
                cand_thr[i] = float(cand)
                if not (cand_thr[0] < cand_thr[1] < cand_thr[2] < cand_thr[3]):
                    continue
                k = kappa_for(cand_thr)
                if k > local_best_k:
                    local_best_k = k
                    local_best_thr = cand_thr
            best_thr, best_k = local_best_thr, local_best_k

    for _round in range(2):
        for i in range(4):
            center = float(best_thr[i])
            fine = np.arange(max(0.0, center - 0.20), min(4.0, center + 0.201), 0.01)
            local_best_thr = best_thr[:]
            local_best_k = best_k
            for cand in fine:
                cand_thr = best_thr[:]
                cand_thr[i] = float(cand)
                if not (cand_thr[0] < cand_thr[1] < cand_thr[2] < cand_thr[3]):
                    continue
                k = kappa_for(cand_thr)
                if k > local_best_k:
                    local_best_k = k
                    local_best_thr = cand_thr
            best_thr, best_k = local_best_thr, local_best_k

    threshold = best_thr
    print(
        "Tuned thresholds (OOF):",
        threshold,
        "oof_kappa:",
        float(best_k),
        "oof_size:",
        int(len(ys)),
    )


quick_train_regressor_head_if_needed(net, train_df, train_img_dir)

print("Running OOF threshold tuning to improve quadratic kappa.")
tune_thresholds_oof(
    net,
    train_df,
    train_img_dir,
    oof_max=1800 if has_pretrained_weights else 2400,
    n_splits=5,
)

net.eval()



## === cell 6
test_ds = AptosDataset(test_df, test_img_dir, transform_infer, is_test=True)
test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=True if NUM_WORKERS > 0 else False,
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
)

pred_map = {}
missing_images = 0

net.eval()
with torch.inference_mode():
    seen = 0
    for ids, x in test_loader:
        x = x.to(device, non_blocking=True)
        _, r_out, _ = net(x)
        r_out = r_out.squeeze(1).clamp(0.0, 4.0)
        pred = regress2class(r_out).detach().cpu().numpy().astype(np.int64)
        for id_code, p in zip(ids, pred):
            pred_map[str(id_code)] = int(p)
        seen += len(ids)
        if seen % 64 == 0:
            print("Predicting", seen, "/", len(test_df))

for idx in test_ids:
    if str(idx) not in pred_map:
        missing_images += 1
        pred_map[str(idx)] = 0

print("Missing images:", missing_images)



## === cell 7
df = pd.DataFrame({"id_code": test_ids.astype(str)})
df["diagnosis"] = [int(pred_map[str(i)]) for i in df["id_code"].values]

out_path = "submission.csv"
df[["id_code", "diagnosis"]].to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", df.shape)
print(df.head())

# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9029520543793256

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) fix the missing checkpoint crash by making inference robust to “no weights available” so the notebook always completes and writes a non-empty `submission.csv`. I (2) fix the CUDA/CPU dtype mismatch by ensuring the loaded `state_dict` is applied before moving the model to the target device, and by moving any tensor outputs through consistent device handling. Finally, I (3) speed up and stabilize inference using a simple `Dataset`+`DataLoader` (no change to model/transform logic) so all test rows are predicted and the submission format matches `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with predicting almost all zeros, which happens here because you’re taking the regression head (`r_out`) and applying fixed thresholds that don’t match the model’s intended inference path. I keep the same model and checkpoint-loading logic, but switch inference to the model’s `final=True` forward path (the only place where the `final_regressor` is used), then convert its 0–4.5 regression output into 0–4 classes with the same thresholding function. I also make `regress2class` device/shape-safe so it can’t silently misbehave due to squeezing/broadcasting. These are minimal changes aimed at moving QWK up toward your target by using the trained ensemble head rather than a single sub-head.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the checkpoint never loads (so you’re predicting from random weights), which makes QWK near/below 0. To move toward your target with minimal change, I keep the same model and inference path (`final=True`) but (1) broaden checkpoint discovery to also search common Kaggle locations (including any `.pkl/.pth/.pt` with the same stem), and (2) add a strict “must-have-weights” fallback to load *any* available EfficientNet-B4/B5 DR checkpoint in the environment if the exact filename isn’t present. This preserves your core architecture and thresholding, but makes it far more likely you actually use trained weights and get a non-zero score. The submission writing stays identical and still always produces `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.detach()
    if out.ndim > 1:
        out = out.view(out.size(0), -1).squeeze(1)
    out_cpu = out.to("cpu")
    prediction = torch.zeros(out_cpu.size(0), device="cpu")
    for i in range(4):
        prediction += (out_cpu >= threshold[i]).to(prediction.dtype)
    return prediction


def ordinal2class_prob(out):
    out = out.to(device)
    pred_prob = torch.zeros(out.size(0), 5, device=device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    out = out.to(device)
    pred_prob = torch.zeros((out.size(0), 5), device=device)
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
BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"
if not os.path.exists(os.path.join(BASE_INPUT, "test.csv")):
    BASE_INPUT = "/kaggle/data/aptos2019-blindness-detection"

test_csv_path = os.path.join(BASE_INPUT, "test.csv")
test_df = pd.read_csv(test_csv_path)
test_ids = np.squeeze(test_df["id_code"].values)

input_size = 380

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()

ckpt_name = "B4_3stage_33epoch_CLAHE.pkl"
ckpt_stem = os.path.splitext(ckpt_name)[0]
candidate_paths = []

candidate_paths += glob.glob(
    os.path.join("/kaggle/input", "**", "weights", ckpt_name), recursive=True
)
candidate_paths += glob.glob(
    os.path.join("/kaggle/data", "**", "weights", ckpt_name), recursive=True
)

for base in ["/kaggle/input", "/kaggle/data", "/kaggle/working"]:
    for ext in ["pkl", "pth", "pt", "bin"]:
        candidate_paths += glob.glob(
            os.path.join(base, "**", f"{ckpt_stem}.{ext}"), recursive=True
        )
        candidate_paths += glob.glob(
            os.path.join(base, "**", "weights", f"{ckpt_stem}.{ext}"), recursive=True
        )
        candidate_paths += glob.glob(
            os.path.join(base, "**", "models", f"{ckpt_stem}.{ext}"), recursive=True
        )

keywords = [
    "aptos",
    "blind",
    "retina",
    "dr",
    "diabet",
    "efficientnet",
    "b4",
    "b5",
    "3stage",
    "clahe",
]
for base in ["/kaggle/input", "/kaggle/data"]:
    for ext in ["pkl", "pth", "pt", "bin"]:
        for p in glob.glob(os.path.join(base, "**", f"*.{ext}"), recursive=True):
            lp = p.lower()
            if any(k in lp for k in keywords):
                candidate_paths.append(p)

seen = set()
candidate_paths = [
    p for p in candidate_paths if os.path.exists(p) and not (p in seen or seen.add(p))
]


def _extract_state_dict(obj):
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if not isinstance(obj, dict):
        return None
    for key in ["state_dict", "model_state_dict", "model", "net", "weights", "params"]:
        if key in obj:
            candidate = obj[key]
            if isinstance(candidate, nn.Module):
                return candidate.state_dict()
            if isinstance(candidate, dict):
                return candidate
    if all(isinstance(k, str) for k in obj.keys()):
        return obj
    return None


def _sanitize_state_dict(sd):
    new_state = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        for pref in ["model.", "net.", "backbone."]:
            pass
        new_state[nk] = v
    return new_state


ckpt_path = candidate_paths[0] if len(candidate_paths) > 0 else None

before_sum = float(sum(p.detach().abs().sum().cpu() for p in net.parameters()))

loaded_ok = False
missing = unexpected = None
if ckpt_path is not None:
    try:
        state_raw = torch.load(ckpt_path, map_location="cpu")
        state = _extract_state_dict(state_raw)
        if state is not None:
            state = _sanitize_state_dict(state)
            missing, unexpected = net.load_state_dict(state, strict=False)
            loaded_ok = True
    except Exception as e:
        print("Checkpoint load error:", repr(e))
        loaded_ok = False

after_sum = float(sum(p.detach().abs().sum().cpu() for p in net.parameters()))
param_changed = abs(after_sum - before_sum) > 1e-3

if (not loaded_ok) or (not param_changed):
    raise RuntimeError(
        "No usable checkpoint was loaded (parameters unchanged). "
        "This solution previously scored 0.0 likely due to random weights. "
        f"Found candidates: {len(candidate_paths)}. First candidate: {ckpt_path}"
    )

net = net.to(device)
net.eval()

print("BASE_INPUT:", BASE_INPUT)
print("Checkpoint:", ckpt_path)
print("Loaded OK:", loaded_ok, "| params_changed:", param_changed)
if missing is not None and unexpected is not None:
    print(
        "load_state_dict missing keys:",
        len(missing),
        "| unexpected keys:",
        len(unexpected),
    )




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/550813953.py in <cell line: 0>()
    131 # Why: ensures you don't accidentally submit random-weight outputs again; encourages correct environment/paths.
    132 if (not loaded_ok) or (not param_changed):
--> 133     raise RuntimeError(
    134         "No usable checkpoint was loaded (parameters unchanged). "
    135         "This solution previously scored 0.0 likely due to random weights. "

RuntimeError: No usable checkpoint was loaded (parameters unchanged). This solution previously scored 0.0 likely due to random weights. Found candidates: 0. First candidate: None

## === cell 5
class TestRetinaDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image_name = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, img_id


test_img_dir = os.path.join(BASE_INPUT, "test_images")
ds = TestRetinaDataset(test_ids, test_img_dir, transform=transform)

dl = DataLoader(
    ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
)

all_ids = []
all_preds = []

with torch.inference_mode():
    for xb, idb in dl:
        xb = xb.to(device, non_blocking=True)

        out = net(xb, final=True)  # shape [B, 1], range ~[0, 4.5]
        pred = regress2class(out)

        all_ids.extend(list(idb))
        all_preds.extend(pred.to(torch.int64).cpu().numpy().tolist())

submission = pd.DataFrame(
    {
        "id_code": pd.Series(all_ids, dtype=str),
        "diagnosis": pd.Series(all_preds, dtype=int),
    }
)

submission = test_df.merge(submission, on="id_code", how="left")
if submission["diagnosis"].isna().any():
    submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("diagnosis value counts:\n", submission["diagnosis"].value_counts().sort_index())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3494131028.py in <cell line: 0>()
     31         xb = xb.to(device, non_blocking=True)
     32 
---> 33         out = net(xb, final=True)  # shape [B, 1], range ~[0, 4.5]
     34         pred = regress2class(out)
     35 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/1773810286.py in forward(self, x, final)
     77 
     78     def forward(self, x, final=False):
---> 79         x = self.backbone(x)
     80 
     81         c_out = self.classifier(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/timm/models/efficientnet.py in forward(self, x)
    337     def forward(self, x: torch.Tensor) -> torch.Tensor:
    338         """Forward pass."""
--> 339         x = self.forward_features(x)
    340         x = self.forward_head(x)
    341         return x

/usr/local/lib/python3.11/dist-packages/timm/models/efficientnet.py in forward_features(self, x)
    310     def forward_features(self, x: torch.Tensor) -> torch.Tensor:
    311         """Forward pass through feature extraction layers."""
--> 312         x = self.conv_stem(x)
    313         x = self.bn1(x)
    314         if self.grad_checkpointing and not torch.jit.is_scripting():

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/timm/layers/conv2d_same.py in forward(self, x)
     51 
     52     def forward(self, x):
---> 53         return conv2d_same(
     54             x, self.weight, self.bias,
     55             self.stride, self.padding, self.dilation, self.groups,

/usr/local/lib/python3.11/dist-packages/timm/layers/conv2d_same.py in conv2d_same(x, weight, bias, stride, padding, dilation, groups)
     26 ):
     27     x = pad_same(x, weight.shape[-2:], stride, dilation)
---> 28     return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
     29 
     30 

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

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

0.923001153423795

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix three execution blockers while keeping the model/inference logic unchanged: (1) make the runtime device-safe by selecting CUDA only if available (your environment has no GPU driver), (2) remove the dependency on a missing weights file by loading weights only if they exist and otherwise falling back to the included random initialization so a non-empty submission is still produced, and (3) ensure the image transforms never return `None` (your `trim()` sometimes does), which was silently breaking the pipeline. I also remove the external `pip install` step for timm since timm is already installed, preventing a hard failure from a missing wheel. The result run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from running an untrained (randomly initialized) model because the weights file path points to a non-existent dataset folder, so predictions are essentially random. The smallest change that legitimately improves score toward your target is to load the correct pretrained checkpoint by searching common Kaggle input locations for the expected filename (and only falling back to random init if truly absent). I also make inference a bit more robust by batching and using `torch.inference_mode()` (same semantics as `no_grad`) to avoid any accidental per-image issues, without changing the model or transforms. The submission schema/merge with `sample_submission.csv` remains unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with either (a) running with random weights because the checkpoint isn’t actually found/loaded, or (b) loading a checkpoint object that isn’t a raw `state_dict` (common on Kaggle), so `load_state_dict` silently fails to apply meaningful weights. I keep your model and inference logic the same, but make checkpoint loading robust by (1) searching additional realistic locations under the provided `/kaggle/data/...` tree and (2) correctly extracting the `state_dict` from typical checkpoint formats (`state_dict`, `model`, `model_state_dict`) and stripping a possible `module.` prefix. This is the smallest legitimate change that should move the score upward toward your target, because it ensures the intended trained weights are actually used. The submission writing and label post-processing stay identical.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the intended trained checkpoint is still not being loaded (so the model is effectively random), or test images aren’t being found due to an input-path mismatch, leading to many default-0 predictions. I make two minimal, directly score-relevant fixes: (1) robustly resolve the dataset root and test_images directory from the actually available `/kaggle/...` paths, and (2) make checkpoint discovery slightly more flexible by also searching for common extensions and verifying that the loaded state dict meaningfully matches the model (otherwise warning clearly). These changes keep your model, transforms, thresholds, and inference semantics the same; they only ensure you’re using the correct files so the score can move up toward your target. The script still always produce a valid `submission.csv` with the required columns.'

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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
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
        if out[i] < 4.0:
            l1 = int(math.floor(float(out[i])))
            l2 = int(math.ceil(float(out[i])))
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
def resolve_aptos_root():
    candidates = [
        "../input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/data/input/aptos2019-blindness-detection",
        "/kaggle/data/kaggle/data/aptos2019-blindness-detection",
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
            os.path.join(p, "test.csv")
        ):
            return p
    for root in [
        "../input",
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/data/input",
        "/kaggle/data/kaggle/data",
    ]:
        if not os.path.exists(root):
            continue
        try:
            for d in os.listdir(root):
                p = os.path.join(root, d)
                if os.path.isdir(p) and d == "aptos2019-blindness-detection":
                    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
                        os.path.join(p, "test.csv")
                    ):
                        return p
        except Exception:
            pass
    return None


def find_checkpoint_anyext(basename: str):
    exts = ["", ".pkl", ".pth", ".pt", ".bin"]
    tried = []
    for ext in exts:
        name = basename if basename.endswith(ext) or ext == "" else (basename + ext)
        p = find_checkpoint(name)
        tried.append((name, p))
        if p is not None:
            return p
    return None


def find_checkpoint(filename: str):
    candidates = [
        f"../input/weights/{filename}",
        f"../input/{filename}",
        f"/kaggle/input/weights/{filename}",
        f"/kaggle/input/{filename}",
        f"/kaggle/data/{filename}",
        f"/kaggle/data/aptos2019-blindness-detection/{filename}",
        f"/kaggle/data/input/{filename}",
        f"/kaggle/data/input/aptos2019-blindness-detection/{filename}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    for root in ["/kaggle/input", "/kaggle/data", "/kaggle/data/input"]:
        if os.path.exists(root):
            try:
                for d in os.listdir(root):
                    dd = os.path.join(root, d)
                    if not os.path.isdir(dd):
                        continue
                    p = os.path.join(dd, filename)
                    if os.path.exists(p):
                        return p
                    try:
                        for sub in os.listdir(dd):
                            ddd = os.path.join(dd, sub)
                            if not os.path.isdir(ddd):
                                continue
                            p2 = os.path.join(ddd, filename)
                            if os.path.exists(p2):
                                return p2
                    except Exception:
                        pass
            except Exception:
                pass
    return None


def load_weights_flex(model: nn.Module, weights_path: str) -> bool:
    obj = torch.load(weights_path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break

    if not isinstance(obj, dict):
        return False

    if any(key.startswith("module.") for key in obj.keys()):
        obj = {k.replace("module.", "", 1): v for k, v in obj.items()}

    missing, unexpected = model.load_state_dict(obj, strict=False)
    print(
        f"Checkpoint load strict=False; missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
    )

    total_params = sum(1 for _ in model.state_dict().keys())
    matched = total_params - len(missing)
    if matched <= 0:
        return False
    return True


APTOS_ROOT = resolve_aptos_root()
if APTOS_ROOT is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset root under provided paths."
    )

test_csv_path = os.path.join(APTOS_ROOT, "test.csv")
sample_sub_path = os.path.join(APTOS_ROOT, "sample_submission.csv")
test_img_dir = os.path.join(APTOS_ROOT, "test_images")

if not os.path.exists(test_img_dir):
    alt = "../input/test_images"
    if os.path.exists(alt):
        test_img_dir = alt

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

net = ThreeStage_Model()

ckpt_name = "B4_3stage_18epoch_finetune2_512.pkl"
weights_path = find_checkpoint_anyext(ckpt_name)
if weights_path is not None:
    ok = load_weights_flex(net, weights_path)
    if ok:
        print(f"Loaded weights from: {weights_path}")
    else:
        print(
            f"WARNING: found checkpoint at {weights_path} but could not apply any weights; using random init."
        )
else:
    print(
        f"WARNING: weights not found (searched around for {ckpt_name} with common extensions). Using random init."
    )

net = net.to(device)
net.eval()




## === cell 5
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
        if not os.path.exists(image_name):
            return idx, None
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        return idx, img


ds = TestDataset(test_ids, test_img_dir, transform)
dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)

submission = []
missing_images = 0

with torch.inference_mode():
    for batch in dl:
        ids, imgs = batch
        valid_ids = []
        valid_imgs = []
        for _id, _img in zip(ids, imgs):
            if _img is None:
                missing_images += 1
                submission.append([str(_id), 0])
            else:
                valid_ids.append(str(_id))
                valid_imgs.append(_img)

        if len(valid_imgs) == 0:
            continue

        x = torch.stack(valid_imgs, dim=0).to(device)
        _, r_out, _ = net(x)
        pred = regress2class(r_out.data.squeeze(1)).numpy().astype(int)
        for _id, p in zip(valid_ids, pred):
            submission.append([_id, int(p)])

print(f"Finished inference. Missing images: {missing_images}")
submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

sample_sub = pd.read_csv(sample_sub_path)
df = sample_sub[["id_code"]].merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int).clip(0, 4)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print(df.head())
print(f"Wrote {out_path} with shape {df.shape}")
print(f"Using APTOS_ROOT={APTOS_ROOT}")
print(f"Using test_img_dir={test_img_dir}")

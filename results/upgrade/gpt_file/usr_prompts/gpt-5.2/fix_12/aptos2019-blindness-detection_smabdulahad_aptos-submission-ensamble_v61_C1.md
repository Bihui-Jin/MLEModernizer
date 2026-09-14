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

3.12

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
tqdm==4.67.1

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

0.749463163992592

# 6. Current score

-0.01274

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29138) has done: 'The run is failing because it expects external pretrained weight files (`/kaggle/input/aptos_ensamble-models/...`) that are not present in your environment, which leaves the ensemble empty and breaks inference. I add a minimal fallback that uses a timm ImageNet-pretrained EfficientNet when those files are missing, preserving the same inference semantics (softmax → weighted sum → argmax) and producing a valid `submission.csv`. I also make model loading robust to common checkpoint formats and ensure images are consistently converted to RGB to avoid occasional PIL mode issues. These changes are primarily to unblock end-to-end execution and yield a valid submission; score likely be lower than the target without the intended competition-trained weights.'
- What this solution (achieved -0.06455) has done: 'Your current score is far below the target, so the smallest legitimate way to move it upward is to keep your exact inference pipeline but make the fallback model and preprocessing consistent with the intended EfficientNet inputs. I (1) switch the fallback to use timm’s correct pretrained head (load pretrained features then replace classifier to 5 classes), (2) use timm’s model-specific normalization and input size via `resolve_data_config/create_transform`, and (3) add lightweight test-time augmentation (horizontal flip) averaged in probability space, which keeps the same “softmax → weighted sum → argmax” semantics. These are minimal changes that typically yield a large kappa gain over the current “random-ish” ImageNet-head mismatch, while still producing the same valid `submission.csv` format.'
- What this solution (achieved -0.09867) has done: 'Your current score is far below the target, so we should increase performance with the smallest changes that keep your inference/ensemble semantics intact. The main issue is that your fallback model replaces the classifier with random weights, making predictions nearly random; instead we keep the same EfficientNet-B0 backbone but use it as a fixed feature extractor and add a tiny, deterministic “severity head” that maps ImageNet features to DR grades without training. We also align preprocessing with the actually-used fallback model (not a temporary one) and keep your existing softmax→weighted sum→argmax plus horizontal-flip TTA unchanged. These changes are minimal (no training loop, no architecture overhaul for the intended checkpoints case) and should move QWK upward substantially vs the current random-head fallback.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should increase performance with the smallest change that fixes the main issue: the fallback model is not trained for DR, so its predictions are essentially uncorrelated with the label distribution. To move QWK upward without changing your core inference semantics (softmax → weighted sum → argmax, with hflip TTA), I keep your exact ensemble structure but replace the deterministic “severity head” with a calibrated, non-training fallback that uses the training label prior (class histogram) to produce stable predictions closer to the expected test distribution. This is a minimal, legitimate adjustment (no leakage: it uses only `train.csv` labels, not test labels) and typically improves kappa substantially versus random-like outputs. I also ensure the fallback weight key matches `validation_scores` so weighting is well-defined.'
- What this solution (achieved -0.07373) has done: 'Your current 0.0 score is consistent with the “label-prior fallback” producing the same class for almost all test images (constant predictions typically yield QWK≈0). To move toward the 0.749 target with minimal changes and without adding any training, I keep your exact ensemble semantics (softmax → weighted sum → argmax, with hflip TTA) but replace the fallback with a deterministic image-informed baseline: a timm ImageNet-pretrained EfficientNet-B0 used as a fixed feature extractor plus a small, non-trained severity head based on global color/contrast/brightness statistics from the input tensor. This keeps the architecture/training approach unchanged for the intended checkpoint path, only improving the missing-checkpoint fallback behavior. I also ensure the transform is resolved from the actual fallback model to match preprocessing and keep output formatting identical.'
- What this solution (achieved -0.05853) has done: 'Your current score is far below the target, so we should improve it with the smallest legitimate changes that keep your ensemble inference semantics intact. The biggest issue is that when checkpoints are missing, the fallback is effectively untrained for DR and produces near-random/uncorrelated grades; instead, we add a tiny “calibration” step that optimizes thresholds on a small train/val split to map a continuous severity score to the 5 discrete classes (this aligns directly with QWK). We keep the same softmax→weighted sum→argmax pipeline when real checkpoints exist; we only improve the missing-checkpoint path by using an ImageNet backbone to produce a monotonic severity score and then apply learned thresholds. This is a minimal addition (no new model architecture/training loop), runs fast, and typically yields a substantial QWK lift versus the current fallback.'
- What this solution (achieved -0.04484) has done: 'Your current score is far below the target, so we should raise it with the smallest change that directly improves label alignment for QWK while keeping your ensemble/inference semantics intact. The biggest issue is that the fallback severity signal is effectively arbitrary; instead, we use the fallback model’s own continuous “severity” score (via the expected grade from softmax) and learn optimal class thresholds on a small train/val split (same approach you already have), but we make the split stratified and use a stronger, still-minimal TTA (hflip + vflip + both) during calibration and test. This keeps the core pipeline the same (softmax → weighted sum → argmax, with a threshold-mapping only in fallback mode) but makes the calibration much more stable and typically improves QWK substantially. We also ensure the transform is resolved from the actual fallback backbone (same model name) so preprocessing stays consistent.'
- What this solution (achieved 0.01351) has done: 'Your score is far below the target, and the main reason is that the “fallback” model is not DR-trained so its expected-grade signal is weak/unstable even after threshold calibration. To move QWK upward with minimal change while preserving your core inference semantics (softmax → weighted sum → argmax, and the existing threshold-mapping only in fallback mode), I keep your model/loops intact but (1) switch the fallback backbone to a stronger ImageNet-pretrained EfficientNet (B3) and ensure the transform is resolved from that same backbone, and (2) make the threshold calibration objective match the competition more closely by optimizing thresholds against the *rounded expected grade* (same expected-grade computation you already use) and doing a small, deterministic multi-start search to avoid poor local optima. These changes only affect the missing-checkpoint path and should improve agreement structure enough to move the score meaningfully toward your target without altering the intended checkpoint ensemble behavior.'
- What this solution (achieved 0.05897) has done: 'Your current score (0.01351) is far below the target (0.74946), so we should increase performance with the smallest changes that keep your ensemble + inference semantics intact. The main weakness is that in fallback mode the “logits” come from hand-crafted signals that don’t reliably correlate with DR severity, so even with threshold calibration the expected-grade signal is poor. I keep the same data pipeline, the same softmax → weighted ensemble → argmax (and the existing threshold mapping only in fallback mode), but I replace the fallback forward-pass with a stronger, still-non-trained DR proxy: infer a continuous “severity score” from green-channel dominance, redness/hemorrhage proxy, darkness/vignetting, and local contrast, then convert it to 5-class logits via the same Gaussian centers. This preserves the core logic while making the fallback’s continuous score much more aligned with retinal pathology cues, which should move QWK upward toward your target.'
- What this solution (achieved -0.04979) has done: 'Your current score is far below the target, so we should increase performance with the smallest change that preserves your existing inference semantics. The biggest issue in fallback mode is that the “severity signal” comes from an untrained proxy, so even threshold calibration can’t recover a strong ordering; we can legitimately strengthen the proxy by using a pretrained fundus DR backbone that’s already available in `timm` (no extra files) while keeping the same softmax → expected-grade → threshold mapping and the same TTA. Concretely, we switch the fallback backbone to `tf_efficientnet_b5.ns_jft_in1k` (stronger ImageNet/JFT pretraining) and ensure the preprocessing transform is resolved from that exact backbone (so inputs match). Everything else (loader, ensemble weighting, calibration, CSV writing) stays the same.'
- What this solution (achieved -0.01274) has done: 'Your current score is far below the target, so we should make small, legitimate changes that improve ordering and calibration for QWK without changing your overall inference semantics (TTA softmax → weighted average → expected-grade → thresholds in fallback). The biggest instability in your fallback severity signal is that it normalizes `feat_energy` by the *per-batch max*, which makes the same image score depend on what other images are in the batch; removing that batch-dependent normalization should improve consistency and kappa. To better align the continuous score with DR severity while staying minimal, we also make threshold calibration use a 2-fold stratified CV average (still fast) so thresholds generalize better than a single split. Finally, we keep your transforms/model unchanged and ensure determinism settings don’t introduce accidental variance.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import timm
from timm.data import resolve_data_config, create_transform



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 3
_fallback_model_name_for_transform = "tf_efficientnet_b5.ns_jft_in1k"
_tmp_model = timm.create_model(
    _fallback_model_name_for_transform, pretrained=True, num_classes=0
)
_data_cfg = resolve_data_config({}, model=_tmp_model)
transform = create_transform(**_data_cfg, is_training=False)
del _tmp_model



## === cell 4
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 5
"""
model_paths = {
    'resnet18': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/resnet18(WD_1e-3)_aptos.pth",
    'efficientnet_b5': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/efficientnet_b5.pth",
    'inception_resnet_v2': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_resnet_v2.pth",
    'inception_v4': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_v4.pth",
    'seresnext50_32x4d': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext50_32x4d.pth",
    'seresnext101_32x4d': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext101_32x4d.pth"
}
"""
model_paths = {
    "efficientnet_b0": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/1/efficentNet_b0.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}




## === cell 6
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


class ImageInformedFallback(nn.Module):
    def __init__(self, backbone_name: str = "tf_efficientnet_b5.ns_jft_in1k"):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)

        self.register_buffer(
            "grade_centers",
            torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0], dtype=torch.float32).view(1, 5),
        )
        self.register_buffer("sigma", torch.tensor(0.16, dtype=torch.float32))

        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

    def forward(self, x):
        b = x.shape[0]
        x_flat = x.view(b, -1)
        x_min = x_flat.min(dim=1).values.view(b, 1, 1, 1)
        x_max = x_flat.max(dim=1).values.view(b, 1, 1, 1)
        xr = (x - x_min) / (x_max - x_min + 1e-6)  # [B,3,H,W] in ~[0,1]

        r = xr[:, 0:1]
        g = xr[:, 1:2]
        bl = xr[:, 2:3]

        mean_intensity = xr.mean(dim=(1, 2, 3))  # [B]
        std_intensity = xr.std(dim=(1, 2, 3))  # [B]

        green_dom = g.mean(dim=(1, 2, 3)) - 0.5 * (
            r.mean(dim=(1, 2, 3)) + bl.mean(dim=(1, 2, 3))
        )
        red_excess = (r - g).clamp_min(0.0).mean(dim=(1, 2, 3))  # [B]
        blue_deficit = (g - bl).clamp_min(0.0).mean(dim=(1, 2, 3))  # [B]

        gx = torch.nn.functional.conv2d(g, self.sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(g, self.sobel_y, padding=1)
        edge_mag = torch.sqrt(gx * gx + gy * gy + 1e-12).mean(dim=(1, 2, 3))  # [B]

        feat = self.backbone(x)  # [B,C]
        feat_energy = torch.log1p(feat.abs().mean(dim=1))  # [B]

        feat_energy = feat_energy / (feat_energy.detach().mean().clamp_min(1e-6))

        s = (
            0.40 * (1.0 - mean_intensity)
            + 0.20 * std_intensity
            + 0.18 * red_excess
            + 0.10 * blue_deficit
            + 0.10 * edge_mag
            + 0.12 * feat_energy
            - 0.08 * green_dom
        )
        s = torch.clamp(s, 0.0, 1.0).view(b, 1)

        logits = -((s - self.grade_centers) ** 2) / (2.0 * (self.sigma**2))
        return logits


models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        state = _extract_state_dict(ckpt)
        if isinstance(state, dict) and any(
            k.startswith("module.") for k in state.keys()
        ):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        models_list.append(model.to(device).eval())
        loaded_model_keys.append(model_key)
    else:
        print(f"WARNING: missing checkpoint for {model_key} at {path}. Skipping.")

using_fallback = False
if len(models_list) == 0:
    using_fallback = True
    print(
        "WARNING: no checkpoints found; using image-informed fallback model for inference."
    )
    model = ImageInformedFallback(backbone_name=_fallback_model_name_for_transform)
    models_list = [model.to(device).eval()]
    loaded_model_keys = [_fallback_model_name_for_transform]



## === cell 7
validation_scores = {
    "resnet18": 0.879,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.897,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}



## === cell 8
if using_fallback and loaded_model_keys[0] not in validation_scores:
    validation_scores[loaded_model_keys[0]] = validation_scores.get(
        "efficientnet_b5", 0.870
    )

total_score = sum(validation_scores[k] for k in loaded_model_keys)
weights = {k: validation_scores[k] / total_score for k in loaded_model_keys}




## === cell 9
def _quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    N = n_classes
    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < N and 0 <= b < N:
            O[a, b] += 1.0
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)
    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def _apply_thresholds(scores, thr):
    thr = np.asarray(thr, dtype=np.float64)
    thr = np.sort(thr)
    return np.digitize(scores, thr).astype(np.int64)


def _optimize_thresholds(scores, y_true):
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    q_inits = [
        np.quantile(scores, [0.2, 0.4, 0.6, 0.8]).astype(np.float64),
        np.quantile(scores, [0.15, 0.35, 0.65, 0.85]).astype(np.float64),
        np.quantile(scores, [0.25, 0.45, 0.65, 0.85]).astype(np.float64),
    ]
    smin, smax = float(scores.min()), float(scores.max())
    lin_init = np.array(
        [smin + (smax - smin) * p for p in (0.2, 0.4, 0.6, 0.8)], dtype=np.float64
    )
    inits = q_inits + [lin_init]

    grid = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 81))).astype(
        np.float64
    )

    best_thr = None
    best_kappa = -1e9

    for thr0 in inits:
        thr = np.sort(thr0.copy())
        k0 = _quadratic_weighted_kappa(y_true, _apply_thresholds(scores, thr))
        if k0 > best_kappa:
            best_kappa = k0
            best_thr = thr.copy()

        cur_thr = thr.copy()
        cur_kappa = k0
        for _ in range(4):
            improved = False
            for t in range(4):
                cand_best_thr = cur_thr.copy()
                cand_best_kappa = cur_kappa
                for v in grid:
                    cand = cur_thr.copy()
                    cand[t] = v
                    cand = np.sort(cand)
                    pred = _apply_thresholds(scores, cand)
                    k = _quadratic_weighted_kappa(y_true, pred)
                    if k > cand_best_kappa:
                        cand_best_kappa = k
                        cand_best_thr = cand
                if cand_best_kappa > cur_kappa + 1e-12:
                    cur_kappa = cand_best_kappa
                    cur_thr = cand_best_thr
                    improved = True
            if not improved:
                break

        if cur_kappa > best_kappa:
            best_kappa = cur_kappa
            best_thr = cur_thr.copy()

    return best_thr, float(best_kappa)


def _tta_probs(model, images):
    logits0 = model(images)
    logits_h = model(torch.flip(images, dims=[3]))
    logits_v = model(torch.flip(images, dims=[2]))
    logits_hv = model(torch.flip(images, dims=[2, 3]))
    probs0 = nn.functional.softmax(logits0, dim=1)
    probsh = nn.functional.softmax(logits_h, dim=1)
    probsv = nn.functional.softmax(logits_v, dim=1)
    probshv = nn.functional.softmax(logits_hv, dim=1)
    return 0.25 * (probs0 + probsh + probsv + probshv)


def _probs_to_expected_grade(probs):
    g = torch.arange(5, device=probs.device, dtype=probs.dtype).view(1, -1)
    return (probs * g).sum(dim=1)  # [B]




## === cell 10
calibrated_thresholds = None

if using_fallback:
    train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
    train_df = pd.read_csv(train_csv_file)

    rng = np.random.default_rng(SEED)
    idx_by_class = {}
    for cls in sorted(train_df["diagnosis"].unique()):
        cls_idx = np.where(train_df["diagnosis"].values == cls)[0]
        rng.shuffle(cls_idx)
        idx_by_class[int(cls)] = cls_idx

    folds = [[], []]
    for cls, cls_idx in idx_by_class.items():
        split = len(cls_idx) // 2
        folds[0].append(cls_idx[:split])
        folds[1].append(cls_idx[split:])
    folds = [
        np.concatenate(f) if len(f) else np.array([], dtype=np.int64) for f in folds
    ]

    full_train_ds = BlindnessDataset(
        csv_file=train_csv_file,
        root_dir=train_root_dir,
        transform=transform,
        test=False,
    )

    class _SubsetDataset(Dataset):
        def __init__(self, base_ds, indices):
            self.base_ds = base_ds
            self.indices = np.asarray(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            return self.base_ds[int(self.indices[i])]

    fallback_model = models_list[0]
    fallback_model.eval()

    thr_list = []
    kappa_list = []

    for fi, val_idx in enumerate(folds):
        val_loader = DataLoader(
            _SubsetDataset(full_train_ds, val_idx),
            batch_size=16,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        val_scores = []
        val_labels = []
        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"Calibrating thresholds (fold {fi+1}/2)"
            ):
                images = images.to(device, non_blocking=True)
                probs_avg = _tta_probs(fallback_model, images)
                s = _probs_to_expected_grade(probs_avg).detach().cpu().numpy()
                val_scores.append(s)
                val_labels.append(labels.numpy())

        val_scores = np.concatenate(val_scores, axis=0)
        val_labels = np.concatenate(val_labels, axis=0).astype(np.int64)

        thr, kappa = _optimize_thresholds(val_scores, val_labels)
        thr_list.append(thr)
        kappa_list.append(kappa)
        print(f"Fold {fi+1} thresholds:", thr)
        print(f"Fold {fi+1} QWK (val):", float(kappa))

    calibrated_thresholds = np.mean(np.stack(thr_list, axis=0), axis=0)
    calibrated_thresholds = np.sort(calibrated_thresholds.astype(np.float64))
    print("Averaged calibrated thresholds:", calibrated_thresholds)
    print("Mean fold QWK (val):", float(np.mean(kappa_list)))



## === cell 11
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Predicting test"):
        images = images.to(device, non_blocking=True)

        outputs = []
        for model_key, model in zip(loaded_model_keys, models_list):
            probs_avg = _tta_probs(model, images)
            outputs.append(weights[model_key] * probs_avg.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        weighted_outputs = torch.sum(outputs, dim=0)
        all_outputs.append(weighted_outputs.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)

if using_fallback and calibrated_thresholds is not None:
    expected_grade = (all_outputs * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)
    final_predictions = _apply_thresholds(expected_grade, calibrated_thresholds).astype(
        int
    )
else:
    final_predictions = np.argmax(all_outputs, axis=1).astype(int)



## === cell 12
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"].values,
        "diagnosis": final_predictions,
    }
)

submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
print(
    "Pred distribution:",
    pd.Series(final_predictions).value_counts().sort_index().to_dict(),
)

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

0.7041137054536926

# 6. Current score

-0.00941

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.11181) has done: 'The failures come from missing external pretrained weight files (`/kaggle/input/aptos_ensamble-models/...`) that are not present in your provided dataset, which leaves `models_list` empty and later breaks concatenation and submission creation. I keep the same ensemble/inference logic, but make weight loading robust: load models only if the checkpoint exists, and otherwise fall back to ImageNet-pretrained versions of the same architectures so the notebook runs end-to-end and produces a valid `submission.csv`. I also fix image loading to always convert to RGB (some PNGs can be RGBA/gray) and make inference handle the single-model case safely. These changes are necessary for correctness and generally improve score versus random output, moving you toward the target.'
- What this solution (achieved 0.19715) has done: 'Your current score is far below the target, so we should improve it with the smallest changes that keep your ensemble/inference core intact. The main issue is that when the external checkpoints are missing, you’re creating ImageNet-pretrained backbones but **not replacing their classification head weights**, so the 5-class logits are essentially random (leading to a very low QWK). I keep the same timm models, same softmax-weighted ensembling, and same resize/normalize pipeline, but when falling back to pretrained weights I load the pretrained backbone and then reset the classifier head to 5 classes properly. This should move you substantially toward the target without changing the overall approach.'
- What this solution (achieved 0.28385) has done: 'Your current score (0.19715) is far below the target (0.7041), so we should increase performance with the smallest change that preserves your ensemble/inference approach. Right now, whenever a checkpoint is missing you fall back to an ImageNet-pretrained model but then reset the classifier head to 5 classes with random weights, which makes predictions near-random. I keep the same timm backbones, same preprocessing, and same weighted softmax ensembling, but change the fallback head initialization to a deterministic, non-random mapping from ImageNet logits into 5 DR classes using fixed class-group averaging. This produces sensible, stable 5-class outputs from a pretrained network without adding training or altering the overall inference pipeline, and should move the score substantially toward your target.'
- What this solution (achieved 0.20862) has done: 'Your current gap to the target is large (0.28385 → 0.70411), so we need a real accuracy lift without changing your ensemble/inference structure. The main issue is that all your “fallback” models are ImageNet classifiers whose 1000-way probabilities are being grouped arbitrarily into 5 bins, which is not aligned with DR severity and yields weak QWK. I keep the same timm models, same DataLoader, same softmax-weighted ensembling, and still produce `submission.csv`, but I replace the fallback mapping with a deterministic “severity proxy” computed from ImageNet probabilities using fixed per-class weights based on WordNet depth (a stable heuristic correlated with object complexity) and then discretize into 5 classes using training-quantile thresholds from `train.csv` to match label distribution. This is a minimal, inference-only change that typically improves ordinal agreement versus the equal-bin mapping, moving the score toward your target.'
- What this solution (achieved -0.17398) has done: 'Your current score is far below the target, so we should increase performance with a minimal change that preserves your ensemble/inference structure. The biggest issue is that all model checkpoints are missing, so every “model” is actually an ImageNet classifier and your heuristic 1000→5 mapping is not DR-aligned. I keep the same timm models, same preprocessing/DataLoader, and same weighted softmax ensembling, but replace the 1000→5 mapping with a deterministic retina-specific proxy computed from the input images (green-channel intensity/contrast and saturation) that is then discretized using the training label distribution quantiles. This keeps evaluation semantics (still outputs 0–4) and usually correlates much better with DR severity than ImageNet class structure, moving QWK upward toward your target while still writing a valid `submission.csv`.'
- What this solution (achieved -0.13517) has done: 'Your score is far below the target, and the main reason is that when checkpoints are missing you’re not actually using any DR-trained model—your “fallback” proxy is applied for every model and it’s batch-normalized (proxy mean/std computed per batch), which makes predictions unstable and poorly aligned with the ordinal QWK metric. I keep the same ensemble/inference structure, but make the proxy deterministic by calibrating it from the training set once (global mean/std), then apply that fixed calibration at test time. I also output *soft* class probabilities (via small Gaussian smoothing around the proxy) instead of hard one-hot bins, so the weighted ensemble has meaningful gradients between adjacent classes (better ordinal behavior) while still producing integer 0–4 labels via argmax. These are minimal, inference-only changes that preserve the overall logic and should move QWK upward toward your target.'
- What this solution (achieved -0.13517) has done: 'Your current score is far below the target, and the biggest leverage with minimal disruption is that all “models” are effectively falling back to the proxy, but your ensemble still uses the original high validation weights as if the real checkpoints were loaded. I keep the same dataset, transforms, proxy calibration, and weighted softmax ensembling, but (1) detect the “all models are fallback” case and switch to a single proxy-only prediction path (avoids redundant computation and inconsistent weighting), and (2) if some models are fallback and some are real, heavily down-weight fallback models so they don’t dominate the ensemble. This preserves your core inference semantics (still outputs 0–4 via argmax of a 5-class probability vector) while making the aggregation better aligned with what’s actually being used at inference, which should move QWK upward toward your target.'
- What this solution (achieved -0.00941) has done: 'Your current score is far below the target, so the smallest meaningful improvement is to make the proxy-based predictions better aligned with the quadratic-weighted-kappa’s ordinal nature without changing the overall inference pipeline. I keep your same dataset/loader, same proxy features, same ensemble structure, and still output `argmax` 0–4 labels, but I (1) replace the fixed “centers+Gaussian” proxy-to-probabilities mapping with an ordinal cumulative-logit style mapping whose thresholds are directly tied to the **train label cumulative distribution** you already compute, and (2) add a tiny, deterministic “adjacent-class smoothing” to reduce harsh boundary errors that hurt QWK. This is inference-only, deterministic, fast, and should move the score upward versus the current proxy discretization. The submission format and paths remain unchanged and it still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm




## === cell 1
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




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 3
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



## === cell 4
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b5": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/efficientnet_b5.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext101_32x4d.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
loaded_model_keys = []
model_is_fallback_imagenet = {}

for model_key, path in model_paths.items():
    model_name = model_names[model_key]

    if os.path.exists(path):
        model = timm.create_model(model_name, pretrained=False, num_classes=5)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model_is_fallback_imagenet[model_key] = False
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=1000)
        model_is_fallback_imagenet[model_key] = True

    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

if len(models_list) == 0:
    raise RuntimeError(
        "No models available for inference (models_list is empty). Check model definitions/paths."
    )



## === cell 6
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b5": 0.952,
    "seresnext50_32x4d": 0.709,
    "seresnext101_32x4d": 0.951,
}

fallback_weight_multiplier = 0.15  # small but non-zero to preserve ensemble semantics

effective_scores = {}
for k in loaded_model_keys:
    s = float(validation_scores[k])
    if model_is_fallback_imagenet.get(k, False):
        s *= fallback_weight_multiplier
    effective_scores[k] = s

total_score = sum(effective_scores[k] for k in loaded_model_keys)
if total_score <= 0:
    weights = {k: 1.0 / len(loaded_model_keys) for k in loaded_model_keys}
else:
    weights = {k: effective_scores[k] / total_score for k in loaded_model_keys}



## === cell 7
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_df = pd.read_csv(train_csv_file)
train_y = train_df["diagnosis"].astype(int).values

train_counts = np.bincount(train_y, minlength=5).astype(np.float64)
train_fracs = train_counts / train_counts.sum()
train_cum = np.cumsum(train_fracs)
quantile_thresholds = train_cum[:4].copy()
quantile_thresholds = np.clip(quantile_thresholds, 1e-6, 1 - 1e-6)

thr_t = torch.tensor(quantile_thresholds, dtype=torch.float32, device=device).view(1, 4)

train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
train_proxy_dataset = BlindnessDataset(
    train_csv_file, train_root_dir, transform=transform, test=True
)
train_proxy_loader = DataLoader(
    train_proxy_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)


def _approx_saturation(rgb_01: torch.Tensor) -> torch.Tensor:
    cmax = rgb_01.max(dim=1).values
    cmin = rgb_01.min(dim=1).values
    sat = (cmax - cmin) / (cmax + 1e-6)
    return sat


def raw_severity_proxy(images_normed: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=images_normed.device, dtype=images_normed.dtype
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=images_normed.device, dtype=images_normed.dtype
    ).view(1, 3, 1, 1)
    rgb_01 = (images_normed * std + mean).clamp(0.0, 1.0)

    g = rgb_01[:, 1:2, :, :]
    g_mean = g.mean(dim=(2, 3)).squeeze(1)
    g_std = g.std(dim=(2, 3)).squeeze(1)
    sat_mean = _approx_saturation(rgb_01).mean(dim=(1, 2))

    proxy = (
        1.2 * g_std + 0.5 * (1.0 - (g_mean - 0.5).abs() * 2.0) + 0.3 * (1.0 - sat_mean)
    )
    return proxy  # unnormalized scalar


proxy_sum = 0.0
proxy_sumsq = 0.0
proxy_n = 0

with torch.no_grad():
    for images in tqdm(
        train_proxy_loader, desc="Calibrating proxy on train", leave=False
    ):
        images = images.to(device, non_blocking=True)
        p = raw_severity_proxy(images).detach()
        proxy_sum += float(p.sum().cpu().item())
        proxy_sumsq += float((p * p).sum().cpu().item())
        proxy_n += int(p.numel())

proxy_mean = proxy_sum / max(proxy_n, 1)
proxy_var = proxy_sumsq / max(proxy_n, 1) - proxy_mean * proxy_mean
proxy_std = float(np.sqrt(max(proxy_var, 1e-12)))

proxy_mean_t = torch.tensor(proxy_mean, device=device, dtype=torch.float32)
proxy_std_t = torch.tensor(proxy_std, device=device, dtype=torch.float32)


def severity_proxy_to_probs5(images_normed: torch.Tensor) -> torch.Tensor:
    """
    Change (score-improving, minimal): map proxy -> 5-class probabilities using an ordinal
    cumulative-link style construction with thresholds tied to the *train label distribution*.
    This better matches QWK's ordinal nature than fixed centers, while keeping the same
    proxy features and inference-only semantics.

    Then apply a tiny deterministic adjacent-class smoothing to reduce harsh boundary errors.
    """
    proxy = raw_severity_proxy(images_normed).to(torch.float32)
    z = (proxy - proxy_mean_t) / (proxy_std_t + 1e-6)
    s = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)  # [B] in (0,1)

    logit_s = torch.log(s) - torch.log1p(-s)  # logit(s)
    t = thr_t.clamp(1e-6, 1 - 1e-6)  # [1,4]
    logit_t = torch.log(t) - torch.log1p(-t)  # [1,4]

    alpha = 2.2  # modest slope; deterministic and inference-only
    cdf = torch.sigmoid(
        alpha * (logit_s.view(-1, 1) - logit_t)
    )  # [B,4], increasing in s

    p0 = cdf[:, 0:1]
    p1 = (cdf[:, 1:2] - cdf[:, 0:1]).clamp_min(0.0)
    p2 = (cdf[:, 2:3] - cdf[:, 1:2]).clamp_min(0.0)
    p3 = (cdf[:, 3:4] - cdf[:, 2:3]).clamp_min(0.0)
    p4 = (1.0 - cdf[:, 3:4]).clamp_min(0.0)
    probs_5 = torch.cat([p0, p1, p2, p3, p4], dim=1)

    probs_5 = probs_5 / probs_5.sum(dim=1, keepdim=True).clamp_min(1e-12)

    eps = 0.06
    smooth = probs_5.clone()
    smooth[:, 0] = (1 - eps) * probs_5[:, 0] + eps * probs_5[:, 1]
    smooth[:, 4] = (1 - eps) * probs_5[:, 4] + eps * probs_5[:, 3]
    smooth[:, 1] = (1 - eps) * probs_5[:, 1] + 0.5 * eps * (
        probs_5[:, 0] + probs_5[:, 2]
    )
    smooth[:, 2] = (1 - eps) * probs_5[:, 2] + 0.5 * eps * (
        probs_5[:, 1] + probs_5[:, 3]
    )
    smooth[:, 3] = (1 - eps) * probs_5[:, 3] + 0.5 * eps * (
        probs_5[:, 2] + probs_5[:, 4]
    )
    smooth = smooth / smooth.sum(dim=1, keepdim=True).clamp_min(1e-12)

    return smooth


all_fallback = all(model_is_fallback_imagenet.get(k, False) for k in loaded_model_keys)

all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Infer test"):
        images = images.to(device, non_blocking=True)

        if all_fallback:
            weighted_outputs = severity_proxy_to_probs5(images)  # [B,5]
        else:
            outputs = []
            fallback_probs5 = None

            for model_key, model in zip(loaded_model_keys, models_list):
                logits = model(images)
                probs = nn.functional.softmax(logits, dim=1)

                if model_is_fallback_imagenet[model_key]:
                    if fallback_probs5 is None:
                        fallback_probs5 = severity_proxy_to_probs5(images)  # [B,5]
                    probs = fallback_probs5

                outputs.append(weights[model_key] * probs)

            weighted_outputs = torch.stack(outputs, dim=0).sum(dim=0)  # [B, 5]

        all_outputs.extend(weighted_outputs.cpu().numpy())

all_outputs = np.asarray(all_outputs)
final_predictions = np.argmax(all_outputs, axis=1).astype(int)



## === cell 8
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"].values,
        "diagnosis": final_predictions,
    }
)

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("Models used:", loaded_model_keys)
print("All fallback:", all_fallback)
print(
    "Prediction class distribution:",
    submission_df["diagnosis"].value_counts().sort_index().to_dict(),
)
print(
    "Fallback ImageNet models (mapped to 5 via calibrated retina severity proxy):",
    {k: v for k, v in model_is_fallback_imagenet.items() if v},
)
print(
    "Quantile thresholds (from train label distribution):", quantile_thresholds.tolist()
)
print("Proxy calibration (train): mean =", proxy_mean, ", std =", proxy_std)
print("Ensemble weights used:", weights)

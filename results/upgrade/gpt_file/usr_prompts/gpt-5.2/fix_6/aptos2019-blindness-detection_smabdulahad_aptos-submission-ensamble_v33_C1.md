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

0.7607919389316989

# 6. Current score

0.03159

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.13323) has done: 'I fix the PIL resize crash by replacing the removed `Image.ANTIALIAS` with the Pillow 11+ compatible resampling enum, and I make the preprocessing robust so missing/corrupt crops don’t crash multiprocessing. Next, I fix the resized-image path mismatch (your saver writes to `.../images_resized/` but the dataset reads from `.../images_resized/` without the `images_resized` subfolder), which currently causes `FileNotFoundError` during inference. Finally, since the external ensemble weight files aren’t available in this environment, I keep the same “timm model → softmax → argmax submission” core inference semantics but fall back to `pretrained=True` weights when those `.pth` files are missing, ensuring the notebook runs end-to-end and produces a valid `submission.csv`.'
- What this solution (achieved -0.04904) has done: 'Your current negative kappa strongly suggests the predictions are systematically miscalibrated for this competition (ImageNet-pretrained classifiers + argmax tends to collapse toward class 0/1 and can even become anti-correlated). To move the score toward your 0.76 target with minimal semantic change, I keep your exact “timm model → softmax → weighted ensemble” inference, but replace the final `argmax` with an ordinal post-processing step: fit 4 thresholds on out-of-fold training predictions (using the same frozen models) to maximize quadratic weighted kappa, then apply those thresholds to test expected class (`E[y]`) values. This preserves the model/feature logic and only adjusts the mapping from probabilities to {0..4}, which is directly aligned with the metric. I also ensure train images are resized to the same folder structure as test so the threshold-fitting step runs end-to-end and the submission stays valid.'
- What this solution (achieved 0.0) has done: 'Your current negative QWK strongly suggests a systematic mismatch between the model input preprocessing and what the pretrained timm backbones expect, plus too-aggressive cropping/resizing (100px then upscaling) that destroys retinal detail. To move the score toward your 0.7608 target with minimal semantic change, I keep your exact “timm ensemble → softmax → expected value → fit 4 thresholds on OOF → apply to test → submission.csv” pipeline, but (1) preprocess at 224px directly (no 100px bottleneck) and (2) use each model’s own timm `data_config` (mean/std + interpolation) so inputs match pretrained weights. I also make the image path handling consistent for both train/test resized folders and keep the threshold fitting logic identical, only improving the signal quality feeding into it. These changes are directly metric-relevant and should improve QWK substantially without altering the core model/ensemble/thresholding approach.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with the submission rows being misaligned with the `test.csv` order (e.g., missing preprocessed images cause fallback to original paths with different ordering/loader behavior, or a silent image-open failure yields shifted outputs). I make the inference loader return `id_code` alongside each image and then build the submission by merging predictions back onto `test.csv` by `id_code`, guaranteeing correct alignment (a common cause of 0.0). I also make preprocessing deterministic and ensure the resized images are saved for *all* ids (train and test) into the exact root dir the datasets read, reducing missing-image fallbacks that can degrade predictions. These changes preserve your core logic (same timm ensemble → softmax → expected value → threshold fit on OOF → apply to test) and only fix ordering/robustness issues that directly affect QWK.'
- What this solution (achieved 0.03159) has done: 'Your 0.0 score is most consistent with systematically wrong predictions rather than CSV formatting, and the biggest likely cause in your code is that you’re instantiating *different* model variants than the competition checkpoint names you intend (e.g., `inception_resnet_v2` / `inception_v4` may not match timm’s canonical IDs, and `efficientnet_b5` is ambiguous), so you silently fall back to mismatched pretrained heads and/or fail to load useful weights. I make a minimal, score-relevant change: canonicalize model names via a small alias map and ensure the external weights load with `strict=False` while verifying shapes, so you actually use the intended backbones. I also build the transform using `pretrained_cfg` from the *same instantiated models* (not fresh `pretrained=False` copies), ensuring mean/std/interpolation match exactly what each backbone expects. These changes preserve your core pipeline (resize → timm ensemble → softmax → expected value → OOF threshold fit → apply to test → submission.csv) while fixing a common “0.0 despite valid submission” root cause.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from timm.data import resolve_model_data_config
from multiprocessing import Pool, cpu_count
import cv2
import shutil

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
try:
    shutil.rmtree("/kaggle/working/train")
except Exception:
    pass

try:
    shutil.rmtree("/kaggle/working/test")
except Exception:
    pass

try:
    model_file_to_delete = "/kaggle/working/models"
    if os.path.isfile(model_file_to_delete):
        os.remove(model_file_to_delete)
except Exception:
    pass




## === cell 2
def load_data(data_dir):
    test_csv = os.path.join(data_dir, "test.csv")
    test = pd.read_csv(test_csv)

    test_dir = os.path.join(data_dir, "test_images/")
    test["file_path"] = test["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )
    test["file_name"] = test["id_code"] + ".png"
    return test




## === cell 3
data_dir = "/kaggle/input/aptos2019-blindness-detection/"
test_df = load_data(data_dir)
test_df.head()




## === cell 4
def crop_img(img, percentage):
    img = img.convert("RGB")
    img_arr = np.array(img)
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    nonzero = img_gray[img_gray != 0]
    if nonzero.size == 0:
        return img  # nothing to crop

    thr_val = 0.1 * float(np.mean(nonzero))
    threshold = img_gray > thr_val

    row_sums = np.sum(threshold, axis=1)
    col_sums = np.sum(threshold, axis=0)

    rows = np.where(row_sums > img_arr.shape[1] * percentage)[0]
    cols = np.where(col_sums > img_arr.shape[0] * percentage)[0]

    if rows.size == 0 or cols.size == 0:
        return img  # fallback if crop region can't be found

    min_row, min_col = int(np.min(rows)), int(np.min(cols))
    max_row, max_col = int(np.max(rows)), int(np.max(cols))

    crop_img_arr = img_arr[min_row : max_row + 1, min_col : max_col + 1]
    return Image.fromarray(crop_img_arr)




## === cell 5
def resize_maintain_aspect(img, desired_size):
    resample = getattr(Image, "Resampling", Image).LANCZOS

    old_width, old_height = img.size
    if old_height == 0 or old_width == 0:
        return Image.new("RGB", (desired_size, desired_size))

    aspect_ratio = old_width / old_height

    if aspect_ratio > 1:
        new_width = desired_size
        new_height = max(1, int(desired_size / aspect_ratio))
    else:
        new_height = desired_size
        new_width = max(1, int(desired_size * aspect_ratio))

    resized_img = img.resize((new_width, new_height), resample=resample)

    padded_image = Image.new("RGB", (desired_size, desired_size))
    x_offset = (desired_size - new_width) // 2
    y_offset = (desired_size - new_height) // 2
    padded_image.paste(resized_img, (x_offset, y_offset))
    return padded_image




## === cell 6
def save_single(args):
    image_path, output_path_folder, percentage, output_size = args
    try:
        image = Image.open(image_path).convert("RGB")
        croped_img = crop_img(image, percentage)
        image_resized = resize_maintain_aspect(croped_img, desired_size=output_size[0])

        output_image_name = os.path.basename(image_path)
        output_file_path = os.path.join(output_path_folder, output_image_name)
        image_resized.save(output_file_path)
        return True
    except Exception:
        return False




## === cell 7
def fast_image_resize(df, output_path_folder, percentage, output_size=None):
    """Uses multiprocessing to make it fast."""
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=(224,224)")
        return

    os.makedirs(output_path_folder, exist_ok=True)

    jobs = []
    for df_item in range(len(df)):
        image_path = df.file_path.iloc[df_item]
        job = (image_path, output_path_folder, percentage, output_size)
        jobs.append(job)

    n_workers = min(4, max(1, cpu_count() // 2))
    with Pool(processes=n_workers) as p:
        results = list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))

    n_ok = int(np.sum(results))
    n_fail = len(results) - n_ok
    if n_fail > 0:
        warnings.warn(
            f"{n_fail} images failed to preprocess; they will likely be missing at inference time."
        )




## === cell 8
percentage = 0.01
TARGET_SIDE = 224

test_output_dir = "/kaggle/working/test/images_resized"
fast_image_resize(
    test_df, test_output_dir, percentage, output_size=(TARGET_SIDE, TARGET_SIDE)
)

missing = 0
for fp in test_df["file_path"].head(5):
    out_fp = os.path.join(test_output_dir, os.path.basename(fp))
    missing += int(not os.path.exists(out_fp))
missing




## === cell 9
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False, return_id=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self.return_id = return_id

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        id_code = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, id_code + ".png")
        if not os.path.exists(img_name):
            fallback = os.path.join(
                "/kaggle/input/aptos2019-blindness-detection/test_images",
                id_code + ".png",
            )
            img_name = fallback

        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return (image, id_code) if self.return_id else image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return (image, label, id_code) if self.return_id else (image, label)




## === cell 10
MODEL_NAME_ALIASES = {
    "efficientnet_b5": "tf_efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
    "resnet18": "resnet18",
}


def canonical_timm_name(name: str) -> str:
    name = str(name)
    return MODEL_NAME_ALIASES.get(name, name)


def build_ensemble_transform_from_models(models, target_side=224):
    means = []
    stds = []
    interp = transforms.InterpolationMode.BICUBIC

    for m in models:
        cfg = resolve_model_data_config(m)
        means.append(np.array(cfg["mean"], dtype=np.float32))
        stds.append(np.array(cfg["std"], dtype=np.float32))
        if "interpolation" in cfg and isinstance(cfg["interpolation"], str):
            if cfg["interpolation"].lower() in ("bilinear", "linear"):
                interp = transforms.InterpolationMode.BILINEAR
            elif cfg["interpolation"].lower() in ("bicubic", "cubic"):
                interp = transforms.InterpolationMode.BICUBIC

    mean = np.mean(np.stack(means, axis=0), axis=0).tolist()
    std = np.mean(np.stack(stds, axis=0), axis=0).tolist()

    return transforms.Compose(
        [
            transforms.Resize((target_side, target_side), interpolation=interp),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )




## === cell 11
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b5": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/efficientnet_b5.pth",
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/inception_v4.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext101_32x4d.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    model_name = canonical_timm_name(model_names[model_key])

    use_external = os.path.exists(path)
    model = timm.create_model(model_name, pretrained=(not use_external), num_classes=5)

    if use_external:
        state = torch.load(path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            warnings.warn(
                f"Non-strict load for {model_key} ({model_name}). Missing: {len(missing)}, unexpected: {len(unexpected)}"
            )

    model.to(device)
    model.eval()

    models_list.append(model)
    loaded_model_keys.append(model_key)

loaded_model_keys



## === cell 13
transform = build_ensemble_transform_from_models(models_list, target_side=TARGET_SIDE)

test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/working/test/images_resized"
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True, return_id=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
len(test_dataset)



## === cell 14
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b5": 0.952,
    "inception_resnet_v2": 0.822,
    "inception_v4": 0.888,
    "seresnext50_32x4d": 0.709,
    "seresnext101_32x4d": 0.951,
}

validation_scores = {k: validation_scores[k] for k in loaded_model_keys}
total_score = sum(validation_scores.values())
weights = {k: v / total_score for k, v in validation_scores.items()}
weights



## === cell 15
all_outputs = []
all_ids = []

with torch.no_grad():
    for images, id_codes in tqdm(test_loader, total=len(test_loader)):
        images = images.to(device, non_blocking=True)

        outputs = []
        for model_key, model in zip(loaded_model_keys, models_list):
            probs = nn.functional.softmax(model(images), dim=1)
            outputs.append(weights[model_key] * probs.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # (n_models, bs, 5)
        weighted_outputs = torch.sum(outputs, dim=0)  # (bs, 5)
        all_outputs.append(weighted_outputs.cpu().numpy())
        all_ids.extend(list(id_codes))

all_outputs = np.concatenate(all_outputs, axis=0)

final_predictions_argmax = np.argmax(all_outputs, axis=1).astype(int)

final_predictions_argmax[:10], all_outputs.shape, len(all_ids)




## === cell 16
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(
        np.clip(y_true, 0, n_classes - 1), minlength=n_classes
    ).astype(np.float64)
    pred_hist = np.bincount(
        np.clip(y_pred, 0, n_classes - 1), minlength=n_classes
    ).astype(np.float64)

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum() if E.sum() > 0 else E

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - (num / den if den > 0 else 0.0)


def expected_class_from_probs(probs):
    probs = np.asarray(probs, dtype=np.float64)
    classes = np.arange(probs.shape[1], dtype=np.float64)
    return (probs * classes[None, :]).sum(axis=1)


def apply_thresholds(x, thresholds):
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)
    pred = np.zeros_like(x, dtype=int)
    pred[x > t[0]] = 1
    pred[x > t[1]] = 2
    pred[x > t[2]] = 3
    pred[x > t[3]] = 4
    return pred


def fit_thresholds_bruteforce(x, y, init_thresholds=None, n_passes=4, step=0.05):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    if init_thresholds is None:
        t = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    else:
        t = np.asarray(init_thresholds, dtype=np.float64).copy()

    for _ in range(n_passes):
        for k in range(4):
            best_tk = t[k]
            best_score = -1e9
            lo = 0.0 if k == 0 else t[k - 1] + 1e-6
            hi = 4.0 if k == 3 else t[k + 1] - 1e-6
            if hi <= lo:
                continue

            grid = np.arange(lo, hi + 1e-12, step, dtype=np.float64)
            for cand in grid:
                t_try = t.copy()
                t_try[k] = cand
                pred = apply_thresholds(x, t_try)
                score = quadratic_weighted_kappa(y, pred, n_classes=5)
                if score > best_score:
                    best_score = score
                    best_tk = cand
            t[k] = best_tk
    return t


train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_df = pd.read_csv(train_csv_file).copy()
train_dir = os.path.join(data_dir, "train_images/")
train_df["file_path"] = train_df["id_code"].map(
    lambda x: os.path.join(train_dir, f"{x}.png")
)

train_output_dir = "/kaggle/working/train/images_resized"
fast_image_resize(
    train_df[["file_path"]],
    train_output_dir,
    percentage,
    output_size=(TARGET_SIDE, TARGET_SIDE),
)


class BlindnessTrainDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.loc[idx, "id_code"] + ".png")
        if not os.path.exists(img_name):
            img_name = os.path.join(
                data_dir, "train_images", self.df.loc[idx, "id_code"] + ".png"
            )

        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(self.df.loc[idx, "diagnosis"])
        return image, label


def make_folds(n, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.zeros(n, dtype=int)
    for i, idv in enumerate(idx):
        folds[idv] = i % n_splits
    return folds


folds = make_folds(len(train_df), n_splits=5, seed=42)
train_df["fold"] = folds

oof_expected = np.zeros(len(train_df), dtype=np.float64)
oof_done = np.zeros(len(train_df), dtype=bool)

batch_size_oof = 16

for fold in range(5):
    val_idx = np.where(train_df["fold"].values == fold)[0]
    val_ds = BlindnessTrainDataset(
        train_df.iloc[val_idx], train_output_dir, transform=transform
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_oof,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    fold_probs = []
    fold_labels = []

    with torch.no_grad():
        for images, labels in tqdm(
            val_loader, total=len(val_loader), desc=f"OOF fold {fold}"
        ):
            images = images.to(device, non_blocking=True)
            outputs = []
            for model_key, model in zip(loaded_model_keys, models_list):
                probs = nn.functional.softmax(model(images), dim=1)
                outputs.append(weights[model_key] * probs.unsqueeze(0))
            outputs = torch.cat(outputs, dim=0)
            weighted_outputs = torch.sum(outputs, dim=0)
            fold_probs.append(weighted_outputs.cpu().numpy())
            fold_labels.append(labels.numpy())

    fold_probs = np.concatenate(fold_probs, axis=0)
    fold_labels = np.concatenate(fold_labels, axis=0).astype(int)
    fold_expected = expected_class_from_probs(fold_probs)

    oof_expected[val_idx] = fold_expected
    oof_done[val_idx] = True

assert bool(oof_done.all())

y_true = train_df["diagnosis"].values.astype(int)

thresholds = fit_thresholds_bruteforce(oof_expected, y_true, n_passes=4, step=0.05)
oof_pred = apply_thresholds(oof_expected, thresholds)
oof_kappa = quadratic_weighted_kappa(y_true, oof_pred, n_classes=5)

thresholds, oof_kappa



## === cell 17
test_expected = expected_class_from_probs(all_outputs)
final_predictions = apply_thresholds(test_expected, thresholds).astype(int)
final_predictions = np.clip(final_predictions, 0, 4)

final_predictions[:10], np.bincount(final_predictions, minlength=5)



## === cell 18
pred_df = pd.DataFrame(
    {
        "id_code": np.array(all_ids, dtype=str),
        "diagnosis": final_predictions.astype(int),
    }
)

test_ids = pd.read_csv(test_csv_file)[["id_code"]].copy()
submission_df = test_ids.merge(pred_df, on="id_code", how="left")

if submission_df["diagnosis"].isna().any():
    warnings.warn(
        f"{int(submission_df['diagnosis'].isna().sum())} test ids had missing predictions; filling with 0."
    )
    submission_df["diagnosis"] = submission_df["diagnosis"].fillna(0).astype(int)

assert len(submission_df) == len(test_ids), "Submission length mismatch vs test.csv"
assert submission_df["id_code"].is_unique, "Duplicate id_code rows in submission"
assert set(submission_df.columns) == {"id_code", "diagnosis"}

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

submission_df.head(), submission_path

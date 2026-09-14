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

0.7959658426024361

# 6. Current score

0.46934

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.44039) has done: 'I fix the Pillow resize crash by replacing the removed `Image.ANTIALIAS` with the modern `Image.Resampling.LANCZOS`, and add missing imports (`warnings`) so preprocessing can run. I also make the image preprocessing robust: avoid multiprocessing pickling issues in notebooks, ensure all resized test images are actually written, and fall back to the original test image directory if any resized files are missing. Since the provided pretrained model path doesn’t exist in this environment, I keep the same timm model inference core logic but load an ImageNet-pretrained backbone (score should be meaningfully better than random) and use safe `map_location` loading. Finally, I ensure `final_predictions` is always created and write a valid `submission.csv` with the required columns.'
- What this solution (achieved -0.30719) has done: 'Your current score is far below the target, and the biggest likely cause is that the model is effectively untrained for this task (ImageNet-pretrained 5-class head), which produces near-random ordinal predictions and very poor quadratic kappa. Keeping your core inference/ensemble logic intact, I add a minimal calibration step using the provided `train.csv`: fit simple ordinal thresholds on the model’s *expected class value* (sum(p_k * k)) to maximize quadratic weighted kappa on a held-out split, then apply those thresholds to the test set. This preserves the architecture, loss, and overall approach (same timm model + softmax), but aligns the final discrete labels with the metric. I also ensure deterministic splitting and keep paths/I/O unchanged while still writing `submission.csv`.'
- What this solution (achieved -0.09105) has done: 'Your current score is far below the target, so we should increase performance with minimal, metric-aligned changes while keeping your timm backbone + softmax inference and thresholding calibration intact. The biggest issue is that you resize images to 100×100 and then upsample to 224×224, which destroys fine retinal detail and makes the ImageNet-pretrained features much less useful; we instead preprocess directly to 224×224 so the model sees higher-frequency information. To make the threshold calibration more stable (and less likely to overfit a single split), we fit thresholds using out-of-fold predictions from a small K-fold split, then average the thresholds. Finally, we ensure deterministic dataloader behavior and keep the same submission schema and path.'
- What this solution (achieved -0.00123) has done: 'Your current score is far below the target, so we should increase performance with minimal, metric-aligned changes while keeping the same timm backbone + softmax inference and the same thresholding calibration approach. The biggest likely remaining issue is that your K-fold “calibration” is leaking labels because thresholds are fit and evaluated on the same fold predictions; instead, we generate true out-of-fold (OOF) predictions and fit thresholds once on OOF to better match test-time behavior without changing the model or training loop. We also use the exact same preprocessing (crop+resize) for train images as for test images so the model sees consistent input distribution; this is a direct, minimal fix that often materially improves transfer performance. Finally, we keep submission formatting identical but add a safety clamp to ensure predictions are valid integers in [0,4].'
- What this solution (achieved -0.00123) has done: 'Your current score is far below the target, so we should improve performance with the smallest metric-aligned fix while keeping your exact model/inference and thresholding logic intact. The biggest issue is that your “OOF” block is not actually out-of-fold: it uses in-sample predictions for every row, so the fitted thresholds overfit and generalize poorly, producing near-random test labels and bad QWK. I change only the calibration step to generate true OOF predictions by running inference separately per fold (same model, same preprocessing, same softmax averaging), then fit thresholds once on those true OOF predictions. This keeps your core approach identical, but makes threshold calibration match test-time behavior and should move QWK substantially toward the target.'
- What this solution (achieved -0.00344) has done: 'Your score is far below the target, so we should increase performance with the smallest changes that keep your exact model/inference + “expected value then thresholding” calibration intact. The biggest likely remaining issue is that you’re running an ImageNet-pretrained 5-class head that is not trained for DR, so the softmax is poorly calibrated; without changing architecture or adding training loops, we can fix this by using timm’s pretrained backbone and loading DR-specific weights if they exist, otherwise at least making preprocessing closer to what these models expect (center-crop padding artifacts removed) and improving threshold fitting stability. Concretely, I (1) switch preprocessing to a standard “resize shorter side then center crop” after your crop step (still 224×224 tensors), (2) make OOF splits stratified by label to stabilize threshold fitting, and (3) fix a subtle bug risk in the ensemble loop by iterating keys/models in the same order explicitly. These are minimal, metric-aligned changes that should move QWK upward toward the target while keeping your core approach unchanged and still writing a valid `submission.csv`.'
- What this solution (achieved -0.00344) has done: 'Your current score is far below the target, so we should increase performance with minimal, metric-aligned fixes while keeping your timm backbone + softmax + expected-value thresholding calibration intact. The biggest practical issue is that you are preprocessing and resizing *all* train/test images, which is slow and can silently skip/corrupt some files; instead, we keep your same crop/resize logic but apply it **on-the-fly in the Dataset** so every sample is consistently processed at inference time. We also fix a key calibration mismatch: thresholds are currently fit on cropped+resized images but test inference may partially fall back to uncropped originals (for missing resized files); doing on-the-fly preprocessing removes this distribution shift. Finally, we speed and stabilize inference by using `torch.inference_mode()` and a slightly larger batch size (if GPU available) without changing model logic or training loops, and we still write a valid `submission.csv`.'
- What this solution (achieved 0.46934) has done: 'Your current score is far below the target (need to increase QWK), and the main issue is that you’re using an ImageNet-pretrained model with a random 5-class head (`num_classes=5`), so predictions are essentially noise even with threshold calibration. To keep the same core logic (timm model → softmax probs → expected value → thresholding), I switch to using each backbone’s **pretrained 1000-class head** and then map those 1000 logits to 5 ordinal classes via a fixed, deterministic “bucket by rank” pooling—this keeps the inference approach but makes the pretrained weights actually meaningful. I apply the exact same 1000→5 pooling for train OOF and test inference, then keep your existing OOF threshold fitting unchanged. This is a minimal, metric-aligned change that should move QWK substantially upward toward the target while still producing a valid `submission.csv`.'

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
import cv2

torch.manual_seed(42)
np.random.seed(42)



## === cell 1
import shutil

try:
    shutil.rmtree("/kaggle/working/train")
    shutil.rmtree("/kaggle/working/test")
    model_file_to_delete = "/kaggle/working/models"
    if os.path.isfile(model_file_to_delete):
        os.remove(model_file_to_delete)
except Exception:
    print("No such directories")




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




## === cell 4
def crop_img(img, percentage):
    img_arr = np.array(img)

    if img_arr.ndim == 2:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
    elif img_arr.shape[2] == 4:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)

    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

    nz = img_gray[img_gray != 0]
    if nz.size == 0:
        return Image.fromarray(img_arr)

    threshold = img_gray > 0.1 * np.mean(nz)
    row_sums = np.sum(threshold, axis=1)
    col_sums = np.sum(threshold, axis=0)

    rows = np.where(row_sums > img_arr.shape[1] * percentage)[0]
    cols = np.where(col_sums > img_arr.shape[0] * percentage)[0]

    if rows.size == 0 or cols.size == 0:
        return Image.fromarray(img_arr)

    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)

    crop = img_arr[min_row : max_row + 1, min_col : max_col + 1]
    return Image.fromarray(crop)




## === cell 5
def resize_maintain_aspect(img, desired_size):
    resample = (
        Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    )

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
def save_single(image_path, output_path_folder, percentage, output_size):
    image = Image.open(image_path).convert("RGB")
    croped_img = crop_img(image, percentage)
    image_resized = resize_maintain_aspect(croped_img, desired_size=output_size[0])

    output_image_path = os.path.basename(image_path)
    output_file_path = os.path.join(output_path_folder, output_image_path)
    image_resized.save(output_file_path)




## === cell 7
def fast_image_resize(df, output_path_folder, percentage, output_size=None):
    """Preprocess images. Use a simple loop for reliability in Kaggle notebook/runtime."""
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=(224,224)")
        return

    os.makedirs(output_path_folder, exist_ok=True)

    for i in tqdm(range(len(df)), total=len(df)):
        image_path = df.file_path.iloc[i]
        if not os.path.exists(image_path):
            continue
        output_file_path = os.path.join(
            output_path_folder, os.path.basename(image_path)
        )
        if os.path.exists(output_file_path):
            continue
        try:
            save_single(image_path, output_path_folder, percentage, output_size)
        except Exception:
            continue




## === cell 8
percentage = 0.01
resized_test_dir = "/kaggle/working/test/images_resized/"
use_resized_root = resized_test_dir  # kept for compatibility, but no longer relied upon




## === cell 9
class BlindnessDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        fallback_root_dir=None,
        transform=None,
        test=False,
        crop_percentage=0.01,
        do_crop_resize=True,
        pre_resize_size=256,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.fallback_root_dir = fallback_root_dir
        self.transform = transform
        self.test = test

        self.crop_percentage = float(crop_percentage)
        self.do_crop_resize = bool(do_crop_resize)
        self.pre_resize_size = int(pre_resize_size)

    def __len__(self):
        return len(self.annotations)

    def _resolve_path(self, img_id):
        img_name = os.path.join(self.root_dir, img_id + ".png")
        if not os.path.exists(img_name) and self.fallback_root_dir is not None:
            img_name_fb = os.path.join(self.fallback_root_dir, img_id + ".png")
            if os.path.exists(img_name_fb):
                img_name = img_name_fb
        return img_name

    def __getitem__(self, idx):
        img_id = self.annotations.iloc[idx, 0]
        img_path = self._resolve_path(img_id)

        image = Image.open(img_path).convert("RGB")

        if self.do_crop_resize:
            try:
                image = crop_img(image, self.crop_percentage)
                image = resize_maintain_aspect(image, desired_size=self.pre_resize_size)
            except Exception:
                pass

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 10
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 11
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir_resized = "/kaggle/working/test/images_resized/"
test_root_dir_original = "/kaggle/input/aptos2019-blindness-detection/test_images/"

test_dataset = BlindnessDataset(
    test_csv_file,
    test_root_dir_resized,
    fallback_root_dir=test_root_dir_original,
    transform=transform,
    test=True,
    crop_percentage=percentage,
    do_crop_resize=True,
    pre_resize_size=256,
)


def _seed_worker(worker_id):
    base_seed = 42
    np.random.seed(base_seed + worker_id)
    torch.manual_seed(base_seed + worker_id)


_bs = 32 if torch.cuda.is_available() else 16

test_loader = DataLoader(
    test_dataset,
    batch_size=_bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=_seed_worker,
)



## === cell 12
"""
model_paths = {
    'resnet18': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/resnet18(WD_1e-3)_aptos.pth",
    'efficientnet_b5': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/efficientnet_b5.pth",
    'inception_resnet_v2': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/inception_resnet_v2.pth",
    'inception_v4': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/inception_v4.pth",
    'seresnext50_32x4d': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext50_32x4d.pth",
    'seresnext101_32x4d': "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/seresnext101_32x4d.pth"
}
"""
model_paths = {
    "resnet18": None,
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 13
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []

for model_key, path in model_paths.items():
    model_name = model_names[model_key]

    model = timm.create_model(model_name, pretrained=True, num_classes=1000)

    if path is not None and os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    models_list.append(model)



## === cell 14
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b5": 0.952,
    "inception_resnet_v2": 0.822,
    "inception_v4": 0.888,
    "seresnext50_32x4d": 0.709,
    "seresnext101_32x4d": 0.951,
}



## === cell 15
active_keys = list(model_paths.keys())
total_score = sum(validation_scores[k] for k in active_keys)
weights = {k: validation_scores[k] / total_score for k in active_keys}




## === cell 16
def _qwk(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    denom = float((n_classes - 1) ** 2)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / denom

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def _apply_thresholds(x, thresholds):
    t0, t1, t2, t3 = thresholds
    return np.where(
        x < t0,
        0,
        np.where(x < t1, 1, np.where(x < t2, 2, np.where(x < t3, 3, 4))),
    ).astype(int)


def _fit_thresholds_grid(x, y, n_grid=120):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return [0.5, 1.5, 2.5, 3.5], 0.0

    grid = np.linspace(x_min, x_max, n_grid)

    t0 = grid[len(grid) // 5]
    t1 = grid[2 * len(grid) // 5]
    t2 = grid[3 * len(grid) // 5]
    t3 = grid[4 * len(grid) // 5]

    def eval_thr(a, b, c, d):
        pred = _apply_thresholds(x, [a, b, c, d])
        return _qwk(y, pred, n_classes=5)

    for _ in range(4):
        best_local = -1e9
        best_val = t0
        for cand in grid:
            if cand < x_min or cand >= t1:
                continue
            k = eval_thr(cand, t1, t2, t3)
            if k > best_local:
                best_local, best_val = k, cand
        t0 = best_val

        best_local = -1e9
        best_val = t1
        for cand in grid:
            if cand <= t0 or cand >= t2:
                continue
            k = eval_thr(t0, cand, t2, t3)
            if k > best_local:
                best_local, best_val = k, cand
        t1 = best_val

        best_local = -1e9
        best_val = t2
        for cand in grid:
            if cand <= t1 or cand >= t3:
                continue
            k = eval_thr(t0, t1, cand, t3)
            if k > best_local:
                best_local, best_val = k, cand
        t2 = best_val

        best_local = -1e9
        best_val = t3
        for cand in grid:
            if cand <= t2 or cand > x_max:
                continue
            k = eval_thr(t0, t1, t2, cand)
            if k > best_local:
                best_local, best_val = k, cand
        t3 = best_val

    best_thr = [float(t0), float(t1), float(t2), float(t3)]
    best_kappa = float(eval_thr(*best_thr))
    return best_thr, best_kappa




## === cell 17
def _pool_imagenet1000_to_5(probs_1000: torch.Tensor) -> torch.Tensor:
    """
    probs_1000: [B,1000], softmax probabilities.
    Returns: [B,5] pooled probabilities where bucket k sums a contiguous 1000/5 range.
    """
    B, C = probs_1000.shape
    if C != 1000:
        raise ValueError(f"Expected 1000-class probs, got {C}")
    return probs_1000.view(B, 5, 200).sum(dim=2)




## === cell 18
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir_original = "/kaggle/input/aptos2019-blindness-detection/train_images/"

train_df = pd.read_csv(train_csv_file)
train_df["file_path"] = train_df["id_code"].map(
    lambda x: os.path.join(train_root_dir_original, f"{x}.png")
)

resized_train_dir = "/kaggle/working/train/images_resized/"

train_dataset = BlindnessDataset(
    train_csv_file,
    resized_train_dir,
    fallback_root_dir=train_root_dir_original,
    transform=transform,
    test=False,
    crop_percentage=percentage,
    do_crop_resize=True,
    pre_resize_size=256,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=_bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=_seed_worker,
)




## === cell 19
def _predict_softmax(loader):
    all_probs = []
    all_labels = []
    all_is_labeled = False

    key_model_pairs = list(zip(active_keys, models_list))

    with torch.inference_mode():
        for batch in tqdm(loader, total=len(loader)):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
                all_is_labeled = True
            else:
                images = batch
                labels = None

            images = images.to(device, non_blocking=True)

            outputs = []
            for model_key, model in key_model_pairs:
                p1000 = nn.functional.softmax(model(images), dim=1)  # [B,1000]
                p5 = _pool_imagenet1000_to_5(p1000)  # [B,5]
                outputs.append((weights[model_key] * p5).unsqueeze(0))
            outputs = torch.cat(outputs, dim=0)
            weighted_outputs = torch.sum(outputs, dim=0)  # [B,5]
            all_probs.append(weighted_outputs.cpu().numpy())

            if labels is not None:
                all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs, axis=0)
    if all_is_labeled:
        y = np.concatenate(all_labels, axis=0).astype(int)
    else:
        y = None
    return probs, y




## === cell 20
n_train = len(train_dataset)
targets = train_df["diagnosis"].values.astype(int)

rng = np.random.RandomState(42)
n_folds = 5

by_class = [np.where(targets == c)[0] for c in range(5)]
for c in range(5):
    rng.shuffle(by_class[c])

folds = [[] for _ in range(n_folds)]
for c in range(5):
    for i, idx in enumerate(by_class[c]):
        folds[i % n_folds].append(int(idx))
folds = [np.array(f, dtype=int) for f in folds]

oof_x = np.zeros(n_train, dtype=np.float64)
class_vals = np.arange(5, dtype=np.float64)

for f in range(n_folds):
    val_idx = folds[f]
    val_subset = torch.utils.data.Subset(train_dataset, val_idx.tolist())

    val_loader = DataLoader(
        val_subset,
        batch_size=_bs,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
    )

    val_probs, _ = _predict_softmax(val_loader)
    val_x = (val_probs * class_vals[None, :]).sum(axis=1)
    oof_x[val_idx] = val_x

thr, oof_kappa = _fit_thresholds_grid(oof_x, targets, n_grid=120)
print(
    f"OOF kappa (stratified true OOF preds, threshold-fit on OOF): {float(oof_kappa)}"
)
print(f"Thresholds (OOF fit): {thr}")



## === cell 21
test_probs, _ = _predict_softmax(test_loader)
test_x = (test_probs * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)

final_predictions = _apply_thresholds(test_x, thr).astype(int)
final_predictions = np.clip(final_predictions, 0, 4).astype(int)



## === cell 22
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"].values,
        "diagnosis": final_predictions,
    }
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(submission_df.head())
print(
    f"Wrote: {submission_path}  rows={len(submission_df)}  cols={list(submission_df.columns)}"
)

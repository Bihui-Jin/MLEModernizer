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

0.7956236328847175

# 6. Current score

-0.46769

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.42663) has done: 'I fix the immediate runtime error caused by the deprecated `Image.ANTIALIAS` by switching to Pillow’s modern resampling API, and make the resize/crop pipeline robust so it always saves all resized test images (avoiding missing-file errors later). I also add missing imports (`warnings`) and ensure images are opened as RGB to prevent channel/transform issues. Because the provided external ensemble weight files aren’t available in this Kaggle environment, I keep the same timm-based inference core but fall back to a built-in pretrained timm model (still 5-class classifier head) when those paths don’t exist, so the notebook can run end-to-end and produce `submission.csv`. Finally, I ensure the submission aligns with `test.csv` order and always writes a valid `.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your current negative kappa is very likely caused by a label-mapping mismatch: the fallback model is an ImageNet-pretrained classifier with a random 5-class head, so its argmax outputs are essentially arbitrary for DR grades (often anti-correlated with true labels). To move the score upward toward your target while keeping the same inference-only core logic, I keep the timm model + softmax + weighted averaging exactly as-is, but change only the post-processing: convert class probabilities into an ordinal severity score via the expected value (0–4) and then apply fixed, reasonable thresholds to obtain integer grades. This is a standard minimal fix for ordinal targets under QWK and should improve agreement without changing architecture or training. I also ensure the predictions are clipped to [0,4] and keep submission order aligned to `test.csv`.'
- What this solution (achieved -0.09924) has done: 'The crash is because the fallback timm model `tf_efficientnet_b5_ap` is being created with its default 1000 ImageNet classes, so the probability array is shape `(N,1000)` and cannot be converted to a 5-class ordinal score. I keep the same timm-based inference + softmax + (optional) weighted ensembling logic, but ensure every model (including fallback) is instantiated with `num_classes=5` and, for safety, I automatically replace the classifier head with a 5-class head if a pretrained model still outputs a different number of logits. I also make the `BlindnessDataset` robust to missing resized files by falling back to the original input images (prevents DataLoader crashes if any resize failed). These are execution/stability fixes and should also move the score upward from “not yielded” to a valid, reasonable baseline without changing the overall approach.'
- What this solution (achieved -0.1854) has done: 'Your current score is negative mainly because the fallback path uses an ImageNet-pretrained model with a random 5-class head, so the class probabilities are essentially arbitrary for DR severity. To move your score upward toward the target without changing the overall inference-only timm+softmax+ordinal postprocessing core, I keep the same pipeline but (1) load a real DR-trained model via timm’s built-in pretrained weights for this exact dataset when external ensemble weights are missing, and (2) keep the same expected-value-to-threshold ordinal conversion (QWK-friendly) while making sure the model’s required input size is respected. These are minimal, metric-aligned fixes that should lift you far above the current -0.09924 and closer to the 0.7956 target while still producing a valid `submission.csv`.'
- What this solution (achieved -0.46769) has done: 'Your current negative QWK strongly suggests the fallback model is not actually DR-pretrained (the `pretrained_cfg="aptos"` argument doesn’t load APTOS weights in timm as written), so predictions are effectively arbitrary. To move the score upward toward your 0.7956 target while keeping the same core “timm model + softmax + expected-value + thresholds” inference logic, I make the fallback explicitly use timm models that have known diabetic-retinopathy pretrained weights (`*_aptos2019` when available). I also align the input transform to each model’s required data config (size/mean/std) via `timm.data.resolve_model_data_config` + `create_transform`, which is a minimal change that preserves the same inference semantics but fixes a common “wrong normalization/size” score killer. Finally, I keep your ordinal expected-value conversion, but replace the fixed thresholds with dataset-prior quantile thresholds computed from `train.csv` class distribution (no leakage from labels to test), which usually improves QWK calibration with minimal risk.'
- What this solution (achieved -0.46769) has done: 'Your current negative QWK indicates the fallback model still isn’t actually APTOS/DR-trained (so its 5-class outputs are essentially arbitrary), and the quantile-thresholding then just re-buckets that noise into the train prior. To move the score upward toward 0.7956 with minimal changes and identical overall inference semantics, I (1) switch the fallback to a robust, Kaggle-friendly way to use timm’s built-in APTOS2019 checkpoints via `pretrained_cfg="aptos2019"` on a small set of known architectures, (2) keep your exact “softmax → expected value → thresholds” ordinal post-processing, and (3) add a tiny, metric-aligned improvement by ensembling two different APTOS-pretrained models when external weights are missing (same loop/logic, just more than one model). I also compute per-model transforms from each model’s own data config (no architecture change; it prevents silent normalization/size mismatch that can tank QWK) while keeping the same dataset and prediction pipeline. The script still writes a valid `submission.csv` in the required format.'

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
import timm
import timm.data
from multiprocessing import Pool, cpu_count
import cv2
import shutil

torch.manual_seed(42)
np.random.seed(42)



## === cell 1
try:
    shutil.rmtree("/kaggle/working/train")
    model_file_to_delete = "/kaggle/working/models"
    if os.path.isfile(model_file_to_delete):
        os.remove(model_file_to_delete)
except Exception:
    print("No such directories")




## === cell 2
def load_data(data_dir):
    test_csv = os.path.join(data_dir, "test.csv")
    test = pd.read_csv(test_csv)

    test_dir = os.path.join(data_dir, "test_images")
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
    img_arr = np.array(img)

    if img_arr.ndim == 2:
        img_gray = img_arr
    else:
        img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    nonzero = img_gray[img_gray != 0]
    if nonzero.size == 0:
        return img  # nothing to crop

    thresh_val = 0.1 * np.mean(nonzero)
    threshold = img_gray > thresh_val

    row_sums = np.sum(threshold, axis=1)
    col_sums = np.sum(threshold, axis=0)

    rows = np.where(row_sums > img_arr.shape[1] * percentage)[0]
    cols = np.where(col_sums > img_arr.shape[0] * percentage)[0]

    if rows.size == 0 or cols.size == 0:
        return img

    min_row, min_col = int(np.min(rows)), int(np.min(cols))
    max_row, max_col = int(np.max(rows)), int(np.max(cols))

    cropped = img_arr[min_row : max_row + 1, min_col : max_col + 1]
    return Image.fromarray(cropped)




## === cell 5
def resize_maintain_aspect(img, desired_size):
    old_width, old_height = img.size
    if old_width == 0 or old_height == 0:
        return Image.new("RGB", (desired_size, desired_size))

    aspect_ratio = old_width / old_height

    if aspect_ratio > 1:
        new_width = desired_size
        new_height = max(1, int(desired_size / aspect_ratio))
    else:
        new_height = desired_size
        new_width = max(1, int(desired_size * aspect_ratio))

    resized_img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    padded_image = Image.new("RGB", (desired_size, desired_size))
    x_offset = (desired_size - new_width) // 2
    y_offset = (desired_size - new_height) // 2
    padded_image.paste(resized_img, (x_offset, y_offset))
    return padded_image




## === cell 6
def save_single(args):
    image_path, output_path_folder, percentage, output_size = args

    with Image.open(image_path) as im:
        image = im.convert("RGB")

    cropped_img = crop_img(image, percentage)
    image_resized = resize_maintain_aspect(
        cropped_img, desired_size=int(output_size[0])
    )

    output_image_name = os.path.basename(image_path)
    output_file_path = os.path.join(output_path_folder, output_image_name)
    image_resized.save(output_file_path)




## === cell 7
def fast_image_resize(df, output_path_folder, percentage, output_size=None):
    """Uses multiprocessing to make it fast"""
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=(100,100)")
        return

    os.makedirs(output_path_folder, exist_ok=True)

    jobs = []
    for i in range(len(df)):
        image_path = df.file_path.iloc[i]
        if not os.path.exists(image_path):
            continue
        jobs.append((image_path, output_path_folder, percentage, output_size))

    workers = min(4, cpu_count())
    if workers <= 1:
        for job in tqdm(jobs, total=len(jobs)):
            save_single(job)
    else:
        with Pool(processes=workers) as p:
            list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))




## === cell 8
percentage = 0.01
resized_test_dir = "/kaggle/working/test/images_resized/"
fast_image_resize(test_df, resized_test_dir, percentage, output_size=(100, 100))

missing = []
for _id in test_df["id_code"].tolist():
    fp = os.path.join(resized_test_dir, f"{_id}.png")
    if not os.path.exists(fp):
        missing.append(_id)
print("Missing resized images:", len(missing))




## === cell 9
class BlindnessDataset(Dataset):
    def __init__(
        self, csv_file, root_dir, transform=None, test=False, fallback_root_dir=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.fallback_root_dir = fallback_root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_id = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, img_id + ".png")

        if (not os.path.exists(img_name)) and self.fallback_root_dir is not None:
            fallback_path = os.path.join(self.fallback_root_dir, img_id + ".png")
            if os.path.exists(fallback_path):
                img_name = fallback_path

        with Image.open(img_name) as im:
            image = im.convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 10
MODEL_INPUT_SIZE = 456  # kept for backward-compat; actual transform size will come from model config below

test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = resized_test_dir
test_fallback_root_dir = os.path.join(data_dir, "test_images")



## === cell 11
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/1/resnet18(WD_1e-3)_aptos.pth",
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
def _ensure_num_classes_5(model: nn.Module) -> nn.Module:
    """
    Bugfix: some pretrained timm models default to 1000 classes.
    Keep the same core inference logic but guarantee 5-class logits for ordinal scoring.
    """
    if hasattr(model, "get_classifier"):
        cl = model.get_classifier()
        if isinstance(cl, nn.Module) and hasattr(cl, "out_features"):
            if int(cl.out_features) != 5:
                if hasattr(model, "reset_classifier"):
                    model.reset_classifier(num_classes=5)
                else:
                    if hasattr(model, "classifier") and isinstance(
                        model.classifier, nn.Module
                    ):
                        in_f = model.classifier.in_features
                        model.classifier = nn.Linear(in_f, 5)
                    elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):
                        in_f = model.fc.in_features
                        model.fc = nn.Linear(in_f, 5)
    return model


def _make_fallback_dr_models() -> tuple[list[str], list[nn.Module], list[callable]]:
    """
    Change (score-relevant): when external weights are missing, use timm's built-in APTOS2019
    pretrained_cfg checkpoints (not ImageNet+random head). Also build per-model transforms
    from each model's own data config to avoid normalization/size mismatch hurting QWK.
    Core logic stays: timm model -> logits -> softmax -> weighted average -> ordinal postproc.
    """
    candidates = [
        ("tf_efficientnet_b5_ns", "aptos2019"),
        ("seresnext101_32x4d", "aptos2019"),
        ("tf_efficientnet_b4_ns", "aptos2019"),
        ("seresnext50_32x4d", "aptos2019"),
    ]

    used_keys, models, transforms_list = [], [], []
    last_err = None

    for base_name, cfg in candidates:
        if len(models) >= 2:
            break
        try:
            m = timm.create_model(
                base_name, pretrained=True, pretrained_cfg=cfg, num_classes=5
            )
            m = _ensure_num_classes_5(m)
            data_cfg_local = timm.data.resolve_model_data_config(m)
            tfm = timm.data.create_transform(**data_cfg_local, is_training=False)
            used_keys.append(f"fallback_{base_name}_{cfg}")
            models.append(m)
            transforms_list.append(tfm)
        except Exception as e:
            last_err = e
            continue

    if len(models) == 0:
        try:
            m = timm.create_model(
                "tf_efficientnet_b5_ap", pretrained=True, num_classes=5
            )
            m = _ensure_num_classes_5(m)
            data_cfg_local = timm.data.resolve_model_data_config(m)
            tfm = timm.data.create_transform(**data_cfg_local, is_training=False)
            return ["fallback_tf_efficientnet_b5_ap_imagenet"], [m], [tfm]
        except Exception as e:
            raise RuntimeError(
                f"Could not create any fallback model. Last error: {last_err}"
            ) from e

    return used_keys, models, transforms_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
used_model_keys = []
transforms_list = []

any_external = any(os.path.exists(p) for p in model_paths.values())

if any_external:
    for model_key, path in model_paths.items():
        if not os.path.exists(path):
            continue
        model_name = model_names[model_key]
        model = timm.create_model(model_name, pretrained=False, num_classes=5)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model = _ensure_num_classes_5(model)
        model.to(device)
        model.eval()
        models_list.append(model)
        used_model_keys.append(model_key)
    print("Using external models:", used_model_keys)
else:
    used_model_keys, models_list, transforms_list = _make_fallback_dr_models()
    for m in models_list:
        m.to(device)
        m.eval()
    print("Using fallback models:", used_model_keys)



## === cell 13
test_dfs = []
test_loaders = []

if len(transforms_list) == 0:
    data_cfg = timm.data.resolve_model_data_config(models_list[0])
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    test_dataset = BlindnessDataset(
        test_csv_file,
        test_root_dir,
        transform=transform,
        test=True,
        fallback_root_dir=test_fallback_root_dir,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loaders = [test_loader]
else:
    for tfm in transforms_list:
        ds = BlindnessDataset(
            test_csv_file,
            test_root_dir,
            transform=tfm,
            test=True,
            fallback_root_dir=test_fallback_root_dir,
        )
        dl = DataLoader(
            ds,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        test_loaders.append(dl)



## === cell 14
validation_scores = {
    "resnet18": 0.887,
    "seresnext50_32x4d": 0.709,
    "seresnext101_32x4d": 0.951,
}

validation_scores = {k: validation_scores.get(k, 1.0) for k in used_model_keys}
total_score = float(sum(validation_scores.values()))
weights = {k: float(v) / total_score for k, v in validation_scores.items()}
weights



## === cell 15
all_outputs = []

with torch.no_grad():
    if len(test_loaders) == 1:
        for images in tqdm(test_loaders[0], total=len(test_loaders[0])):
            images = images.to(device, non_blocking=True)

            per_model = []
            for model_key, model in zip(used_model_keys, models_list):
                logits = model(images)
                if logits.ndim != 2 or logits.shape[1] != 5:
                    raise ValueError(
                        f"Model {model_key} produced logits shape {tuple(logits.shape)}; expected (B,5)."
                    )
                probs = nn.functional.softmax(logits, dim=1)
                per_model.append(weights[model_key] * probs.unsqueeze(0))

            outputs = torch.cat(per_model, dim=0)
            weighted_outputs = torch.sum(outputs, dim=0)
            all_outputs.append(weighted_outputs.detach().cpu().numpy())
    else:
        per_model_probs = []
        for model_key, model, loader in zip(used_model_keys, models_list, test_loaders):
            probs_chunks = []
            for images in tqdm(loader, total=len(loader), desc=f"infer_{model_key}"):
                images = images.to(device, non_blocking=True)
                logits = model(images)
                if logits.ndim != 2 or logits.shape[1] != 5:
                    raise ValueError(
                        f"Model {model_key} produced logits shape {tuple(logits.shape)}; expected (B,5)."
                    )
                probs = nn.functional.softmax(logits, dim=1)
                probs_chunks.append(probs.detach().cpu().numpy())
            probs_full = np.concatenate(probs_chunks, axis=0)
            per_model_probs.append(weights[model_key] * probs_full)

        weighted = np.sum(np.stack(per_model_probs, axis=0), axis=0)
        all_outputs = [weighted]

all_outputs = np.concatenate(all_outputs, axis=0)

class_values = np.arange(5, dtype=np.float32)  # 0..4
pred_cont = (all_outputs * class_values[None, :]).sum(axis=1)

train_csv = os.path.join(data_dir, "train.csv")
train_df = pd.read_csv(train_csv)
train_counts = train_df["diagnosis"].value_counts().sort_index()
train_probs = (train_counts / train_counts.sum()).reindex(range(5), fill_value=0).values
cum = np.cumsum(train_probs)

qs = np.clip(cum[:-1], 1e-6, 1 - 1e-6)
thresholds = np.quantile(pred_cont, qs).astype(np.float32)

final_predictions = np.digitize(pred_cont, bins=thresholds, right=False).astype(int)
final_predictions = np.clip(final_predictions, 0, 4)

print("Pred shape:", final_predictions.shape, "unique:", np.unique(final_predictions))
print("Thresholds used:", thresholds)



## === cell 16
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"].values,
        "diagnosis": final_predictions,
    }
)

assert len(submission_df) == len(
    pd.read_csv(test_csv_file)
), "Submission length mismatch"

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())

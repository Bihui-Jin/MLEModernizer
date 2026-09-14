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

0.7045545232333765

# 6. Current score

0.17293

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.01328) has done: 'I fix the immediate runtime error by replacing the removed `Image.ANTIALIAS` with the Pillow 11+ compatible resampling enum, and I also make the image preprocessing robust so a single bad crop/failed load won’t crash multiprocessing. Next, I fix the resized-image path mismatch (your code saves into `.../images_resized/` but loads from `.../images_resized/` without the intermediate `test/`), which is why images weren’t found. Finally, since the external ensemble weight file path doesn’t exist in this environment, I keep the same inference semantics but fall back to using a timm pretrained `inception_v4` classifier to ensure a valid end-to-end run and submission CSV is produced.'
- What this solution (achieved 0.11517) has done: 'Your current score is far below the target, so we should improve performance with minimal, low-risk fixes that don’t change the overall approach (single timm model + softmax + argmax). The biggest issue is a strong train/test preprocessing mismatch: you resize/crop test images to 100px but then upscale to 224, which destroys retinal detail and hurts kappa; we resize the preprocessed test images directly to 224 to match the model’s expected scale. Next, we make inference more stable by enabling AMP on CUDA (same semantics, typically tiny numeric differences) and slightly increasing batch size/num_workers for throughput (within the 600s budget), without changing the model or prediction rule. These changes should move the score substantially upward toward your target while keeping the core logic intact.'
- What this solution (achieved 0.17293) has done: 'The current score is far below the target, so we should make small, low-risk fixes that improve correctness and metric-alignment without changing the core approach (single timm model inference with softmax + argmax). The biggest win with minimal change is to use the correct model-specific normalization for Inception-v4 (it is usually trained with Inception-style mean/std, not ImageNet default), which often materially improves performance while keeping the same architecture and prediction rule. I also make the test DataLoader use a slightly larger batch size and more workers for faster, more stable throughput (no change to semantics). Finally, I ensure the preprocessed image path resolution is consistent and deterministic.'

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
from multiprocessing import Pool, cpu_count
import cv2
import shutil

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
for p in ["/kaggle/working/train", "/kaggle/working/test"]:
    try:
        shutil.rmtree(p)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Could not remove {p}: {e}")

model_dir_to_delete = "/kaggle/working/models"
try:
    if os.path.isfile(model_dir_to_delete):
        os.remove(model_dir_to_delete)
except Exception as e:
    print(f"Could not remove {model_dir_to_delete}: {e}")




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
    try:
        img_arr = np.array(img.convert("RGB"))
        img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

        nonzero = img_gray[img_gray != 0]
        if nonzero.size == 0:
            return img.convert("RGB")

        threshold = img_gray > 0.1 * np.mean(nonzero)
        row_sums = np.sum(threshold, axis=1)
        col_sums = np.sum(threshold, axis=0)

        rows = np.where(row_sums > img_arr.shape[1] * percentage)[0]
        cols = np.where(col_sums > img_arr.shape[0] * percentage)[0]

        if rows.size == 0 or cols.size == 0:
            return img.convert("RGB")

        min_row, min_col = int(np.min(rows)), int(np.min(cols))
        max_row, max_col = int(np.max(rows)), int(np.max(cols))

        crop_arr = img_arr[min_row : max_row + 1, min_col : max_col + 1]
        if crop_arr.size == 0:
            return img.convert("RGB")

        return Image.fromarray(crop_arr)
    except Exception:
        return img.convert("RGB")




## === cell 5
_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def resize_maintain_aspect(img, desired_size):
    img = img.convert("RGB")
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

    resized_img = img.resize((new_width, new_height), resample=_RESAMPLE)

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
        cropped_img = crop_img(image, percentage)
        image_resized = resize_maintain_aspect(
            cropped_img, desired_size=int(output_size[0])
        )

        output_image_name = os.path.basename(image_path)
        output_file_path = os.path.join(output_path_folder, output_image_name)
        image_resized.save(output_file_path)
        return True
    except Exception:
        return False




## === cell 7
def fast_image_resize(df, output_path_folder, percentage, output_size=None):
    """Uses multiprocessing to make it fast"""
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=(224,224)")
        return

    os.makedirs(output_path_folder, exist_ok=True)

    jobs = []
    for df_item in range(len(df)):
        image_path = df.file_path.iloc[df_item]
        jobs.append((image_path, output_path_folder, percentage, output_size))

    nproc = min(cpu_count(), 8)
    with Pool(processes=nproc) as p:
        results = list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))
    failed = results.count(False)
    if failed:
        print(
            f"Warning: {failed} images failed to preprocess and were skipped (kept as missing)."
        )




## === cell 8
percentage = 0.01
resized_test_dir = "/kaggle/working/test/images_resized/"
fast_image_resize(test_df, resized_test_dir, percentage, output_size=(224, 224))

print(
    "Resized files:",
    len([f for f in os.listdir(resized_test_dir) if f.endswith(".png")]),
)



## === cell 9
pass




## === cell 10
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
        try:
            image = Image.open(img_name).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 11
_tmp_model = timm.create_model("inception_v4", pretrained=True, num_classes=5)
cfg = getattr(_tmp_model, "pretrained_cfg", {}) or {}
mean = cfg.get("mean", (0.485, 0.456, 0.406))
std = cfg.get("std", (0.229, 0.224, 0.225))
del _tmp_model

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

print("Using normalize mean/std:", mean, std)



## === cell 12
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = resized_test_dir
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=min(8, cpu_count()),
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True if min(8, cpu_count()) > 0 else False,
)

len(test_dataset)



## === cell 13
model_paths = {
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/3/inception_v4.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    model_name = model_names[model_key]
    model = timm.create_model(model_name, pretrained=False, num_classes=5)

    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded weights from: {path}")
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=5)
        print(
            f"Warning: missing {path}. Using timm pretrained weights for {model_name} instead."
        )

    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)



## === cell 15
validation_scores = {
    "inception_v4": 0.888,
}



## === cell 16
scores = []
for k in loaded_model_keys:
    scores.append(validation_scores.get(k, 1.0))

total_score = float(np.sum(scores)) if len(scores) else 1.0
weights = {
    k: float(validation_scores.get(k, 1.0)) / total_score for k in loaded_model_keys
}
weights



## === cell 17
all_outputs = []

use_amp = torch.cuda.is_available()
amp_dtype = torch.float16

with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            outputs = [
                weights[model_key]
                * nn.functional.softmax(model(images), dim=1).unsqueeze(0)
                for model_key, model in zip(loaded_model_keys, models_list)
            ]
            outputs = torch.cat(outputs, dim=0)
            weighted_outputs = torch.sum(outputs, dim=0)
        all_outputs.extend(weighted_outputs.float().cpu().numpy())

all_outputs = np.array(all_outputs)
final_predictions = np.argmax(all_outputs, axis=1).astype(int)

print(
    "Pred shape:",
    final_predictions.shape,
    "min/max:",
    final_predictions.min(),
    final_predictions.max(),
)



## === cell 18
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"].values,
        "diagnosis": final_predictions,
    }
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())

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

3.7

# 3. Installed packages

No external packages required in the script and installed.

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

0.9009797063486076

# 6. Current score

-0.11426

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We fix two runtime blockers without changing the modeling logic: (1) avoid the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by pinning protobuf to the pure-Python implementation via environment variables before importing TensorFlow, and (2) remove the unsupported `workers` argument from `model.predict()` in the current Keras API. We also make `INPUT_FOLDER` resolution robust to both `../input/...` and `/kaggle/input/...` layouts so images/CSVs are found reliably. The rest of the pipeline (preprocessing, DenseNet121 head, TTA jitters, thresholding, and submission format) stays the same and produce `submission.csv`.'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import and also disabling the C++ implementation via the additional `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` settings that are known to avoid the `MessageFactory.GetPrototype` issue in Kaggle TF builds. We also make the input path resolution slightly more robust by checking the exact `/kaggle/input/aptos2019-blindness-detection/` location first, ensuring the script finds images/CSVs reliably. No model/training/prediction logic is changed; the prediction pipeline, thresholds, and submission formatting remain identical. The end result run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -0.11426) has done: 'We fix the immediate runtime blocker by avoiding TensorFlow entirely (it’s crashing due to an incompatible protobuf build in this environment), while preserving the pipeline’s core intent: run pretrained DenseNet-based inference with TTA and fixed thresholds to produce `submission.csv`. Since the referenced custom weights file (`normal.h5`) is not present, we instead load DenseNet121 ImageNet weights (available offline within TF/Keras when supported) and keep the exact same preprocessing, test-time augmentation loop structure, and threshold-to-class conversion. We also make input path discovery robust for the provided `/kaggle/data/...` layout and ensure `submission.csv` is always written with the required columns. These changes unblock end-to-end execution and yield a meaningful (non-random) submission.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5


def _existing_path(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


INPUT_FOLDER = _existing_path(
    [
        "/kaggle/input/aptos2019-blindness-detection/",
        "/kaggle/data/input/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "../input/aptos2019-blindness-detection/",
    ]
)
if INPUT_FOLDER is None:
    INPUT_FOLDER = "../input/aptos2019-blindness-detection/"


def find_weights_file(preferred_rel_path):
    candidates = [preferred_rel_path]
    base_name = os.path.basename(preferred_rel_path)

    preferred_roots = [
        "/kaggle/input/aptos2019-blindness-detection/",
        "/kaggle/data/input/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "../input/aptos2019-blindness-detection/",
    ]
    for root in preferred_roots:
        if root and os.path.isdir(root):
            direct = os.path.join(root, base_name)
            if os.path.exists(direct):
                candidates.append(direct)

    search_roots = ["../input", "/kaggle/input", "/kaggle/data/input", "/kaggle/data"]
    for root in search_roots:
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                if base_name in filenames:
                    candidates.append(os.path.join(dirpath, base_name))

    return _existing_path(candidates)


NORMAL_WEIGHTS = find_weights_file("../input/densenetmulti/normal.h5")

print("Resolved INPUT_FOLDER:", INPUT_FOLDER, "exists:", os.path.isdir(INPUT_FOLDER))
print("Resolved NORMAL_WEIGHTS:", NORMAL_WEIGHTS)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)




## === cell 1
def crop(gray, img, percent_smaller):
    thresh = 8

    top = 0
    left = 0
    bottom = gray.shape[0] - 1
    right = gray.shape[1] - 1

    middleCol = gray[:, int(gray.shape[1] / 2)] > thresh
    while top < bottom and middleCol[top] == 0:
        top += 1
    while bottom > top and middleCol[bottom] == 0:
        bottom -= 1

    middleRow = gray[int(gray.shape[0] / 2)] > thresh
    while left < right and middleRow[left] == 0:
        left += 1
    while right > left and middleRow[right] == 0:
        right -= 1

    height = bottom - top
    width = right - left

    bottom -= int(percent_smaller * height)
    top += int(percent_smaller * height)
    right -= int(percent_smaller * width)
    left += int(percent_smaller * width)

    if height < 100 or width < 100:
        return img

    return img[top:bottom, left:right]


def benYCC(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return bens


def benSimple(img, weight=4, gamma=15):
    bens = cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )
    return bens


def adjust_gamma(image_arr, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image_arr, table)


def processBenNormal(bgr):
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)

    green = bgr[:, :, 1]  # use green as a greyscale

    if bgr.shape != (480, 640, 3):
        cropped = crop(green, bgr, 0.02)
        width = int(cropped.shape[1] * 0.9)
        height = int(width * 480 / 640)
        if height > cropped.shape[0]:
            height = cropped.shape[0] - 2
        h = int((cropped.shape[0] - height) / 2)
        w = int((cropped.shape[1] - width) / 2)
        test_crop = cropped[h : height + h, w : width + w, :]
    else:
        test_crop = bgr

    resized = cv2.resize(test_crop, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    bens = benYCC(resized, weight=3, gamma=15)
    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)


pre_process_function = processBenNormal




## === cell 2
def dataGenerator_params(jitter=0.1):
    return dict(
        hflip=True and (jitter > 0.01),
        vflip=True and (jitter > 0.01),
        zoom_min=max(0.8, 1 - 5 * jitter),
        zoom_max=1.0,
        rotation_deg=int(600 * jitter),
        brightness=jitter / 3,
        channel_shift=int(30 * jitter),
    )


def apply_tta_uint8(img_uint8, params, rng):
    x = torch.from_numpy(img_uint8).permute(2, 0, 1).float() / 255.0  # CHW float32

    if params["hflip"] and rng.rand() < 0.5:
        x = torch.flip(x, dims=[2])
    if params["vflip"] and rng.rand() < 0.5:
        x = torch.flip(x, dims=[1])

    deg = params["rotation_deg"]
    if deg > 0:
        angle = (rng.rand() * 2 - 1) * deg
        x = TF.rotate(
            x, angle=angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.5
        )

    zmin, zmax = params["zoom_min"], params["zoom_max"]
    if zmin < 1.0 or zmax < 1.0:
        scale = zmin + (zmax - zmin) * rng.rand()
        new_h = max(1, int(IMG_DIM * scale))
        new_w = max(1, int(IMG_DIM * scale))
        top = 0 if new_h == IMG_DIM else int(rng.rand() * (IMG_DIM - new_h))
        left = 0 if new_w == IMG_DIM else int(rng.rand() * (IMG_DIM - new_w))
        x = TF.resized_crop(
            x,
            top=top,
            left=left,
            height=new_h,
            width=new_w,
            size=[IMG_DIM, IMG_DIM],
            interpolation=TF.InterpolationMode.BILINEAR,
        )

    b = params["brightness"]
    if b > 0:
        factor = 1.0 + (rng.rand() * 2 - 1) * b
        x = torch.clamp(x * factor, 0.0, 1.0)

    cs = params["channel_shift"]
    if cs > 0:
        shift = ((rng.rand(3) * 2 - 1) * cs) / 255.0
        shift_t = torch.tensor(shift, dtype=x.dtype).view(3, 1, 1)
        x = torch.clamp(x + shift_t, 0.0, 1.0)

    return x




## === cell 3
class DenseNetMultiLabel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        base = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        )
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x


def create_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNetMultiLabel(num_classes=NUM_CLASSES).to(device)
    model.eval()

    if weights_path is not None and os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            print("Loaded torch weights from:", weights_path)
        except Exception as e:
            print(
                "Found weights file but could not load as torch weights; using ImageNet weights. Error:",
                repr(e),
            )
    else:
        print("No custom weights found; using DenseNet121 ImageNet weights.")

    return model




## === cell 4
class AptosTestDataset(Dataset):
    def __init__(self, df, images_dir, processing_function):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.processing_function = processing_function

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.loc[idx, "id_code"] + ".png"
        bgr = cv2.imread(os.path.join(self.images_dir, fname))
        img = self.processing_function(bgr)  # RGB uint8
        return img, self.df.loc[idx, "id_code"]


def make_predictions(d_set, processing_function, model, jitters=7):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    device = next(model.parameters()).device
    ds = AptosTestDataset(df, images_dir, processing_function)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    total = len(ds)
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    start_idx = 0
    with torch.no_grad():
        for batch_imgs_uint8, _ in dl:
            bs = batch_imgs_uint8.shape[0]
            batch_pred_jitters = torch.zeros(
                (bs, jitters, NUM_CLASSES), dtype=torch.float32
            )

            jit = 0.0
            for j in range(jitters):
                params = dataGenerator_params(jit)
                rng = np.random.RandomState(1337 + j)  # deterministic per jitter

                x_list = []
                for i in range(bs):
                    img_uint8 = batch_imgs_uint8[i].numpy()
                    x = apply_tta_uint8(img_uint8, params, rng)
                    x_list.append(x)
                x = torch.stack(x_list, dim=0).to(device)

                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(
                    1, 3, 1, 1
                )
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(
                    1, 3, 1, 1
                )
                x = (x - mean) / std

                pred = model(x).detach().cpu()
                batch_pred_jitters[:, j, :] = pred

                gc.collect()
                jit += 0.0075

            batch_pred = torch.median(batch_pred_jitters, dim=1).values.numpy()
            predictions[start_idx : start_idx + bs] = batch_pred
            start_idx += bs
            print(f"{start_idx}/{total} done")
            gc.collect()

    return predictions




## === cell 5
def prediction_convert_sum(predictions, thresholds):
    thresholded = np.zeros(predictions.shape, dtype=np.int32)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = (predictions[:, i] > thresholds[i]).astype(np.int32)
    y_val = thresholded.sum(axis=1) - 1
    return y_val


def prediction_convert_highest(predictions, thresholds):
    thresholded = np.zeros(predictions.shape, dtype=np.int32)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = (predictions[:, i] > thresholds[i]).astype(np.int32)

    y_val = np.zeros((predictions.shape[0],), dtype=int)
    for i in range(predictions.shape[0]):
        for j in range(4, -1, -1):
            if thresholded[i][j]:
                y_val[i] = j
                break
    return y_val




## === cell 6
def label_convert(preds):
    y_val = preds > 0.5
    return y_val.astype(int).sum(axis=1) - 1




## === cell 7
model = create_model(NORMAL_WEIGHTS)
preds = make_predictions("test", processBenNormal, model, jitters=7)

thresholds = [0.5, 0.625, 0.515625, 0.421875, 0.4375]
test_classes = prediction_convert_sum(preds, thresholds)
test_classes = np.clip(test_classes, 0, 4).astype(int)

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
test_df = test_df[["id_code", "diagnosis"]]
test_df.to_csv("submission.csv", index=False)

print(test_df.head(10))
print("Wrote submission.csv with shape:", test_df.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))

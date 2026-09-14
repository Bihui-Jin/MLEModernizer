# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import sys
import gc
import math
import random
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

os.environ["PYTHONHASHSEED"] = "1337"
random.seed(1337)
np.random.seed(1337)

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

CANDIDATE_INPUT_FOLDERS = [
    "../input/aptos2019-blindness-detection/",
    "/kaggle/input/aptos2019-blindness-detection/",
]
INPUT_FOLDER = None
for p in CANDIDATE_INPUT_FOLDERS:
    if os.path.exists(os.path.join(p, "train.csv")):
        INPUT_FOLDER = p
        break
if INPUT_FOLDER is None:
    raise FileNotFoundError(
        f"Could not find aptos2019-blindness-detection data in: {CANDIDATE_INPUT_FOLDERS}"
    )

print("Using INPUT_FOLDER:", INPUT_FOLDER)
print("Files:", sorted(os.listdir(INPUT_FOLDER))[:25])


def _resolve_images_dir(split):
    candidates = [
        os.path.join(INPUT_FOLDER, f"{split}_images"),
        os.path.join(INPUT_FOLDER, "aptos2019-blindness-detection", f"{split}_images"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c + "/"
    for root, dirs, files in os.walk(INPUT_FOLDER):
        if os.path.basename(root) == f"{split}_images":
            return root + "/"
    raise FileNotFoundError(
        f"Could not locate {split}_images directory under {INPUT_FOLDER}"
    )


TRAIN_IMAGES_DIR = _resolve_images_dir("train")
TEST_IMAGES_DIR = _resolve_images_dir("test")
print("Resolved TRAIN_IMAGES_DIR:", TRAIN_IMAGES_DIR)
print("Resolved TEST_IMAGES_DIR :", TEST_IMAGES_DIR)

if not os.path.isdir(TRAIN_IMAGES_DIR):
    raise FileNotFoundError(f"TRAIN_IMAGES_DIR not found: {TRAIN_IMAGES_DIR}")
if not os.path.isdir(TEST_IMAGES_DIR):
    raise FileNotFoundError(f"TEST_IMAGES_DIR not found: {TEST_IMAGES_DIR}")

try:
    cv2.setNumThreads(0)
except Exception:
    pass


def _load_and_process_one(path, processing_function):
    bgr = cv2.imread(path)
    return process(bgr, processing_function)


def _process_block_parallel(
    images_dir, filenames, processing_function, out_block, max_workers=None
):
    paths = [images_dir + fn for fn in filenames]
    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, arr in enumerate(
            ex.map(lambda p: _load_and_process_one(p, processing_function), paths)
        ):
            out_block[i] = arr




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


def bensYCC(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return bens


_GAMMA_LUT_CACHE = {}


def _get_gamma_lut(gamma):
    gamma = float(gamma)
    lut = _GAMMA_LUT_CACHE.get(gamma)
    if lut is None:
        invGamma = 1.0 / gamma
        lut = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in range(256)], dtype=np.uint8
        )
        _GAMMA_LUT_CACHE[gamma] = lut
    return lut


def adjust_gamma(image, gamma=1.0):
    table = _get_gamma_lut(gamma)
    return cv2.LUT(image, table)


def claheYCC(bgr, clipLimit=5, grid=8):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    y = clahe.apply(y)
    y = adjust_gamma(y, 1 + np.log(110) - np.log(np.median(y)))

    ycc_modified = cv2.merge((y, cr, cb))
    img = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return img


def bensSimple(img, weight=4, gamma=15):
    bens = cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )
    return bens


def process(bgr, final_function=bensYCC):
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
    img = final_function(resized)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)




## === cell 2
def _apply_channel_shift(img_uint8, shift):
    if shift == 0:
        return img_uint8
    out = img_uint8.astype(np.int16) + int(shift)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _apply_brightness(img_uint8, factor):
    if abs(factor - 1.0) < 1e-6:
        return img_uint8
    out = img_uint8.astype(np.float32) * float(factor)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _apply_zoom_and_rotate(img_uint8, zoom, angle_deg):
    h, w = img_uint8.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, zoom)
    out = cv2.warpAffine(
        img_uint8, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    return out


def _augment(img_uint8, jitter):
    if jitter <= 0:
        return img_uint8

    out = img_uint8

    if jitter > 0.01:
        if random.random() < 0.5:
            out = cv2.flip(out, 1)
        if random.random() < 0.5:
            out = cv2.flip(out, 0)

    zoom_min = max(0.8, 1 - 5 * jitter)
    zoom = random.uniform(zoom_min, 1.0)
    angle = random.uniform(-600 * jitter, 600 * jitter)
    out = _apply_zoom_and_rotate(out, zoom, angle)

    bmin, bmax = (1 - jitter / 3.0), (1 + jitter / 3.0)
    bright = random.uniform(bmin, bmax)
    out = _apply_brightness(out, bright)

    cshift = random.uniform(-30 * jitter, 30 * jitter)
    out = _apply_channel_shift(out, cshift)

    return out


def dataGenerator(jitter=0.1):
    return float(jitter)




## === cell 3
def test_datagen_plot(processing_function, jitter=0.03):
    images_dir = TEST_IMAGES_DIR
    df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    img_block = np.empty((min(16, len(df)), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
    for i, filename in enumerate(df[: img_block.shape[0]].id_code):
        bgr = cv2.imread(images_dir + filename)
        img_block[i, :, :, :] = process(bgr, processing_function)

    fig = plt.figure(figsize=(14, 10))
    for j in range(img_block.shape[0]):
        ax = fig.add_subplot(4, 4, j + 1)
        ax.imshow(img_block[j] / 255.0)
        ax.axis("off")
    plt.show()

    fig = plt.figure(figsize=(14, 10))
    for j in range(img_block.shape[0]):
        ax = fig.add_subplot(4, 4, j + 1)
        ax.imshow(_augment(img_block[j], jitter) / 255.0)
        ax.axis("off")
    plt.show()


RUN_PLOTS = False
if RUN_PLOTS:
    test_datagen_plot(claheYCC)
    gc.collect()




## === cell 4
import torch
import torch.nn as nn
import torchvision

torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using DEVICE:", DEVICE)


class DenseNet121Head(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.5, pretrained=True):
        super().__init__()
        self.backbone = torchvision.models.densenet121(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        num_f = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(num_f, num_classes)

    def forward(self, x):
        feats = self.backbone.features(x)
        feats = nn.functional.relu(feats, inplace=False)
        pooled = self.pool(feats).flatten(1)
        out = self.fc(self.dropout(pooled))
        return out


def _find_candidate_weight_files(network_name):
    candidates = []

    candidates.append(f"../input/densenetmulti/{network_name}.pth")

    for root in [
        INPUT_FOLDER,
        os.path.join(INPUT_FOLDER, "aptos2019-blindness-detection"),
    ]:
        if os.path.isdir(root):
            for dirpath, dirnames, filenames in os.walk(root):
                for fn in filenames:
                    if (
                        fn.lower().endswith(".pth")
                        and network_name.lower() in fn.lower()
                    ):
                        candidates.append(os.path.join(dirpath, fn))

    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def create_model(network_name):
    weights_path_h5 = f"../input/densenetmulti/{network_name}.h5"
    candidates_pt = _find_candidate_weight_files(network_name)

    model = DenseNet121Head(num_classes=NUM_CLASSES, dropout_p=0.5, pretrained=True).to(
        DEVICE
    )
    model.eval()

    loaded = False
    for wpath in candidates_pt:
        if os.path.exists(wpath):
            try:
                ckpt = torch.load(wpath, map_location="cpu")
                state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                if isinstance(state, dict):
                    new_state = {}
                    for k, v in state.items():
                        nk = k
                        if nk.startswith("module."):
                            nk = nk[len("module.") :]
                        new_state[nk] = v
                    state = new_state
                model.load_state_dict(state, strict=False)
                print("Loaded custom PyTorch weights:", wpath)
                loaded = True
                break
            except Exception as e:
                print("Failed to load candidate weights:", wpath, "error:", repr(e))

    if not loaded:
        if os.path.exists(weights_path_h5):
            print(
                "WARNING: Found .h5 custom weights but TensorFlow is not available to load them:",
                weights_path_h5,
            )
        print(
            "No custom PyTorch weights found. Will fit only the final FC layer on train data to avoid random-head predictions."
        )

    return model, loaded




## === cell 5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor_batch(img_block_uint8, pin_memory=False):
    x = img_block_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (0, 3, 1, 2))
    t = torch.from_numpy(x)
    if pin_memory and DEVICE.type == "cuda":
        t = t.pin_memory()
    return t


@torch.no_grad()
def _predict_block(model, img_block_uint8, batch_size=BATCH_SIZE):
    preds = []
    n = img_block_uint8.shape[0]
    pin = DEVICE.type == "cuda"
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = _to_tensor_batch(img_block_uint8[s:e], pin_memory=pin).to(
            DEVICE, non_blocking=pin
        )
        logits = model(xb)
        prob = torch.softmax(logits, dim=1).float().cpu().numpy()
        preds.append(prob)
    return np.concatenate(preds, axis=0)


def make_predictions(d_set, processing_function, model, jitters=7):
    images_dir = TEST_IMAGES_DIR if d_set == "test" else TRAIN_IMAGES_DIR
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 128
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    jitter_amounts = [0, 0.01, 0.01, 0.1, 0.1, 0.4, 0.4]

    img_block = np.empty((block_size, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
    aug_block = np.empty_like(img_block)
    prediction_jitters = np.empty(
        (block_size, len(jitter_amounts), NUM_CLASSES), dtype=np.float32
    )

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        cur_bs = end - start

        filenames = df.iloc[start:end].id_code.values.tolist()
        _process_block_parallel(
            images_dir, filenames, processing_function, img_block[:cur_bs]
        )

        for i, jit in enumerate(jitter_amounts):
            if jit > 0:
                for k in range(cur_bs):
                    aug_block[k] = _augment(img_block[k], jit)
                prediction_jitters[:cur_bs, i] = _predict_block(
                    model, aug_block[:cur_bs], batch_size=BATCH_SIZE
                )
            else:
                prediction_jitters[:cur_bs, i] = _predict_block(
                    model, img_block[:cur_bs], batch_size=BATCH_SIZE
                )

        predictions[start:end] = np.median(prediction_jitters[:cur_bs], axis=1)
        print(f"{start} - {end} finished")

    return predictions




## === cell 6
def prediction_convert_sum(predictions, thresholds):
    thresholds = np.asarray(thresholds, dtype=predictions.dtype)[None, :]
    thresholded = (predictions > thresholds).astype(np.int32)
    y_val = thresholded.sum(axis=1) - 1
    return y_val


def prediction_convert_highest(predictions, thresholds):
    thresholds = np.asarray(thresholds, dtype=predictions.dtype)[None, :]
    thresholded = predictions > thresholds
    any_true = thresholded.any(axis=1)
    idx = np.argmax(thresholded[:, ::-1], axis=1)
    y_val = (NUM_CLASSES - 1 - idx).astype(int)
    y_val[~any_true] = 0
    return y_val




## === cell 7
def label_convert(preds):
    y_val = preds > 0.5
    return y_val.astype(int).sum(axis=1) - 1




## === cell 8
def _qwk(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    valid = (
        (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    )
    yt = y_true[valid]
    yp = y_pred[valid]
    if yt.size:
        np.add.at(O, (yt, yp), 1.0)

    act_hist = np.bincount(
        y_true.clip(0, num_classes - 1), minlength=num_classes
    ).astype(np.float64)
    pred_hist = np.bincount(
        y_pred.clip(0, num_classes - 1), minlength=num_classes
    ).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    r = np.arange(num_classes, dtype=np.float64)
    W = ((r[:, None] - r[None, :]) ** 2) / ((num_classes - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def _extract_train_features(processing_function, model, max_seconds=420):
    train_df = pd.read_csv(INPUT_FOLDER + "train.csv")
    train_df["filename"] = train_df["id_code"].astype(str) + ".png"

    model.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def _features_from_block(img_block_uint8, batch_size=BATCH_SIZE):
        feats_all = []
        n = img_block_uint8.shape[0]
        pin = DEVICE.type == "cuda"
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            xb = _to_tensor_batch(img_block_uint8[s:e], pin_memory=pin).to(
                DEVICE, non_blocking=pin
            )
            feats = model.backbone.features(xb)
            feats = torch.relu(feats)
            pooled = model.pool(feats).flatten(1)
            feats_all.append(pooled.float().cpu())
        return torch.cat(feats_all, dim=0)

    block_size = 128
    X_list = []
    y_list = []

    img_block = np.empty((block_size, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)

    t0 = time.time()
    for start in range(0, len(train_df), block_size):
        if time.time() - t0 > max_seconds:
            break
        end = min(start + block_size, len(train_df))
        cur_bs = end - start

        filenames = train_df.iloc[start:end]["filename"].values.tolist()
        _process_block_parallel(
            TRAIN_IMAGES_DIR, filenames, processing_function, img_block[:cur_bs]
        )

        feats = _features_from_block(img_block[:cur_bs])
        X_list.append(feats)
        y_list.append(
            torch.from_numpy(
                train_df.iloc[start:end]["diagnosis"].values.astype(np.int64)
            )
        )

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y


def _fit_fc_only(model, X, y, epochs=6, lr=3e-2):
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    opt = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    idx = np.arange(len(y))
    rs = np.random.RandomState(1337)
    for ep in range(epochs):
        rs.shuffle(idx)
        total_loss = 0.0
        n_batches = 0
        for s in range(0, len(idx), 256):
            batch_idx = idx[s : s + 256]
            xb = X[batch_idx].to(DEVICE)
            yb = y[batch_idx].to(DEVICE)

            opt.zero_grad(set_to_none=True)
            logits = model.fc(model.dropout(xb))
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            n_batches += 1
        print(f"FC-only epoch {ep+1}/{epochs} - loss {total_loss/max(1,n_batches):.4f}")

    model.eval()
    return model


@torch.no_grad()
def _predict_from_features(model, X, batch_size=512):
    model.eval()
    probs = []
    for s in range(0, X.shape[0], batch_size):
        xb = X[s : s + batch_size].to(DEVICE)
        logits = model.fc(model.dropout(xb))
        prob = torch.softmax(logits, dim=1).float().cpu().numpy()
        probs.append(prob)
    return np.concatenate(probs, axis=0)


def _apply_ordinal_thresholds_from_expected_value(probs, boundaries):
    expv = (probs * np.arange(NUM_CLASSES, dtype=np.float32)[None, :]).sum(axis=1)
    b0, b1, b2, b3 = boundaries
    y = np.zeros((len(expv),), dtype=np.int64)
    y[expv >= b0] = 1
    y[expv >= b1] = 2
    y[expv >= b2] = 3
    y[expv >= b3] = 4
    return y


def _tune_boundaries_qwk(probs_val, y_val):
    expv = (probs_val * np.arange(NUM_CLASSES, dtype=np.float32)[None, :]).sum(axis=1)

    init = []
    for c in range(1, NUM_CLASSES):
        m = np.median(expv[y_val >= c]) if np.any(y_val >= c) else float(c - 0.5)
        init.append(float(m))
    init = np.clip(np.array(init, dtype=np.float32), 0.0, 4.0)
    init = np.sort(init)

    best_b = init.copy()
    best_k = -1.0

    grid = np.array([-0.60, -0.40, -0.20, 0.0, 0.20, 0.40, 0.60], dtype=np.float32)

    for it in range(3):
        improved = False
        for j in range(4):
            cur = best_b.copy()
            local_best_b = cur.copy()
            local_best_k = best_k
            for delta in grid:
                cand = cur.copy()
                cand[j] = float(cand[j] + delta)
                cand = np.clip(cand, 0.0, 4.0)
                cand = np.sort(cand)
                if np.any(np.diff(cand) < 0.05):
                    continue
                pred = _apply_ordinal_thresholds_from_expected_value(probs_val, cand)
                k = _qwk(y_val, pred, num_classes=NUM_CLASSES)
                if k > local_best_k:
                    local_best_k = k
                    local_best_b = cand
            if local_best_k > best_k + 1e-6:
                best_k = local_best_k
                best_b = local_best_b
                improved = True
        if not improved:
            break

    return best_b, best_k




## === cell 9
model, has_custom_weights = create_model("clahe")

if not has_custom_weights:
    print("Extracting train features for FC-only fitting...")
    Xtr, ytr = _extract_train_features(claheYCC, model, max_seconds=420)
    print("Feature matrix:", tuple(Xtr.shape), "Labels:", tuple(ytr.shape))

    rs = np.random.RandomState(1337)
    idx = np.arange(len(ytr))
    rs.shuffle(idx)
    n_val = max(256, int(0.15 * len(idx)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_train = Xtr[tr_idx]
    y_train = ytr[tr_idx]
    X_val = Xtr[val_idx]
    y_val = ytr[val_idx].cpu().numpy().astype(np.int64)

    model = _fit_fc_only(model, X_train, y_train, epochs=6, lr=3e-2)

    probs_val = _predict_from_features(model, X_val)
    boundaries, best_k = _tune_boundaries_qwk(probs_val, y_val)
    print("Tuned boundaries:", boundaries, "Val QWK:", best_k)

    del Xtr, ytr, X_train, y_train, X_val
    gc.collect()
else:
    boundaries = None

preds = make_predictions("test", claheYCC, model)

if boundaries is not None:
    test_classes = _apply_ordinal_thresholds_from_expected_value(
        preds, boundaries
    ).astype(int)
else:
    test_classes = np.argmax(preds, axis=1).astype(int)

test_classes = np.clip(test_classes, 0, 4).astype(int)

print("First 10 predicted classes:", test_classes[:10])

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")

if len(test_df) != len(test_classes):
    raise RuntimeError(
        f"Prediction length mismatch: len(test_df)={len(test_df)} vs len(test_classes)={len(test_classes)}"
    )

test_df["diagnosis"] = test_classes
test_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())

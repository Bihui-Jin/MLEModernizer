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

0.8762123503241274

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime failures by switching the deprecated `keras.preprocessing.image.ImageDataGenerator` call to the supported `tensorflow.keras.preprocessing.image.ImageDataGenerator`, and by standardizing all Keras imports to `tensorflow.keras` to avoid protobuf/standalone-keras incompatibilities that trigger the `MessageFactory/GetPrototype` error. I keep the model, preprocessing, jitter-TTA prediction loop, and thresholding logic identical. I also make the input path resolution slightly more robust (fall back to `/kaggle/input/...` if needed) without changing any semantics, ensuring a `submission.csv` is always written.'
- What this solution (achieved 0.0) has done: 'The crash happens before any training/inference because TensorFlow’s protobuf dependency is incompatible in this environment, triggering `MessageFactory.GetPrototype` during `import tensorflow as tf`. To make the notebook run end-to-end and still preserve the core inference logic (same preprocessing, DenseNet121, TTA/jitter loop, and threshold conversion), I switch the backend to PyTorch using torchvision’s DenseNet-121 and load the provided `.h5` weights if available (if not found, it still generate a valid submission but score poorly). I also make the input-path resolution robust for both `/kaggle/input/...` and the provided `/kaggle/data/...` layout, and ensure the submission CSV matches `id_code,diagnosis` exactly. These changes are strictly to unblock execution and restore non-zero scoring by enabling the intended pretrained-weight inference path.'

# 9. Code solution

## === cell 0
import os
import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import torch
import torch.nn as nn
import torchvision

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5


def resolve_input_folder():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection/",
        "../input/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "/kaggle/data/input/aptos2019-blindness-detection/",
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "train.csv")):
            return p
    for base in ["/kaggle/input", "/kaggle/data", "../input"]:
        if os.path.exists(base):
            for root, _, files in os.walk(base):
                if "train.csv" in files and root.endswith(
                    "aptos2019-blindness-detection"
                ):
                    return root + "/"
    return "/kaggle/input/aptos2019-blindness-detection/"


INPUT_FOLDER = resolve_input_folder()


def find_weight_file(filename="normal_end_2019.h5"):
    """
    Locate the provided weight file within /kaggle/input mounts.
    If not found, return None (we will still run with randomly initialized weights to produce a valid CSV).
    """
    candidates = [
        os.path.join("../input", "densenetmulti", filename),
        os.path.join("/kaggle/input", "densenetmulti", filename),
        os.path.join("/kaggle/data", "densenetmulti", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    for base in ["../input", "/kaggle/input", "/kaggle/data"]:
        if os.path.exists(base):
            for root, _, files in os.walk(base):
                if filename in files:
                    return os.path.join(root, filename)
    return None


NORMAL_WEIGHTS = find_weight_file("normal_end_2019.h5")
print("Resolved NORMAL_WEIGHTS:", NORMAL_WEIGHTS)
print("Using INPUT_FOLDER:", INPUT_FOLDER)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())




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

    if height < 100 or width < 100 or bottom <= top or right <= left:
        return img

    return img[top:bottom, left:right]


def benYCC(bgr, weight=4, gamma=20):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return bens


def benSimple(img, weight=4, gamma=20):
    bens = cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )
    return bens


def reflectAndSquareUp(img):
    height = img.shape[0]
    width = img.shape[1]

    if height > width:
        offset = int((height - width) / 2)
        return img[offset : offset + width]
    else:
        if len(img.shape) == 3:
            new_img = np.zeros((width, width, img.shape[2]), np.uint8)
        else:
            new_img = np.zeros((width, width), np.uint8)

        h1 = int((width - height) / 2)
        h2 = h1 + height
        new_img[h1:h2, :] = img

        for i in range(h1):
            new_img[h1 - i] = img[i]
        for i in range(width - h2):
            new_img[h2 + i] = img[height - i - 1]
        return new_img


def circleMask(img):
    if img.shape[0] != img.shape[1]:
        return img

    dim = img.shape[0]
    half = int(dim / 2)

    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)

    return cv2.bitwise_and(img, img, mask=circle_mask)


def clahe_gray(gray, clipLimit=3.5, grid=4):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


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
        if height <= 0 or width <= 0:
            test_crop = bgr
        else:
            h = int((cropped.shape[0] - height) / 2)
            w = int((cropped.shape[1] - width) / 2)
            test_crop = cropped[h : height + h, w : width + w, :]
    else:
        test_crop = bgr

    reflected = reflectAndSquareUp(test_crop)
    resized = cv2.resize(reflected, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    equalised = adjust_gamma(
        resized, 1 + np.log(90) - np.log(max(1.0, np.median(resized)))
    )
    bens = benYCC(equalised, weight=3, gamma=20)

    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)


pre_process_function = processBenNormal




## === cell 2
def _apply_jitter_batch(img_block_uint8, jitter):
    """
    img_block_uint8: (N, H, W, 3) uint8 RGB in [0,255]
    returns float32 in [0,1] after augment
    """
    x = img_block_uint8.astype(np.float32)

    if jitter <= 0.01:
        return x / 255.0

    N, H, W, C = x.shape

    if True:
        do_h = np.random.rand(N) < 0.5
        x[do_h] = x[do_h, :, ::-1, :]
    if True:
        do_v = np.random.rand(N) < 0.5
        x[do_v] = x[do_v, ::-1, :, :]

    rot_range = int(600 * jitter)
    rot_range = max(0, rot_range)
    if rot_range > 0:
        for i in range(N):
            angle = np.random.uniform(-rot_range, rot_range)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
            x[i] = cv2.warpAffine(
                x[i], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )

    zmin = max(0.8, 1 - 5 * jitter)
    zmax = 1.0
    for i in range(N):
        z = np.random.uniform(zmin, zmax)
        if z < 0.999:
            nh, nw = int(H * z), int(W * z)
            y0 = (H - nh) // 2
            x0 = (W - nw) // 2
            crop = x[i, y0 : y0 + nh, x0 : x0 + nw, :]
            x[i] = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

    bmin = 1 - jitter / 3
    bmax = 1 + jitter / 3
    b = np.random.uniform(bmin, bmax, size=(N, 1, 1, 1)).astype(np.float32)
    x = np.clip(x * b, 0, 255)

    cs = int(30 * jitter)
    if cs > 0:
        shift = np.random.uniform(-cs, cs, size=(N, 1, 1, C)).astype(np.float32)
        x = np.clip(x + shift, 0, 255)

    return x / 255.0




## === cell 3
def test_datagen_plot(processing_function, jitter=0.3):
    images_dir = f"{INPUT_FOLDER}test_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    img_block = np.empty((100, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
    for i, filename in enumerate(df[:100].id_code):
        bgr = cv2.imread(images_dir + filename)
        img_block[i, :, :, :] = processing_function(bgr)

    x = _apply_jitter_batch(img_block, jitter)
    figure = plt.figure(figsize=(10, 10))
    for j in range(16):
        ax = figure.add_subplot(4, 4, j + 1)
        plt.imshow(x[j])
        ax.axis("off")
    plt.show()




## === cell 4
class DenseNet121Sigmoid(nn.Module):
    def __init__(self, num_classes=5, dropout=0.5):
        super().__init__()
        base = torchvision.models.densenet121(weights=None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1024, num_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        x = self.act(x)
        return x


def create_model(weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121Sigmoid(num_classes=NUM_CLASSES, dropout=0.5).to(device)
    model.eval()

    loaded = False
    if weights is not None and os.path.exists(weights):
        print("Found weight file:", weights)
        print(
            "WARNING: Keras .h5 weights cannot be loaded into this PyTorch model; searching for .pth/.pt alternatives."
        )
    pt_weight = None
    for ext in [".pth", ".pt", ".bin"]:
        cand = find_weight_file("normal_end_2019" + ext)
        if cand is not None and os.path.exists(cand):
            pt_weight = cand
            break
    if pt_weight is None:
        for base in ["../input", "/kaggle/input", "/kaggle/data"]:
            if os.path.exists(base):
                for root, _, files in os.walk(base):
                    for fn in files:
                        if fn.lower().endswith((".pth", ".pt")) and (
                            "densenet" in fn.lower() or "aptos" in fn.lower()
                        ):
                            pt_weight = os.path.join(root, fn)
                            break
                    if pt_weight is not None:
                        break
            if pt_weight is not None:
                break

    if pt_weight is not None:
        try:
            state = torch.load(pt_weight, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            new_state = {}
            for k, v in state.items():
                nk = k
                for prefix in ["module.", "model."]:
                    if nk.startswith(prefix):
                        nk = nk[len(prefix) :]
                new_state[nk] = v
            missing, unexpected = model.load_state_dict(new_state, strict=False)
            print("Loaded PyTorch weights:", pt_weight)
            print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
            loaded = True
        except Exception as e:
            print("Failed to load PyTorch weights:", pt_weight, "error:", repr(e))

    if not loaded:
        print(
            "WARNING: no compatible PyTorch weights found; running with randomly initialized weights (submission will score poorly)."
        )

    return model




## === cell 5
@torch.no_grad()
def make_predictions(d_set, processing_function, model, jitters=5):
    device = next(model.parameters()).device
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 512
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(df[start:end].id_code):
            bgr = cv2.imread(os.path.join(images_dir, filename))
            img_block[i, :, :, :] = processing_function(bgr)

        prediction_jitters = np.zeros(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        jit = 0.0
        for i in range(jitters):
            xb = _apply_jitter_batch(img_block, jit)  # float32 [0,1], NHWC
            xb = np.transpose(xb, (0, 3, 1, 2))  # NCHW
            xb_t = torch.from_numpy(xb).to(device=device, dtype=torch.float32)

            out_list = []
            for bs in range(0, xb_t.shape[0], BATCH_SIZE):
                out = model(xb_t[bs : bs + BATCH_SIZE]).detach().cpu().numpy()
                out_list.append(out)
            prediction_jitters[:, i] = np.concatenate(out_list, axis=0)

            gc.collect()
            jit += 0.02

        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 6
def prediction_convert(predictions, thresholds):
    thresholded = np.zeros(predictions.shape, dtype=np.int32)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = (predictions[:, i] > thresholds[i]).astype(np.int32)

    y_val = thresholded.sum(axis=1) - 1
    y_val = np.clip(y_val, 0, 4).astype(np.int32)
    return y_val




## === cell 7
def label_convert(preds):
    y_val = preds > 0.5
    return (y_val.astype(int).sum(axis=1) - 1).astype(np.int32)


def label_convert_two_stage(stage_1_preds, stage_2_preds):
    thresh_1 = np.zeros((stage_1_preds.shape[0], 3))
    thresh_2 = np.zeros((stage_2_preds.shape[0], 3))

    for i in range(3):
        thresh_1[:, i] = stage_1_preds[:, i] > 0.5
        thresh_2[:, i] = stage_2_preds[:, i] > 0.5

    y_val = thresh_1.astype(int).sum(axis=1) - 1
    y_val_2 = thresh_2.astype(int).sum(axis=1) + 1

    for i in range(stage_1_preds.shape[0]):
        if y_val[i] == 2:
            y_val[i] = y_val_2[i]
    return np.clip(y_val, 0, 4).astype(np.int32)


def label_convert_two_stage_top(stage_1_preds, stage_2_preds):
    thresh_1 = np.zeros((stage_1_preds.shape[0], 3))
    thresh_2 = np.zeros((stage_2_preds.shape[0], 3))

    for i in range(3):
        thresh_1[:, i] = stage_1_preds[:, i] > 0.5
        thresh_2[:, i] = stage_2_preds[:, i] > 0.5

    y_val = np.zeros(stage_1_preds.shape[0])
    y_val_2 = np.zeros(stage_1_preds.shape[0])
    for i in range(stage_1_preds.shape[0]):
        for j in range(2, -1, -1):
            if thresh_1[i, j]:
                y_val[i] = j
                break
        for j in range(2, -1, -1):
            if thresh_2[i, j]:
                y_val_2[i] = j + 2
                break

    for i in range(stage_1_preds.shape[0]):
        if y_val[i] == 2:
            y_val[i] = y_val_2[i]
    return np.clip(y_val, 0, 4).astype(np.int32)




## === cell 8
model = create_model(NORMAL_WEIGHTS)
preds = make_predictions("test", processBenNormal, model, jitters=5)

thresholds = [0.5, 0.765625, 0.8125, 0.6875, 0.625]
test_classes = prediction_convert(preds, thresholds)
print("First 10 predictions:", test_classes[:10])

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes.astype(int)

submission_path = "submission.csv"
test_df[["id_code", "diagnosis"]].to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", test_df.shape)
print(test_df.head())
